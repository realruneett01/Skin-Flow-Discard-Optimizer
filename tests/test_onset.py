"""Task 3.2: onset detection."""
import numpy as np
import pytest

from skinflow_discard_optimizer.core.observer.onset import (
    BOCPD,
    OnsetGLR,
    detect_onset,
    naive_second_derivative,
    observable_onset,
)
from skinflow_discard_optimizer.core.observer.ukf import StrokeContext, UKFParams, run_stroke
from skinflow_discard_optimizer.sim.force_model import Alloy, Press, nominal_stroke, pressures_to_force, simulate_stroke

A_REF = 0.25e6


@pytest.fixture(scope="module")
def press():
    return Press.load()


def _synthetic(h_on=30.0, lam=6.0, sd=8e3, seed=0, h_from=120.0, h_to=12.0):
    rng = np.random.default_rng(seed)
    h = np.arange(h_from, h_to, -0.24)
    a = A_REF * np.exp(h_on / lam)
    r = a * np.exp(-h / lam) + rng.normal(0, sd, h.size)
    return h, r, np.full(h.size, sd**2)


def test_glr_posterior_recovers_onset_on_synthetic_residuals():
    for seed in range(5):
        h, r, S = _synthetic(h_on=30.0 + seed, seed=seed)
        g = OnsetGLR(A_REF)
        for hh, rr, ss in zip(h, r, S):
            g.update(hh, rr, ss)
        s = g.summary()
        assert s["h_onset_mean"] == pytest.approx(30.0 + seed, abs=0.3)
        assert s["h_onset_q05"] < s["h_onset_mean"] < s["h_onset_q95"]


def test_glr_quiet_without_upturn():
    rng = np.random.default_rng(1)
    g = OnsetGLR(A_REF, threshold=12.0)
    for hh in np.arange(120, 12, -0.24):
        g.update(hh, rng.normal(0, 8e3), 8e3**2)
    assert g.fired_at_h is None


def test_glr_fires_before_the_onset_is_reached():
    h, r, S = _synthetic(h_on=30.0)
    res = detect_onset(h, r, S, A_REF)
    assert res.glr_fired_at_h is not None and res.glr_fired_at_h > 35.0


def test_bocpd_detects_sign_agnostic_change_and_confidence_logic():
    rng = np.random.default_rng(2)
    b = BOCPD()
    h = np.arange(120, 20, -0.25)
    z = rng.normal(0, 1, h.size)
    z[h < 50] -= 4.0                      # a drop, not an upturn
    for hh, zz in zip(h, z):
        b.update(hh, zz)
    assert b.fired_at_h is not None and 44 < b.fired_at_h <= 50
    # GLR (upturn-only) stays quiet on a drop, so confidence is low
    res = detect_onset(h, z * 8e3, np.full(h.size, 64e6), A_REF)
    assert res.glr_fired_at_h is None and res.confidence == "low"


def test_observable_onset_definition():
    assert observable_onset(30.0, 6.0, A_REF, A_REF) == pytest.approx(30.0)
    assert observable_onset(30.0, 6.0, 2 * A_REF, A_REF) == pytest.approx(30.0 + 6.0 * np.log(2))


def test_end_to_end_on_simulated_stroke_beats_naive(press):
    spec = nominal_stroke("AA6063", press, h_onset_mm=31.0)
    d = simulate_stroke(spec, press, np.random.default_rng(5))
    ctx = StrokeContext(spec.L0_mm, spec.extrusion_ratio, spec.T_front_C, Alloy.from_config("AA6063"), press)
    hg = 0.15 * spec.L0_mm
    tr = run_stroke(d.t_s, d.x_mm, d.p_cap_bar, d.p_rod_bar, ctx, UKFParams.for_press(press), block=20,
                    h_stop_mm=12, theta_every=10**9, freeze_h_mm=hg)
    g = tr.h_mm < hg
    before = g & (tr.h_mm >= 31.0)          # causal: only data until the ram reaches the onset
    res = detect_onset(tr.h_mm[before], tr.innovation[before], tr.S[before], A_REF, H=tr.H[before],
                       baseline_cov=tr.final_phi_cov, noise_var=tr.r_var)
    assert res.confidence == "high"
    assert res.h_onset_mean == pytest.approx(31.0, abs=0.5)
    F = pressures_to_force(d.p_cap_bar, d.p_rod_bar, press)
    n_det, _ = naive_second_derivative(d.x_mm, F, spec.L0_mm, hg, 12)
    assert n_det is not None and n_det < res.glr_fired_at_h      # naive alarm comes later in the stroke
