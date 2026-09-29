"""Task 2.1: preprocessing, derivatives and the dummy-block gate.

Documented derivative error bounds (asserted below, reported in docs/phase2_signal.md):
relative RMS error of dF/dx and d2F/dx2 against the noise-free model, in the mid
stroke and in the gated tail.
"""
import numpy as np
import pytest

from skinflow_discard_optimizer.core.preprocess import (
    GATE_FRACTION,
    choose_sg_window,
    lowpass,
    onset_gate,
    preprocess_stroke,
    resample_to_position,
    robust_noise_sd,
    sg_derivative,
)
from skinflow_discard_optimizer.sim.cycle import regenerate_stroke
from skinflow_discard_optimizer.sim.force_model import (
    Press,
    force_breakdown,
    nominal_stroke,
    ram_kinematics,
    simulate_stroke,
)
from skinflow_discard_optimizer.sim.scenario import Scenario, run_scenario

# documented bounds (relative RMS error vs. true derivative); measured values are ~2%, <1%, ~5%
D1_MID_REL, D1_TAIL_REL = 0.05, 0.03
D2_TAIL_REL = 0.12


@pytest.fixture(scope="module")
def press():
    return Press.load()


@pytest.fixture(scope="module")
def healthy(press):
    spec = nominal_stroke("AA6063", press, h_onset_mm=31.0)
    d = simulate_stroke(spec, press, np.random.default_rng(0))
    pp = preprocess_stroke(d.t_s, d.x_mm, d.p_cap_bar, d.p_rod_bar, press, spec.L0_mm)
    return spec, d, pp


def test_lowpass_is_zero_phase():
    fs = 1000.0
    t = np.arange(0, 2, 1 / fs)
    y = np.exp(-0.5 * ((t - 0.8) / 0.05) ** 2)   # sharp pulse: any phase lag moves its peak
    yf = lowpass(y, fs, 10.0)
    assert np.argmax(yf) == np.argmax(y)
    assert np.allclose(yf, y, atol=0.02)


def test_noise_sd_estimate():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 5000)
    y = 3 * x**2 + rng.normal(0, 0.5, x.size)
    assert robust_noise_sd(y) == pytest.approx(0.5, rel=0.08)


def test_sg_derivative_exact_on_cubic_and_causal_uses_past_only():
    x = np.arange(0, 10, 0.1)
    y = x**3 - 2 * x
    d = sg_derivative(y, 11, 1, 0.1, polyorder=3)
    assert np.allclose(d[10:-10], 3 * x[10:-10] ** 2 - 2, atol=1e-8)
    dc = sg_derivative(y, 11, 1, 0.1, polyorder=3, causal=True)
    assert np.isnan(dc[:10]).all()
    assert np.allclose(dc[10:], 3 * x[10:] ** 2 - 2, atol=1e-6)
    # changing a future sample must not change a causal estimate
    y2 = y.copy()
    y2[60:] += 100
    assert np.allclose(sg_derivative(y2, 11, 1, 0.1, 3, causal=True)[:60], dc[:60], equal_nan=True)


def test_window_grows_with_noise():
    rng = np.random.default_rng(2)
    x = np.arange(0, 100, 0.1)
    clean = 1e4 * np.sin(x / 8)
    low = choose_sg_window(clean + rng.normal(0, 5, x.size), 0.1, deriv=1).window
    high = choose_sg_window(clean + rng.normal(0, 200, x.size), 0.1, deriv=1).window
    assert high > low


def test_resample_to_position_uniform_grid(press):
    x = np.sort(np.random.default_rng(3).uniform(0, 50, 20000))
    c = resample_to_position(x, 2 * x, dx_mm=0.5, x_min=0.0)
    assert np.allclose(np.diff(c.x_mm), 0.5)
    assert np.allclose(c.force_N, 2 * c.x_mm, atol=0.1)


def _rel_rms(est, true, mask):
    return np.sqrt(np.mean((est[mask] - true[mask]) ** 2)) / np.sqrt(np.mean(true[mask] ** 2))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_derivatives_within_documented_error(press, seed):
    spec = nominal_stroke("AA6063", press, h_onset_mm=31.0)
    d = simulate_stroke(spec, press, np.random.default_rng(seed))
    pp = preprocess_stroke(d.t_s, d.x_mm, d.p_cap_bar, d.p_rod_bar, press, spec.L0_mm)
    x = pp.x_mm
    t, xk, vk = ram_kinematics(spec, press.sample_rate_hz)   # true speed profile incl. acceleration
    Ft = np.interp(x, xk, force_breakdown(xk, spec, press, v_ram_mm_s=np.maximum(vk, 1e-6)).total_N)
    d1t = np.gradient(Ft, x)
    d2t = np.gradient(d1t, x)
    mid = (x > 100) & (x < 600)
    tail = pp.gate & (pp.h_mm > 12)          # stay clear of the edge-padded end
    assert _rel_rms(pp.dF_dx, d1t, mid) < D1_MID_REL
    assert _rel_rms(pp.dF_dx, d1t, tail) < D1_TAIL_REL
    assert _rel_rms(pp.d2F_dx2, d2t, tail) < D2_TAIL_REL


def test_gate_position_rule(healthy):
    spec, _, pp = healthy
    xg = pp.x_mm[pp.gate]
    assert xg.min() >= GATE_FRACTION * spec.L0_mm
    assert pp.transients.transient_end_mm <= 0.05 * spec.L0_mm + 1e-9
    assert not pp.transients.flash_windows_mm            # healthy stroke: no flash found


def test_gate_rejects_flash_window_even_late():
    g = onset_gate(np.arange(0, 800, 1.0), 800.0, 20.0, [(700.0, 705.0)])
    assert not g[700:706].any() and g[706] and not g[:680].any()


def test_flash_detected_and_gate_never_fires_during_flash(press):
    scn = Scenario.load("flash_spike").with_cycles(6010)
    hits = 0
    for row, _ in run_scenario(scn, start=6000):
        d = regenerate_stroke(row, press)
        pp = preprocess_stroke(d.t_s, d.x_mm, d.p_cap_bar, d.p_rod_bar, press, row["spec_L0_mm"])
        xf, w = row["eff_flash_x_mm"], row["eff_flash_width_mm"]
        spike = (pp.x_mm > xf - 2 * w) & (pp.x_mm < xf + 2 * w)
        assert not (pp.gate & spike).any()
        found = any(a <= xf <= b for a, b in pp.transients.flash_windows_mm)
        hits += found
        # the transient end moves past the flash
        assert pp.transients.transient_end_mm >= xf
    assert hits >= 9   # detected in at least 9 of 10 cycles
