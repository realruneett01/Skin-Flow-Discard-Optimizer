"""Task 3.1: within-stroke UKF."""
import numpy as np
import pytest

from skinflow_discard_optimizer.core.observer.ukf import (
    UKF,
    StrokeContext,
    UKFParams,
    causal_speed,
    nis_bounds,
    phi_to_theta,
    run_stroke,
    theta_to_phi,
    unscented_transform,
    windowed_nis,
)
from skinflow_discard_optimizer.sim.force_model import Alloy, Press, nominal_stroke, simulate_stroke

TUNED = UKFParams(np.array([1e-12, 1e-12, 1e-3]), 37_075.0, 1.0, np.array([0.10, 0.05, 0.3e6]))


@pytest.fixture(scope="module")
def press():
    return Press.load()


def _run(press, seed, noise=True, **spec_kw):
    spec = nominal_stroke("AA6063", press, h_onset_mm=31.0, **spec_kw)
    d = simulate_stroke(spec, press, np.random.default_rng(seed), noise=noise)
    ctx = StrokeContext(spec.L0_mm, spec.extrusion_ratio, spec.T_front_C, Alloy.from_config("AA6063"), press)
    return spec, d, ctx


def test_unscented_transform_exact_for_linear_map():
    m, P = np.array([1.0, 2.0, 3.0]), np.diag([0.1, 0.2, 0.3])
    A = np.array([[1, 2, 0], [0, 1, -1.0]])
    mu, cov = unscented_transform(m, P, lambda X: X @ A.T)
    assert np.allclose(mu, A @ m) and np.allclose(cov, A @ P @ A.T)


def test_theta_phi_roundtrip():
    X = np.array([[1.1, 0.5, 6e5], [0.9, 0.6, 7e5]])
    assert np.allclose(phi_to_theta(theta_to_phi(X)), X)


def test_causal_speed_uses_past_only_and_is_accurate():
    t = np.arange(0, 5, 1e-3)
    x = 12.0 * t + np.random.default_rng(0).normal(0, 0.02, t.size)
    v = causal_speed(t, x, 1000)
    assert np.isnan(v[:999]).all()
    assert np.nanstd(v) < 0.01 and abs(np.nanmean(v) - 12.0) < 0.005
    x2 = x.copy()
    x2[3000:] += 50
    assert np.allclose(causal_speed(t, x2, 1000)[:3000], v[:3000], equal_nan=True)


def test_kalman_update_matches_batch_bayes_for_static_parameters(press):
    """In phi the model is linear, so the filter must equal the exact batch posterior."""
    spec, d, ctx = _run(press, 1)
    params = UKFParams(np.zeros(3), TUNED.force_noise_sd_N, 1.0, TUNED.p0_sd)
    tr = run_stroke(d.t_s, d.x_mm, d.p_cap_bar, d.p_rod_bar, ctx, params, block=40,
                    h_stop_mm=0.4 * spec.L0_mm, theta_every=10**9)
    # rebuild the same regression by hand
    from skinflow_discard_optimizer.core.observer.ukf import default_prior
    from skinflow_discard_optimizer.sim.force_model import pressures_to_force
    f = UKF(*default_prior(ctx.alloy, press, params), params)
    m0, P0 = f.phi, f.phi_cov
    F = pressures_to_force(d.p_cap_bar, d.p_rod_bar, press)
    lag = 1000
    v = causal_speed(d.t_s, d.x_mm, lag)
    i0 = max(int(np.searchsorted(d.x_mm, 0.12 * spec.L0_mm)), lag)
    i1 = int(np.argmax(spec.L0_mm - d.x_mm < 0.4 * spec.L0_mm))
    n = (i1 - i0) // 40
    y = F[i0:i0 + 40 * n].reshape(n, 40).mean(1)
    x = d.x_mm[i0:i0 + 40 * n].reshape(n, 40).mean(1)
    last = np.arange(i0 + 39, i0 + 40 * n, 40)
    H = np.array([ctx.regressors(xx, vv) for xx, vv in zip(x, v[last])])
    rv = params.force_noise_sd_N**2 / 40 + 1.0
    Pinv = np.linalg.inv(P0) + H.T @ H / rv
    post = np.linalg.solve(Pinv, np.linalg.solve(P0, m0) + H.T @ y / rv)
    assert np.allclose(tr.final_phi, post, rtol=1e-4)


def test_noise_free_stroke_recovers_theta_exactly(press):
    spec, d, ctx = _run(press, 2, noise=False, sigma_scale=1.08, mu=0.6, F_tool_N=0.8e6)
    tr = run_stroke(d.t_s, d.x_mm, d.p_cap_bar, d.p_rod_bar, ctx, TUNED, block=20,
                    h_stop_mm=0.15 * spec.L0_mm, theta_every=10**9)
    assert tr.theta[-1][:2] == pytest.approx([1.08, 0.6], rel=2e-3)
    assert tr.theta[-1][2] == pytest.approx(0.8e6, rel=1e-2)   # F_tool sits on the weak ridge


def test_converges_by_60_percent_and_nis_consistent(press):
    # 30 strokes: the three components fail together along the s/F_tool ridge, so small samples are lumpy
    zs, nis_out, nis_mean = [], [], []
    lo, hi = nis_bounds(50)
    for seed in range(30):
        spec, d, ctx = _run(press, 100 + seed)
        tr = run_stroke(d.t_s, d.x_mm, d.p_cap_bar, d.p_rod_bar, ctx, TUNED, block=20,
                        h_stop_mm=0.15 * spec.L0_mm, theta_every=5)
        k60 = int(np.searchsorted(tr.x_mm, 0.6 * spec.L0_mm))
        truth = np.array([1.0, spec.mu, spec.F_tool_N])
        zs.append((tr.theta[k60] - truth) / tr.theta_sd[k60])
        wn = windowed_nis(tr.nis, 50)
        nis_out.append(np.mean((wn < lo) | (wn > hi)))
        nis_mean.append(tr.nis.mean())
    zs = np.array(zs)
    assert np.mean(np.abs(zs) <= 2) >= 0.85          # nominal 95%
    assert np.all(np.abs(zs.mean(axis=0)) < 0.5)      # unbiased
    assert np.all((zs.std(axis=0) > 0.6) & (zs.std(axis=0) < 1.3))   # calibrated spread
    assert np.mean(nis_mean) == pytest.approx(1.0, abs=0.1)
    assert np.mean(nis_out) < 0.05


def test_nis_flags_end_of_stroke_mismatch(press):
    spec, d, ctx = _run(press, 3)
    tr = run_stroke(d.t_s, d.x_mm, d.p_cap_bar, d.p_rod_bar, ctx, TUNED, block=20, h_stop_mm=12,
                    theta_every=10**9)
    body = tr.h_mm > 80
    end = tr.h_mm < 25
    assert tr.nis[body].mean() < 1.5 and tr.nis[end].mean() > 20


def test_live_step_matches_block_run(press):
    spec, d, ctx = _run(press, 4)
    tr = run_stroke(d.t_s, d.x_mm, d.p_cap_bar, d.p_rod_bar, ctx, TUNED, block=1,
                    h_stop_mm=0.8 * spec.L0_mm, theta_every=10**9)
    from skinflow_discard_optimizer.core.observer.ukf import default_prior
    from skinflow_discard_optimizer.sim.force_model import pressures_to_force
    f = UKF(*default_prior(ctx.alloy, press, TUNED), TUNED)
    F = pressures_to_force(d.p_cap_bar, d.p_rod_bar, press)
    v = causal_speed(d.t_s, d.x_mm, 1000)
    i0 = max(int(np.searchsorted(d.x_mm, 0.12 * spec.L0_mm)), 1000)
    r_var = TUNED.force_noise_sd_N**2 + TUNED.model_sd_N**2
    xprev = d.x_mm[i0]
    for k in range(i0, i0 + len(tr.x_mm)):
        f.step(F[k], d.x_mm[k], d.x_mm[k] - xprev, v[k], ctx, r_var)
        xprev = d.x_mm[k]
    assert np.allclose(f.phi, tr.final_phi, rtol=1e-9)
