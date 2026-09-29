"""Unit tests for Task 4.1 Latent wear-state tracker and regime switching filter."""
import numpy as np
import pytest

from skinflow_discard_optimizer.aware.state_tracker import (
    LatentWearTracker,
    Regime,
    TrackerConfig,
)


@pytest.fixture
def tracker() -> LatentWearTracker:
    cfg = TrackerConfig.load(n_particles=200)
    return LatentWearTracker(config=cfg, seed=42)


def test_tracker_initialization(tracker: LatentWearTracker):
    assert tracker.particles.shape == (200, 4)
    assert np.all(tracker.particles >= tracker.bounds_lo)
    assert np.all(tracker.particles <= tracker.bounds_hi)
    assert np.all(tracker.regimes == Regime.NORMAL)
    assert np.isclose(np.sum(tracker.weights), 1.0)


def test_tracker_single_cycle_update(tracker: LatentWearTracker):
    features = {
        "theta_F_tool_N": 6.15e5,    # 0.615 MN -> wear ~ (0.615 - 0.60) / 0.15 = 0.10
        "theta_mu": 0.556,          # friction -> scale ~ (0.556 - 0.55) / 0.06 = 0.10
        "theta_sigma_scale": 1.00,
        "spec_taper_K": 25.0,
    }
    est = tracker.update(cycle=1, features=features)
    assert est.cycle == 1
    assert 0.0 <= est.die_wear_mean <= 1.5
    assert 0.0 <= est.liner_scale_mean <= 3.0
    assert 0.8 <= est.sensor_gain_mean <= 1.3
    assert est.die_wear_ci[0] <= est.die_wear_mean <= est.die_wear_ci[1]
    assert np.isclose(est.p_normal + est.p_ramping + est.p_step, 1.0)


def test_tracker_follows_wear_drift(tracker: LatentWearTracker):
    # Simulate a 100-cycle increasing wear trend
    wear_true = np.linspace(0.10, 0.40, 100)
    estimates = []
    for c, w in enumerate(wear_true, start=1):
        f_tool_n = (0.60 + 0.15 * w) * 1e6 + np.random.normal(0, 1e4)
        feats = {
            "theta_F_tool_N": f_tool_n,
            "theta_mu": 0.556,
            "theta_sigma_scale": 1.00,
            "spec_taper_K": 25.0,
        }
        est = tracker.update(cycle=c, features=feats)
        estimates.append(est.die_wear_mean)

    estimates = np.array(estimates)
    # Estimated wear should track upwards
    assert estimates[-1] > estimates[0]
    # Late estimates should track towards higher wear
    late_err = np.mean(np.abs(estimates[-20:] - wear_true[-20:]))
    assert late_err < 0.25


def test_step_fault_flips_regime():
    tracker = LatentWearTracker(TrackerConfig.load(n_particles=300), seed=99)
    # Run 20 healthy cycles
    for c in range(1, 21):
        tracker.update(c, {
            "theta_F_tool_N": 6e5,
            "theta_mu": 0.558,
            "theta_sigma_scale": 1.00,
            "spec_taper_K": 25.0,
        })
    assert tracker.history[-1].p_normal > 0.80

    # Inject sudden step fault in friction at cycle 21
    # mu steps from 0.558 to 0.650 (large jump)
    flip_detected = False
    for c in range(21, 35):
        est = tracker.update(c, {
            "theta_F_tool_N": 6e5,
            "theta_mu": 0.650,
            "theta_sigma_scale": 1.00,
            "spec_taper_K": 25.0,
        })
        if (est.p_step + est.p_ramping) > 0.60:
            flip_detected = True
            break

    assert flip_detected, "Regime failed to flip after sudden step fault"
