"""Unit tests for Task 4.3 Forward trajectory forecasting and time-to-limit prediction."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from skinflow_discard_optimizer.aware.forecast import (
    DEFAULT_LIMITS,
    Forecaster,
    ForecastResult,
    ResidualTCN,
    _compute_ttl_metric,
    _propagate_single_step,
)
from skinflow_discard_optimizer.aware.state_tracker import (
    LatentWearTracker,
    Regime,
    TrackerConfig,
)
from skinflow_discard_optimizer.sim.defect_model import DefectModel


@pytest.fixture
def initialized_tracker() -> LatentWearTracker:
    cfg = TrackerConfig.load(n_particles=150)
    tracker = LatentWearTracker(cfg, seed=42)
    # Feed 10 baseline cycles
    for c in range(1, 11):
        tracker.update(c, {
            "theta_F_tool_N": 6.0e5,
            "theta_mu": 0.556,
            "theta_sigma_scale": 1.00,
            "spec_taper_K": 25.0,
        })
    return tracker


@pytest.fixture
def forecaster() -> Forecaster:
    return Forecaster(seed=42)


def test_forecaster_initialization(forecaster: Forecaster):
    assert forecaster.dm is not None
    assert forecaster.rng is not None
    assert not forecaster.use_learned_residual


def test_propagate_single_step():
    cfg = TrackerConfig.load(n_particles=50)
    rng = np.random.default_rng(123)
    particles = np.tile(cfg.baseline, (50, 1))
    regimes = np.zeros(50, dtype=int)

    new_p, new_r = _propagate_single_step(particles, regimes, cfg, rng)
    assert new_p.shape == (50, 4)
    assert new_r.shape == (50,)
    assert np.all(new_p >= 0.0)


def test_forecaster_trajectory_output_shapes(
    forecaster: Forecaster, initialized_tracker: LatentWearTracker
):
    horizon = 30
    res = forecaster.forecast(
        tracker=initialized_tracker,
        horizon_steps=horizon,
        n_simulations=100,
        limits=DEFAULT_LIMITS,
    )
    assert res.current_cycle == 10
    assert res.horizon_steps == horizon
    assert len(res.summaries) == horizon
    assert len(res.limits) == 3

    # Check monotonicity of cycles
    for i, s in enumerate(res.summaries):
        assert s.step == i + 1
        assert s.cycle == 10 + i + 1
        assert s.h_cut_ci[0] <= s.h_cut_median <= s.h_cut_ci[1]
        assert s.h_crit_ci[0] <= s.h_crit_mean <= s.h_crit_ci[1]
        assert 0.0 <= s.defect_prob_mean <= 1.0


def test_compute_ttl_metric():
    # 10 trajectories, horizon 20
    traj = np.zeros((10, 20))
    # 5 trajectories breach at step 8
    traj[:5, 7:] = 35.0
    # 5 trajectories never breach (stay at 25.0)
    traj[5:, :] = 25.0

    ttl = _compute_ttl_metric(traj, "test_limit", 30.0, compare_greater=True)
    assert np.isclose(ttl.crossing_probability, 0.50)
    assert ttl.median_ttl_cycles == 8.0
    assert ttl.ttl_p10_cycles == 8.0
    assert ttl.ttl_p90_cycles == 8.0


def test_residual_tcn_forward():
    model = ResidualTCN(in_channels=4, horizon_steps=25, hidden=16)
    x = torch.randn(2, 4, 10)  # batch 2, 4 channels, 10 history cycles
    out = model(x)
    assert out.shape == (2, 25)


def test_forecast_lead_time_on_elevated_wear(forecaster: Forecaster):
    cfg = TrackerConfig.load(n_particles=150)
    tracker = LatentWearTracker(cfg, seed=99)
    # Feed cycles with high wear (F_tool at 0.68 MN -> wear ~ 0.53)
    for c in range(1, 15):
        tracker.update(c, {
            "theta_F_tool_N": 6.8e5,
            "theta_mu": 0.585,
            "theta_sigma_scale": 1.00,
            "spec_taper_K": 25.0,
        })

    # Strict limits: cut upper bound at 29.0 mm
    strict_limits = {
        "cut_upper_bound_mm": 29.0,
        "defect_prob_pct": 0.05,
        "mewma_alarm_stat": 140.0,
    }
    res = forecaster.forecast(tracker, horizon_steps=40, n_simulations=100, limits=strict_limits)
    assert res.recommended_lead_warning
    assert res.lead_time_gained_cycles > 0
