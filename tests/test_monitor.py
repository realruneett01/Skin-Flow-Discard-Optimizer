"""Unit tests for Task 4.2 Multivariate statistical process monitoring with attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skinflow_discard_optimizer.aware.monitor import (
    DEFAULT_CROSS_COLS,
    DEFAULT_FPCA_COLS,
    Attribution,
    CycleMonitorResult,
    MonitorConfig,
    MultivariateMonitor,
)


@pytest.fixture
def synthetic_healthy_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 2000
    data = {}
    for i, col in enumerate(DEFAULT_FPCA_COLS):
        # Varying scale per component
        scale = 1e6 / (i + 1)
        data[col] = rng.normal(0.0, scale, size=n)

    data["fpca_spe"] = rng.exponential(1e8, size=n)

    data["oil_temp_C"] = rng.normal(45.0, 1.5, size=n)
    data["supply_pressure_min_bar"] = rng.normal(295.0, 3.0, size=n)
    data["pump_energy_kwh"] = rng.normal(12.0, 0.8, size=n)
    data["shear_stroke_s"] = rng.normal(2.5, 0.1, size=n)
    data["dead_cycle_s"] = rng.normal(18.0, 0.5, size=n)
    data["billet_temp_C"] = rng.normal(470.0, 5.0, size=n)
    data["cycle"] = np.arange(n)

    return pd.DataFrame(data)


@pytest.fixture
def fitted_monitor(synthetic_healthy_df: pd.DataFrame) -> MultivariateMonitor:
    return MultivariateMonitor.fit(
        df_healthy=synthetic_healthy_df,
        far_target=0.01,
        mewma_lambda=0.15,
    )


def _build_nominal_features(cfg: MonitorConfig, **overrides: float) -> dict[str, float]:
    feats = {c: float(cfg.mu_fpca[i]) for i, c in enumerate(cfg.fpca_cols)}
    feats["fpca_spe"] = 0.0
    for i, c in enumerate(cfg.cross_cols):
        feats[c] = float(cfg.mu_cross[i])
    feats.update(overrides)
    return feats


def _assert_alarm_attribution(res: CycleMonitorResult, alarm_type: str = "any") -> Attribution:
    assert res.is_alarm
    assert res.attribution is not None
    if alarm_type == "t2":
        assert res.t2_alarm
    elif alarm_type == "spe":
        assert res.spe_alarm
    return res.attribution


def _assert_nominal(res: CycleMonitorResult) -> None:
    assert not (res.is_alarm or res.t2_alarm or res.spe_alarm or res.mewma_alarm)
    assert res.attribution is None


def test_monitor_fit_and_limits(fitted_monitor: MultivariateMonitor):
    cfg = fitted_monitor.cfg
    assert cfg.t2_limit > 0.0
    assert cfg.spe_limit > 0.0
    assert cfg.mewma_limit > 0.0
    assert len(cfg.mu_fpca) == len(DEFAULT_FPCA_COLS)
    assert cfg.inv_cov_fpca.shape == (len(DEFAULT_FPCA_COLS), len(DEFAULT_FPCA_COLS))


def test_monitor_single_cycle_nominal(fitted_monitor: MultivariateMonitor):
    nominal_features = _build_nominal_features(fitted_monitor.cfg)
    res = fitted_monitor.update(cycle=1, features=nominal_features)
    _assert_nominal(res)


def test_monitor_hotelling_t2_alarm(fitted_monitor: MultivariateMonitor):
    cfg = fitted_monitor.cfg
    # Inject large instantaneous shift in fpca_1 (10x std dev)
    std_fpca_1 = float(np.sqrt(1.0 / cfg.inv_cov_fpca[0, 0]))
    shocked_features = _build_nominal_features(cfg, fpca_1=float(cfg.mu_fpca[0] + 10.0 * std_fpca_1))

    res = fitted_monitor.update(cycle=10, features=shocked_features)
    attr = _assert_alarm_attribution(res, "t2")
    assert attr.top_scores[0][0] == "fpca_1"
    assert attr.top_scores[0][1] > 80.0  # Dominates T2 contribution


def test_monitor_spe_alarm(fitted_monitor: MultivariateMonitor):
    cfg = fitted_monitor.cfg
    # Inject out-of-subspace SPE anomaly
    features = _build_nominal_features(cfg, fpca_spe=cfg.spe_limit * 5.0)

    res = fitted_monitor.update(cycle=15, features=features)
    attr = _assert_alarm_attribution(res, "spe")
    assert attr.dominant_channel == "spe"


def test_monitor_mewma_detects_small_sustained_shift(fitted_monitor: MultivariateMonitor):
    cfg = fitted_monitor.cfg
    fitted_monitor.reset()

    # Shift is small (1.2 sigma), so a single cycle does NOT trip T2 limit
    std_fpca_2 = float(np.sqrt(1.0 / cfg.inv_cov_fpca[1, 1]))
    shift_val = float(cfg.mu_fpca[1] + 1.2 * std_fpca_2)
    features = _build_nominal_features(cfg, fpca_2=shift_val, fpca_spe=float(cfg.spe_limit * 0.1))

    # First cycle: T2 does not trip (T2 ~ 1.44 << 17.8)
    first_res = fitted_monitor.update(cycle=1, features=features)
    assert not first_res.t2_alarm

    # Sustain shift: MEWMA accumulates and trips
    mewma_tripped = False
    for c in range(2, 45):
        res = fitted_monitor.update(cycle=c, features=features)
        if res.mewma_alarm:
            mewma_tripped = True
            break

    assert mewma_tripped, "MEWMA failed to detect small sustained shift"


def test_monitor_attribution_cross_features(fitted_monitor: MultivariateMonitor):
    cfg = fitted_monitor.cfg
    # Inject large oil temperature spike (+6 sigma)
    features = _build_nominal_features(
        cfg,
        fpca_spe=cfg.spe_limit * 2.0,
        oil_temp_C=float(cfg.mu_cross[0] + 6.0 * cfg.std_cross[0]),
    )

    res = fitted_monitor.update(cycle=100, features=features)
    attr = _assert_alarm_attribution(res)
    top_cross_name, top_cross_z = attr.top_cross_features[0]
    assert top_cross_name == "oil_temp_C"
    assert top_cross_z > 5.0
