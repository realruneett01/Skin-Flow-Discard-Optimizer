"""Unit and integration tests for joint ROI calculation engine (Task 6.2)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from skinflow_discard_optimizer.roi.engine import (
    ROIEngine,
    ROIParameters,
    ValueRange,
)


@pytest.fixture
def engine() -> ROIEngine:
    return ROIEngine()


def test_value_range_operations() -> None:
    v1 = ValueRange(low=10.0, expected=20.0, high=30.0, unit="EUR")
    scaled = v1.scale(2.5)
    assert scaled.low == 25.0
    assert scaled.expected == 50.0
    assert scaled.high == 75.0

    v2 = ValueRange(low=5.0, expected=15.0, high=25.0, unit="EUR")
    added = v1.add(v2)
    assert added.low == 15.0
    assert added.expected == 35.0
    assert added.high == 55.0


def test_default_joint_roi_calculation(engine: ROIEngine) -> None:
    res = engine.calculate_roi()

    # Rule 6 check: All modules must report low, expected, high intervals
    assert res.skdo.annual_value_keur.low < res.skdo.annual_value_keur.expected < res.skdo.annual_value_keur.high
    assert res.dcto.annual_value_keur.low < res.dcto.annual_value_keur.expected < res.dcto.annual_value_keur.high
    assert res.hpeo.annual_value_keur.low < res.hpeo.annual_value_keur.expected < res.hpeo.annual_value_keur.high

    # Combined platform valuation
    total = res.total_annual_value_keur
    assert total.low < total.expected < total.high
    assert 200.0 < total.expected < 350.0  # expected annual platform return is ~293.7 kEUR
    assert res.oracle_annual_ceiling_keur > total.expected


def test_roi_parameter_sensitivity(engine: ROIEngine) -> None:
    base = engine.default_params
    base_res = engine.calculate_roi(base)

    # Doubling metal spread (from 0.50 to 1.00 EUR/kg) should double SKDO return
    high_spread = ROIParameters(
        metal_price_eur_per_kg=3.10,
        remelt_credit_eur_per_kg=2.10,
        cycles_per_year=base.cycles_per_year,
    )
    high_res = engine.calculate_roi(high_spread)
    ratio = high_res.skdo.annual_value_keur.expected / base_res.skdo.annual_value_keur.expected
    assert abs(ratio - 2.0) < 1e-3

    # Doubling annual volume should double all module returns
    double_vol = ROIParameters(
        cycles_per_year=base.cycles_per_year * 2,
    )
    vol_res = engine.calculate_roi(double_vol)
    vol_ratio = vol_res.total_annual_value_keur.expected / base_res.total_annual_value_keur.expected
    assert abs(vol_ratio - 2.0) < 1e-3


def test_sensitivity_table_generation(engine: ROIEngine) -> None:
    tables = engine.evaluate_sensitivity()
    assert "metal_spread_sensitivity" in tables
    assert "tariff_sensitivity" in tables
    assert "volume_sensitivity" in tables

    spreads = tables["metal_spread_sensitivity"]
    assert len(spreads) == 6
    assert spreads[0]["skdo_annual_exp"] < spreads[-1]["skdo_annual_exp"]


def test_save_roi_summary(engine: ROIEngine, tmp_path: Path) -> None:
    target = tmp_path / "roi_summary.json"
    engine.save_roi_summary(target)
    assert target.exists()

    with open(target, encoding="utf-8") as f:
        data = json.load(f)
    assert "total_annual_value_keur" in data
    assert "skdo" in data
    assert "dcto" in data
    assert "hpeo" in data
