"""Tests for streaming optimization service and advisory tags (Task 6.1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from skinflow_discard_optimizer.core.features import CycleInputs, inputs_from_row
from skinflow_discard_optimizer.service.replay import cycle_to_records
from skinflow_discard_optimizer.service.run import (
    ServiceConfig,
    StreamingOptimizerService,
)
from skinflow_discard_optimizer.sim.scenario import Scenario, run_scenario


@pytest.fixture
def service() -> StreamingOptimizerService:
    cfg = ServiceConfig(latency_budget_ms=200.0, block=40, mock_cycles=2)
    return StreamingOptimizerService(config=cfg)


def test_service_initial_health(service: StreamingOptimizerService) -> None:
    health = service.health_check()
    assert health.status == "HEALTHY"
    assert health.cycles_processed == 0
    assert health.fallbacks_count == 0
    assert health.budget_violations == 0
    assert health.uptime_seconds >= 0.0


def test_service_run_mock_cycles(service: StreamingOptimizerService) -> None:
    entries = service.run_mock(n_cycles=2, scenario_name="healthy_baseline")
    assert len(entries) == 2

    for e in entries:
        assert e.within_budget is True
        assert e.latency_ms <= 200.0
        assert 12.0 <= e.cut_recommendation_mm <= 60.0
        assert e.confidence == "high"
        assert e.is_fallback is False
        assert e.drift_alarm is False
        assert e.fault_class == "none"
        assert e.published_tags["Predictor.ButtCutMm"] == e.cut_recommendation_mm
        for tag in e.published_tags:
            assert tag.startswith("Predictor.")

    health = service.health_check()
    assert health.cycles_processed == 2
    assert health.budget_violations == 0
    assert health.status == "HEALTHY"


def test_service_corrupt_telemetry_engages_fallback(service: StreamingOptimizerService) -> None:
    sc = Scenario.load("healthy_baseline").with_cycles(1)
    row, stroke = next(run_scenario(sc, keep_strokes=True))
    inp = inputs_from_row(row, stroke)

    # Corrupt stroke position with NaN
    corrupt_x = inp.x_mm.copy()
    corrupt_x[50] = np.nan
    corrupt_inp = CycleInputs(
        cycle_id=inp.cycle_id,
        alloy_id=inp.alloy_id,
        die_id=inp.die_id,
        t_s=inp.t_s,
        x_mm=corrupt_x,
        p_cap_bar=inp.p_cap_bar,
        p_rod_bar=inp.p_rod_bar,
        billet_temp_C=inp.billet_temp_C,
        billet_length_mm=inp.billet_length_mm,
        liner_temps_C=inp.liner_temps_C,
        oil_temp_C=inp.oil_temp_C,
        supply_pressure_min_bar=inp.supply_pressure_min_bar,
        pump_energy_kwh=inp.pump_energy_kwh,
        phase_durations_s=inp.phase_durations_s,
    )

    entry = service.process_cycle_inputs(corrupt_inp)
    assert entry.is_fallback is True
    assert entry.cut_recommendation_mm == 40.0
    assert any("MISSING_SAMPLES" in r for r in entry.fallback_reasons)


def test_service_replay_recording(service: StreamingOptimizerService) -> None:
    sc = Scenario.load("healthy_baseline").with_cycles(1)
    row, stroke = next(run_scenario(sc, keep_strokes=True))
    rec, _ = cycle_to_records(row, stroke)

    entries = service.replay_recording(rec)
    assert len(entries) == 1
    assert entries[0].within_budget is True
    assert entries[0].cut_recommendation_mm > 0.0


def test_service_json_audit_and_health_dict(service: StreamingOptimizerService, tmp_path: Path) -> None:
    entries = service.run_mock(n_cycles=1, scenario_name="healthy_baseline")
    out_file = tmp_path / "decisions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([e.published_tags for e in entries], f)

    assert out_file.exists()
    health = service.health_check().to_dict()
    assert "mean_latency_ms" in health
    assert health["cycles_processed"] == 1
