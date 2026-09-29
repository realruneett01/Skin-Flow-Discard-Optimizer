"""Tests for safety gates and fallback mechanisms (Task 5.3)."""
from __future__ import annotations

import numpy as np
import pytest

from skinflow_discard_optimizer.core.safety import (
    CutCandidate,
    GuardedDecision,
    SafetyConfig,
    SafetyGate,
    StrokeTelemetry,
)


@pytest.fixture
def clean_telemetry() -> StrokeTelemetry:
    t_s = np.linspace(0.0, 10.0, 500)
    x_mm = np.linspace(0.0, 800.0, 500)
    p_cap_bar = 200.0 + 10.0 * np.sin(np.linspace(0, 3.14, 500))
    p_rod_bar = 30.0 + 2.0 * np.cos(np.linspace(0, 3.14, 500))
    return StrokeTelemetry(t_s=t_s, x_mm=x_mm, p_cap_bar=p_cap_bar, p_rod_bar=p_rod_bar)


def test_clean_telemetry_validation(clean_telemetry: StrokeTelemetry) -> None:
    gate = SafetyGate()
    issues = gate.validate_stroke_telemetry(clean_telemetry)
    assert len(issues) == 0


def test_missing_samples_triggers_fallback(clean_telemetry: StrokeTelemetry) -> None:
    gate = SafetyGate()
    corrupt_x = clean_telemetry.x_mm.copy()
    corrupt_x[100] = np.nan
    corrupt = StrokeTelemetry(
        t_s=clean_telemetry.t_s,
        x_mm=corrupt_x,
        p_cap_bar=clean_telemetry.p_cap_bar,
        p_rod_bar=clean_telemetry.p_rod_bar,
    )
    issues = gate.validate_stroke_telemetry(corrupt)
    assert any("MISSING_SAMPLES" in issue for issue in issues)

    candidate = CutCandidate(raw_cut_mm=25.0, confidence="high", telemetry_issues=tuple(issues))
    decision = gate.guard_cut_decision(candidate)
    assert decision.is_fallback is True
    assert decision.h_cut_mm == 40.0
    assert decision.alert_level == "CRITICAL_FALLBACK"


def test_frozen_sensor_triggers_fallback(clean_telemetry: StrokeTelemetry) -> None:
    gate = SafetyGate()
    frozen_p = np.full_like(clean_telemetry.p_cap_bar, 200.0)
    frozen = StrokeTelemetry(
        t_s=clean_telemetry.t_s,
        x_mm=clean_telemetry.x_mm,
        p_cap_bar=frozen_p,
        p_rod_bar=clean_telemetry.p_rod_bar,
    )
    issues = gate.validate_stroke_telemetry(frozen)
    assert any("FROZEN_SENSOR" in issue for issue in issues)

    candidate = CutCandidate(raw_cut_mm=25.0, telemetry_issues=tuple(issues))
    decision = gate.guard_cut_decision(candidate)
    assert decision.is_fallback is True
    assert decision.h_cut_mm == 40.0


def test_timestamp_jitter_triggers_fallback(clean_telemetry: StrokeTelemetry) -> None:
    gate = SafetyGate()
    jitter_t = clean_telemetry.t_s.copy()
    jitter_t[50] = jitter_t[49]  # dt = 0
    jitter = StrokeTelemetry(
        t_s=jitter_t,
        x_mm=clean_telemetry.x_mm,
        p_cap_bar=clean_telemetry.p_cap_bar,
        p_rod_bar=clean_telemetry.p_rod_bar,
    )
    issues = gate.validate_stroke_telemetry(jitter)
    assert any("TIMESTAMP_JITTER" in issue for issue in issues)


def test_out_of_range_detection(clean_telemetry: StrokeTelemetry) -> None:
    gate = SafetyGate()
    blown_p = clean_telemetry.p_cap_bar.copy()
    blown_p[10] = 550.0  # above 450 bar hydraulic limit
    blown = StrokeTelemetry(
        t_s=clean_telemetry.t_s,
        x_mm=clean_telemetry.x_mm,
        p_cap_bar=blown_p,
        p_rod_bar=clean_telemetry.p_rod_bar,
    )
    issues = gate.validate_stroke_telemetry(blown)
    assert any("OUT_OF_RANGE" in issue for issue in issues)


def test_context_validation_unseen_alloy_and_temperatures() -> None:
    gate = SafetyGate()
    alloy_issues = gate.validate_process_context(alloy_id="AA7075", billet_temp_C=480.0, liner_temp_C=430.0)
    assert any("UNSEEN_ALLOY" in issue for issue in alloy_issues)

    temp_issues = gate.validate_process_context(alloy_id="AA6063", billet_temp_C=200.0, liner_temp_C=430.0)
    assert any("OUT_OF_RANGE" in issue for issue in temp_issues)


def test_clamping_bounds_and_advisory_protocol() -> None:
    gate = SafetyGate(SafetyConfig(min_cut_mm=12.0, max_cut_mm=60.0, static_cut_mm=40.0))

    # Test lower clamping
    low_candidate = CutCandidate(raw_cut_mm=5.0, confidence="high")
    low_decision = gate.guard_cut_decision(low_candidate)
    assert low_decision.h_cut_mm == 12.0
    assert low_decision.is_fallback is False

    # Test upper clamping
    high_candidate = CutCandidate(raw_cut_mm=85.0, confidence="high")
    high_decision = gate.guard_cut_decision(high_candidate)
    assert high_decision.h_cut_mm == 60.0
    assert high_decision.is_fallback is False

    # Test advisory tag structure
    assert high_decision.is_advisory is True
    for key in high_decision.published_tags:
        assert key.startswith("Predictor.")
