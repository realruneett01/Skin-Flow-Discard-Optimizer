"""Tests for stress test suite execution and artifact generation (Task 5.3)."""
from __future__ import annotations

import json
from pathlib import Path

from skinflow_discard_optimizer.eval.stress import (
    generate_stress_report,
    run_stress_suite,
    save_stress_summary,
)


def test_run_stress_suite_all_pass() -> None:
    suite_result = run_stress_suite()
    assert suite_result.total_scenarios == 7
    assert suite_result.passed_scenarios == 7
    assert suite_result.all_safe_and_bounded is True

    for scenario in suite_result.scenarios:
        assert scenario.passed is True
        assert scenario.is_fallback is True
        assert scenario.alert_level == "CRITICAL_FALLBACK"
        assert 12.0 <= scenario.h_cut_mm <= 60.0
        assert scenario.h_cut_mm == 40.0
        for tag_key in scenario.published_tags:
            assert tag_key.startswith("Predictor.")


def test_save_and_report_stress_artifacts(tmp_path: Path) -> None:
    suite_result = run_stress_suite()
    json_path = tmp_path / "artifacts" / "stress_summary.json"
    save_stress_summary(suite_result, json_path)
    assert json_path.exists()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_scenarios"] == 7
    assert data["passed_scenarios"] == 7

    report_md = generate_stress_report(suite_result)
    assert "# Task 5.3: Stress Tests and Safety Gates Validation Report" in report_md
    assert "PASS - ALL BOUNDED & SAFE" in report_md
    assert "missing_samples" in report_md
    assert "frozen_sensor" in report_md
    assert "unseen_alloy" in report_md
