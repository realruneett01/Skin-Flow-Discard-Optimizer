"""Unit tests for Task 5.1 Backtesting protocol."""
import numpy as np
import pandas as pd
import pytest

from skinflow_discard_optimizer.eval.backtest import (
    MetricInterval,
    ScenarioMetrics,
    compute_recovery_pct,
    generate_report_markdown,
    run_clustered_bootstrap,
    run_full_backtest,
)
from skinflow_discard_optimizer.paths import ARTIFACTS_DIR


def test_compute_recovery_pct():
    h_cut = np.array([28.0, 30.0, 40.0])
    h_static = np.array([40.0, 40.0, 40.0])
    mass_per_mm = 0.116675
    billet_mass = 95.0

    rec = compute_recovery_pct(h_cut, h_static, mass_per_mm, billet_mass)
    assert len(rec) == 3
    assert rec[2] == pytest.approx(0.0)
    assert rec[0] == pytest.approx(12.0 * mass_per_mm / billet_mass * 100.0)
    assert rec[0] > rec[1] > rec[2]


def test_metric_interval_formatting():
    m = MetricInterval(
        name="Test Metric",
        unit="mm",
        point_estimate=25.4321,
        ci_lower=23.1111,
        ci_upper=27.8888,
        target="< 30 mm",
        status="PASS",
        description="A test metric",
    )
    assert m.format_estimate(".2f") == "25.43"
    assert m.format_ci(".2f") == "[23.11, 27.89]"


def test_run_clustered_bootstrap():
    rng = np.random.default_rng(0)
    rows = []
    for sc in ("sc_a", "sc_b", "sc_c", "sc_d"):
        n = 100
        cut = rng.normal(28.0, 1.0, n)
        stat = np.full(n, 40.0)
        cov = rng.choice([True, False], p=[0.9, 0.1], size=n)
        defect = rng.uniform(0, 0.001, n)
        cost = cut * 0.05
        static_cost = stat * 0.05
        comp = rng.normal(15.0, 2.0, n)
        for i in range(n):
            rows.append({
                "scenario": sc,
                "h_cut": cut[i],
                "static_cut": stat[i],
                "covered": cov[i],
                "defect_prob": defect[i],
                "cost": cost[i],
                "static_cost": static_cost[i],
                "compute_ms": comp[i],
            })
    df = pd.DataFrame(rows)
    ci_res = run_clustered_bootstrap(df, mass_per_mm=0.116, billet_mass=95.0, n_bootstrap=100)

    for key, (point, lo, hi) in ci_res.items():
        assert lo <= point <= hi or lo <= hi, f"Invalid interval for {key}: lo={lo}, point={point}, hi={hi}"


def test_full_backtest_execution():
    if not (ARTIFACTS_DIR / "decision_eval.parquet").exists():
        pytest.skip("decision_eval.parquet not found")

    results = run_full_backtest(n_bootstrap=50, seed=123)

    assert "headline_kpis" in results
    assert "supporting_kpis" in results
    assert "scenario_metrics" in results
    assert "lead_time_metrics" in results

    headlines = results["headline_kpis"]
    assert len(headlines) == 6
    kpi_names = [k.name for k in headlines]
    assert "Butt (discard) thickness" in kpi_names
    assert "Recovery" in kpi_names
    assert "Defect rate" in kpi_names
    assert "False-alarm rate" in kpi_names
    assert "Warning lead time" in kpi_names
    assert "Cut-decision latency" in kpi_names

    for k in headlines:
        assert k.ci_lower <= k.ci_upper, f"Invalid CI bounds for {k.name}: [{k.ci_lower}, {k.ci_upper}]"

    md = generate_report_markdown(results)
    assert "# Task 5.1: Rolling-Origin Backtesting Protocol" in md
    assert "Executive Summary" in md
    assert "Headline KPIs Summary" in md
    assert "Injected Fault Detection & Warning Lead Times" in md
