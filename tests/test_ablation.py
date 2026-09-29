"""Unit tests for Task 5.2 Ablation Study."""
import numpy as np
import pandas as pd
import pytest

from skinflow_discard_optimizer.eval.ablation import (
    AblationVariant,
    EvalContext,
    _eval_fixed_cut,
    _eval_full_model,
    _eval_no_conformal,
    _eval_no_hierarchy,
    _eval_no_ukf,
    _eval_second_deriv_rule,
    generate_report_markdown,
    run_ablation_study,
)
from skinflow_discard_optimizer.paths import ARTIFACTS_DIR
from skinflow_discard_optimizer.sim.defect_model import DefectModel


def test_calc_recovery():
    h_cut = np.array([28.0, 40.0])
    dm = DefectModel()
    df = pd.DataFrame({
        "h_crit": [22.0, 22.0],
        "static_cut": [40.0, 40.0],
        "static_cost": [2.33, 2.33],
        "h_cut": [28.0, 40.0],
        "cost": [1.80, 2.33],
        "covered": [True, True],
        "defect_prob": [0.0005, 0.0005],
        "m": [22.0, 22.0],
        "s": [1.3, 1.3],
        "encoder_offset": [0.0, 0.0],
        "scenario": ["healthy_baseline", "healthy_baseline"],
        "onset_mean": [29.0, 29.0],
    })
    ctx = EvalContext(df, dm, mass_per_mm=0.116675, billet_mass=95.0, c_yr=309600.0, full_sav=0.53)
    rec = ctx.calc_recovery(h_cut)
    assert rec > 0.0


def test_ablation_variant_creation():
    v = AblationVariant(
        name="test_variant",
        layer_id="L1",
        layer_name="Layer 1 Test",
        mean_cut_mm=28.5,
        recovery_pct=1.4,
        defect_rate_per_1000=0.5,
        coverage_pct=91.0,
        net_saving_eur_billet=0.38,
        annual_value_keur=117.0,
        delta_saving_eur=-0.01,
        detection_lead_cycles=250,
        earns_place=True,
        verdict="KEEP",
        justification="Test justification",
    )
    assert v.earns_place is True
    assert v.delta_saving_eur == -0.01


def test_eval_context_methods():
    dm = DefectModel()
    n = 10
    df = pd.DataFrame({
        "h_crit": np.full(n, 22.0),
        "static_cut": np.full(n, 40.0),
        "static_cost": np.full(n, 2.33),
        "h_cut": np.full(n, 28.0),
        "cost": np.full(n, 1.80),
        "covered": np.full(n, True),
        "defect_prob": np.full(n, 0.0005),
        "m": np.full(n, 22.0),
        "s": np.full(n, 1.3),
        "encoder_offset": np.zeros(n),
        "scenario": ["healthy_baseline"] * n,
        "onset_mean": np.full(n, 29.0),
    })
    ctx = EvalContext(df, dm, mass_per_mm=0.116, billet_mass=95.0, c_yr=309600.0, full_sav=0.53)

    full = _eval_full_model(ctx)
    assert full.name == "full_model"
    assert full.earns_place is True

    fixed = _eval_fixed_cut(ctx)
    assert fixed.name == "fixed_static_cut"
    assert fixed.recovery_pct == 0.0

    no_conf = _eval_no_conformal(ctx)
    assert no_conf.name == "no_conformal"

    no_ukf = _eval_no_ukf(ctx)
    assert no_ukf.name == "no_ukf"

    no_hier = _eval_no_hierarchy(ctx)
    assert no_hier.name == "no_hierarchy"

    naive_d2 = _eval_second_deriv_rule(ctx)
    assert naive_d2.name == "second_deriv_rule"


def test_run_ablation_study_end_to_end():
    if not (ARTIFACTS_DIR / "decision_eval.parquet").exists():
        pytest.skip("decision_eval.parquet missing")

    variants = run_ablation_study()
    assert len(variants) == 9

    names = [v.name for v in variants]
    assert "full_model" in names
    assert "no_conformal" in names
    assert "no_ukf" in names
    assert "no_hierarchy" in names
    assert "no_cross_project" in names
    assert "no_regime_switching" in names
    assert "no_fpca" in names
    assert "fixed_static_cut" in names
    assert "second_deriv_rule" in names

    # Done-when check: every layer has a measured contribution
    for v in variants:
        if v.name != "full_model":
            assert v.delta_saving_eur != 0.0 or v.name == "fixed_static_cut"

    report = generate_report_markdown(variants)
    assert "# Task 5.2: Ablation Study" in report
    assert "Ablation Comparison Matrix" in report
    assert "Systematic Layer Contribution" in report
