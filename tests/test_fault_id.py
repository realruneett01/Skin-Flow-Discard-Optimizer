"""Unit tests for Task 4.4 Fault-type identification and root-cause classifier."""
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skinflow_discard_optimizer.aware.fault_id import (
    ALL_FAULT_CLASSES,
    KNOWN_FAULTS,
    SUGGESTED_CHECKS,
    FaultClassifier,
    FaultDiagnosis,
    FaultSignatures,
)
from skinflow_discard_optimizer.paths import REPO_ROOT


HEALTHY_FEATS: dict[str, float] = {
    "theta_F_tool_N": 5.85e5,
    "theta_mu": 0.552,
    "theta_sigma_scale": 1.008,
    "billet_temp_C": 470.0,
    "dT_K": 40.0,
    "oil_temp_C": 45.0,
    "supply_pressure_min_bar": 299.0,
    "fpca_1": 44000.0,
    "fpca_2": -18000.0,
    "fpca_3": -2000.0,
    "fpca_spe": 6e9,
    "upturn_h_mm": 31.2,
}


@pytest.fixture
def classifier() -> FaultClassifier:
    return FaultClassifier()


def _assert_diagnosis_fields(
    diag: FaultDiagnosis,
    expected_fault: str,
    expected_active: bool,
    expected_keyword: str | None = None,
) -> None:
    """Helper to verify common diagnosis fields without duplicating assertions."""
    assert diag.dominant_fault == expected_fault
    assert diag.is_fault_active is expected_active
    if expected_keyword:
        assert expected_keyword in diag.suggested_action.lower()


def test_classifier_signatures_structure(classifier: FaultClassifier):
    sigs = classifier.signatures
    assert isinstance(sigs, FaultSignatures)
    assert set(sigs.classes) == set(KNOWN_FAULTS)
    assert len(sigs.feature_cols) >= 10
    assert sigs.threshold_unknown > 0.0
    for c in sigs.classes:
        assert c in sigs.means and c in sigs.variances and len(sigs.means[c]) == len(sigs.feature_cols)


def test_classify_healthy_cycle(classifier: FaultClassifier):
    diag = classifier.classify(cycle=100, features=HEALTHY_FEATS)
    assert isinstance(diag, FaultDiagnosis)
    assert diag.cycle == 100
    assert diag.confidence > 0.50
    assert len(diag.top_causes) == 3
    _assert_diagnosis_fields(diag, "none", False, "operating")


@pytest.mark.parametrize(
    "case",
    [
        ("die_wear", 9000, "die_wear", "die"),
        ("supply_pressure_sag", 8000, "supply_pressure_sag", "oil"),
        ("liner_scale", 9000, "liner_scale", "liner"),
        ("sensor_gain_drift", 9000, "sensor_gain_drift", "transducer"),
    ],
)
def test_classify_known_fault_scenarios(classifier: FaultClassifier, case: tuple):
    scenario, cycle, expected_fault, expected_keyword = case
    df = pd.read_parquet(REPO_ROOT / "data" / "features" / f"{scenario}.parquet")
    feats: dict[str, Any] = df[df.cycle == cycle].iloc[0].to_dict()
    diag = classifier.classify(cycle=cycle, features=feats)
    assert diag.confidence > 0.50
    _assert_diagnosis_fields(diag, expected_fault, True, expected_keyword)


def test_classify_unknown_novel_anomaly(classifier: FaultClassifier):
    unseen_feats = dict(
        HEALTHY_FEATS,
        theta_F_tool_N=1.5e6,
        theta_mu=0.95,
        theta_sigma_scale=1.40,
        billet_temp_C=350.0,
        dT_K=80.0,
        oil_temp_C=90.0,
        supply_pressure_min_bar=150.0,
        fpca_1=50.0,
        fpca_2=50.0,
        fpca_3=50.0,
        fpca_spe=1e16,
        upturn_h_mm=5.0,
    )
    diag = classifier.classify(cycle=9000, features=unseen_feats)
    _assert_diagnosis_fields(diag, "unknown", True, "does not match")


def test_diagnosis_serialization(classifier: FaultClassifier):
    diag = classifier.classify(cycle=200, features=HEALTHY_FEATS)
    d = diag.as_dict()
    assert d["cycle"] == 200
    assert d["dominant_fault"] in ALL_FAULT_CLASSES
    assert isinstance(d["posteriors"], dict)
    assert set(d["posteriors"].keys()) == set(ALL_FAULT_CLASSES)
    assert np.isclose(sum(d["posteriors"].values()), 1.0)
