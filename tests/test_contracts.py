"""Task 0.2: signal contract, generated models and validators."""
import copy

import numpy as np
import pytest
import yaml

from skinflow_discard_optimizer.contracts.schema import SignalContract, load_contract
from skinflow_discard_optimizer.contracts.validator import (
    AdvisoryViolation,
    ContractViolation,
    check_array,
    ensure_advisory,
    validate_array,
    validate_output,
    validate_record,
    validate_value,
)
from skinflow_discard_optimizer.paths import SIGNALS_YAML

REQUIRED_INPUTS = [
    "ram_cap_pressure", "ram_rod_pressure", "ram_position",
    "container_liner_temp_1", "container_liner_temp_2",
    "container_liner_temp_3", "container_liner_temp_4",
    "billet_temp", "oil_temp", "pump_supply_pressure", "pump_power",
    "cycle_phase", "butt_shear_position",
]
REQUIRED_OUTPUTS = [
    "Predictor.ButtCutMm", "Predictor.CutConfidence",
    "Predictor.DriftAlarm", "Predictor.FaultClass",
]

GOOD_RECORD = {
    "cycle_id": 1204,
    "cycle_phase": "extrusion",
    "billet_temp": 470.0,
    "billet_length": 850.0,
    "container_liner_temp_1": 425.0,
    "oil_temp": 45.5,
    "alloy_id": "AA6063",
    "die_id": "D-117",
}


@pytest.fixture(scope="module")
def raw_yaml():
    with open(SIGNALS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_contract_loads_and_has_required_signals():
    c = load_contract()
    for name in REQUIRED_INPUTS:
        assert name in c.signals, name
    for name in REQUIRED_OUTPUTS:
        assert name in c.outputs.signals, name


def test_every_signal_has_tag_unit_rate_type_owner():
    c = load_contract()
    for name, s in c.signals.items():
        assert s.tag.startswith("ns=2;s="), name
        assert s.unit and s.type and s.owner, name
        if s.type in ("float", "int"):
            assert s.range[0] < s.range[1], name


def test_phase_aliases_map_both_projects():
    c = load_contract()
    assert c.canonical_phase("main_extrusion_stroke", "hpeo") == "extrusion"
    assert c.canonical_phase("die_slide_shift", "hpeo") == "die_slide"
    assert c.canonical_phase("rapid_advance", "dcto") == "rapid_advance"
    with pytest.raises(KeyError):
        c.canonical_phase("warp_drive", "dcto")


# --- the validator passes on sample data -----------------------------------------

def test_valid_record_passes():
    rec = validate_record(GOOD_RECORD)
    assert rec.billet_temp == 470.0
    assert rec.ram_cap_pressure is None  # fields are optional


def test_numpy_scalars_accepted():
    rec = validate_record({"cycle_id": np.int64(5), "oil_temp": np.float64(40.0)})
    assert rec.cycle_id == 5


def test_valid_high_rate_array_passes():
    cap = np.linspace(20, 310, 1000)
    out = validate_array("ram_cap_pressure", cap)
    assert out.shape == (1000,)


def test_valid_outputs_pass():
    assert validate_output("Predictor.ButtCutMm", 32.5) == ("ns=2;s=Predictor.ButtCutMm", 32.5)
    assert validate_output("Predictor.CutConfidence", "high")[1] == "high"
    assert validate_output("Predictor.DriftAlarm", False)[1] is False
    assert validate_output("Predictor.FaultClass", "die_wear")[1] == "die_wear"


# --- and fails on bad data ---------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("billet_temp", 900.0),          # above range
    ("billet_temp", float("nan")),   # not finite
    ("oil_temp", -20.0),             # below range
    ("cycle_phase", "warp_drive"),   # not in enum
    ("cycle_id", "12"),              # wrong type (strict)
    ("cycle_id", 1.5),               # float for int
    ("alloy_id", ""),                # empty string
    ("not_a_signal", 1.0),           # unknown key
])
def test_bad_record_rejected(field, value):
    rec = dict(GOOD_RECORD)
    rec[field] = value
    with pytest.raises(ContractViolation) as exc:
        validate_record(rec)
    assert field in str(exc.value)


def test_bad_array_rejected_and_mask_marks_bad_samples():
    cap = np.full(100, 150.0)
    cap[[3, 50]] = [500.0, np.nan]
    mask = check_array("ram_cap_pressure", cap)
    assert mask.sum() == 98 and not mask[3] and not mask[50]
    with pytest.raises(ContractViolation, match="2 of 100"):
        validate_array("ram_cap_pressure", cap)


@pytest.mark.parametrize("name,value", [
    ("Predictor.ButtCutMm", -1.0),
    ("Predictor.ButtCutMm", 1e6),
    ("Predictor.CutConfidence", "certain"),
    ("Predictor.DriftAlarm", "yes"),
])
def test_bad_outputs_rejected(name, value):
    with pytest.raises(ContractViolation):
        validate_output(name, value)


def test_single_value_validator():
    assert validate_value("ram_position", 412.0) == 412.0
    with pytest.raises(ContractViolation):
        validate_value("ram_position", 1e5)


# --- advisory-only rule --------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "ram_cap_pressure",          # an input signal
    "Press01.Shear.Setpoint",    # a setpoint
    "PumpSetpoints/StagingCode", # HPEO setpoint
    "Predictor.Unregistered",    # right prefix, not in contract
])
def test_publishing_outside_predictor_namespace_refused(name):
    with pytest.raises(AdvisoryViolation):
        ensure_advisory(name)


def test_outputs_confined_to_predictor_namespace(raw_yaml):
    bad = copy.deepcopy(raw_yaml)
    bad["outputs"]["signals"]["Shear.CutSetpoint"] = {
        "tag": "ns=2;s=Shear.CutSetpoint", "unit": "mm", "type": "float",
        "range": [0, 100], "description": "not allowed",
    }
    with pytest.raises(ValueError, match="outside the Predictor"):
        SignalContract.model_validate(bad)


def test_contract_rejects_duplicate_tags_and_bad_ranges(raw_yaml):
    dup = copy.deepcopy(raw_yaml)
    dup["signals"]["oil_temp"]["tag"] = dup["signals"]["billet_temp"]["tag"]
    with pytest.raises(ValueError, match="duplicate"):
        SignalContract.model_validate(dup)

    rng = copy.deepcopy(raw_yaml)
    rng["signals"]["oil_temp"]["range"] = [90.0, 0.0]
    with pytest.raises(ValueError, match="min < max"):
        SignalContract.model_validate(rng)
