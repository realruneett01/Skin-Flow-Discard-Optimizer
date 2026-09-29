"""Tasks 1.3 and 1.4: drift/fault injection, scenario library, dataset builder."""
import numpy as np
import pandas as pd
import pytest

from skinflow_discard_optimizer.sim.build_dataset import EMBARGO, assign_split, build
from skinflow_discard_optimizer.sim.cycle import regenerate_stroke
from skinflow_discard_optimizer.sim.defect_model import PressState
from skinflow_discard_optimizer.sim.faults import FAULT_KINDS, FaultSpec, PressProcess
from skinflow_discard_optimizer.sim.scenario import Scenario, list_scenarios, run_scenario

# --- fault primitives --------------------------------------------------------------------


@pytest.mark.parametrize("ramp,expect", [
    ("step", [0, 1, 1, 1]),
    ("linear", [0, 0.01, 0.5, 1.0]),
])
def test_fault_activation_shapes(ramp, expect):
    f = FaultSpec("die_wear", onset_cycle=100, severity=1.0, ramp=ramp, ramp_cycles=100)
    got = [f.activation(n) for n in (99, 100, 149, 400)]
    assert got == pytest.approx(expect)


def test_fault_end_cycle_and_validation():
    f = FaultSpec("lubricant_loss", 10, ramp="step", end_cycle=20)
    assert f.activation(15) == 1.0 and f.activation(20) == 0.0
    with pytest.raises(ValueError):
        FaultSpec("gremlins", 0)


def test_ou_wander_is_stationary_without_faults():
    proc = PressProcess(PressState(), rng=np.random.default_rng(0))
    temps = np.array([proc.step(n).state.billet_temp_C for n in range(4000)])
    th, sd = proc.config.ou_theta["billet_temp_C"], proc.config.ou_sigma["billet_temp_C"]
    stationary_sd = sd / np.sqrt(1 - (1 - th) ** 2)
    assert abs(temps.mean() - 470.0) < 1.0
    assert np.std(temps[500:]) == pytest.approx(stationary_sd, rel=0.2)


def test_supply_sag_couples_to_oil_temperature():
    proc = PressProcess(PressState(), [FaultSpec("supply_pressure_sag", 0, 1.0, "step")],
                        rng=np.random.default_rng(1), ou_scale=0.0)
    c = proc.step(5)
    assert c.effects.oil_temp_C > 60
    assert c.effects.supply_sag_bar > 0 and c.effects.speed_scale < 1 and c.effects.cap_bias_bar < 0
    healthy = PressProcess(PressState(), rng=np.random.default_rng(1), ou_scale=0.0).step(5)
    assert healthy.effects.supply_sag_bar == pytest.approx(0.0, abs=0.05)


def test_sensor_faults_do_not_touch_physical_state():
    kw = dict(rng=np.random.default_rng(2), ou_scale=0.0)
    a = PressProcess(PressState(), **kw).step(10)
    b = PressProcess(PressState(), [FaultSpec("sensor_gain_drift", 0, 1.0, "step"),
                                    FaultSpec("encoder_offset", 0, 1.0, "step")], **kw).step(10)
    assert a.state == b.state
    assert b.effects.cap_gain > 1 and b.effects.encoder_offset_mm > 0


# --- Done-when: "die wear starts at cycle 2000, severity 0.5" and the labels record it -----


def test_scenario_file_specifies_fault_and_labels_record_it(tmp_path):
    p = tmp_path / "wear.yaml"
    p.write_text(
        "name: wear_test\nseed: 7\nn_cycles: 2600\n"
        "faults:\n  - {kind: die_wear, onset_cycle: 2000, severity: 0.5, ramp: linear, ramp_cycles: 500}\n"
    )
    scn = Scenario.load(p)
    df = pd.DataFrame([r for r, _ in run_scenario(scn)])
    assert (df.loc[df.cycle < 2000, "fault_die_wear"] == 0).all()
    assert (df.loc[df.cycle < 2000, "fault_class"] == "none").all()
    assert df.loc[df.cycle == 2000, "fault_die_wear"].item() > 0
    assert df.loc[df.cycle >= 2499, "fault_die_wear"].max() == pytest.approx(0.5)
    assert (df.loc[df.cycle >= 2000, "fault_class"] == "die_wear").all()
    # the physical latent state follows the label
    late = df[df.cycle > 2500].true_die_wear.mean() - df[df.cycle < 2000].true_die_wear.mean()
    assert late == pytest.approx(0.5 * 0.8, abs=0.05)


def test_all_library_scenarios_parse_and_cover_every_fault():
    names = list_scenarios()
    assert len(names) >= 12
    kinds = {f.kind for n in names for f in Scenario.load(n).faults}
    assert kinds == set(FAULT_KINDS)
    types = {e.type for n in names for e in Scenario.load(n).events}
    assert types == {"die_change", "alloy_change", "cold_start"}
    assert sum(Scenario.load(n).n_cycles for n in names) >= 200_000


def test_events_change_context():
    scn = Scenario.load("alloy_change").with_cycles(7002)
    rows = [r for r, _ in run_scenario(scn, start=6998)]
    assert [r["alloy_id"] for r in rows] == ["AA6063"] * 2 + ["AA6082"] * 2
    scn = Scenario.load("die_change").with_cycles(7001)
    rows = [r for r, _ in run_scenario(scn, start=6999)]
    assert rows[0]["die_id"] == "D-105" and rows[1]["die_id"] == "D-106"
    assert rows[1]["true_die_wear"] < 0.1 < rows[0]["true_die_wear"]


def test_billet_length_and_speed_vary_per_cycle():
    scn = Scenario.load("healthy_baseline").with_cycles(400)
    df = pd.DataFrame([r for r, _ in run_scenario(scn)])
    assert df.billet_length_mm.std() == pytest.approx(6.0, rel=0.2)
    assert df.spec_ram_speed_mm_s.std() == pytest.approx(0.8, rel=0.2)
    # upset length follows the billet length
    assert np.corrcoef(df.billet_length_mm, df.spec_L0_mm)[0, 1] > 0.999


def test_stroke_regenerates_exactly_from_row():
    scn = Scenario.load("flash_spike").with_cycles(6003)
    (row, stroke), = list(run_scenario(scn, keep_strokes=True, start=6002))
    assert row["eff_flash_amp_frac"] > 0
    again = regenerate_stroke(row)
    assert np.array_equal(stroke.p_cap_bar, again.p_cap_bar)
    assert np.array_equal(stroke.x_mm, again.x_mm)


def test_rows_identical_with_and_without_strokes():
    scn = Scenario.load("encoder_offset").with_cycles(30)
    a = [r for r, _ in run_scenario(scn)]
    b = [r for r, _ in run_scenario(scn, keep_strokes=True)]
    for ra, rb in zip(a, b):
        for k in ra:
            if k not in ("pump_energy_kwh", "stroke_duration_s"):
                assert ra[k] == rb[k] or (ra[k] != ra[k] and rb[k] != rb[k]), k
        assert ra["pump_energy_kwh"] == pytest.approx(rb["pump_energy_kwh"], rel=2e-3)


# --- splits and reproducible build ----------------------------------------------------------


def test_split_is_by_time_with_embargo():
    cyc = np.arange(10_000)
    s = assign_split("die_wear", cyc, 10_000)
    order = {"train": 0, "val": 1, "test": 2}
    ranks = [order[x] for x in s if x != "gap"]
    assert ranks == sorted(ranks)                           # never interleaved
    gaps = cyc[s == "gap"]
    assert len(gaps) == 2 * EMBARGO
    assert set(assign_split("die_change", cyc, 10_000)) == {"test_scenario"}


def test_dataset_builds_reproducibly(tmp_path):
    kw = dict(scale=0.01, scenarios=["healthy_baseline", "die_wear", "die_change"], workers=1)
    m1 = build(tmp_path / "a", **kw)
    m2 = build(tmp_path / "b", **kw)
    assert m1["table_hash"] == m2["table_hash"]
    df = pd.read_parquet(tmp_path / "a" / "cycles.parquet")
    assert {"h_crit_mm", "oracle_cut_mm", "fault_class", "split", "stroke_seed"} <= set(df.columns)
    assert df.groupby("scenario").cycle.is_monotonic_increasing.all()
