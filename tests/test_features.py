"""Task 2.3: feature store; identical features from simulated data and a replayed tag stream."""
import numpy as np
import pandas as pd
import pytest

from skinflow_discard_optimizer.core.assembler import CycleAssembler
from skinflow_discard_optimizer.core.features import FeatureExtractor, fit_theta, inputs_from_row
from skinflow_discard_optimizer.core.functional import FPCA, curve_from_stroke
from skinflow_discard_optimizer.core.store import ParquetStore
from skinflow_discard_optimizer.service.replay import build_recording, load_recording, replay, save_recording
from skinflow_discard_optimizer.sim.cycle import regenerate_stroke
from skinflow_discard_optimizer.sim.force_model import Press
from skinflow_discard_optimizer.sim.scenario import Scenario, run_scenario


@pytest.fixture(scope="module")
def press():
    return Press.load()


@pytest.fixture(scope="module")
def extractor(press):
    rows = [r for r, _ in run_scenario(Scenario.load("healthy_baseline").with_cycles(80))]
    X = np.array([curve_from_stroke(regenerate_stroke(r, press), press) for r in rows])
    return FeatureExtractor(press, FPCA.fit(X, n_components=4))


@pytest.fixture(scope="module")
def cycles(press):
    scn = Scenario.load("die_change").with_cycles(7003)
    out = list(run_scenario(scn, keep_strokes=True, start=6998))   # spans the die change
    return [r for r, _ in out], [s for _, s in out]


def test_theta_fit_recovers_truth(press, cycles, extractor):
    rows, strokes = cycles
    for r, s in zip(rows, strokes):
        f = extractor.extract(inputs_from_row(r, s))
        assert f["theta_mu"] == pytest.approx(r["spec_mu"], abs=0.02)
        assert f["theta_sigma_scale"] == pytest.approx(1.0, abs=0.03)
        assert f["theta_F_tool_N"] == pytest.approx(r["spec_F_tool_N"], rel=0.15)


def test_upturn_location_matches_true_onset(press, extractor):
    rows = [r for r, _ in run_scenario(Scenario.load("healthy_baseline").with_cycles(15))]
    errs = []
    for r in rows:
        f = extractor.extract(inputs_from_row(r, regenerate_stroke(r, press)))
        errs.append(f["upturn_h_mm"] - r["spec_h_onset_mm"])
    errs = np.array(errs)
    assert np.all(np.isfinite(errs))
    assert abs(errs.mean()) < 0.8 and errs.std() < 1.2


def test_no_sample_past_the_stop_point_is_used(press, extractor):
    """Causality: features must not change if the data past the stop point changes."""
    r = next(run_scenario(Scenario.load("healthy_baseline").with_cycles(1)))[0]
    s = regenerate_stroke(r, press)
    fx = FeatureExtractor(press, extractor.fpca, h_stop_mm=40.0)   # e.g. stopped at the static cut
    a = fx.extract(inputs_from_row(r, s))
    L0 = press.upset_length(r["billet_length_mm"])
    past = (L0 - s.x_mm) < 40.0
    s.p_cap_bar = s.p_cap_bar.copy()
    s.p_cap_bar[past] += 50.0
    b = fx.extract(inputs_from_row(r, s))
    for k, v in a.items():
        assert v == b[k] or (v != v and b[k] != b[k]), k
    # the upturn is already rising above 40 mm, so its onset can be extrapolated before it is reached
    assert a["upturn_detected"] and a["upturn_h_mm"] < 40.0


def test_same_features_from_simulation_and_replayed_tag_stream(cycles, extractor, tmp_path):
    rows, strokes = cycles
    batch = pd.DataFrame([extractor.extract(inputs_from_row(r, s)) for r, s in zip(rows, strokes)])

    rec = build_recording(rows, strokes)
    save_recording(rec, tmp_path / "rec.parquet")
    asm = CycleAssembler()
    replay(load_recording(tmp_path / "rec.parquet"), asm)
    live_inputs = asm.pop_completed()
    assert len(live_inputs) == len(rows)
    live = pd.DataFrame([extractor.extract(ci) for ci in live_inputs])

    assert list(live.cycle) == list(batch.cycle)
    assert list(live.die_id) == list(batch.die_id)
    num = [c for c in batch.columns if pd.api.types.is_numeric_dtype(batch[c]) and c != "upturn_detected"]
    for c in num:
        assert np.allclose(live[c], batch[c], rtol=1e-6, atol=1e-6, equal_nan=True), c


def test_assembler_rejects_out_of_range_samples(cycles):
    rows, strokes = cycles
    rec = build_recording(rows[:1], strokes[:1])
    bad = rec.signal == "ram_cap_pressure"
    rec.loc[rec.index[bad][:5], "value"] = 9999.0
    asm = CycleAssembler()
    replay(rec, asm)
    assert asm.rejected["ram_cap_pressure"] == 5
    assert len(asm.pop_completed()) == 1


def test_store_upserts_and_filters(tmp_path):
    st = ParquetStore(tmp_path)
    st.write("press01", pd.DataFrame({"cycle": [1, 2, 3], "a": [1.0, 2.0, 3.0]}))
    st.write("press01", pd.DataFrame({"cycle": [3, 4], "a": [30.0, 4.0]}))
    df = st.read("press01")
    assert list(df.cycle) == [1, 2, 3, 4] and df.a.tolist() == [1.0, 2.0, 30.0, 4.0]
    assert st.read("press01", ["a"], cycle_from=2, cycle_to=4).cycle.tolist() == [2, 3]
    assert st.streams() == ["press01"]


def test_fit_theta_exact_on_model_curve(press):
    from skinflow_discard_optimizer.sim.force_model import Alloy, force_breakdown, nominal_stroke
    spec = nominal_stroke("AA6063", press, sigma_scale=1.1, mu=0.6, F_tool_N=0.9e6)
    x = np.linspace(80, spec.L0_mm - 60, 400)
    F = force_breakdown(x, spec, press).total_N
    th = fit_theta(x, F, spec.L0_mm, spec.extrusion_ratio, spec.ram_speed_mm_s, spec.T_front_C,
                   Alloy.from_config("AA6063"), press)
    # not bit-exact: the model curve still carries a small upturn tail (h >= 60 mm) the fit omits
    assert th.sigma_scale == pytest.approx(1.1, rel=5e-3)
    assert th.mu == pytest.approx(0.6, rel=5e-3)
    assert th.F_tool_N == pytest.approx(0.9e6, rel=1e-2)
