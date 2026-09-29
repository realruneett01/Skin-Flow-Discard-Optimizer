"""GPU batch trajectories must match the CPU engine (same algorithm, float64)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from skinflow_discard_optimizer.accel import get_device  # noqa: E402
from skinflow_discard_optimizer.accel.gpu_engine import batch_trajectories  # noqa: E402
from skinflow_discard_optimizer.core.estimator.decision import AdaptiveConformal, DecisionSettings  # noqa: E402
from skinflow_discard_optimizer.core.estimator.hierarchical import HierarchicalModel  # noqa: E402
from skinflow_discard_optimizer.core.estimator.pipeline import (  # noqa: E402
    GROUPS, ONSET_FEATURES, PRIOR_FEATURES, CutEngine, inputs_from_row, resolve_commit,
)
from skinflow_discard_optimizer.sim.scenario import Scenario, run_scenario  # noqa: E402


def _toy_models():
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([rng.normal(0.55, 0.01, n), rng.normal(40, 3, n), rng.normal(470, 3, n)])
    g = {"die_id": rng.choice(["D-101", "D-102"], n), "alloy_id": np.array(["AA6063"] * n)}
    prior = HierarchicalModel(PRIOR_FEATURES, GROUPS).fit(X, 21 + 5 * (X[:, 0] - 0.55) + rng.normal(0, 1.3, n),
                                                          g, 200, 100)
    onset = HierarchicalModel(ONSET_FEATURES, GROUPS).fit(X, -10 + rng.normal(0, 1.2, n), g, 200, 100, seed=1)
    return prior, onset


@pytest.fixture(scope="module")
def setup():
    prior, onset = _toy_models()
    eng = CutEngine(prior_model=prior, onset_model=onset)
    rows = []
    for name, start in (("healthy_baseline", 0), ("lubricant_loss", 6000), ("flash_spike", 6000),
                        ("encoder_offset", 7000)):
        for r, s in run_scenario(Scenario.load(name).with_cycles(start + 3), keep_strokes=True, start=start):
            rows.append((r, s))
    inputs = [inputs_from_row(r, s) for r, s in rows]
    return eng, inputs


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_batch_trajectories_match_cpu_engine(setup, dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    eng, inputs = setup
    gpu = batch_trajectories(eng, inputs, device=torch.device(dev), chunk=5)
    for si, tg in zip(inputs, gpu):
        tc = eng.trajectory(si)
        assert np.allclose(tg.h, tc.h, atol=1e-9)
        assert np.array_equal(tg.ck_idx, tc.ck_idx)
        assert tg.conf == tc.conf
        assert np.allclose(tg.m, tc.m, atol=1e-6) and np.allclose(tg.s, tc.s, atol=1e-6)
        assert np.allclose(tg.onset_mean, tc.onset_mean, atol=1e-6, equal_nan=True)
        assert np.isclose(tg.glr_fired_h, tc.glr_fired_h, equal_nan=True)
        assert np.isclose(tg.bocpd_fired_h, tc.bocpd_fired_h, equal_nan=True)
        assert tg.features["theta_mu"] == pytest.approx(tc.features["theta_mu"], abs=1e-9)


def test_resolved_decisions_match(setup):
    eng, inputs = setup
    st = DecisionSettings.load()
    cal = AdaptiveConformal(np.abs(np.random.default_rng(1).normal(0, 1, 500)), st.coverage_target, 0.0)
    for si, tg in zip(inputs, batch_trajectories(eng, inputs, device=get_device())):
        a = resolve_commit(tg, st, cal)
        b = resolve_commit(eng.trajectory(si), st, cal)
        assert a.h_cut_mm == pytest.approx(b.h_cut_mm, abs=1e-6)
        assert a.late == b.late and a.onset_confidence == b.onset_confidence
