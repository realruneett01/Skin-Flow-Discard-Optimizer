"""End-to-end fuzz: corrupted cycles through the whole service must never crash or publish an unbounded or non-advisory cut."""
from dataclasses import replace

import numpy as np
import pytest

from skinflow_discard_optimizer.core.features import inputs_from_row
from skinflow_discard_optimizer.deploy.bootstrap import REQUIRED_ARTIFACTS
from skinflow_discard_optimizer.paths import ARTIFACTS_DIR
from skinflow_discard_optimizer.service.run import ServiceConfig, StreamingOptimizerService
from skinflow_discard_optimizer.sim.scenario import Scenario, run_scenario

pytestmark = pytest.mark.skipif(not all((ARTIFACTS_DIR / a).exists() for a in REQUIRED_ARTIFACTS),
                                reason="models not built (run `skdo bootstrap`)")

CORRUPTIONS = ["none", "nan_x", "nan_cap", "inf_rod", "frozen_cap", "short", "tiny", "neg_time", "dup_time", "huge_cap",
               "neg_cap", "unseen_alloy", "cold_billet", "nan_temp", "nan_liner", "empty_alloy", "shuffled", "x_far", "stops_short", "bad_length"]


@pytest.fixture(scope="module")
def base_cycles():
    sc = Scenario.load("healthy_baseline").with_cycles(8)
    return [inputs_from_row(r, s) for r, s in run_scenario(sc, keep_strokes=True)]


@pytest.fixture(scope="module")
def service():
    return StreamingOptimizerService(ServiceConfig(latency_budget_ms=1e9))


def corrupt(inp, kind, rng):
    t, x, cap, rod = (a.copy() for a in (inp.t_s, inp.x_mm, inp.p_cap_bar, inp.p_rod_bar))
    n = len(t)
    i = int(rng.integers(1, n - 1))
    kw = {}
    if kind == "nan_x":
        x[i] = np.nan
    elif kind == "nan_cap":
        cap[i:i + 5] = np.nan
    elif kind == "inf_rod":
        rod[i] = np.inf
    elif kind == "frozen_cap":
        cap[:] = cap[0]
    elif kind == "short":
        k = int(rng.integers(10, 49))
        t, x, cap, rod = t[:k], x[:k], cap[:k], rod[:k]
    elif kind == "tiny":
        t, x, cap, rod = t[:2], x[:2], cap[:2], rod[:2]
    elif kind == "neg_time":
        t[i] = t[i - 1] - 0.5
    elif kind == "dup_time":
        t[i] = t[i - 1]
    elif kind == "huge_cap":
        cap[i] = 1e6
    elif kind == "neg_cap":
        cap[i] = -50.0
    elif kind == "unseen_alloy":
        kw["alloy_id"] = "AA7075"
    elif kind == "cold_billet":
        kw["billet_temp_C"] = 200.0
    elif kind == "nan_temp":
        kw["billet_temp_C"] = float("nan")
    elif kind == "nan_liner":
        kw["liner_temps_C"] = (float("nan"),) * 4
    elif kind == "empty_alloy":
        kw["alloy_id"] = ""
    elif kind == "shuffled":
        p = rng.permutation(n)
        x, cap, rod = x[p], cap[p], rod[p]
    elif kind == "x_far":
        x = x + 5000.0
    elif kind == "stops_short":
        x = x * float(rng.uniform(0.5, 0.88))
    elif kind == "bad_length":
        kw["billet_length_mm"] = float(rng.choice([float("nan"), 0.0, -5.0, 1e9]))
    return replace(inp, t_s=t, x_mm=x, p_cap_bar=cap, p_rod_bar=rod, **kw)


@pytest.mark.parametrize("seed", range(len(CORRUPTIONS) * 12))
def test_corrupted_cycle_is_bounded_advisory_and_never_crashes(seed, service, base_cycles):
    rng = np.random.default_rng(700_000 + seed)
    kind = CORRUPTIONS[seed % len(CORRUPTIONS)]
    inp = corrupt(base_cycles[int(rng.integers(0, len(base_cycles)))], kind, rng)
    entry = service.process_cycle_inputs(inp)
    assert np.isfinite(entry.cut_recommendation_mm) and 12.0 <= entry.cut_recommendation_mm <= 60.0, kind
    assert all(k.startswith("Predictor.") for k in entry.published_tags), kind
    tags = entry.published_tags
    assert np.isfinite(tags["Predictor.CutIntervalLowMm"]) and np.isfinite(tags["Predictor.CutIntervalHighMm"])
    assert tags["Predictor.CutConfidence"] in ("high", "medium", "low", "fallback")
    if kind != "none":
        assert entry.is_fallback, f"{kind} should have engaged the safety fallback"
    if entry.is_fallback:
        assert entry.cut_recommendation_mm == 40.0
