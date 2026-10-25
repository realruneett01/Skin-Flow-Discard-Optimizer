"""Randomised checks of the safety gate and the contract validator: whatever comes in, the output is bounded and advisory."""
import numpy as np
import pytest

from skinflow_discard_optimizer.contracts.schema import load_contract
from skinflow_discard_optimizer.contracts.validator import check_array
from skinflow_discard_optimizer.core.safety import CutCandidate, SafetyGate, StrokeTelemetry

GATE = SafetyGate()


def clean_stroke(rng, n=None):
    n = int(n or rng.integers(60, 400))
    t = np.cumsum(rng.uniform(0.0005, 0.002, n))
    x = np.linspace(0.0, rng.uniform(300, 800), n)
    cap = 200 + 50 * np.sin(np.linspace(0, 3, n)) + rng.normal(0, 1.0, n)
    rod = 20 + 5 * np.cos(np.linspace(0, 2, n)) + rng.normal(0, 0.5, n)
    return t, x, cap, rod


DEFECTS = ["nan", "inf", "frozen_cap", "frozen_x", "frozen_rod", "dt_zero", "dt_negative", "x_high", "x_low",
           "cap_high", "cap_negative", "rod_high", "too_short", "nan_time"]


def inject(defect, rng, t, x, cap, rod):
    t, x, cap, rod = (a.copy() for a in (t, x, cap, rod))
    i = int(rng.integers(1, len(t) - 1))
    if defect == "nan":
        [x, cap, rod][int(rng.integers(0, 3))][i] = np.nan
    elif defect == "inf":
        [x, cap, rod][int(rng.integers(0, 3))][i] = np.inf
    elif defect == "frozen_cap":
        cap[:] = cap[0]
    elif defect == "frozen_x":
        x[:] = x[0]
    elif defect == "frozen_rod":
        rod[:] = rod[0]
    elif defect == "dt_zero":
        t[i] = t[i - 1]
    elif defect == "dt_negative":
        t[i] = t[i - 1] - 0.01
    elif defect == "x_high":
        x[i] = 1501.0 + rng.uniform(0, 1000)
    elif defect == "x_low":
        x[i] = -1.0 - rng.uniform(0.1, 100)
    elif defect == "cap_high":
        cap[i] = 450.1 + rng.uniform(0, 500)
    elif defect == "cap_negative":
        cap[i] = -rng.uniform(0.1, 50)
    elif defect == "rod_high":
        rod[i] = 450.1 + rng.uniform(0, 500)
    elif defect == "too_short":
        k = int(rng.integers(0, 49))
        t, x, cap, rod = t[:k], x[:k], cap[:k], rod[:k]
    elif defect == "nan_time":
        t[i] = np.nan
    return t, x, cap, rod


@pytest.mark.parametrize("seed", range(150))
def test_clean_telemetry_has_no_issues(seed):
    rng = np.random.default_rng(seed)
    t, x, cap, rod = clean_stroke(rng)
    assert GATE.validate_stroke_telemetry(StrokeTelemetry(t, x, cap, rod)) == []


@pytest.mark.parametrize("seed", range(len(DEFECTS) * 20))
def test_every_injected_defect_is_caught(seed):
    rng = np.random.default_rng(1000 + seed)
    defect = DEFECTS[seed % len(DEFECTS)]
    t, x, cap, rod = inject(defect, rng, *clean_stroke(rng))
    issues = GATE.validate_stroke_telemetry(StrokeTelemetry(t, x, cap, rod))
    assert issues, f"{defect} was not detected"


@pytest.mark.parametrize("seed", range(300))
def test_guarded_decision_is_always_bounded_and_advisory(seed):
    rng = np.random.default_rng(5000 + seed)
    raw = rng.choice([None, np.nan, np.inf, -np.inf, rng.uniform(-100, 300), rng.uniform(12, 60)])
    cand = CutCandidate(
        raw_cut_mm=None if raw is None else float(raw),
        confidence=str(rng.choice(["high", "medium", "low", "none"])),
        fault_class=str(rng.choice(["none", "die_wear", "unknown", "liner_scale"])),
        is_novel_fault=bool(rng.integers(0, 2)),
        telemetry_issues=tuple(["X"] * int(rng.integers(0, 3))),
    )
    g = GATE.guard_cut_decision(cand)
    assert np.isfinite(g.h_cut_mm) and GATE.cfg.min_cut_mm <= g.h_cut_mm <= GATE.cfg.max_cut_mm
    assert g.is_advisory
    assert all(k.startswith("Predictor.") for k in g.published_tags)
    should_fall_back = (cand.raw_cut_mm is None or not np.isfinite(cand.raw_cut_mm) or cand.confidence in ("none", "low")
                        or cand.is_novel_fault or cand.fault_class == "unknown" or bool(cand.telemetry_issues))
    assert g.is_fallback is should_fall_back
    if g.is_fallback:
        assert g.h_cut_mm == GATE.cfg.static_cut_mm and g.alert_level == "CRITICAL_FALLBACK"


@pytest.mark.parametrize("seed", range(120))
def test_process_context(seed):
    rng = np.random.default_rng(9000 + seed)
    alloy = str(rng.choice(["AA6063", "AA6082", "AA7075", "", "aa6063"]))
    bt, lt = float(rng.uniform(250, 650)), float(rng.uniform(200, 600))
    if seed % 10 == 0:
        bt = float(rng.choice([np.nan, np.inf, -np.inf]))
    issues = GATE.validate_process_context(alloy, bt, lt)
    ok = alloy in ("AA6063", "AA6082") and 350 <= bt <= 580 and 300 <= lt <= 520
    assert (issues == []) is ok


NUMERIC = [n for n, s in load_contract().signals.items() if s.type in ("float", "int")]


@pytest.mark.parametrize("seed", range(200))
def test_check_array_matches_definition(seed):
    rng = np.random.default_rng(12_000 + seed)
    name = NUMERIC[seed % len(NUMERIC)]
    lo, hi = load_contract().signals[name].range
    span = hi - lo
    v = rng.uniform(lo - span, hi + span, 40)
    v[rng.integers(0, 40, 3)] = rng.choice([np.nan, np.inf, -np.inf], 3)
    got = check_array(name, v)
    with np.errstate(invalid="ignore"):
        want = np.isfinite(v) & (v >= lo) & (v <= hi)
    assert np.array_equal(got, want)
