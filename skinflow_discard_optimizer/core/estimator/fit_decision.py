"""Fit, calibrate and evaluate the cut decision (Task 3.3).

    python -m skinflow_discard_optimizer.core.estimator.fit_decision [--n-train 6000] [--stride 5]

1. **Train:** decision-time features (``CutEngine.features_at``, the live code path)
   for a sample of *train*-split cycles from every scenario. The horizon is where a
   commit would typically happen (``max(oracle cut, observable onset - 3 mm)``).
   Two hierarchical models are fitted: pre-onset (state only) and onset.
2. **Calibrate:** run the full causal engine on *val*-split cycles and take the
   conformity scores ``|h_crit - m|/s``.
3. **Evaluate:** run the engine in time order over the *test* split of every
   scenario and over the two fully held-out scenarios, with adaptive conformal
   updates from audited labels (1 in ``audit_every``, delayed ``audit_delay``
   cycles). Coverage is measured on *all* test cycles.

Writes ``artifacts/hcrit_models.npz``, ``artifacts/decision_eval.parquet`` and
``reports/task_3_3_decision.md``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from skinflow_discard_optimizer.config import load_config, value
from skinflow_discard_optimizer.core.estimator.decision import AdaptiveConformal, DecisionSettings
from skinflow_discard_optimizer.core.estimator.hierarchical import HierarchicalModel
from skinflow_discard_optimizer.core.estimator.pipeline import (
    GROUPS, ONSET_FEATURES, PRIOR_FEATURES, CutEngine, inputs_from_row, load_models, resolve_commit, save_models,
)
from skinflow_discard_optimizer.core.observer.onset import observable_onset
from skinflow_discard_optimizer.paths import REPO_ROOT, REPORTS_DIR, default_workers
from skinflow_discard_optimizer.sim.build_dataset import HELD_OUT_SCENARIOS, load_cycles
from skinflow_discard_optimizer.sim.cycle import regenerate_stroke
from skinflow_discard_optimizer.sim.defect_model import DefectModel

EVAL_PATH = REPO_ROOT / "artifacts" / "decision_eval.parquet"
_engine: CutEngine | None = None


# ---------------------------------------------------------------------------- workers

def _parse_named_args(args: tuple, kwargs: dict, spec: list[tuple[str, any]]) -> dict:
    res = {}
    for i, (k, default) in enumerate(spec):
        if i < len(args):
            res[k] = args[i]
        else:
            res[k] = kwargs.get(k, default)
    return res


def _init(with_models: bool) -> None:
    global _engine
    if with_models:
        prior, onset = load_models()
        _engine = CutEngine(prior_model=prior, onset_model=onset)
    else:
        _engine = CutEngine()


def _train_row(row: dict) -> dict:
    assert _engine is not None
    press = _engine.press
    horizon = _horizon(row, press.amplitude_at_onset_N)
    f = _engine.features_at(inputs_from_row(row, regenerate_stroke(row, press)), horizon)
    f.update(h_crit=row["h_crit_mm"], scenario=row["scenario"], cycle=row["cycle"])
    return f


def _decide_row(row: dict) -> dict:
    """Raw decision without calibration (scale 1). Calibration is replayed afterwards in time order."""
    assert _engine is not None
    d = _engine.decide(inputs_from_row(row, regenerate_stroke(row, _engine.press)))
    r = d.recommendation
    return {"scenario": row["scenario"], "cycle": row["cycle"], "split": row["split"],
            "h_crit": row["h_crit_mm"], "oracle_cut": row["oracle_cut_mm"], "static_cut": row["static_cut_mm"],
            "m": r.h_crit_mean_mm, "s": r.h_crit_sd_mm, "h_commit": d.h_commit_mm, "late_raw": d.late,
            "onset_conf": d.onset_confidence, "onset_mean": d.onset_mean_mm, "onset_fired": d.onset_fired_h,
            "compute_ms": d.compute_ms, "fault_class": row["fault_class"]}


def _run_stream(args: tuple[list[dict], np.ndarray]) -> list[dict]:
    """One scenario's test stream, in time order, with the live calibrator and a delayed audit queue.

    Each decision (including *when* it commits) uses the calibrator state at that
    cycle, which only knows labels from audited cycles at least ``audit_delay``
    cycles earlier. This is exactly the online behaviour.
    """
    assert _engine is not None
    rows, cal_scores = args
    eng = _engine

    def decide(row: dict, cal: AdaptiveConformal):
        eng.cal = cal
        return eng.decide(inputs_from_row(row, regenerate_stroke(row, eng.press)))

    return stream_loop(rows, cal_scores, eng.st, decide)


def stream_loop(rows: list[dict], cal_scores: np.ndarray, st: DecisionSettings, decide) -> list[dict]:
    """The online loop for one stream: calibrator updates from delayed audits, then decide each cycle.

    ``decide(row, cal)`` returns a ``CycleDecision``. The CPU path runs the full engine
    there; the GPU path resolves a precomputed trajectory. The calibration and audit
    logic is this one function, whichever path is used.
    """
    cal = AdaptiveConformal(cal_scores, st.coverage_target, st.aci_gamma)
    # Audits are counted in *real* production cycles (A-26: 1 billet in audit_every), not in
    # evaluated cycles. The stream may be strided, so an audited cycle is one whose cycle number
    # is a multiple of audit_every, and its label arrives audit_delay real cycles later.
    pending: list[tuple[int, float, float, float]] = []   # (available_from_cycle, y, m, s)
    out = []
    for row in rows:
        cyc = int(row["cycle"])
        while pending and pending[0][0] <= cyc:
            _, y, m, s = pending.pop(0)
            cal.update(y, m, s)
        d = decide(row, cal)
        r = d.recommendation
        # The engine works in the press's measured coordinates and the ram stops on the measured
        # position. An encoder offset of +o mm in x makes measured h read o mm low, so the true
        # thickness of the cut (and of the interval) is o mm more than the recorded number.
        off = float(row.get("eff_encoder_offset_mm", 0.0))
        # Audited labels are measured on the sectioned discard, i.e. in true thickness; the
        # model's (m, s) live in measured coordinates, so the label is mapped into those.
        if cyc % st.audit_every == 0:
            pending.append((cyc + st.audit_delay, row["h_crit_mm"] - off, r.h_crit_mean_mm, r.h_crit_sd_mm))
        lo_t, hi_t = r.interval_lo_mm + off, r.interval_hi_mm + off
        out.append({"scenario": row["scenario"], "cycle": row["cycle"], "split": row["split"],
                    "fault_class": row["fault_class"], "h_crit": row["h_crit_mm"], "oracle_cut": row["oracle_cut_mm"],
                    "static_cut": row["static_cut_mm"], "m": r.h_crit_mean_mm, "s": r.h_crit_sd_mm,
                    "s_cal": r.h_crit_sd_cal_mm, "h_cut": d.h_cut_mm + off, "h_star": r.h_cut_mm, "h_cc": r.h_cut_cc_mm,
                    "lo": lo_t, "hi": hi_t, "encoder_offset": off, "confidence": r.confidence,
                    "onset_conf": d.onset_confidence, "onset_mean": d.onset_mean_mm, "onset_fired": d.onset_fired_h,
                    "late": d.late, "h_commit": d.h_commit_mm, "alpha_t": cal.alpha_t, "compute_ms": d.compute_ms,
                    "covered": lo_t <= row["h_crit_mm"] <= hi_t})
    return out


def _pool_map(fn, rows, with_models: bool = False, **kwargs) -> list:
    workers = kwargs.get("workers", None)
    chunksize = kwargs.get("chunksize", 8)
    with ProcessPoolExecutor(max_workers=workers or default_workers(), initializer=_init,
                             initargs=(with_models,)) as ex:
        return list(ex.map(fn, rows, chunksize=chunksize))


# ---------------------------------------------------------------------------- steps

def _horizon(row: dict, a_ref: float) -> float:
    """Training horizon: roughly where a commit would happen (uses truth only to pick the horizon)."""
    on = observable_onset(row["spec_h_onset_mm"], row["spec_lam_mm"], row["spec_upturn_amp_N"], a_ref)
    return max(row["oracle_cut_mm"], on - 3.0)


def train(df: pd.DataFrame, n_train: int, gpu: bool = False) -> tuple[HierarchicalModel, HierarchicalModel, pd.DataFrame]:
    tr = df[df.split == "train"].sample(n=min(n_train, int((df.split == "train").sum())), random_state=0)
    recs = tr.to_dict("records")
    if gpu:
        from skinflow_discard_optimizer.accel.gpu_engine import batch_features_at, prepare_rows

        eng = CutEngine()
        a_ref = eng.press.amplitude_at_onset_N
        rows = batch_features_at(eng, prepare_rows(recs), np.array([_horizon(r, a_ref) for r in recs]))
        for f, r in zip(rows, recs):
            f.update(h_crit=r["h_crit_mm"], scenario=r["scenario"], cycle=r["cycle"])
        feats = pd.DataFrame(rows)
    else:
        feats = pd.DataFrame(_pool_map(_train_row, recs, with_models=False))
    it = int(value(load_config("decision"), "gibbs.iterations"))
    burn = int(value(load_config("decision"), "gibbs.burn_in"))
    groups = {g: feats[g].to_numpy() for g in GROUPS}
    prior = HierarchicalModel(PRIOR_FEATURES, GROUPS).fit(feats[PRIOR_FEATURES].to_numpy(), feats.h_crit.to_numpy(),
                                                          groups, it, burn)
    ok = feats.glr_fired.to_numpy()
    gap = (feats.h_crit - feats.onset_mm).to_numpy()          # onset model target: h_crit - onset
    onset = HierarchicalModel(ONSET_FEATURES, GROUPS).fit(feats.loc[ok, ONSET_FEATURES].to_numpy(), gap[ok],
                                                          {g: v[ok] for g, v in groups.items()}, it, burn, seed=1)
    save_models(prior, onset)
    return prior, onset, feats


def stream_rows(df: pd.DataFrame, splits: tuple[str, ...], stride: int) -> pd.DataFrame:
    sub = df[df.split.isin(splits) & (df.cycle % stride == 0)]
    return sub.sort_values(["scenario", "cycle"])


def evaluate_streams(rows: pd.DataFrame, cal_scores: np.ndarray, model: DefectModel,
                     gpu: bool = False) -> pd.DataFrame:
    """Run every scenario's test stream causally and score it exactly.

    CPU: one process per scenario runs the full engine in time order.
    GPU: all trajectories are computed in batches first (they do not depend on the
    calibrator), then each stream is replayed in time order with ``resolve_commit``.
    """
    if gpu:
        import time

        from skinflow_discard_optimizer.accel.gpu_engine import batch_trajectories, prepare_rows

        prior, onset = load_models()
        eng = CutEngine(prior_model=prior, onset_model=onset)
        recs = rows.to_dict("records")
        t0 = time.perf_counter()
        trajs = batch_trajectories(eng, prepared=prepare_rows(recs))
        t_batch_ms = (time.perf_counter() - t0) * 1000.0 / max(len(recs), 1)
        by_key = {(r["scenario"], int(r["cycle"])): tr for r, tr in zip(recs, trajs)}

        def _decide_gpu(row: dict, cal):
            d = resolve_commit(by_key[(row["scenario"], int(row["cycle"]))], eng.st, cal)
            d.compute_ms = t_batch_ms
            return d

        parts = []
        for _, g in rows.groupby("scenario", sort=True):
            parts.append(stream_loop(g.to_dict("records"), cal_scores, eng.st, _decide_gpu))
    else:
        streams = [(g.to_dict("records"), cal_scores) for _, g in rows.groupby("scenario", sort=True)]
        parts = _pool_map(_run_stream, streams, with_models=True, chunksize=1)
    ev = pd.DataFrame([r for part in parts for r in part])
    ev["defect_prob"] = model.defect_probability(ev.h_cut, ev.h_crit)
    ev["cost"] = model.expected_cost(ev.h_cut, ev.h_crit)
    ev["static_cost"] = model.expected_cost(ev.static_cut, ev.h_crit)
    ev["oracle_cost"] = model.expected_cost(ev.oracle_cut, ev.h_crit)
    ev["static_defect_prob"] = model.defect_probability(ev.static_cut, ev.h_crit)
    return ev


def _summarize_scenario_metrics(ev: pd.DataFrame, model: DefectModel) -> pd.DataFrame:
    mass_per_mm = float(model.discard_mass_kg(1.0))
    billet = model.econ.billet_mass_kg
    rows = []
    for name, g in ev.groupby("scenario", sort=True):
        rows.append({
            "scenario": name + (" (held out)" if name in HELD_OUT_SCENARIOS else ""),
            "n": len(g),
            "coverage %": 100 * g.covered.mean(),
            "mean cut mm": g.h_cut.mean(),
            "recovery vs static % billet": 100 * ((g.static_cut - g.h_cut) * mass_per_mm).mean() / billet,
            "defects /1000": 1000 * g.defect_prob.mean(),
            "static defects /1000": 1000 * g.static_defect_prob.mean(),
            "saving vs static EUR/billet": (g.static_cost - g.cost).mean(),
            "oracle saving EUR/billet": (g.static_cost - g.oracle_cost).mean(),
            "late %": 100 * g.late.mean(),
        })
    return pd.DataFrame(rows)


def _format_model_sections(prior: HierarchicalModel, onset: HierarchicalModel, ev: pd.DataFrame) -> list[str]:
    coef = lambda mdl: "\n".join(f"| {n} | {m:.4g} | {s:.2g} |" for n, m, s in mdl.coef_table())  # noqa: E731
    return [
        "## Models",
        "",
        f"Pre-onset model (state features only): residual sd {prior.diagnostics['sigma_mm']:.2f} mm.",
        "",
        "| term | posterior mean (per unit) | sd |", "|---|---|---|", coef(prior),
        "",
        f"Onset model (target: the gap `h_crit - onset`, so the onset enters with coefficient 1; assumption A-27): "
        f"residual sd {onset.diagnostics['sigma_mm']:.2f} mm.",
        "",
        "| term | posterior mean (per unit) | sd |", "|---|---|---|", coef(onset),
        "",
        f"Gibbs diagnostics (naive Geweke z, |z|<2 suggests convergence): prior sigma {prior.diagnostics['geweke_z_sigma']:.2f}, "
        f"onset sigma {onset.diagnostics['geweke_z_sigma']:.2f}.",
        "",
        "Die random effects (onset model, posterior mean mm): "
        + ", ".join(f"{lv} {onset.effects['die_id'][:, i].mean() * onset.y_sd:+.2f}"
                    for i, lv in enumerate(onset.levels["die_id"])),
        "",
        f"Decision latency (engine compute per cycle, offline Python): median {ev.compute_ms.median():.0f} ms, "
        f"p99 {ev.compute_ms.quantile(0.99):.0f} ms. This covers the whole stroke replay; the live service only "
        "does one incremental update per sample (Task 6.1 measures that).",
    ]


def report(*args, **kwargs) -> str:
    spec_list = [
        ("ev", None),
        ("prior", None),
        ("onset", None),
        ("st", None),
        ("model", None),
        ("n_train", None),
        ("n_cal", None),
    ]
    p = _parse_named_args(args, kwargs, spec_list)
    ev, prior, onset = p["ev"], p["prior"], p["onset"]
    st, model, n_train, n_cal = p["st"], p["model"], p["n_train"], p["n_cal"]

    t = _summarize_scenario_metrics(ev, model)
    target = 100 * st.coverage_target
    held = ev.scenario.isin(HELD_OUT_SCENARIOS)
    drift = ev.scenario.isin(["die_wear", "liner_scale", "temperature_drift", "supply_pressure_sag",
                              "sensor_gain_drift", "combined_wear_and_scale"])
    cov_all = 100 * ev.covered.mean()
    cov_held = 100 * ev[held].covered.mean()
    cov_drift = 100 * ev[drift].covered.mean()
    worst = t["coverage %"].sub(target).abs().max()
    ok = all(abs(c - target) <= 2 for c in (cov_all, cov_held, cov_drift))
    lines = [
        "# Task 3.3: critical-thickness posterior and cost-optimal cut",
        "",
        f"Models fitted on decision-time features of {n_train} training cycles (all scenarios, split `train`), "
        f"conformal scores from {n_cal} validation cycles, evaluated causally in time order on the `test` split "
        f"of every scenario and the two fully held-out scenarios ({len(ev)} cycles, every "
        f"{int(ev.groupby('scenario').cycle.diff().median())}th cycle). Adaptive conformal inference sees a true "
        f"`h_crit` label for 1 real production cycle in {st.audit_every}, {st.audit_delay} cycles late "
        "(assumption A-26); audits are counted in real cycles, so a strided stream sees every one of them.",
        "",
        "## Done-when: interval coverage within 2 points of target",
        "",
        f"Target {target:.0f}%. Overall **{cov_all:.1f}%**; held-out scenarios **{cov_held:.1f}%**; "
        f"drifting scenarios **{cov_drift:.1f}%**. Result: **{'PASS' if ok else 'FAIL'}**. "
        f"Largest deviation for a single scenario: {worst:.1f} points.",
        "",
        t.to_markdown(index=False, floatfmt=".2f"),
        "",
        "Recovery is metal saved against the static 40 mm cut, as % of billet mass. 'saving' is the true "
        "expected cost difference (metal plus defect risk) per billet against the static cut; 'oracle' is the "
        "ceiling if `h_crit` were known exactly. 'late' means the recommendation arrived after the ram had passed "
        "it and the cut landed where the ram could stop.",
        "",
        *_format_model_sections(prior, onset, ev),
    ]
    return "\n".join(lines)


SCORES_PATH = REPO_ROOT / "artifacts" / "conformal_scores.npy"


def _cal_scores_gpu(prior: HierarchicalModel, onset: HierarchicalModel, val: pd.DataFrame) -> np.ndarray:
    from skinflow_discard_optimizer.accel.gpu_engine import batch_trajectories, prepare_rows

    eng = CutEngine(prior_model=prior, onset_model=onset)
    recs = val.to_dict("records")
    trajs = batch_trajectories(eng, prepared=prepare_rows(recs))
    ms = [resolve_commit(tr, eng.st, None).recommendation.h_crit_mean_mm for tr in trajs]
    ss = [resolve_commit(tr, eng.st, None).recommendation.h_crit_sd_mm for tr in trajs]
    h_crits = np.array([r["h_crit_mm"] for r in recs])
    return (np.abs(h_crits - np.array(ms)) / np.array(ss))


def main(*args, **kwargs) -> None:
    spec_list = [
        ("n_train", 6000),
        ("n_cal", 1500),
        ("stride", 5),
        ("eval_only", False),
        ("device", "cpu"),
    ]
    p = _parse_named_args(args, kwargs, spec_list)
    n_train, n_cal = p["n_train"], p["n_cal"]
    stride, eval_only, device = p["stride"], p["eval_only"], p["device"]

    df = load_cycles()
    st = DecisionSettings.load()
    model = DefectModel()
    gpu = device == "cuda"
    if eval_only:        # reuse the fitted models and calibration scores from the last full run
        prior, onset = load_models()
        cal_scores = np.load(SCORES_PATH)
        n_cal = len(cal_scores)
    else:
        prior, onset, _ = train(df, n_train, gpu=gpu)
        val = df[df.split == "val"].sample(n=n_cal, random_state=1)
        if gpu:
            cal_scores = _cal_scores_gpu(prior, onset, val)
        else:
            cal_raw = pd.DataFrame(_pool_map(_decide_row, val.to_dict("records"), with_models=True))
            cal_scores = (np.abs(cal_raw.h_crit - cal_raw.m) / cal_raw.s).to_numpy()
        np.save(SCORES_PATH, cal_scores)
    rows = stream_rows(df, ("test", "test_scenario"), stride)
    ev = evaluate_streams(rows, cal_scores, model, gpu=gpu)
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(EVAL_PATH, index=False)
    text = report(ev, prior, onset, st, model, n_train, n_cal)
    (REPORTS_DIR / "task_3_3_decision.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    from skinflow_discard_optimizer.accel import gpu_available

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-cal", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--eval-only", action="store_true", help="reuse saved models and conformal scores")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if gpu_available() else "cpu",
                    help="device to run on (cpu or cuda)")
    a = ap.parse_args()
    main(a.n_train, a.n_cal, a.stride, a.eval_only, a.device)
