"""Causal per-stroke cut decision: UKF -> onset -> h_crit posterior -> cut (Tasks 3.1-3.3 together).

``CutEngine.decide`` replays one stroke in time order, exactly as it would run
live:

1. The UKF tracks theta over the body of the stroke and freezes at the gate
   (h = 15% of L0).
2. After the gate, frozen-baseline residuals feed the onset GLR and BOCPD.
3. Every ``update_every_mm`` of travel the engine predicts ``h_crit``: from the
   onset model once the GLR has fired, otherwise from the pre-onset model
   (state features only). It then computes the calibrated recommendation.
4. The cut commits as soon as the ram is within ``latency_margin_mm`` of the
   current recommendation. If the ram is already past it, the cut lands where the
   ram can stop, and the decision is flagged ``late``.

Nothing after the commit point is used. Training features come from the same
code (``features_at``), so training and live stay identical.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skinflow_discard_optimizer.core.estimator.decision import (
    AdaptiveConformal,
    CutRecommendation,
    DecisionSettings,
    recommend,
)

__all__ = ["CutEngine", "CycleDecision", "DecisionTrajectory", "StrokeInputs", "resolve_commit",
           "inputs_from_row", "load_models", "save_models", "AdaptiveConformal"]
from skinflow_discard_optimizer.core.estimator.hierarchical import HierarchicalModel
from skinflow_discard_optimizer.core.observer.onset import BOCPD, OnsetGLR
from skinflow_discard_optimizer.core.observer.ukf import (
    StrokeContext,
    UKFParams,
    UKFTrace,
    phi_to_theta,
    run_stroke,
    unscented_transform,
)
from skinflow_discard_optimizer.paths import REPO_ROOT
from skinflow_discard_optimizer.sim.force_model import Alloy, Press

GATE_FRAC = 0.15
LATE_TOLERANCE_MM = 0.5
# Only well-identified inputs. sigma_scale and F_tool sit on a weak ridge (Task 3.1);
# as regressors their coefficients came out meaningless (posterior sd ~100) and the
# model extrapolated along the ridge under drift. mu, temperatures and the onset are solid.
PRIOR_FEATURES = ["theta_mu", "dT_K", "billet_temp_C"]
# The onset model predicts the *gap* h_crit - onset from the same state features, i.e.
# the onset enters with coefficient 1 (docs/assumptions.md A-27). A free regression
# coefficient came out ~0.5 on mostly-healthy training data (it blends onset with the
# state features) and so under-followed drift, which moves h_crit and the onset together.
ONSET_FEATURES = PRIOR_FEATURES
GROUPS = ["die_id", "alloy_id"]
MODELS_PATH = REPO_ROOT / "artifacts" / "hcrit_models.npz"


@dataclass
class StrokeInputs:
    """What the engine needs for one cycle (live: from ``CycleInputs``; offline: from a dataset row)."""

    t_s: np.ndarray
    x_mm: np.ndarray
    p_cap_bar: np.ndarray
    p_rod_bar: np.ndarray
    billet_length_mm: float
    billet_temp_C: float
    liner_temp_mean_C: float
    extrusion_ratio: float
    alloy_id: str
    die_id: str


@dataclass
class CycleDecision:
    h_cut_mm: float                  # where the cut actually lands
    recommendation: CutRecommendation
    h_commit_mm: float               # ram position (as thickness) when the decision was committed
    late: bool
    onset_fired_h: float | None
    onset_mean_mm: float | None
    onset_sd_mm: float | None
    onset_confidence: str
    features: dict = field(default_factory=dict)
    compute_ms: float = 0.0


def state_features_from_phi(phi: np.ndarray, phi_cov: np.ndarray, si: StrokeInputs) -> dict:
    """Decision-time state features from the filter state at the freeze point.

    The filter stops updating at the gate, so its final phi and covariance *are* the
    state at freeze. theta comes from them by the unscented transform. (An earlier
    version read theta from a trace computed only every 1e9 updates, which silently
    repeated the value after the *first* update; fixed 2026-09-28.)
    """
    th, _ = unscented_transform(np.asarray(phi, float), np.asarray(phi_cov, float), phi_to_theta)
    return {"theta_sigma_scale": float(th[0]), "theta_mu": float(th[1]), "theta_F_tool_MN": float(th[2]) / 1e6,
            "dT_K": si.billet_temp_C - si.liner_temp_mean_C, "billet_temp_C": si.billet_temp_C,
            "die_id": si.die_id, "alloy_id": si.alloy_id}


def save_models(prior: HierarchicalModel, onset: HierarchicalModel, path: Path = MODELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {f"prior__{k}": v for k, v in prior.to_npz().items()}
    d.update({f"onset__{k}": v for k, v in onset.to_npz().items()})
    np.savez(path, **d)


def load_models(path: Path = MODELS_PATH) -> tuple[HierarchicalModel, HierarchicalModel]:
    z = np.load(path)

    class _Sub(dict):
        pass

    out = []
    for pre in ("prior__", "onset__"):
        sub = _Sub({k[len(pre):]: z[k] for k in z.files if k.startswith(pre)})
        out.append(HierarchicalModel.from_npz(sub))
    return out[0], out[1]


class CutEngine:
    def __init__(self, press: Press | None = None, settings: DecisionSettings | None = None,
                 prior_model: HierarchicalModel | None = None, onset_model: HierarchicalModel | None = None,
                 **kwargs):
        self.press = press or Press.load()
        self.st = settings or DecisionSettings.load()
        self.prior_model, self.onset_model = prior_model, onset_model
        self.cal = kwargs.get("calibrator")
        self.ukf_params = kwargs.get("ukf_params") or UKFParams.for_press(self.press)
        self.block = kwargs.get("block", 20)
        self._alloys: dict[str, Alloy] = {}

    # ------------------------------------------------------------------ building blocks
    def _alloy(self, name: str) -> Alloy:
        if name not in self._alloys:
            self._alloys[name] = Alloy.from_config(name)
        return self._alloys[name]

    def trace(self, si: StrokeInputs) -> tuple[UKFTrace, StrokeContext]:
        L0 = self.press.upset_length(si.billet_length_mm)
        ctx = StrokeContext(L0, si.extrusion_ratio, si.billet_temp_C, self._alloy(si.alloy_id), self.press)
        tr = run_stroke(si.t_s, si.x_mm, si.p_cap_bar, si.p_rod_bar, ctx, self.ukf_params, block=self.block,
                        h_stop_mm=self.st.min_cut_mm - 2.0, theta_every=10**9, freeze_h_mm=GATE_FRAC * L0)
        return tr, ctx

    @staticmethod
    def state_features(tr: UKFTrace, si: StrokeInputs) -> dict:
        return state_features_from_phi(tr.final_phi, tr.final_phi_cov, si)

    def _glr(self, tr: UKFTrace) -> tuple[OnsetGLR, BOCPD]:
        return (OnsetGLR(self.press.amplitude_at_onset_N, baseline_cov=tr.final_phi_cov, noise_var=tr.r_var),
                BOCPD())

    def features_at(self, si: StrokeInputs, h_horizon_mm: float) -> dict:
        """Decision-time features using data down to ``h_horizon_mm`` (training-set builder)."""
        tr, _ = self.trace(si)
        f = self.state_features(tr, si)
        glr, _ = self._glr(tr)
        g = (tr.h_mm < GATE_FRAC * tr.L0_mm) & (tr.h_mm >= h_horizon_mm)
        for hh, rr, ss, H in zip(tr.h_mm[g], tr.innovation[g], tr.S[g], tr.H[g]):
            glr.update(hh, rr, ss, H)
        s = glr.summary()
        f.update({"onset_mm": s["h_onset_mean"], "onset_sd_mm": s["h_onset_sd"],
                  "glr_fired": glr.fired_at_h is not None})
        return f

    def _predict(self, f: dict, onset: tuple[float, float] | None) -> tuple[float, float]:
        groups = {g: np.array([f[g]]) for g in GROUPS}
        if onset is not None and self.onset_model is not None:
            x = np.array([[f[c] for c in ONSET_FEATURES]])
            gap, v = self.onset_model.predict(x, groups)
            # h_crit = onset + gap; the onset estimate's own uncertainty adds directly
            return float(onset[0] + gap[0]), float(np.sqrt(v[0] + onset[1] ** 2))
        x = np.array([[f[c] for c in PRIOR_FEATURES]])
        m, v = self.prior_model.predict(x, groups)
        return float(m[0]), float(np.sqrt(v[0]))

    # ------------------------------------------------------------------ the decision
    def trajectory(self, si: StrokeInputs) -> "DecisionTrajectory":
        """Everything the decision needs that does not depend on the calibrator, for the whole gate.

        At every checkpoint (each ``update_every_mm`` of travel after the gate) it holds
        the raw ``h_crit`` predictive (m, s), the onset state and the confidence. The
        calibrator only enters in ``resolve_commit``. ``accel.gpu_engine`` produces the
        same object for many strokes at once.
        """
        tr, _ = self.trace(si)
        f = self.state_features(tr, si)
        glr, bo = self._glr(tr)
        idx = np.flatnonzero(tr.h_mm < GATE_FRAC * tr.L0_mm)
        st = self.st
        m0, s0 = self._predict(f, None)
        ck, ms, ss, confs, on_m, on_s = [], [], [], [], [], []
        next_eval_h = np.inf
        onset = None
        conf = "none"
        for j, k in enumerate(idx):
            h_now = float(tr.h_mm[k])
            glr.update(h_now, tr.innovation[k], tr.S[k], tr.H[k])
            bo.update(h_now, tr.innovation[k] / np.sqrt(tr.S[k]))
            if h_now > next_eval_h:
                continue
            next_eval_h = h_now - st.update_every_mm
            onset, conf = _eval_onset_alarm(glr, bo, st, onset, conf)
            m, s = self._predict(f, onset)
            ck.append(j)
            ms.append(m)
            ss.append(s)
            confs.append(conf)
            on_m.append(onset[0] if onset else np.nan)
            on_s.append(onset[1] if onset else np.nan)
        return DecisionTrajectory(
            h=tr.h_mm[idx].astype(float), ck_idx=np.array(ck, int), m=np.array(ms), s=np.array(ss),
            conf=confs, onset_mean=np.array(on_m), onset_sd=np.array(on_s),
            glr_fired_h=np.nan if glr.fired_at_h is None else float(glr.fired_at_h),
            bocpd_fired_h=np.nan if bo.fired_at_h is None else float(bo.fired_at_h),
            m0=m0, s0=s0, h_last=float(tr.h_mm[-1]), features=f)

    def decide(self, si: StrokeInputs) -> CycleDecision:
        t0 = time.perf_counter()
        d = resolve_commit(self.trajectory(si), self.st, self.cal)
        d.compute_ms = (time.perf_counter() - t0) * 1e3
        return d


def _eval_onset_alarm(glr: OnsetGLR, bo: BOCPD, st: DecisionSettings,
                      prev_onset: tuple[float, float] | None, prev_conf: str) -> tuple[tuple[float, float] | None, str]:
    if glr.fired_at_h is not None:
        sm = glr.summary()
        agree = bo.fired_at_h is not None and abs(bo.fired_at_h - glr.fired_at_h) <= 12.0
        conf = "high" if agree else "low"
        onset = (sm["h_onset_mean"], sm["h_onset_sd"]) if sm["h_onset_sd"] <= st.onset_sd_max_mm else prev_onset
        return onset, conf
    if bo.fired_at_h is not None:
        return prev_onset, "low"
    return prev_onset, prev_conf


@dataclass
class DecisionTrajectory:
    h: np.ndarray                 # remaining thickness at each gated update (time order)
    ck_idx: np.ndarray            # update index of each checkpoint
    m: np.ndarray                 # raw h_crit predictive mean at each checkpoint
    s: np.ndarray                 # raw predictive sd
    conf: list[str]               # onset confidence at each checkpoint
    onset_mean: np.ndarray        # onset used by the prediction (nan if the prior model was used)
    onset_sd: np.ndarray
    glr_fired_h: float            # nan if it never fired
    bocpd_fired_h: float
    m0: float                     # prior-model predictive, used only if the gate saw no update
    s0: float
    h_last: float
    features: dict = field(default_factory=dict)


def resolve_commit(tr: DecisionTrajectory, st: DecisionSettings, cal=None) -> CycleDecision:
    """Apply the calibrator and find where the cut commits.

    Between checkpoints the recommendation is fixed; the commit is checked at every
    update: the first update with ``h - latency_margin <= h_cut``.
    """
    snap = cal.snapshot() if hasattr(cal, "snapshot") else cal
    n_ck = len(tr.ck_idx)
    if n_ck == 0:
        rec = recommend(tr.m0, tr.s0, st, snap, "none")
        h_now = tr.h_last
        return _finish(tr, st, rec, h_now, committed=False, j=None)
    bounds = np.r_[tr.ck_idx, len(tr.h)]
    for j in range(n_ck):
        rec = recommend(float(tr.m[j]), float(tr.s[j]), st, snap, tr.conf[j])
        seg = tr.h[bounds[j]:bounds[j + 1]]
        hit = np.flatnonzero(seg - st.latency_margin_mm <= rec.h_cut_mm)
        if hit.size:
            return _finish(tr, st, rec, float(seg[hit[0]]), committed=True, j=j)
    return _finish(tr, st, rec, float(tr.h[-1]), committed=False, j=n_ck - 1)


def _compute_cut_position(h_now: float, rec: CutRecommendation, st: DecisionSettings, committed: bool) -> tuple[float, bool]:
    if not committed:
        return max(h_now, st.min_cut_mm), True
    late = (h_now - st.latency_margin_mm) < (rec.h_cut_mm - LATE_TOLERANCE_MM)
    if late:
        return max(h_now - st.latency_margin_mm, st.min_cut_mm), True
    return rec.h_cut_mm, False


def _extract_trajectory_onset(tr: DecisionTrajectory, j: int | None, h_now: float) -> tuple[float | None, float | None, str, float | None]:
    if j is None:
        return None, None, "none", None
    om = float(tr.onset_mean[j]) if np.isfinite(tr.onset_mean[j]) else None
    osd = float(tr.onset_sd[j]) if om is not None else None
    fired = float(tr.glr_fired_h) if (np.isfinite(tr.glr_fired_h) and tr.glr_fired_h >= h_now) else None
    return om, osd, tr.conf[j], fired


def _finish(tr: DecisionTrajectory, st: DecisionSettings, rec: CutRecommendation, h_now: float,
            **kwargs) -> CycleDecision:
    committed: bool = kwargs.get("committed", True)
    j: int | None = kwargs.get("j", None)
    h_cut, late = _compute_cut_position(h_now, rec, st, committed)
    om, osd, conf, fired = _extract_trajectory_onset(tr, j, h_now)
    f = dict(tr.features)
    f["onset_mm"] = om if om is not None else np.nan
    f["onset_sd_mm"] = osd if osd is not None else np.nan
    return CycleDecision(h_cut, rec, h_now, late, fired, om, osd, conf, f)


def inputs_from_row(row: dict, stroke) -> StrokeInputs:
    liner = np.mean([row[f"container_liner_temp_{i}"] for i in range(1, 5)])
    return StrokeInputs(stroke.t_s, stroke.x_mm, stroke.p_cap_bar, stroke.p_rod_bar, float(row["billet_length_mm"]),
                        float(row["billet_temp_C"]), float(liner), float(row["extrusion_ratio"]),
                        row["alloy_id"], row["die_id"])
