"""Cost-optimal discard cut with conformal calibration (Task 3.3, layer L5).

Part of the Bayesian Discard Thickness Estimator.

Given a Gaussian predictive ``h_crit | data ~ N(m, s^2)``:

* expected cost per billet (plan section 1.2):
  ``C(h) = c_m*rho*A_b*h + c_d*L_d*P(defect | h)``.
  With ``use_transition_width`` the clean-to-defective transition is the smooth
  logistic of width ``w`` from ``config/defect.yaml``. By the probit approximation
  ``E[logistic((h_crit - h)/w)] ~ Phi((m - h)/s_eff)`` with
  ``s_eff^2 = s^2 + (1.702*w)^2``. Without it, ``s_eff = s`` (the plan's step
  ``P(h < h_crit)``).
* ``dC/dh = 0`` gives ``phi(z) = c*s_eff/L`` with ``z = (h - m)/s_eff``, so
  ``h* = m + s_eff * sqrt(-2*ln(c*s_eff*sqrt(2*pi)/L))``, clipped to the safety
  bounds. If ``c*s_eff*sqrt(2*pi) >= L`` there is no interior minimum and the lower
  bound wins (metal would cost more than the defects it prevents).
* chance-constrained cut: ``h_cc = m + s*z_{1-alpha}``, so ``P(h_crit > h_cc) = alpha``.

Calibration (``AdaptiveConformal``)
    The conformity score is ``|h_crit - m| / s``. Split-conformal takes the
    ``(1 - alpha_t)`` quantile ``q`` of calibration scores. The interval is
    ``m +- q*s``, and the decision uses the calibrated scale ``s * q / z_nominal``.
    Adaptive conformal inference (Gibbs & Candes 2021) then moves ``alpha_t`` online
    as audited labels arrive: ``alpha_{t+1} = alpha_t + gamma*(alpha - err_t)``, so
    coverage holds under drift.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from skinflow_discard_optimizer.config import load_config, value
from skinflow_discard_optimizer.sim.defect_model import DefectModel

PROBIT_LOGISTIC = 1.702


@dataclass(frozen=True)
class DecisionSettings:
    metal_cost_per_mm: float       # c_m * rho * A_b, EUR/mm
    defect_cost: float             # c_d * L_d, EUR per defect
    width_mm: float                # logistic transition width (0 = step)
    min_cut_mm: float
    max_cut_mm: float
    static_cut_mm: float
    coverage_target: float
    aci_gamma: float
    audit_every: int
    audit_delay: int
    latency_margin_mm: float
    update_every_mm: float
    chance_alpha: float
    onset_sd_max_mm: float = 2.0

    @classmethod
    def load(cls, model: DefectModel | None = None) -> "DecisionSettings":
        m = model or DefectModel()
        t = load_config("decision")
        v = lambda k: value(t, k)  # noqa: E731
        return cls(
            metal_cost_per_mm=m.metal_cost_per_mm,
            defect_cost=m.defect_cost,
            width_mm=m.p.width_mm if int(v("decision.use_transition_width")) else 0.0,
            min_cut_mm=m.p.min_cut_mm, max_cut_mm=m.p.max_cut_mm, static_cut_mm=m.p.static_cut_mm,
            coverage_target=float(v("conformal.coverage_target")), aci_gamma=float(v("conformal.aci_gamma")),
            audit_every=int(v("conformal.audit_every")), audit_delay=int(v("conformal.audit_delay")),
            latency_margin_mm=float(v("decision.latency_margin_mm")),
            update_every_mm=float(v("decision.update_every_mm")), chance_alpha=float(v("decision.chance_alpha")),
            onset_sd_max_mm=float(v("decision.onset_sd_max_mm")),
        )

    def s_eff(self, s):
        return np.sqrt(np.asarray(s) ** 2 + (PROBIT_LOGISTIC * self.width_mm) ** 2)


def expected_cost(h, m, s, st: DecisionSettings):
    """Expected cost per billet of cutting at ``h`` when h_crit ~ N(m, s^2)."""
    se = st.s_eff(s)
    return st.metal_cost_per_mm * np.asarray(h) + st.defect_cost * norm.sf((np.asarray(h) - m) / se)


def cost_optimal_cut(m, s, st: DecisionSettings):
    """Closed-form minimiser of ``expected_cost``, clipped to the safety bounds."""
    m, se = np.asarray(m, float), st.s_eff(s)
    k = st.metal_cost_per_mm * se * np.sqrt(2 * np.pi) / st.defect_cost
    z = np.sqrt(-2.0 * np.log(np.clip(k, 1e-300, None)))
    h = np.where(k < 1.0, m + se * z, st.min_cut_mm)
    # Safety: when the risk-optimal cut lies beyond the maximum allowed cut, return the maximum.
    # A pure cost comparison would pick the *thinnest* cut there ("the defect is unavoidable
    # anyway"), which is the wrong advice for an advisory system. The lower bound only wins
    # when metal genuinely costs more than the defect (k >= 1).
    return np.clip(h, st.min_cut_mm, st.max_cut_mm)


def chance_constrained_cut(m, s, st: DecisionSettings):
    """Cut such that P(h_crit > h) = alpha (plan's quantile version), clipped to the bounds."""
    return np.clip(np.asarray(m) + np.asarray(s) * norm.isf(st.chance_alpha), st.min_cut_mm, st.max_cut_mm)


class AdaptiveConformal:
    """Split-conformal interval on h_crit with adaptive conformal inference for drift."""

    def __init__(self, calibration_scores: np.ndarray, target: float = 0.9, gamma: float = 0.01,
                 window: int = 2000):
        self.target = target
        self.alpha_goal = 1.0 - target
        self.alpha_t = self.alpha_goal
        self.gamma = gamma
        self.scores = deque(np.asarray(calibration_scores, float).tolist(), maxlen=window)
        self.history: list[tuple[float, int]] = []   # (alpha_t, miss) per audited label

    def q(self) -> float:
        """Conformal quantile at the current alpha_t (finite-sample corrected)."""
        a = float(np.clip(self.alpha_t, 1e-4, 0.9999))
        s = np.sort(np.fromiter(self.scores, float))
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - a))) - 1
        if k >= n:
            return float(s[-1] * 1.5)                 # asked for more than the data can give: widen
        return float(s[max(k, 0)])

    def interval(self, m, s) -> tuple[np.ndarray, np.ndarray]:
        q = self.q()
        return np.asarray(m) - q * np.asarray(s), np.asarray(m) + q * np.asarray(s)

    def scale_factor(self) -> float:
        """Calibrated / nominal sd ratio: q divided by the Gaussian quantile for the same coverage."""
        return float(self.q() / norm.isf(self.alpha_goal / 2))

    def snapshot(self) -> "FrozenConformal":
        """The calibrator as it stands now. It does not change within one stroke, so one snapshot serves all its checkpoints."""
        return FrozenConformal(self.q(), self.alpha_goal)

    def update(self, y: float, m: float, s: float) -> bool:
        """Feed one audited label. Returns True if it fell outside the interval."""
        lo, hi = self.interval(m, s)
        miss = int(not (lo <= y <= hi))
        self.history.append((self.alpha_t, miss))
        self.alpha_t += self.gamma * (self.alpha_goal - miss)
        self.scores.append(abs(y - m) / s)
        return bool(miss)


@dataclass(frozen=True)
class FrozenConformal:
    """Read-only calibrator state with a cached quantile (same interface ``recommend`` uses)."""

    q_value: float
    alpha_goal: float

    def q(self) -> float:
        return self.q_value

    def interval(self, m, s):
        return np.asarray(m) - self.q_value * np.asarray(s), np.asarray(m) + self.q_value * np.asarray(s)

    def scale_factor(self) -> float:
        return float(self.q_value / norm.isf(self.alpha_goal / 2))


@dataclass
class CutRecommendation:
    h_cut_mm: float                 # cost-optimal, calibrated, bounded
    h_cut_cc_mm: float              # chance-constrained alternative
    h_crit_mean_mm: float
    h_crit_sd_mm: float             # raw model sd
    h_crit_sd_cal_mm: float         # conformally calibrated sd
    interval_lo_mm: float           # conformal interval on h_crit
    interval_hi_mm: float
    confidence: str                 # high | medium | low
    onset_used: bool
    notes: list[str] = field(default_factory=list)


def recommend(m: float, s: float, st: DecisionSettings, cal: "AdaptiveConformal | FrozenConformal | None",
              onset_confidence: str = "none") -> CutRecommendation:
    """Turn a raw h_crit predictive into the published recommendation."""
    factor = cal.scale_factor() if cal is not None else 1.0
    s_cal = s * max(factor, 1e-3)
    h = float(cost_optimal_cut(m, s_cal, st))
    h_cc = float(chance_constrained_cut(m, s_cal, st))
    if cal is not None:
        lo, hi = (float(v) for v in cal.interval(m, s))
    else:
        z = float(norm.isf((1 - st.coverage_target) / 2))
        lo, hi = float(m - z * s), float(m + z * s)
    notes = []
    if h <= st.min_cut_mm + 1e-9 or h >= st.max_cut_mm - 1e-9:
        notes.append("cut at safety bound")
    onset_used = onset_confidence in ("high", "low")
    width = hi - lo
    if onset_confidence == "high" and width < 6.0:
        conf = "high"
    elif onset_confidence in ("high", "low") and width < 10.0:
        conf = "medium"
    else:
        conf = "low"
    return CutRecommendation(h, h_cc, float(m), float(s), float(s_cal), lo, hi, conf, onset_used, notes)
