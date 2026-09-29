"""Ground-truth critical discard thickness and defect model (Task 1.2).

Every simulated cycle carries its true ``h_crit``, so a cut can be scored exactly:
``defect_probability(h_cut, h_crit)`` gives the true defect probability of any cut,
``expected_cost`` its true cost, and ``oracle_cut`` the cost-optimal cut for a
decision-maker who knew ``h_crit``. The oracle is the ceiling no model can beat.

The same latent state drives both ``h_crit`` and the force curve (friction and
tooling force, plus an onset of the end-of-stroke change that sits a little before
``h_crit``), so the force signal carries real but imperfect information about the
answer. Coefficients live in ``config/defect.yaml`` and are all PLACEHOLDER.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from skinflow_discard_optimizer.config import Economics, load_config, load_economics, value
from skinflow_discard_optimizer.sim.force_model import (
    Alloy,
    Press,
    Shape,
    StrokeSpec,
    nominal_stroke,
)


@dataclass(frozen=True)
class PressState:
    """Physical latent state of press, tooling and billet for one cycle (ground truth)."""

    liner_scale_mm: float = 0.10
    die_wear: float = 0.10           # 0 = new die, 1 = end of life
    billet_temp_C: float = 470.0     # front temperature
    liner_temp_C: float = 430.0      # mean of the four liner zones
    mu_base: float = 0.55            # clean-liner friction, before scale
    taper_K: float = 25.0
    ram_speed_mm_s: float = 12.0
    billet_length_mm: float = 850.0

    @property
    def dT_K(self) -> float:
        return self.billet_temp_C - self.liner_temp_C


@dataclass(frozen=True)
class DefectParams:
    base_mm: float
    k_liner_scale: float
    k_die_wear: float
    k_dT: float
    dT_ref: float
    k_mu: float
    noise_sd: float
    h_crit_min: float
    offset_mean: float
    offset_sd: float
    lam_sd_frac: float
    amp_sd_frac: float
    width_mm: float
    mu_per_scale: float
    F_tool_per_wear: float
    static_cut_mm: float
    min_cut_mm: float
    max_cut_mm: float
    mu_ref: float

    @classmethod
    def load(cls, mu_ref: float = 0.55) -> "DefectParams":
        t = load_config("defect")
        v = lambda k: float(value(t, k))  # noqa: E731
        return cls(
            base_mm=v("h_crit.base_mm"),
            k_liner_scale=v("h_crit.k_liner_scale_mm_per_mm"),
            k_die_wear=v("h_crit.k_die_wear_mm"),
            k_dT=v("h_crit.k_dT_mm_per_K"),
            dT_ref=v("h_crit.dT_ref_K"),
            k_mu=v("h_crit.k_mu_mm"),
            noise_sd=v("h_crit.noise_sd_mm"),
            h_crit_min=v("h_crit.min_mm"),
            offset_mean=v("onset.offset_mean_mm"),
            offset_sd=v("onset.offset_sd_mm"),
            lam_sd_frac=v("onset.lam_sd_frac"),
            amp_sd_frac=v("onset.amp_sd_frac"),
            width_mm=v("defect.width_mm"),
            mu_per_scale=v("coupling.mu_per_liner_scale_mm"),
            F_tool_per_wear=v("coupling.F_tool_per_die_wear_N"),
            static_cut_mm=v("baseline.static_cut_mm"),
            min_cut_mm=v("baseline.min_cut_mm"),
            max_cut_mm=v("baseline.max_cut_mm"),
            mu_ref=mu_ref,
        )


def _logistic(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


@dataclass(frozen=True)
class CycleTruth:
    """Hidden answers for one simulated cycle."""

    h_crit_mm: float
    h_crit_mean_mm: float    # deterministic part; the rest is irreducible billet noise
    h_onset_mm: float
    mu: float
    F_tool_N: float
    shape: Shape


class DefectModel:
    """Maps latent state to ``h_crit``, defect probability and per-billet cost."""

    def __init__(self, params: DefectParams | None = None, press: Press | None = None,
                 alloy: Alloy | None = None, economics: Economics | None = None):
        self.press = press or Press.load()
        self.alloy = alloy or Alloy.from_config("AA6063")
        self.p = params or DefectParams.load(mu_ref=self.alloy.mu_nominal)
        self.econ = economics or load_economics()

    # ---- coupling from latent state to force-model parameters

    def effective_mu(self, s: PressState) -> float:
        return s.mu_base + self.p.mu_per_scale * s.liner_scale_mm

    def tool_force_N(self, s: PressState) -> float:
        return self.press.F_tool_N + self.p.F_tool_per_wear * s.die_wear

    # ---- critical thickness

    def h_crit_mean(self, s: PressState) -> float:
        p = self.p
        h = (p.base_mm
             + p.k_liner_scale * s.liner_scale_mm
             + p.k_die_wear * s.die_wear
             + p.k_dT * (s.dT_K - p.dT_ref)
             + p.k_mu * (self.effective_mu(s) - p.mu_ref))
        return max(h, p.h_crit_min)

    def sample_truth(self, s: PressState, rng: np.random.Generator, shape: Shape = "upturn") -> CycleTruth:
        mean = self.h_crit_mean(s)
        h_crit = max(mean + rng.normal(0.0, self.p.noise_sd), self.p.h_crit_min)
        onset = h_crit + max(rng.normal(self.p.offset_mean, self.p.offset_sd), 0.0)
        return CycleTruth(h_crit, mean, onset, self.effective_mu(s), self.tool_force_N(s), shape)

    # ---- scoring: exact for any cut

    def defect_probability(self, h_cut, h_crit):
        """True probability that a cut at ``h_cut`` lets skin-flow material into the product."""
        return _logistic((np.asarray(h_crit) - np.asarray(h_cut)) / self.p.width_mm)

    def discard_mass_kg(self, h_mm):
        """Mass of a discard of thickness ``h`` (it fills the container bore)."""
        return self.alloy.density * self.press.container_area_mm2 * np.asarray(h_mm) * 1e-9

    @property
    def metal_cost_per_mm(self) -> float:
        """c_m * rho * A_b in EUR per mm of discard."""
        return self.econ.net_metal_loss_per_kg * float(self.discard_mass_kg(1.0))

    @property
    def defect_cost(self) -> float:
        """c_d * L_d in EUR per defect event."""
        return self.econ.defect_loss_per_metre * self.econ.defect_affected_length_m

    def expected_cost(self, h_cut, h_crit):
        """True expected cost per billet: C(h) = c_m*rho*Ab*h + c_d*L_d*P(defect)."""
        return self.metal_cost_per_mm * np.asarray(h_cut) + self.defect_cost * self.defect_probability(h_cut, h_crit)

    def oracle_cut(self, h_crit):
        """Cost-optimal cut if ``h_crit`` were known, clipped to the configured cut bounds.

        Setting dC/dh = 0 gives p(1-p) = c*w/L for the logistic p; the smaller root is
        the minimum on the safe side of h_crit. If c*w/L >= 1/4 there is no interior
        minimum and the thinnest allowed cut is optimal.
        """
        h_crit = np.asarray(h_crit, dtype=float)
        k = self.metal_cost_per_mm * self.p.width_mm / self.defect_cost
        lo, hi = self.p.min_cut_mm, self.p.max_cut_mm
        if k >= 0.25:
            return np.full_like(h_crit, lo)
        p = 0.5 * (1.0 - np.sqrt(1.0 - 4.0 * k))
        h_star = h_crit - self.p.width_mm * np.log(p / (1.0 - p))
        h_star = np.clip(h_star, lo, hi)
        # guard the bounds: the clipped interior point vs. the lower bound
        return np.where(self.expected_cost(lo, h_crit) < self.expected_cost(h_star, h_crit), lo, h_star)

    def oracle_margin_mm(self) -> float:
        """How far above h_crit the oracle cuts (unclipped)."""
        return float(self.oracle_cut(np.array(30.0)) - 30.0)

    # ---- one simulated cycle

    def stroke_for(self, s: PressState, truth: CycleTruth, rng: np.random.Generator,
                   extrusion_ratio: float | None = None) -> StrokeSpec:
        """Force-model parameters consistent with the latent state and the hidden truth."""
        pr = self.press
        lam = pr.decay_length_mm * max(1.0 + rng.normal(0.0, self.p.lam_sd_frac), 0.3)
        amp = pr.amplitude_at_onset_N * max(1.0 + rng.normal(0.0, self.p.amp_sd_frac), 0.2)
        return nominal_stroke(
            self.alloy, pr,
            L0_mm=pr.upset_length(s.billet_length_mm),
            extrusion_ratio=extrusion_ratio or pr.extrusion_ratio,
            ram_speed_mm_s=s.ram_speed_mm_s,
            T_front_C=s.billet_temp_C,
            taper_K=s.taper_K,
            mu=truth.mu,
            F_tool_N=truth.F_tool_N,
            h_onset_mm=truth.h_onset_mm,
            upturn_amp_N=amp,
            lam_mm=lam,
            shape=truth.shape,
        )

    def with_alloy(self, alloy: Alloy) -> "DefectModel":
        m = DefectModel(replace(self.p, mu_ref=alloy.mu_nominal), self.press, alloy, self.econ)
        return m
