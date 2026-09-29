"""Latent wear-state tracker and regime switching filter (Task 4.1).

Tracks slow-moving latent physical states of the press cycle by cycle:
1. **die_wear:** Tooling wear index [0.0, 1.5] (drives tooling force F_tool).
2. **liner_scale_mm:** Container liner scale thickness [0.0, 3.0] mm (drives friction mu).
3. **temperature_taper_K:** Billet front-to-back thermal taper [0.0, 80.0] K.
4. **sensor_gain:** Cap-pressure transducer gain error [0.8, 1.3] (multiplies sigma_scale).

Uses a particle filter with Markov regime switching:
- **normal:** Steady operation with Ornstein-Uhlenbeck (OU) wander around baseline.
- **ramping:** Sustained gradual degradation (linear wear/scale accumulation).
- **step:** Sudden shock or abrupt shift (lubricant failure, sensor jump).

Outputs posterior state distributions (mean, sd, 90% credible intervals) and
regime probabilities P(normal), P(ramping), P(step) each cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from skinflow_discard_optimizer.config import load_config, value


class Regime(IntEnum):
    NORMAL = 0
    RAMPING = 1
    STEP = 2


REGIME_NAMES = {
    Regime.NORMAL: "normal",
    Regime.RAMPING: "ramping",
    Regime.STEP: "step",
}

STATE_NAMES = ("die_wear", "liner_scale_mm", "taper_K", "sensor_gain")
N_STATES = len(STATE_NAMES)


@dataclass(frozen=True)
class TrackerEstimate:
    """Posterior state estimates and regime probabilities for one cycle."""

    cycle: int
    die_wear_mean: float
    liner_scale_mean: float
    taper_mean: float
    sensor_gain_mean: float

    die_wear_sd: float
    liner_scale_sd: float
    taper_sd: float
    sensor_gain_sd: float

    die_wear_ci: tuple[float, float]
    liner_scale_ci: tuple[float, float]
    taper_ci: tuple[float, float]
    sensor_gain_ci: tuple[float, float]

    p_normal: float
    p_ramping: float
    p_step: float
    dominant_regime: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "cycle": self.cycle,
            "die_wear_mean": self.die_wear_mean,
            "die_wear_sd": self.die_wear_sd,
            "liner_scale_mean": self.liner_scale_mean,
            "liner_scale_sd": self.liner_scale_sd,
            "taper_mean": self.taper_mean,
            "taper_sd": self.taper_sd,
            "sensor_gain_mean": self.sensor_gain_mean,
            "sensor_gain_sd": self.sensor_gain_sd,
            "p_normal": self.p_normal,
            "p_ramping": self.p_ramping,
            "p_step": self.p_step,
            "dominant_regime": self.dominant_regime,
        }


@dataclass
class TrackerConfig:
    """Physical coefficients, noise variances, and transition probabilities."""

    n_particles: int
    ou_theta: np.ndarray
    ou_sigma: np.ndarray
    baseline: np.ndarray
    ramp_sigma: np.ndarray
    step_sigma: np.ndarray
    obs_sigma: np.ndarray
    regime_transition: np.ndarray

    f_tool_base_mn: float
    k_tool_wear_mn: float
    mu_base: float
    k_mu_scale: float
    taper_base_k: float

    @classmethod
    def load(cls, n_particles: int = 500) -> "TrackerConfig":
        faults_cfg = load_config("faults")
        defect_cfg = load_config("defect")
        press_cfg = load_config("press")

        v_f = lambda k: float(value(faults_cfg, k))
        v_d = lambda k: float(value(defect_cfg, k))
        v_p = lambda k: float(value(press_cfg, k))

        baseline = np.array([0.10, 0.10, v_p("thermal.taper_K"), 1.00], dtype=float)

        ou_theta = np.array([
            v_f("ou.die_wear.theta"),
            v_f("ou.liner_scale_mm.theta"),
            0.05,
            0.02,
        ], dtype=float)

        ou_sigma = np.array([
            v_f("ou.die_wear.sigma"),
            v_f("ou.liner_scale_mm.sigma"),
            0.40,
            0.001,
        ], dtype=float)

        # Drift tracking noise in RAMPING regime (diffusive random walk allowing fast/slow drift)
        ramp_sigma = np.array([0.008, 0.008, 1.00, 0.003], dtype=float)

        # Shock noise in STEP regime (allows rapid jumping to new offsets)
        step_sigma = np.array([0.40, 0.60, 10.0, 0.05], dtype=float)

        # Observation noise standard deviations [y_tool_mn, y_mu, y_taper, y_gain]
        obs_sigma = np.array([0.035, 0.010, 2.0, 0.012], dtype=float)

        # Regime transition matrix: [NORMAL, RAMPING, STEP]
        trans = np.array([
            [0.985, 0.010, 0.005],
            [0.010, 0.985, 0.005],
            [0.020, 0.900, 0.080],
        ], dtype=float)

        f_base = v_p("tooling.F_tool_N") * 1e-6
        k_wear = v_d("coupling.F_tool_per_die_wear_N") * 1e-6
        k_scale = v_d("coupling.mu_per_liner_scale_mm")

        return cls(
            n_particles=n_particles,
            ou_theta=ou_theta,
            ou_sigma=ou_sigma,
            baseline=baseline,
            ramp_sigma=ramp_sigma,
            step_sigma=step_sigma,
            obs_sigma=obs_sigma,
            regime_transition=trans,
            f_tool_base_mn=f_base,
            k_tool_wear_mn=k_wear,
            mu_base=0.55,
            k_mu_scale=k_scale,
            taper_base_k=baseline[2],
        )


class LatentWearTracker:
    """Particle filter with Markov regime switching for latent press state estimation."""

    def __init__(self, config: TrackerConfig | None = None, seed: int = 42):
        self.cfg = config or TrackerConfig.load()
        self.rng = np.random.default_rng(seed)
        self.history: list[TrackerEstimate] = []
        self.last_billet_temp_C: float = 470.0
        self.last_liner_temp_C: float = 430.0
        self._init_particles()

    def _init_particles(self) -> None:
        cfg = self.cfg
        n = cfg.n_particles
        self.bounds_lo = np.array([0.0, 0.0, 0.0, 0.80], dtype=float)
        self.bounds_hi = np.array([1.5, 3.0, 80.0, 1.30], dtype=float)

        init_sd = np.array([0.03, 0.03, 2.0, 0.01], dtype=float)
        self.particles = self.rng.normal(cfg.baseline, init_sd, size=(n, N_STATES))
        self.particles = np.clip(self.particles, self.bounds_lo, self.bounds_hi)

        self.regime_probs = np.array([0.96, 0.03, 0.01], dtype=float)
        self.regimes = np.zeros(n, dtype=int)
        self.weights = np.full(n, 1.0 / n, dtype=float)

    def _propagate_particles(self) -> np.ndarray:
        """Sample regimes and advance particles with regime-dependent dynamics.

        Returns proposal ratio p(r_prior) / q(r) for importance weighting.
        """
        cfg = self.cfg
        n = cfg.n_particles
        alpha = 0.15

        # Particle-specific Markov transition with exploration mixture
        p_trans = cfg.regime_transition[self.regimes]  # (n, 3)
        q_trans = (1.0 - alpha) * p_trans + (alpha / 3.0)

        u = self.rng.uniform(0.0, 1.0, size=n)
        cum_q = np.cumsum(q_trans, axis=1)
        self.regimes = (u[:, None] > cum_q[:, :2]).sum(axis=1)

        prop_weights = p_trans[np.arange(n), self.regimes] / q_trans[np.arange(n), self.regimes]

        # 1. Normal: OU wander around baseline
        norm_mask = (self.regimes == Regime.NORMAL)
        if np.any(norm_mask):
            x = self.particles[norm_mask]
            drift = -cfg.ou_theta * (x - cfg.baseline)
            noise = self.rng.normal(0.0, cfg.ou_sigma, size=x.shape)
            self.particles[norm_mask] = x + drift + noise

        # 2. Ramping: diffusive random walk tracking drift without baseline pull
        ramp_mask = (self.regimes == Regime.RAMPING)
        if np.any(ramp_mask):
            x = self.particles[ramp_mask]
            noise = self.rng.normal(0.0, cfg.ramp_sigma, size=x.shape)
            self.particles[ramp_mask] = x + noise

        # 3. Step: shock jumps to capture abrupt changes
        step_mask = (self.regimes == Regime.STEP)
        if np.any(step_mask):
            x = self.particles[step_mask]
            shock = self.rng.normal(0.0, cfg.step_sigma, size=x.shape)
            self.particles[step_mask] = x + shock

        self.particles = np.clip(self.particles, self.bounds_lo, self.bounds_hi)
        return prop_weights

    def _predict_observations(self, particles: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        w = particles[:, 0]
        scale = particles[:, 1]
        taper = particles[:, 2]
        gain = particles[:, 3]

        y_tool = cfg.f_tool_base_mn + cfg.k_tool_wear_mn * w
        y_mu = cfg.mu_base + cfg.k_mu_scale * scale
        y_taper = taper
        y_gain = gain

        return np.column_stack([y_tool, y_mu, y_taper, y_gain])

    def _resample(self) -> None:
        """Systematic resampling."""
        n = len(self.weights)
        positions = (self.rng.uniform(0.0, 1.0) + np.arange(n)) / n
        indexes = np.zeros(n, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.regimes = self.regimes[indexes]
        self.weights = np.full(n, 1.0 / n, dtype=float)

    def update(self, cycle: int, features: dict[str, float]) -> TrackerEstimate:
        """Process one cycle of observed features and compute posterior estimates."""
        f_tool_mn = float(features.get("theta_F_tool_N", 6e5)) * 1e-6
        mu = float(features.get("theta_mu", 0.55))
        gain = float(features.get("theta_sigma_scale", 1.0))
        taper = float(features.get("spec_taper_K", features.get("taper_K", 25.0)))

        self.last_billet_temp_C = float(features.get("billet_temp_C", self.last_billet_temp_C))
        self.last_liner_temp_C = float(features.get("liner_temp_mean_C", features.get("container_liner_temp_1", self.last_liner_temp_C)))

        obs = np.array([f_tool_mn, mu, taper, gain], dtype=float)

        # 1. Propagate particles with mixture proposal
        prop_weights = self._propagate_particles()

        # 2. Measurement likelihood
        pred = self._predict_observations(self.particles)
        innov = (obs - pred) / self.cfg.obs_sigma
        log_lik = -0.5 * np.sum(innov ** 2, axis=1)

        # Unnormalized importance weights
        weights_unnorm = prop_weights * np.exp(log_lik - np.max(log_lik))
        tot_w = np.sum(weights_unnorm)
        if tot_w > 0:
            self.weights = weights_unnorm / tot_w
        else:
            self.weights = np.full_like(self.weights, 1.0 / len(self.weights))

        # 3. Posterior moments and regime probabilities
        w = self.weights
        means = np.sum(self.particles * w[:, None], axis=0)
        variances = np.sum(w[:, None] * (self.particles - means) ** 2, axis=0)
        sds = np.sqrt(np.maximum(variances, 1e-12))

        cis = []
        for d in range(N_STATES):
            vals = self.particles[:, d]
            idx = np.argsort(vals)
            cum_w = np.cumsum(w[idx])
            lo = float(vals[idx[np.searchsorted(cum_w, 0.05)]])
            hi = float(vals[idx[min(np.searchsorted(cum_w, 0.95), len(vals) - 1)]])
            cis.append((lo, hi))

        p_norm = float(np.sum(w[self.regimes == Regime.NORMAL]))
        p_ramp = float(np.sum(w[self.regimes == Regime.RAMPING]))
        p_step = float(np.sum(w[self.regimes == Regime.STEP]))

        self.regime_probs = np.array([p_norm, p_ramp, p_step], dtype=float)
        dom_idx = int(np.argmax(self.regime_probs))
        dom_regime = REGIME_NAMES[Regime(dom_idx)]

        est = TrackerEstimate(
            cycle=cycle,
            die_wear_mean=float(means[0]),
            liner_scale_mean=float(means[1]),
            taper_mean=float(means[2]),
            sensor_gain_mean=float(means[3]),
            die_wear_sd=float(sds[0]),
            liner_scale_sd=float(sds[1]),
            taper_sd=float(sds[2]),
            sensor_gain_sd=float(sds[3]),
            die_wear_ci=cis[0],
            liner_scale_ci=cis[1],
            taper_ci=cis[2],
            sensor_gain_ci=cis[3],
            p_normal=p_norm,
            p_ramping=p_ramp,
            p_step=p_step,
            dominant_regime=dom_regime,
        )

        # 4. Resample
        self._resample()

        self.history.append(est)
        return est

    def run_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run tracker over a sequence of cycle features."""
        recs = df.to_dict("records")
        out = []
        for r in recs:
            cyc = int(r["cycle"])
            est = self.update(cyc, r)
            row_dict = est.as_dict()
            for true_col in ("y_true_die_wear", "true_die_wear", "y_true_liner_scale_mm",
                             "true_liner_scale_mm", "y_eff_cap_gain", "eff_cap_gain",
                             "spec_taper_K", "fault_class", "y_fault_class"):
                if true_col in r:
                    row_dict[true_col] = r[true_col]
            out.append(row_dict)
        return pd.DataFrame(out)
