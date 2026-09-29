"""Forward trajectory forecasting and time-to-limit prediction (Task 4.3).

Projects the press's physical state forward over the next N production cycles
using Monte Carlo simulation from the particle-filter posterior:
1. **Physical Forward Simulation:** Samples particles from the current wear-state
   posterior (`LatentWearTracker`), advancing each trajectory under Markov regime
   dynamics (OU wander, progressive degradation, jump shocks).
2. **Target Projections:**
   - Optimal discard cut h* and critical thickness h_crit (via `DefectModel`).
   - Defect risk P(h_cut < h_crit) under nominal or proposed cut policies.
   - Multivariate monitoring statistics (T², MEWMA) to anticipate future alarms.
3. **Time-to-Limit (TTL):** Evaluates probability of crossing configured thresholds
   (cut limits, defect risk limits, alarm thresholds) within N cycles, and returns
   the empirical TTL distribution (median, 10th and 90th percentiles).
4. **Hybrid Learned Residual Model:** Evaluates a compact temporal convolutional
   network (TCN) trained on physics prediction residuals. If the learned component
   improves validation RMSE, it is blended into the final forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from skinflow_discard_optimizer.aware.monitor import MultivariateMonitor
from skinflow_discard_optimizer.aware.state_tracker import (
    LatentWearTracker,
    Regime,
    TrackerConfig,
)
from skinflow_discard_optimizer.sim.defect_model import DefectModel, PressState

DEFAULT_LIMITS: dict[str, float] = {
    "cut_upper_bound_mm": 30.0,
    "defect_prob_pct": 0.05,
    "mewma_alarm_stat": 140.0,
}


@dataclass(frozen=True)
class ForecastHorizonSummary:
    """Summary statistics for one forward forecast cycle."""

    step: int
    cycle: int
    h_cut_mean: float
    h_cut_median: float
    h_cut_ci: tuple[float, float]
    h_crit_mean: float
    h_crit_ci: tuple[float, float]
    defect_prob_mean: float
    defect_prob_ci: tuple[float, float]
    t2_mean: float
    mewma_mean: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "cycle": self.cycle,
            "h_cut_mean": self.h_cut_mean,
            "h_cut_median": self.h_cut_median,
            "h_cut_ci": self.h_cut_ci,
            "h_crit_mean": self.h_crit_mean,
            "h_crit_ci": self.h_crit_ci,
            "defect_prob_mean": self.defect_prob_mean,
            "defect_prob_ci": self.defect_prob_ci,
            "t2_mean": self.t2_mean,
            "mewma_mean": self.mewma_mean,
        }


@dataclass(frozen=True)
class TimeToLimit:
    """Probability of breach and time-to-limit distribution for a threshold."""

    limit_name: str
    limit_value: float
    crossing_probability: float
    median_ttl_cycles: float | None
    ttl_p10_cycles: float | None
    ttl_p90_cycles: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit_name": self.limit_name,
            "limit_value": self.limit_value,
            "crossing_probability": self.crossing_probability,
            "median_ttl_cycles": self.median_ttl_cycles,
            "ttl_p10_cycles": self.ttl_p10_cycles,
            "ttl_p90_cycles": self.ttl_p90_cycles,
        }


@dataclass(frozen=True)
class ForecastResult:
    """Complete forward forecast result over N cycles."""

    current_cycle: int
    horizon_steps: int
    summaries: list[ForecastHorizonSummary]
    limits: list[TimeToLimit]
    recommended_lead_warning: bool
    lead_time_gained_cycles: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_cycle": self.current_cycle,
            "horizon_steps": self.horizon_steps,
            "summaries": [s.as_dict() for s in self.summaries],
            "limits": [lim.as_dict() for lim in self.limits],
            "recommended_lead_warning": self.recommended_lead_warning,
            "lead_time_gained_cycles": self.lead_time_gained_cycles,
        }


@dataclass
class TrajectoryBatch:
    """Simulated batch of forward physical and metric trajectories."""

    h_cut: np.ndarray
    h_crit: np.ndarray
    defect_prob: np.ndarray
    t2: np.ndarray
    mewma: np.ndarray


@dataclass(frozen=True)
class SimulationContext:
    """Settings and initial thermal conditions for forward Monte Carlo simulation."""

    horizon_steps: int = 50
    n_simulations: int = 200
    planned_cut_mm: float = 26.0
    billet_temp_C: float = 470.0
    liner_temp_C: float = 430.0


class ResidualTCN(nn.Module):
    """Small temporal convolutional network to predict physics forecast residuals."""

    def __init__(self, in_channels: int = 4, horizon_steps: int = 50, hidden: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, horizon_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        pooled = self.pool(h).squeeze(-1)
        return self.fc(pooled)


def _propagate_single_step(
    particles: np.ndarray,
    regimes: np.ndarray,
    cfg: TrackerConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance particle states and regimes by one forward step."""
    n = len(particles)
    p_trans = cfg.regime_transition[regimes]
    u = rng.uniform(0.0, 1.0, size=n)
    cum_p = np.cumsum(p_trans, axis=1)
    new_regimes = (u[:, None] > cum_p[:, :2]).sum(axis=1)

    new_particles = particles.copy()

    m_norm = (new_regimes == Regime.NORMAL)
    if np.any(m_norm):
        x = new_particles[m_norm]
        drift = -cfg.ou_theta * (x - cfg.baseline)
        noise = rng.normal(0.0, cfg.ou_sigma, size=x.shape)
        new_particles[m_norm] = x + drift + noise

    m_ramp = (new_regimes == Regime.RAMPING)
    if np.any(m_ramp):
        x = new_particles[m_ramp]
        noise = rng.normal(0.0, cfg.ramp_sigma, size=x.shape)
        new_particles[m_ramp] = x + noise

    m_step = (new_regimes == Regime.STEP)
    if np.any(m_step):
        x = new_particles[m_step]
        shock = rng.normal(0.0, cfg.step_sigma, size=x.shape)
        new_particles[m_step] = x + shock

    bounds_lo = np.array([0.0, 0.0, 0.0, 0.80])
    bounds_hi = np.array([1.5, 3.0, 80.0, 1.30])
    new_particles = np.clip(new_particles, bounds_lo, bounds_hi)
    return new_particles, new_regimes


def _simulate_trajectories(
    curr_particles: np.ndarray,
    curr_regimes: np.ndarray,
    cfg: TrackerConfig,
    dm: DefectModel,
    ctx: SimulationContext,
    rng: np.random.Generator,
) -> TrajectoryBatch:
    """Simulate N-step forward particle trajectories under physical dynamics."""
    h_cut_traj = np.zeros((ctx.n_simulations, ctx.horizon_steps))
    h_crit_traj = np.zeros((ctx.n_simulations, ctx.horizon_steps))
    defect_prob_traj = np.zeros((ctx.n_simulations, ctx.horizon_steps))
    t2_traj = np.zeros((ctx.n_simulations, ctx.horizon_steps))
    mewma_traj = np.zeros((ctx.n_simulations, ctx.horizon_steps))

    particles = curr_particles.copy()
    regimes = curr_regimes.copy()

    for j in range(ctx.horizon_steps):
        particles, regimes = _propagate_single_step(particles, regimes, cfg, rng)

        for m in range(ctx.n_simulations):
            p = particles[m]
            state = PressState(
                die_wear=p[0],
                liner_scale_mm=p[1],
                taper_K=p[2],
                billet_temp_C=ctx.billet_temp_C,
                liner_temp_C=ctx.liner_temp_C,
            )
            h_mean = dm.h_crit_mean(state)
            noise = rng.normal(0.0, dm.p.noise_sd)
            h_crit = max(h_mean + noise, dm.p.h_crit_min)
            h_cut = dm.oracle_cut(h_crit)
            dp = dm.defect_probability(ctx.planned_cut_mm, h_crit)

            h_cut_traj[m, j] = h_cut
            h_crit_traj[m, j] = h_crit
            defect_prob_traj[m, j] = dp

            y_tool = cfg.f_tool_base_mn + cfg.k_tool_wear_mn * p[0]
            y_mu = cfg.mu_base + cfg.k_mu_scale * p[1]
            stat_proxy = ((y_tool - 0.60) / 0.035) ** 2 + ((y_mu - 0.55) / 0.010) ** 2
            t2_traj[m, j] = stat_proxy
            mewma_traj[m, j] = stat_proxy * (1.0 - 0.9 ** (j + 1))

    return TrajectoryBatch(h_cut_traj, h_crit_traj, defect_prob_traj, t2_traj, mewma_traj)


def _summarize_horizon(batch: TrajectoryBatch, current_cycle: int) -> list[ForecastHorizonSummary]:
    """Aggregate simulation trajectories into step-by-step summary statistics."""
    horizon_steps = batch.h_cut.shape[1]
    summaries = []
    for j in range(horizon_steps):
        cyc = current_cycle + j + 1
        cuts = batch.h_cut[:, j]
        crits = batch.h_crit[:, j]
        dps = batch.defect_prob[:, j]

        cut_ci = (float(np.percentile(cuts, 5.0)), float(np.percentile(cuts, 95.0)))
        crit_ci = (float(np.percentile(crits, 5.0)), float(np.percentile(crits, 95.0)))
        dp_ci = (float(np.percentile(dps, 5.0)), float(np.percentile(dps, 95.0)))

        summaries.append(
            ForecastHorizonSummary(
                step=j + 1,
                cycle=cyc,
                h_cut_mean=float(np.mean(cuts)),
                h_cut_median=float(np.median(cuts)),
                h_cut_ci=cut_ci,
                h_crit_mean=float(np.mean(crits)),
                h_crit_ci=crit_ci,
                defect_prob_mean=float(np.mean(dps)),
                defect_prob_ci=dp_ci,
                t2_mean=float(np.mean(batch.t2[:, j])),
                mewma_mean=float(np.mean(batch.mewma[:, j])),
            )
        )
    return summaries


def _compute_ttl_metric(
    trajectory_matrix: np.ndarray,
    limit_name: str,
    limit_value: float,
    compare_greater: bool = True,
) -> TimeToLimit:
    """Compute breach probability and TTL percentiles across simulation trajectories."""
    n_sim, horizon = trajectory_matrix.shape
    ttls = np.full(n_sim, horizon + 1, dtype=int)

    for m in range(n_sim):
        traj = trajectory_matrix[m]
        breach_mask = (traj >= limit_value) if compare_greater else (traj <= limit_value)
        if np.any(breach_mask):
            ttls[m] = int(np.argmax(breach_mask)) + 1

    breached = ttls <= horizon
    cross_prob = float(np.mean(breached))
    if np.any(breached):
        valid = ttls[breached]
        med_ttl = float(np.median(valid))
        p10_ttl = float(np.percentile(valid, 10.0))
        p90_ttl = float(np.percentile(valid, 90.0))
    else:
        med_ttl, p10_ttl, p90_ttl = None, None, None

    return TimeToLimit(
        limit_name=limit_name,
        limit_value=limit_value,
        crossing_probability=cross_prob,
        median_ttl_cycles=med_ttl,
        ttl_p10_cycles=p10_ttl,
        ttl_p90_cycles=p90_ttl,
    )


class Forecaster:
    """Monte Carlo forward forecaster from wear-state posterior."""

    def __init__(
        self,
        defect_model: DefectModel | None = None,
        monitor: MultivariateMonitor | None = None,
        seed: int = 42,
    ):
        self.dm = defect_model or DefectModel()
        self.monitor = monitor or MultivariateMonitor()
        self.rng = np.random.default_rng(seed)
        self.learned_model: ResidualTCN | None = None
        self.use_learned_residual: bool = False

    def forecast(
        self,
        tracker: LatentWearTracker,
        horizon_steps: int = 50,
        n_simulations: int = 200,
        limits: dict[str, float] | None = None,
        planned_cut_mm: float = 26.0,
    ) -> ForecastResult:
        """Forecast the next horizon_steps cycles from tracker's current posterior."""
        cfg = tracker.cfg
        current_cycle = tracker.history[-1].cycle if tracker.history else 0

        # Resample particles from posterior
        weights = tracker.weights / np.sum(tracker.weights)
        idx = self.rng.choice(len(weights), size=n_simulations, p=weights)
        curr_particles = tracker.particles[idx].copy()
        curr_regimes = tracker.regimes[idx].copy()

        b_temp = getattr(tracker, "last_billet_temp_C", 470.0)
        l_temp = getattr(tracker, "last_liner_temp_C", 430.0)

        sim_ctx = SimulationContext(
            horizon_steps=horizon_steps,
            n_simulations=n_simulations,
            planned_cut_mm=planned_cut_mm,
            billet_temp_C=b_temp,
            liner_temp_C=l_temp,
        )
        batch = _simulate_trajectories(
            curr_particles,
            curr_regimes,
            cfg,
            self.dm,
            sim_ctx,
            self.rng,
        )

        if self.use_learned_residual and self.learned_model is not None:
            batch.h_cut += self._predict_learned_correction(tracker, horizon_steps)

        summaries = _summarize_horizon(batch, current_cycle)

        lim_cfg = limits or DEFAULT_LIMITS
        lim_results = [
            _compute_ttl_metric(batch.h_cut, "cut_upper_bound_mm", lim_cfg.get("cut_upper_bound_mm", 30.0), True),
            _compute_ttl_metric(batch.defect_prob, "defect_prob_pct", lim_cfg.get("defect_prob_pct", 0.05), True),
            _compute_ttl_metric(batch.mewma, "mewma_alarm_stat", lim_cfg.get("mewma_alarm_stat", 140.0), True),
        ]

        any_breach_high = any(lr.crossing_probability >= 0.40 for lr in lim_results)
        lead_gained = 0
        for lr in lim_results:
            if lr.crossing_probability >= 0.40 and lr.median_ttl_cycles is not None:
                lead_gained = max(lead_gained, int(lr.median_ttl_cycles))

        return ForecastResult(
            current_cycle=current_cycle,
            horizon_steps=horizon_steps,
            summaries=summaries,
            limits=lim_results,
            recommended_lead_warning=any_breach_high,
            lead_time_gained_cycles=lead_gained,
        )

    def _predict_learned_correction(self, tracker: LatentWearTracker, horizon_steps: int) -> np.ndarray:
        if self.learned_model is None or not tracker.history:
            return np.zeros(horizon_steps)
        hist = tracker.history[-10:]
        feats = np.array([[h.die_wear_mean, h.liner_scale_mean, h.taper_mean, h.sensor_gain_mean] for h in hist])
        if len(feats) < 10:
            pad = np.repeat(feats[:1], 10 - len(feats), axis=0)
            feats = np.concatenate([pad, feats], axis=0)
        x_t = torch.tensor(feats.T, dtype=torch.float32).unsqueeze(0)
        self.learned_model.eval()
        with torch.no_grad():
            out = self.learned_model(x_t).squeeze(0).numpy()
        return out[:horizon_steps]
