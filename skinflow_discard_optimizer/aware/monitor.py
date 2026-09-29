"""Multivariate statistical process monitoring with root-cause attribution (Task 4.2).

Implements three complementary monitoring statistics on force curve FPCA scores and
cross-project process telemetry:
1. **Hotelling T²:** Detects large instantaneous departures within the FPCA subspace.
2. **Squared Prediction Error (SPE / Q-statistic):** Detects novel shape distortions
   breaking out of the healthy eigen-subspace.
3. **Multivariate EWMA (MEWMA):** Detects small, slow, sustained drifts (e.g. progressive
   die wear, gradual scale accumulation, thermal drift) by exponential smoothing.

When an alarm triggers, computes fine-grained attribution:
- **Subspace score breakdown:** Relative percentage contribution of each FPCA mode.
- **Spatial curve reconstruction:** Projects score deviations back onto physical
  stroke eigenfunctions to identify which curve region (entry/upset, steady body,
  deceleration, dead-metal zone) and stroke coordinate (x_mm) departed.
- **Cross-project feature z-scores:** Ranks hydraulic, thermal, and dead-cycle signals
  (oil temperature, pump pressure sag, pump energy, dead-cycle step durations).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skinflow_discard_optimizer.paths import ARTIFACTS_DIR

DEFAULT_FPCA_COLS: tuple[str, ...] = ("fpca_1", "fpca_2", "fpca_3", "fpca_4", "fpca_5")
DEFAULT_SPE_COL: str = "fpca_spe"
DEFAULT_CROSS_COLS: tuple[str, ...] = (
    "oil_temp_C",
    "supply_pressure_min_bar",
    "pump_energy_kwh",
    "shear_stroke_s",
    "dead_cycle_s",
    "billet_temp_C",
)

REGION_BOUNDARIES = (
    ("entry_upset", 0.0, 50.0),
    ("body_steady", 50.0, 650.0),
    ("tail_deceleration", 650.0, 730.0),
    ("dead_metal_zone", 730.0, 800.0),
)


@dataclass(frozen=True)
class Attribution:
    """Attribution analysis for an off-course drift or fault alarm."""

    dominant_channel: str
    top_scores: list[tuple[str, float]]
    top_cross_features: list[tuple[str, float]]
    dominant_curve_region: str
    region_contributions: dict[str, float]
    peak_deviation_x_mm: float
    peak_deviation_force_n: float
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dominant_channel": self.dominant_channel,
            "top_scores": [list(x) for x in self.top_scores],
            "top_cross_features": [list(x) for x in self.top_cross_features],
            "dominant_curve_region": self.dominant_curve_region,
            "region_contributions": self.region_contributions,
            "peak_deviation_x_mm": self.peak_deviation_x_mm,
            "peak_deviation_force_n": self.peak_deviation_force_n,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class MonitorStats:
    """Raw monitoring statistics and alarm states for one cycle."""

    t2: float
    spe: float
    mewma: float
    t2_alarm: bool
    spe_alarm: bool
    mewma_alarm: bool


@dataclass(frozen=True)
class CycleMonitorResult:
    """Monitoring statistics, alarm states, and attribution for one cycle."""

    cycle: int
    t2_statistic: float
    t2_limit: float
    t2_alarm: bool

    spe_statistic: float
    spe_limit: float
    spe_alarm: bool

    mewma_statistic: float
    mewma_limit: float
    mewma_alarm: bool

    is_alarm: bool
    attribution: Attribution | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "cycle": self.cycle,
            "t2_statistic": self.t2_statistic,
            "t2_limit": self.t2_limit,
            "t2_alarm": self.t2_alarm,
            "spe_statistic": self.spe_statistic,
            "spe_limit": self.spe_limit,
            "spe_alarm": self.spe_alarm,
            "mewma_statistic": self.mewma_statistic,
            "mewma_limit": self.mewma_limit,
            "mewma_alarm": self.mewma_alarm,
            "is_alarm": self.is_alarm,
        }
        if self.attribution is not None:
            d["attribution"] = self.attribution.as_dict()
        return d


@dataclass
class MonitorConfig:
    """Calibrated statistical control limits, baseline moments, and eigencomponents."""

    fpca_cols: tuple[str, ...]
    spe_col: str
    cross_cols: tuple[str, ...]

    mu_fpca: np.ndarray
    inv_cov_fpca: np.ndarray
    t2_limit: float
    spe_limit: float

    mewma_lambda: float
    mewma_limit: float
    target_far: float

    mu_cross: np.ndarray
    std_cross: np.ndarray

    eigencomponents: np.ndarray | None = None
    grid_x: np.ndarray | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "fpca_cols": np.array(self.fpca_cols),
            "spe_col": np.array(self.spe_col),
            "cross_cols": np.array(self.cross_cols),
            "mu_fpca": self.mu_fpca,
            "inv_cov_fpca": self.inv_cov_fpca,
            "t2_limit": np.array(self.t2_limit),
            "spe_limit": np.array(self.spe_limit),
            "mewma_lambda": np.array(self.mewma_lambda),
            "mewma_limit": np.array(self.mewma_limit),
            "target_far": np.array(self.target_far),
            "mu_cross": self.mu_cross,
            "std_cross": self.std_cross,
        }
        if self.eigencomponents is not None:
            save_dict["eigencomponents"] = self.eigencomponents
        if self.grid_x is not None:
            save_dict["grid_x"] = self.grid_x
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: Path) -> "MonitorConfig":
        d = np.load(path, allow_pickle=True)
        return cls(
            fpca_cols=tuple(d["fpca_cols"]),
            spe_col=str(d["spe_col"]),
            cross_cols=tuple(d["cross_cols"]),
            mu_fpca=d["mu_fpca"],
            inv_cov_fpca=d["inv_cov_fpca"],
            t2_limit=float(d["t2_limit"]),
            spe_limit=float(d["spe_limit"]),
            mewma_lambda=float(d["mewma_lambda"]),
            mewma_limit=float(d["mewma_limit"]),
            target_far=float(d["target_far"]),
            mu_cross=d["mu_cross"],
            std_cross=d["std_cross"],
            eigencomponents=d.get("eigencomponents"),
            grid_x=d.get("grid_x"),
        )


def _calibrate_limits(
    t2_vals: np.ndarray,
    spe_vals: np.ndarray,
    mewma_vals: np.ndarray,
    target_far: float,
) -> tuple[float, float, float]:
    """Search for joint quantile to meet target FAR on training data."""
    q_low, q_high = 0.990, 0.9999
    best_q = 0.9985
    for _ in range(25):
        q_mid = 0.5 * (q_low + q_high)
        t2_lim = float(np.percentile(t2_vals, q_mid * 100.0))
        spe_lim = float(np.percentile(spe_vals, q_mid * 100.0))
        mewma_lim = float(np.percentile(mewma_vals, q_mid * 100.0))

        joint_alarm = (t2_vals > t2_lim) | (spe_vals > spe_lim) | (mewma_vals > mewma_lim)
        if float(np.mean(joint_alarm)) > target_far:
            q_low = q_mid
        else:
            q_high = q_mid
            best_q = q_mid

    return (
        float(np.percentile(t2_vals, best_q * 100.0)),
        float(np.percentile(spe_vals, best_q * 100.0)),
        float(np.percentile(mewma_vals, best_q * 100.0)),
    )


def _attribute_scores(
    diff: np.ndarray, inv_cov: np.ndarray, fpca_cols: tuple[str, ...]
) -> list[tuple[str, float]]:
    """Compute percentage contributions of each FPCA score mode to Hotelling T2."""
    contrib_vec = diff * (inv_cov @ diff)
    pos_contrib = np.maximum(contrib_vec, 0.0)
    tot = float(np.sum(pos_contrib))
    pcts = (pos_contrib / tot * 100.0) if tot > 0.0 else np.zeros_like(pos_contrib)
    return sorted(
        [(name, float(pct)) for name, pct in zip(fpca_cols, pcts)],
        key=lambda x: x[1],
        reverse=True,
    )


def _attribute_cross_features(
    features: dict[str, float],
    cross_cols: tuple[str, ...],
    mu_cross: np.ndarray,
    std_cross: np.ndarray,
) -> list[tuple[str, float]]:
    """Rank cross-project features by standardized z-score deviation."""
    cross_vals = np.array([float(features.get(c, mu_cross[i])) for i, c in enumerate(cross_cols)])
    z_scores = (cross_vals - mu_cross) / std_cross
    return sorted(
        [(name, float(z)) for name, z in zip(cross_cols, z_scores)],
        key=lambda x: abs(x[1]),
        reverse=True,
    )


def _attribute_spatial_curve(
    diff: np.ndarray,
    components: np.ndarray | None,
    grid_x: np.ndarray | None,
) -> tuple[str, dict[str, float], float, float]:
    """Reconstruct force profile deviation to identify physical stroke region and peak location."""
    reg_contribs: dict[str, float] = {r[0]: 0.0 for r in REGION_BOUNDARIES}
    dom_region = "body_steady"
    peak_x = 350.0
    peak_force = 0.0

    if components is None or grid_x is None:
        return dom_region, reg_contribs, peak_x, peak_force

    delta_y = diff @ components
    abs_dev = np.abs(delta_y)
    peak_idx = int(np.argmax(abs_dev))
    peak_x = float(grid_x[peak_idx])
    peak_force = float(delta_y[peak_idx])

    tot_energy = 0.0
    for name, x_lo, x_hi in REGION_BOUNDARIES:
        mask = (grid_x >= x_lo) & (grid_x < x_hi)
        if np.any(mask):
            energy = float(np.sum(delta_y[mask] ** 2))
            reg_contribs[name] = energy
            tot_energy += energy

    if tot_energy > 0.0:
        for k in reg_contribs:
            reg_contribs[k] = float(reg_contribs[k] / tot_energy * 100.0)
        dom_region = max(reg_contribs, key=reg_contribs.get)

    return dom_region, reg_contribs, peak_x, peak_force


class MultivariateMonitor:
    """Multivariate statistical process monitor with MEWMA and spatial attribution."""

    def __init__(self, config: MonitorConfig | None = None, fpca_path: Path | None = None):
        if config is None:
            default_path = ARTIFACTS_DIR / "monitor_config.npz"
            if default_path.exists():
                self.cfg = MonitorConfig.load(default_path)
            else:
                self.cfg = self._default_config()
        else:
            self.cfg = config

        if self.cfg.eigencomponents is None:
            self._load_eigencomponents(fpca_path or ARTIFACTS_DIR / "fpca.npz")

        self.reset()

    def _default_config(self) -> MonitorConfig:
        n_fpca = len(DEFAULT_FPCA_COLS)
        n_cross = len(DEFAULT_CROSS_COLS)
        return MonitorConfig(
            fpca_cols=DEFAULT_FPCA_COLS,
            spe_col=DEFAULT_SPE_COL,
            cross_cols=DEFAULT_CROSS_COLS,
            mu_fpca=np.zeros(n_fpca),
            inv_cov_fpca=np.eye(n_fpca),
            t2_limit=18.0,
            spe_limit=8.0e9,
            mewma_lambda=0.10,
            mewma_limit=140.0,
            target_far=0.005,
            mu_cross=np.zeros(n_cross),
            std_cross=np.ones(n_cross),
        )

    def _load_eigencomponents(self, fpca_path: Path) -> None:
        if not fpca_path.exists():
            return
        d = np.load(fpca_path)
        self.cfg.eigencomponents = d["components"]
        n_pts = self.cfg.eigencomponents.shape[1]
        if n_pts == 360:
            body_x = np.linspace(2.0, 680.0, 300)
            tail_x = np.linspace(681.0, 740.0, 60)
            self.cfg.grid_x = np.concatenate([body_x, tail_x])
        else:
            self.cfg.grid_x = np.linspace(2.0, 750.0, n_pts)

    def reset(self) -> None:
        """Reset MEWMA state between runs."""
        self._mewma_z = np.zeros_like(self.cfg.mu_fpca)
        self._step_k = 0
        sigma_z_factor = self.cfg.mewma_lambda / (2.0 - self.cfg.mewma_lambda)
        self._inv_cov_z = self.cfg.inv_cov_fpca / sigma_z_factor

    @classmethod
    def fit(
        cls,
        df_healthy: pd.DataFrame,
        fpca_path: Path | None = None,
        far_target: float = 0.005,
        mewma_lambda: float = 0.10,
    ) -> "MultivariateMonitor":
        """Calibrate control limits on healthy baseline telemetry to achieve stated FAR."""
        fpca_cols = tuple([c for c in DEFAULT_FPCA_COLS if c in df_healthy.columns])
        spe_col = DEFAULT_SPE_COL if DEFAULT_SPE_COL in df_healthy.columns else "fpca_spe"
        cross_cols = tuple([c for c in DEFAULT_CROSS_COLS if c in df_healthy.columns])

        x_fpca = df_healthy[list(fpca_cols)].to_numpy(dtype=float)
        mu_fpca = np.mean(x_fpca, axis=0)
        cov_fpca = np.cov(x_fpca, rowvar=False)
        inv_cov_fpca = np.linalg.pinv(cov_fpca)

        diff_fpca = x_fpca - mu_fpca
        t2_vals = np.sum((diff_fpca @ inv_cov_fpca) * diff_fpca, axis=1)

        spe_vals = df_healthy[spe_col].to_numpy(dtype=float) if spe_col in df_healthy.columns else np.zeros(len(df_healthy))

        mewma_vals = np.zeros(len(x_fpca))
        z = np.zeros_like(mu_fpca)
        sigma_z_factor = mewma_lambda / (2.0 - mewma_lambda)
        inv_cov_z = inv_cov_fpca / sigma_z_factor

        for k in range(len(x_fpca)):
            z = mewma_lambda * diff_fpca[k] + (1.0 - mewma_lambda) * z
            corr = 1.0 - (1.0 - mewma_lambda) ** (2 * (k + 1))
            mewma_vals[k] = float(z @ (inv_cov_z / corr) @ z)

        x_cross = df_healthy[list(cross_cols)].to_numpy(dtype=float) if cross_cols else np.zeros((len(df_healthy), 1))
        mu_cross = np.mean(x_cross, axis=0)
        std_cross = np.std(x_cross, axis=0)
        std_cross = np.where(std_cross < 1e-6, 1.0, std_cross)

        t2_lim, spe_lim, mewma_lim = _calibrate_limits(t2_vals, spe_vals, mewma_vals, far_target)

        config = MonitorConfig(
            fpca_cols=fpca_cols,
            spe_col=spe_col,
            cross_cols=cross_cols,
            mu_fpca=mu_fpca,
            inv_cov_fpca=inv_cov_fpca,
            t2_limit=t2_lim,
            spe_limit=spe_lim,
            mewma_lambda=mewma_lambda,
            mewma_limit=mewma_lim,
            target_far=far_target,
            mu_cross=mu_cross,
            std_cross=std_cross,
        )
        return cls(config=config, fpca_path=fpca_path)

    def update(self, cycle: int, features: dict[str, float]) -> CycleMonitorResult:
        """Stream update for one cycle's extracted features."""
        cfg = self.cfg
        self._step_k += 1

        scores = np.array([float(features.get(c, 0.0)) for c in cfg.fpca_cols], dtype=float)
        diff = scores - cfg.mu_fpca

        t2_stat = float(diff @ cfg.inv_cov_fpca @ diff)
        t2_alarm = t2_stat > cfg.t2_limit

        spe_stat = float(features.get(cfg.spe_col, 0.0))
        spe_alarm = spe_stat > cfg.spe_limit

        self._mewma_z = cfg.mewma_lambda * diff + (1.0 - cfg.mewma_lambda) * self._mewma_z
        corr = 1.0 - (1.0 - cfg.mewma_lambda) ** (2 * self._step_k)
        mewma_stat = float(self._mewma_z @ (self._inv_cov_z / corr) @ self._mewma_z)
        mewma_alarm = mewma_stat > cfg.mewma_limit

        stats = MonitorStats(
            t2=t2_stat,
            spe=spe_stat,
            mewma=mewma_stat,
            t2_alarm=t2_alarm,
            spe_alarm=spe_alarm,
            mewma_alarm=mewma_alarm,
        )

        is_alarm = t2_alarm or spe_alarm or mewma_alarm
        attrib = None
        if is_alarm:
            attrib = self._build_attribution(diff, stats, features)

        return CycleMonitorResult(
            cycle=cycle,
            t2_statistic=t2_stat,
            t2_limit=cfg.t2_limit,
            t2_alarm=t2_alarm,
            spe_statistic=spe_stat,
            spe_limit=cfg.spe_limit,
            spe_alarm=spe_alarm,
            mewma_statistic=mewma_stat,
            mewma_limit=cfg.mewma_limit,
            mewma_alarm=mewma_alarm,
            is_alarm=is_alarm,
            attribution=attrib,
        )

    def _build_attribution(
        self,
        diff: np.ndarray,
        stats: MonitorStats,
        features: dict[str, float],
    ) -> Attribution:
        cfg = self.cfg
        ratios = {
            "t2": stats.t2 / max(cfg.t2_limit, 1e-6),
            "spe": stats.spe / max(cfg.spe_limit, 1e-6),
            "mewma": stats.mewma / max(cfg.mewma_limit, 1e-6),
        }
        dominant_channel = max(ratios, key=ratios.get)

        top_scores = _attribute_scores(diff, cfg.inv_cov_fpca, cfg.fpca_cols)
        top_cross = _attribute_cross_features(features, cfg.cross_cols, cfg.mu_cross, cfg.std_cross)
        dom_reg, reg_contribs, peak_x, peak_force = _attribute_spatial_curve(diff, cfg.eigencomponents, cfg.grid_x)

        top_sc = f"{top_scores[0][0]} ({top_scores[0][1]:.1f}%)" if top_scores else "none"
        top_cr = f"{top_cross[0][0]} ({top_cross[0][1]:+.1f}σ)" if top_cross else "none"
        active_tags = [
            name
            for name, is_active in [
                ("T²", stats.t2_alarm),
                ("SPE", stats.spe_alarm),
                ("MEWMA", stats.mewma_alarm),
            ]
            if is_active
        ]
        active_str = "+".join(active_tags)

        summary = (
            f"Drift alarm [{active_str}] on {dominant_channel.upper()} channel. "
            f"Curve departure centered in {dom_reg} at x={peak_x:.1f} mm ({peak_force*1e-3:+.1f} kN). "
            f"Top score: {top_sc}; key telemetry shift: {top_cr}."
        )

        return Attribution(
            dominant_channel=dominant_channel,
            top_scores=top_scores,
            top_cross_features=top_cross,
            dominant_curve_region=dom_reg,
            region_contributions=reg_contribs,
            peak_deviation_x_mm=peak_x,
            peak_deviation_force_n=peak_force,
            summary=summary,
        )

    def run_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run monitor sequentially over a feature dataframe."""
        self.reset()
        recs = df.to_dict("records")
        results = []
        for r in recs:
            cyc = int(r["cycle"])
            res = self.update(cyc, r)
            row_dict = res.as_dict()
            for true_col in ("fault_class", "y_fault_class", "y_n_active_faults", "y_h_crit_mm", "y_oracle_cut_mm"):
                if true_col in r:
                    row_dict[true_col] = r[true_col]
            results.append(row_dict)
        return pd.DataFrame(results)
