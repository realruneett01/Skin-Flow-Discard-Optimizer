"""Fault-type identification and root-cause classifier (Task 4.4).

Maintains a library of physics-informed and statistical fault signatures:
1. **die_wear:** Tooling wear index elevation (F_tool residual, dead-metal zone departure).
2. **liner_scale:** Gradual container scale accumulation (friction mu drift, steady body region).
3. **sensor_gain_drift:** Cap-pressure transducer gain error (sigma_scale drift without physical heat/energy changes).
4. **encoder_offset:** Ram position transducer zero shift (huge SPE, upturn position translation).
5. **lubricant_loss:** Abrupt lubrication breakdown (step friction jump, STEP regime).
6. **temperature_drift:** Furnace thermal taper and cold-billet drift (billet_temp_C drop, dT_K sag).
7. **supply_pressure_sag:** Hydraulic power unit degradation (oil temp surge, supply pressure drop).
8. **unknown:** Novel, unmodeled, or out-of-library anomalies (e.g. flash_spike or unmodeled shocks).
9. **none:** Healthy baseline operation within statistical limits.

Computes the posterior distribution over candidate fault types by comparing how well
each signature explains recent process features and residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skinflow_discard_optimizer.paths import ARTIFACTS_DIR, REPO_ROOT

DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "theta_F_tool_N",
    "theta_mu",
    "theta_sigma_scale",
    "billet_temp_C",
    "dT_K",
    "oil_temp_C",
    "supply_pressure_min_bar",
    "fpca_1",
    "fpca_2",
    "fpca_3",
    "fpca_spe",
    "upturn_h_mm",
)

KNOWN_FAULTS: tuple[str, ...] = (
    "none",
    "die_wear",
    "liner_scale",
    "sensor_gain_drift",
    "encoder_offset",
    "lubricant_loss",
    "temperature_drift",
    "supply_pressure_sag",
)

ALL_FAULT_CLASSES: tuple[str, ...] = KNOWN_FAULTS + ("unknown",)

SUGGESTED_CHECKS: dict[str, str] = {
    "none": "Process operating within normal tolerances. No maintenance action required.",
    "die_wear": "Inspect die bearing surface, check die deflection and wear profile; verify die change history.",
    "liner_scale": "Inspect container liner for alumina/oxide scale accumulation; verify liner heating uniformity and clean liner bore.",
    "sensor_gain_drift": "Verify cap-pressure transducer calibration, excitation voltage, and zero-offset drift.",
    "encoder_offset": "Check ram position encoder optical tape, reader head alignment, and zero-reference switch.",
    "lubricant_loss": "Inspect billet and dummy-block lubrication delivery nozzles; check lubricant tank level and pump delivery pressure.",
    "temperature_drift": "Inspect billet induction/gas furnace heating zones, pyrometer calibration, and billet taper setpoints.",
    "supply_pressure_sag": "Check HPU oil cooler heat exchanger, cooling water flow, hydraulic pump leakage, and relief valve settings.",
    "unknown": "Anomaly does not match known fault library; perform general visual press and hydraulic inspection.",
}


@dataclass(frozen=True)
class FaultCause:
    """Ranked candidate fault cause with confidence and operational guidance."""

    fault_class: str
    probability: float
    suggested_check: str
    log_likelihood: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "fault_class": self.fault_class,
            "probability": self.probability,
            "suggested_check": self.suggested_check,
            "log_likelihood": self.log_likelihood,
        }


@dataclass(frozen=True)
class FaultDiagnosis:
    """Comprehensive fault classification and root-cause analysis for one cycle."""

    cycle: int
    dominant_fault: str
    confidence: float
    is_fault_active: bool
    posteriors: dict[str, float]
    top_causes: list[FaultCause]
    suggested_action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "dominant_fault": self.dominant_fault,
            "confidence": self.confidence,
            "is_fault_active": self.is_fault_active,
            "posteriors": self.posteriors,
            "top_causes": [c.as_dict() for c in self.top_causes],
            "suggested_action": self.suggested_action,
        }


@dataclass
class FaultSignatures:
    """Calibrated statistical signatures for all modeled fault modes."""

    classes: tuple[str, ...]
    feature_cols: tuple[str, ...]
    means: dict[str, np.ndarray]
    variances: dict[str, np.ndarray]
    pooled_variance: np.ndarray
    threshold_unknown: float = 45.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data_dict: dict[str, Any] = {
            "classes": np.array(self.classes),
            "feature_cols": np.array(self.feature_cols),
            "pooled_variance": self.pooled_variance,
            "threshold_unknown": np.array([self.threshold_unknown]),
        }
        for c in self.classes:
            data_dict[f"mean_{c}"] = self.means[c]
            data_dict[f"var_{c}"] = self.variances[c]
        np.savez_compressed(path, **data_dict)

    @classmethod
    def load(cls, path: Path) -> "FaultSignatures":
        data = np.load(path, allow_pickle=True)
        classes = tuple(str(x) for x in data["classes"])
        feat_cols = tuple(str(x) for x in data["feature_cols"])
        means = {c: data[f"mean_{c}"] for c in classes}
        vars_ = {c: data[f"var_{c}"] for c in classes}
        pooled = data["pooled_variance"]
        thr = float(data["threshold_unknown"][0])
        return cls(
            classes=classes,
            feature_cols=feat_cols,
            means=means,
            variances=vars_,
            pooled_variance=pooled,
            threshold_unknown=thr,
        )


def _extract_feature_vector(
    features: dict[str, Any],
    feature_cols: tuple[str, ...],
    default_means: np.ndarray,
) -> np.ndarray:
    """Extract and normalize numerical feature values into an array."""
    vec = np.zeros(len(feature_cols), dtype=float)
    for i, col in enumerate(feature_cols):
        raw_val = features.get(col, np.nan)
        val = float(raw_val) if (raw_val is not None and not np.isnan(raw_val)) else default_means[i]

        if col == "theta_F_tool_N":
            val *= 1e-6
        elif col == "fpca_spe":
            val = float(np.log10(max(val, 1.0)))

        vec[i] = val
    return vec


def _compute_class_log_lik(
    vec: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> float:
    """Compute Gaussian log-likelihood for a feature vector under signature parameters."""
    diff = vec - mean
    sq_dist = float(np.sum((diff ** 2) / variance))
    log_det = float(np.sum(np.log(variance)))
    return -0.5 * (sq_dist + log_det)


def _compute_posteriors(
    vec: np.ndarray,
    sigs: FaultSignatures,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute posterior probabilities and raw log-likelihoods over all fault classes."""
    log_liks: dict[str, float] = {}
    for c in sigs.classes:
        log_liks[c] = _compute_class_log_lik(vec, sigs.means[c], sigs.variances[c])

    # Unknown hypothesis: log-likelihood corresponding to distance threshold
    log_det_pooled = float(np.sum(np.log(sigs.pooled_variance)))
    log_liks["unknown"] = -0.5 * (sigs.threshold_unknown + log_det_pooled)

    max_ll = max(log_liks.values())
    unnorm_probs = {c: float(np.exp(ll - max_ll)) for c, ll in log_liks.items()}
    total_p = sum(unnorm_probs.values())
    posteriors = {c: unnorm_probs[c] / max(total_p, 1e-12) for c in log_liks}
    return posteriors, log_liks


class FaultClassifier:
    """Bayesian multivariate fault classifier with reject option for novel anomalies."""

    def __init__(self, signatures: FaultSignatures | None = None):
        self.signatures = signatures or self._load_or_build_signatures()
        self._default_vector = np.mean(list(self.signatures.means.values()), axis=0)

    @classmethod
    def _signatures_path(cls) -> Path:
        return ARTIFACTS_DIR / "fault_signatures.npz"

    def _load_or_build_signatures(self) -> FaultSignatures:
        p = self._signatures_path()
        if p.exists():
            return FaultSignatures.load(p)
        return self.fit()

    def fit(self, features_dir: Path | None = None) -> FaultSignatures:
        """Fit empirical fault signature models from scenario training datasets."""
        data_dir = features_dir or (REPO_ROOT / "data" / "features")
        means: dict[str, np.ndarray] = {}
        vars_: dict[str, np.ndarray] = {}

        sc_map = [
            ("healthy_baseline", "none", 0, 3500),
            ("die_wear", "die_wear", 6500, 9500),
            ("liner_scale", "liner_scale", 6500, 9500),
            ("sensor_gain_drift", "sensor_gain_drift", 6500, 9500),
            ("encoder_offset", "encoder_offset", 6500, 9500),
            ("lubricant_loss", "lubricant_loss", 6010, 8000),
            ("temperature_drift", "temperature_drift", 6500, 9500),
            ("supply_pressure_sag", "supply_pressure_sag", 6500, 9500),
        ]

        var_floor = np.array([1e-3, 1e-4, 1e-4, 1.0, 1.0, 0.5, 0.5, 1e8, 1e8, 1e8, 0.05, 1.0])
        for sc_name, f_class, c_lo, c_hi in sc_map:
            p = data_dir / f"{sc_name}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            sub = df[(df.cycle >= c_lo) & (df.cycle < c_hi)]
            x_df = sub[list(DEFAULT_FEATURE_COLS)].copy()
            x_df["fpca_spe"] = np.log10(np.maximum(x_df["fpca_spe"], 1.0))
            x_df["theta_F_tool_N"] = x_df["theta_F_tool_N"] * 1e-6
            x_df = x_df.fillna(x_df.median())

            means[f_class] = x_df.mean().to_numpy()
            vars_[f_class] = np.maximum(x_df.var().to_numpy(), var_floor)

        pooled_var = np.mean(list(vars_.values()), axis=0)
        sigs = FaultSignatures(
            classes=KNOWN_FAULTS,
            feature_cols=DEFAULT_FEATURE_COLS,
            means=means,
            variances=vars_,
            pooled_variance=pooled_var,
            threshold_unknown=120.0,
        )
        sigs.save(self._signatures_path())
        self.signatures = sigs
        return sigs

    def classify(
        self,
        cycle: int,
        features: dict[str, Any],
    ) -> FaultDiagnosis:
        """Classify fault type and generate root-cause diagnosis from cycle telemetry."""
        vec = _extract_feature_vector(
            features,
            self.signatures.feature_cols,
            self._default_vector,
        )

        posteriors, log_liks = _compute_posteriors(vec, self.signatures)

        sorted_causes = sorted(
            [
                FaultCause(
                    fault_class=c,
                    probability=posteriors[c],
                    suggested_check=SUGGESTED_CHECKS[c],
                    log_likelihood=log_liks[c],
                )
                for c in posteriors
            ],
            key=lambda x: x.probability,
            reverse=True,
        )

        dominant = sorted_causes[0].fault_class
        confidence = sorted_causes[0].probability
        is_active = dominant != "none"
        action = SUGGESTED_CHECKS[dominant]

        return FaultDiagnosis(
            cycle=cycle,
            dominant_fault=dominant,
            confidence=confidence,
            is_fault_active=is_active,
            posteriors=posteriors,
            top_causes=sorted_causes[:3],
            suggested_action=action,
        )
