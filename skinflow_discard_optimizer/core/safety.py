"""Safety gates, fallback mechanisms, and advisory output enforcement (Task 5.3).

Part of the Extrusion Charge-Weld Risk Controller.

Responsibilities:
1. Telemetry Validation: Detects corrupt signals (missing samples, NaNs, frozen transducers,
   timestamp jitter/non-monotonicity, out-of-range physical bounds).
2. Operational Context Validation: Detects unsupported alloys, extreme operating temperatures,
   and uncalibrated process windows.
3. Fallback Clamping: If sensor data is corrupt, confidence is compromised, or an unknown
   process state is observed, safely falls back to the conservative static cut (40.0 mm)
   and raises operational alarms.
4. Hard Boundary Enforcement: Strictly constrains all published cut recommendations within
   [min_cut_mm, max_cut_mm] (12.0 mm to 60.0 mm), preventing unbounded or negative cuts.
5. Advisory Protocol Enforcement: Confines outputs exclusively to the ``Predictor.*`` namespace;
   never emits machine or actuator setpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

from skinflow_discard_optimizer.config import load_config, value

logger = logging.getLogger(__name__)

DEFAULT_STATIC_CUT_MM: float = 40.0
DEFAULT_MIN_CUT_MM: float = 12.0
DEFAULT_MAX_CUT_MM: float = 60.0
DEFAULT_SUPPORTED_ALLOYS: tuple[str, ...] = ("AA6063", "AA6082")


@dataclass(frozen=True)
class SafetyConfig:
    static_cut_mm: float = DEFAULT_STATIC_CUT_MM
    min_cut_mm: float = DEFAULT_MIN_CUT_MM
    max_cut_mm: float = DEFAULT_MAX_CUT_MM
    min_samples: int = 50
    max_missing_fraction: float = 0.05
    min_pressure_variance_bar: float = 1e-3
    supported_alloys: tuple[str, ...] = DEFAULT_SUPPORTED_ALLOYS

    @classmethod
    def load(cls) -> "SafetyConfig":
        try:
            cfg = load_config("defect")
            return cls(
                static_cut_mm=float(value(cfg, "baseline.static_cut_mm")),
                min_cut_mm=float(value(cfg, "baseline.min_cut_mm")),
                max_cut_mm=float(value(cfg, "baseline.max_cut_mm")),
            )
        except (KeyError, ValueError, FileNotFoundError, TypeError) as err:
            logger.debug("Falling back to default SafetyConfig due to load error: %s", err)
            return cls()


@dataclass(frozen=True)
class StrokeTelemetry:
    """Encapsulates 1 kHz ram stroke telemetry arrays."""

    t_s: np.ndarray
    x_mm: np.ndarray
    p_cap_bar: np.ndarray
    p_rod_bar: np.ndarray

    @property
    def sample_count(self) -> int:
        return len(self.t_s)


@dataclass(frozen=True)
class CutCandidate:
    """Candidate cut recommendation and process diagnosis awaiting safety review."""

    raw_cut_mm: float | None
    confidence: str = "high"
    fault_class: str = "none"
    is_novel_fault: bool = False
    telemetry_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardedDecision:
    """Safe, bounded, advisory cut recommendation."""

    h_cut_mm: float
    is_fallback: bool
    fallback_reasons: tuple[str, ...]
    alert_level: str  # "NORMAL", "WARNING", "CRITICAL_FALLBACK"
    confidence: str  # "high", "low", "none"
    is_advisory: bool
    published_tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if np.isnan(self.h_cut_mm) or np.isinf(self.h_cut_mm):
            raise ValueError(f"Unbounded cut recommendation: {self.h_cut_mm}")


def _check_timestamps(t_s: np.ndarray) -> list[str]:
    if len(t_s) < 2:
        return ["INSUFFICIENT_TIMESTAMPS: fewer than 2 samples"]
    dt = np.diff(t_s)
    issues: list[str] = []
    if np.any(dt <= 0.0):
        issues.append("TIMESTAMP_JITTER: non-strictly-increasing timestamps (dt <= 0)")
    if np.any(np.isnan(t_s)) or np.any(np.isinf(t_s)):
        issues.append("TIMESTAMP_CORRUPT: NaN or Inf in timestamps")
    return issues


def _check_arrays_health(x_mm: np.ndarray, p_cap: np.ndarray, p_rod: np.ndarray, min_var: float) -> list[str]:
    issues: list[str] = []
    for name, arr in (("x_mm", x_mm), ("p_cap_bar", p_cap), ("p_rod_bar", p_rod)):
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            issues.append(f"MISSING_SAMPLES: NaN or Inf detected in {name}")
            continue
        if len(arr) > 1 and float(np.var(arr)) < min_var:
            issues.append(f"FROZEN_SENSOR: zero variance detected in {name}")
    return issues


def _check_physical_ranges(x_mm: np.ndarray, p_cap: np.ndarray, p_rod: np.ndarray) -> list[str]:
    issues: list[str] = []
    if np.any(x_mm < -1.0) or np.any(x_mm > 1500.0):
        issues.append("OUT_OF_RANGE: ram position x_mm outside physical bounds [-1, 1500]")
    if np.any(p_cap < 0.0) or np.any(p_cap > 450.0):
        issues.append("OUT_OF_RANGE: p_cap_bar outside hydraulic bounds [0, 450]")
    if np.any(p_rod < 0.0) or np.any(p_rod > 450.0):
        issues.append("OUT_OF_RANGE: p_rod_bar outside hydraulic bounds [0, 450]")
    return issues


def _evaluate_fallback_reasons(candidate: CutCandidate) -> list[str]:
    """Evaluates candidate recommendation against safety criteria."""
    reasons = list(candidate.telemetry_issues)
    if candidate.confidence in ("none", "low"):
        reasons.append(f"LOW_CONFIDENCE: onset estimator reported '{candidate.confidence}' confidence")
    if candidate.is_novel_fault or candidate.fault_class == "unknown":
        reasons.append(f"NOVEL_FAULT: classifier flagged out-of-library anomaly ('{candidate.fault_class}')")
    if candidate.raw_cut_mm is None or np.isnan(candidate.raw_cut_mm) or np.isinf(candidate.raw_cut_mm):
        reasons.append("INVALID_CUT: raw cut recommendation is missing or non-finite")
    return reasons


def _build_published_tags(h_cut_mm: float, conf: str, is_fallback: bool, fault_class: str) -> dict[str, Any]:
    """Generates strictly advisory Predictor.* telemetry tags."""
    return {
        "Predictor.ButtCutMm": float(h_cut_mm),
        "Predictor.CutConfidence": conf,
        "Predictor.DriftAlarm": bool(is_fallback),
        "Predictor.FaultClass": str(fault_class),
        "Predictor.IsFallback": bool(is_fallback),
    }


class SafetyGate:
    """Safety supervisor validating telemetry and enforcing bounded fallbacks."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.cfg = config or SafetyConfig.load()

    def validate_stroke_telemetry(self, telemetry: StrokeTelemetry) -> list[str]:
        """Validates 1 kHz stroke sensor streams."""
        n_samples = telemetry.sample_count
        if n_samples < self.cfg.min_samples:
            return [f"INSUFFICIENT_SAMPLES: got {n_samples} samples, min {self.cfg.min_samples}"]

        issues: list[str] = []
        issues.extend(_check_timestamps(telemetry.t_s))
        issues.extend(
            _check_arrays_health(
                telemetry.x_mm,
                telemetry.p_cap_bar,
                telemetry.p_rod_bar,
                self.cfg.min_pressure_variance_bar,
            )
        )
        issues.extend(_check_physical_ranges(telemetry.x_mm, telemetry.p_cap_bar, telemetry.p_rod_bar))
        return issues

    def validate_process_context(
        self,
        alloy_id: str,
        billet_temp_C: float,
        liner_temp_C: float,
    ) -> list[str]:
        """Validates alloy specification and operating thermal envelope."""
        issues: list[str] = []
        if alloy_id not in self.cfg.supported_alloys:
            issues.append(f"UNSEEN_ALLOY: '{alloy_id}' not in calibrated alloy library {self.cfg.supported_alloys}")
        if not (350.0 <= billet_temp_C <= 580.0):
            issues.append(f"OUT_OF_RANGE: billet_temp_C ({billet_temp_C:.1f} C) outside envelope [350, 580]")
        if not (300.0 <= liner_temp_C <= 520.0):
            issues.append(f"OUT_OF_RANGE: liner_temp_C ({liner_temp_C:.1f} C) outside envelope [300, 520]")
        return issues

    def guard_cut_decision(self, candidate: CutCandidate) -> GuardedDecision:
        """Applies safety gates, clamping bounds, and fallback resolution."""
        reasons = _evaluate_fallback_reasons(candidate)

        if reasons:
            h_raw = self.cfg.static_cut_mm
            is_fallback = True
            alert = "CRITICAL_FALLBACK"
            conf = "none"
            logger.warning("Safety fallback engaged: %s. Emitting static cut %.1f mm.", "; ".join(reasons), h_raw)
        else:
            assert candidate.raw_cut_mm is not None
            h_raw = candidate.raw_cut_mm
            is_fallback = False
            alert = "NORMAL"
            conf = candidate.confidence

        h_bounded = float(np.clip(h_raw, self.cfg.min_cut_mm, self.cfg.max_cut_mm))
        tags = _build_published_tags(h_bounded, conf, is_fallback, candidate.fault_class)

        return GuardedDecision(
            h_cut_mm=h_bounded,
            is_fallback=is_fallback,
            fallback_reasons=tuple(reasons),
            alert_level=alert,
            confidence=conf,
            is_advisory=True,
            published_tags=tags,
        )
