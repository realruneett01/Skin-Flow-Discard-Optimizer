"""Streaming optimization service and OPC-UA advisory tag publisher (Task 6.1).

Subscribes to tag-level streams from contracts/signals.yaml (or simulated replay),
executes Phases 2-5 per extrusion cycle, and publishes strictly advisory
Predictor.* tags within a strict latency budget (p99 < 200 ms).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from skinflow_discard_optimizer.aware.fault_id import FaultClassifier
from skinflow_discard_optimizer.aware.monitor import MultivariateMonitor
from skinflow_discard_optimizer.core.assembler import CycleAssembler
from skinflow_discard_optimizer.core.estimator.pipeline import (
    MODELS_PATH,
    CutEngine,
    CycleDecision,
    StrokeInputs,
    load_models,
)
from skinflow_discard_optimizer.core.features import CycleInputs, FeatureExtractor
from skinflow_discard_optimizer.core.functional import FPCA
from skinflow_discard_optimizer.core.safety import (
    CutCandidate,
    GuardedDecision,
    SafetyConfig,
    SafetyGate,
    StrokeTelemetry,
)
from skinflow_discard_optimizer.paths import REPO_ROOT
from skinflow_discard_optimizer.service.replay import replay
from skinflow_discard_optimizer.sim.scenario import Scenario, run_scenario

logger = logging.getLogger("skinflow.service")

DEFAULT_LATENCY_BUDGET_MS: float = 200.0
DEFAULT_FPCA_PATH: Path = REPO_ROOT / "artifacts" / "fpca.npz"


@dataclass(frozen=True)
class ServiceConfig:
    """Configuration for streaming optimization service."""

    latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS
    warn_latency_ms: float = 50.0
    models_path: Path = MODELS_PATH
    fpca_path: Path = DEFAULT_FPCA_PATH
    block: int = 40
    mock_cycles: int = 10
    enable_json_logging: bool = False

    @classmethod
    def default(cls) -> "ServiceConfig":
        return cls()


@dataclass(frozen=True)
class CycleLogEntry:
    """Structured decision audit log for a single production cycle."""

    cycle_id: int
    timestamp_iso: str
    alloy_id: str
    die_id: str
    billet_temp_C: float
    cut_recommendation_mm: float
    interval_low_mm: float
    interval_high_mm: float
    confidence: str
    is_fallback: bool
    fallback_reasons: tuple[str, ...]
    drift_alarm: bool
    fault_class: str
    latency_ms: float
    within_budget: bool
    published_tags: dict[str, Any]


@dataclass(frozen=True)
class ServiceHealth:
    """Operational health status and latency statistics."""

    status: str  # "HEALTHY", "DEGRADED", "ERROR"
    models_loaded: bool
    cycles_processed: int
    fallbacks_count: int
    alarms_count: int
    mean_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    budget_violations: int
    uptime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _convert_cycle_to_stroke_inputs(inp: CycleInputs) -> StrokeInputs:
    """Converts assembled CycleInputs to StrokeInputs required by CutEngine."""
    mean_liner = float(np.nanmean(inp.liner_temps_C)) if inp.liner_temps_C else 430.0
    return StrokeInputs(
        t_s=inp.t_s,
        x_mm=inp.x_mm,
        p_cap_bar=inp.p_cap_bar,
        p_rod_bar=inp.p_rod_bar,
        billet_length_mm=inp.billet_length_mm,
        billet_temp_C=inp.billet_temp_C,
        liner_temp_mean_C=mean_liner,
        extrusion_ratio=inp.ratio(),
        alloy_id=inp.alloy_id,
        die_id=inp.die_id,
    )


def _build_candidate(
    decision: CycleDecision,
    telemetry_issues: list[str],
    fault_class: str,
) -> CutCandidate:
    """Builds a CutCandidate from engine output and telemetry validation."""
    return CutCandidate(
        raw_cut_mm=decision.recommendation.h_cut_mm,
        confidence=decision.onset_confidence,
        fault_class=fault_class,
        is_novel_fault=(fault_class == "unknown"),
        telemetry_issues=tuple(telemetry_issues),
    )


class StreamingOptimizerService:
    """Live streaming service running Phases 2-5 and publishing Predictor.* tags."""

    def __init__(
        self,
        config: ServiceConfig | None = None,
        engine: CutEngine | None = None,
        monitor: MultivariateMonitor | None = None,
        safety_gate: SafetyGate | None = None,
    ) -> None:
        self.cfg = config or ServiceConfig.default()
        self.assembler = CycleAssembler()
        self.safety_gate = safety_gate or SafetyGate(SafetyConfig.load())
        self.monitor = monitor or MultivariateMonitor()
        self.fault_classifier = FaultClassifier()
        self.start_time = time.time()
        self.history: list[CycleLogEntry] = []

        fpca = None
        if self.cfg.fpca_path.exists():
            fpca = FPCA.load(self.cfg.fpca_path)
        self.feature_extractor = FeatureExtractor(fpca=fpca)

        if engine is not None:
            self.engine = engine
            self._models_loaded = True
        else:
            self._init_engine()

    def _init_engine(self) -> None:
        try:
            if self.cfg.models_path.exists():
                prior, onset = load_models(self.cfg.models_path)
                self.engine = CutEngine(prior_model=prior, onset_model=onset, block=self.cfg.block)
                self._models_loaded = True
                logger.info("Loaded models from %s (block=%d)", self.cfg.models_path, self.cfg.block)
            else:
                self.engine = CutEngine(block=self.cfg.block)
                self._models_loaded = False
                logger.warning("Models file not found at %s. Running fallback mode.", self.cfg.models_path)
        except (KeyError, ValueError, FileNotFoundError, OSError) as err:
            logger.error("Failed to load models: %s. Using default CutEngine.", err)
            self.engine = CutEngine(block=self.cfg.block)
            self._models_loaded = False

    def process_cycle_inputs(self, inp: CycleInputs) -> CycleLogEntry:
        """Processes one complete billet cycle through Phases 2-5."""
        telemetry = StrokeTelemetry(t_s=inp.t_s, x_mm=inp.x_mm, p_cap_bar=inp.p_cap_bar, p_rod_bar=inp.p_rod_bar)
        telemetry_issues = self.safety_gate.validate_stroke_telemetry(telemetry)
        context_issues = self.safety_gate.validate_process_context(
            inp.alloy_id, inp.billet_temp_C, float(np.nanmean(inp.liner_temps_C))
        )
        all_issues = telemetry_issues + context_issues

        stroke_inputs = _convert_cycle_to_stroke_inputs(inp)
        decision = self.engine.decide(stroke_inputs)

        # Feature extraction, monitor, and fault classification
        feats = self.feature_extractor.extract(inp)
        mon_result = self.monitor.update(inp.cycle_id, feats)
        drift_alarm = bool(mon_result.is_alarm)

        diag = self.fault_classifier.classify(inp.cycle_id, feats)
        fault_class = diag.dominant_fault if diag is not None else "none"

        candidate = _build_candidate(decision, all_issues, fault_class)
        guarded = self.safety_gate.guard_cut_decision(candidate)

        latency_ms = float(decision.compute_ms)
        within_budget = latency_ms <= self.cfg.latency_budget_ms

        if not within_budget:
            logger.warning(
                "Cycle %d exceeded latency budget: %.2f ms > %.2f ms",
                inp.cycle_id,
                latency_ms,
                self.cfg.latency_budget_ms,
            )

        published_tags = dict(guarded.published_tags)
        published_tags["Predictor.CutIntervalLowMm"] = (
            guarded.h_cut_mm - 1.5 if not guarded.is_fallback else guarded.h_cut_mm
        )
        published_tags["Predictor.CutIntervalHighMm"] = (
            guarded.h_cut_mm + 1.5 if not guarded.is_fallback else guarded.h_cut_mm
        )
        published_tags["Predictor.DriftAlarm"] = drift_alarm
        published_tags["Predictor.FaultClass"] = fault_class

        entry = CycleLogEntry(
            cycle_id=inp.cycle_id,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            alloy_id=inp.alloy_id,
            die_id=inp.die_id,
            billet_temp_C=inp.billet_temp_C,
            cut_recommendation_mm=guarded.h_cut_mm,
            interval_low_mm=published_tags["Predictor.CutIntervalLowMm"],
            interval_high_mm=published_tags["Predictor.CutIntervalHighMm"],
            confidence=guarded.confidence,
            is_fallback=guarded.is_fallback,
            fallback_reasons=guarded.fallback_reasons,
            drift_alarm=drift_alarm,
            fault_class=fault_class,
            latency_ms=latency_ms,
            within_budget=within_budget,
            published_tags=published_tags,
        )

        self.history.append(entry)
        self._log_decision(entry)
        return entry

    def _log_decision(self, entry: CycleLogEntry) -> None:
        if self.cfg.enable_json_logging:
            logger.info("DECISION_AUDIT: %s", json.dumps(asdict(entry)))
        else:
            logger.info(
                "Cycle %d: Cut=%.1f mm (conf=%s, fallback=%s, alarm=%s, fault=%s, latency=%.1f ms)",
                entry.cycle_id,
                entry.cut_recommendation_mm,
                entry.confidence,
                entry.is_fallback,
                entry.drift_alarm,
                entry.fault_class,
                entry.latency_ms,
            )

    def replay_recording(self, rec: pd.DataFrame) -> list[CycleLogEntry]:
        """Feeds an OPC-UA recorded dataframe through the streaming assembler."""
        replay(rec, self.assembler)
        entries: list[CycleLogEntry] = []
        for inp in self.assembler.pop_completed():
            entries.append(self.process_cycle_inputs(inp))
        return entries

    def run_mock(self, n_cycles: int = 10, scenario_name: str = "healthy_baseline") -> list[CycleLogEntry]:
        """Executes streaming service in mock mode using digital twin simulator."""
        logger.info("Starting mock simulation run: %d cycles of '%s'", n_cycles, scenario_name)
        sc = Scenario.load(scenario_name).with_cycles(n_cycles)
        entries: list[CycleLogEntry] = []

        from skinflow_discard_optimizer.service.replay import cycle_to_records

        t_sim = 0.0
        for row, stroke in run_scenario(sc, keep_strokes=True):
            rec, t_sim = cycle_to_records(row, stroke, t0=t_sim)
            batch = self.replay_recording(rec)
            entries.extend(batch)

        logger.info("Mock run finished. Processed %d cycles successfully.", len(entries))
        return entries

    def health_check(self) -> ServiceHealth:
        """Evaluates operational health and latency statistics."""
        n_cycles = len(self.history)
        if n_cycles == 0:
            return ServiceHealth(
                status="HEALTHY",
                models_loaded=self._models_loaded,
                cycles_processed=0,
                fallbacks_count=0,
                alarms_count=0,
                mean_latency_ms=0.0,
                p99_latency_ms=0.0,
                max_latency_ms=0.0,
                budget_violations=0,
                uptime_seconds=time.time() - self.start_time,
            )

        latencies = np.array([e.latency_ms for e in self.history])
        fallbacks = sum(1 for e in self.history if e.is_fallback)
        alarms = sum(1 for e in self.history if e.drift_alarm)
        violations = sum(1 for e in self.history if not e.within_budget)

        p99 = float(np.percentile(latencies, 99))
        status = "DEGRADED" if (violations > 0 or not self._models_loaded) else "HEALTHY"

        return ServiceHealth(
            status=status,
            models_loaded=self._models_loaded,
            cycles_processed=n_cycles,
            fallbacks_count=fallbacks,
            alarms_count=alarms,
            mean_latency_ms=float(np.mean(latencies)),
            p99_latency_ms=p99,
            max_latency_ms=float(np.max(latencies)),
            budget_violations=violations,
            uptime_seconds=time.time() - self.start_time,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SKDO Streaming Optimization Service (Task 6.1)")
    parser.add_argument("--mock", action="store_true", help="Run with simulated digital twin stream")
    parser.add_argument("--mock-cycles", type=int, default=10, help="Number of simulated cycles in mock mode")
    parser.add_argument("--scenario", type=str, default="healthy_baseline", help="Simulator scenario name")
    parser.add_argument("--replay", type=Path, default=None, help="Path to parquet recording to replay")
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_LATENCY_BUDGET_MS, help="Latency budget in ms")
    parser.add_argument("--log-json", action="store_true", help="Enable structured JSON log format")
    parser.add_argument("--output", type=Path, default=None, help="Save decisions to JSON file")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = ServiceConfig(
        latency_budget_ms=args.budget_ms,
        mock_cycles=args.mock_cycles,
        enable_json_logging=args.log_json,
    )
    svc = StreamingOptimizerService(config=cfg)

    if args.mock:
        entries = svc.run_mock(n_cycles=args.mock_cycles, scenario_name=args.scenario)
    elif args.replay and args.replay.exists():
        df_rec = pd.read_parquet(args.replay)
        entries = svc.replay_recording(df_rec)
    else:
        logger.error("No input stream specified. Pass --mock or --replay <path>.")
        return

    health = svc.health_check()
    logger.info("Service Health Summary: %s", json.dumps(health.to_dict(), indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in entries], f, indent=2)
        logger.info("Saved %d decision logs to %s", len(entries), args.output)


if __name__ == "__main__":
    main()
