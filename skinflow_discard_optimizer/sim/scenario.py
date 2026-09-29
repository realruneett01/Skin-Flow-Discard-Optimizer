"""Scenario files: a named, seeded sequence of cycles with faults and events (Tasks 1.3, 1.4).

Example (``sim/scenarios/die_wear.yaml``)::

    name: die_wear
    seed: 1002
    n_cycles: 8000
    context: {alloy: AA6063, die_id: D-101, extrusion_ratio: 40}
    faults:
      - {kind: die_wear, onset_cycle: 2000, severity: 0.5, ramp: linear, ramp_cycles: 4000}

Events change the operating context at a given cycle::

    events:
      - {cycle: 3000, type: die_change, die_id: D-202, extrusion_ratio: 55, die_wear: 0.02}
      - {cycle: 5000, type: alloy_change, alloy: AA6082}
      - {cycle: 0, type: cold_start, liner_drop_K: 60, billet_drop_K: 12, tau_cycles: 40}
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from skinflow_discard_optimizer.sim.cycle import CycleContext, simulate_cycle
from skinflow_discard_optimizer.sim.defect_model import DefectModel, PressState
from skinflow_discard_optimizer.sim.faults import FaultConfig, FaultSpec, PressProcess
from skinflow_discard_optimizer.sim.force_model import Alloy, Press, StrokeData

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
EVENT_TYPES = ("die_change", "alloy_change", "cold_start")


@dataclass(frozen=True)
class Event:
    cycle: int
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type!r}; expected one of {EVENT_TYPES}")


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    n_cycles: int
    context: CycleContext = CycleContext()
    baseline: dict[str, float] = field(default_factory=dict)
    faults: tuple[FaultSpec, ...] = ()
    events: tuple[Event, ...] = ()
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        events = []
        for e in d.get("events", []) or []:
            e = dict(e)
            events.append(Event(cycle=int(e.pop("cycle")), type=e.pop("type"), params=e))
        return cls(
            name=d["name"],
            seed=int(d["seed"]),
            n_cycles=int(d["n_cycles"]),
            context=CycleContext(**(d.get("context") or {})),
            baseline=dict(d.get("baseline") or {}),
            faults=tuple(FaultSpec(**f) for f in d.get("faults", []) or []),
            events=tuple(sorted(events, key=lambda e: e.cycle)),
            description=d.get("description", ""),
        )

    @classmethod
    def load(cls, path_or_name: str | Path) -> "Scenario":
        p = Path(path_or_name)
        if not p.suffix:
            p = SCENARIO_DIR / f"{path_or_name}.yaml"
        with open(p, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def with_cycles(self, n: int) -> "Scenario":
        return replace(self, n_cycles=n)


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def stroke_seed(scenario_seed: int, cycle: int) -> int:
    """Independent, reproducible seed for one cycle's sensor noise."""
    return int(np.random.SeedSequence([scenario_seed, cycle]).generate_state(1)[0])


def run_scenario(scn: Scenario, *, keep_strokes: bool = False, start: int = 0,
                 stop: int | None = None) -> Iterator[tuple[dict, StrokeData | None]]:
    """Yield ``(row, stroke)`` for each cycle. ``stroke`` is None unless ``keep_strokes``.

    The whole trajectory is always simulated from cycle 0 (the latent state is a
    Markov chain). ``start``/``stop`` only limit what is yielded.
    """
    press = Press.load()
    alloy = Alloy.from_config(scn.context.alloy)
    model = DefectModel(press=press, alloy=alloy)
    base = PressState(
        billet_temp_C=press.billet_temp_C, liner_temp_C=press.liner_temp_C,
        mu_base=alloy.mu_nominal, taper_K=press.taper_K,
        ram_speed_mm_s=press.ram_speed_mm_s, billet_length_mm=press.billet_length_mm,
    )
    base = replace(base, **scn.baseline)
    ss = np.random.SeedSequence(scn.seed)
    proc_seed, truth_seed = ss.spawn(2)
    truth_rng = np.random.default_rng(truth_seed)
    process = PressProcess(base, list(scn.faults), FaultConfig.load(press.supply_pressure_nominal_bar),
                           np.random.default_rng(proc_seed))
    ctx = scn.context
    cold: tuple[float, float, float, int] | None = None   # (liner_drop, billet_drop, tau, start)
    events = list(scn.events)
    stop = scn.n_cycles if stop is None else min(stop, scn.n_cycles)

    for n in range(stop):
        while events and events[0].cycle <= n:
            ev = events.pop(0)
            if ev.type == "die_change":
                ctx = replace(ctx, die_id=ev.params.get("die_id", ctx.die_id),
                              extrusion_ratio=float(ev.params.get("extrusion_ratio", ctx.extrusion_ratio)))
                process.set_baseline(die_wear=float(ev.params.get("die_wear", 0.02)))
                process.reset_ou("die_wear")
                # a new die restarts wear-type faults
                process.faults = [f for f in process.faults if f.kind != "die_wear"
                                  or f.onset_cycle > n]
            elif ev.type == "alloy_change":
                alloy = Alloy.from_config(ev.params["alloy"])
                model = model.with_alloy(alloy)
                ctx = replace(ctx, alloy=alloy.name)
                process.set_baseline(mu_base=alloy.mu_nominal)
            elif ev.type == "cold_start":
                cold = (float(ev.params.get("liner_drop_K", 60.0)), float(ev.params.get("billet_drop_K", 10.0)),
                        float(ev.params.get("tau_cycles", 40.0)), n)
        extra = None
        if cold is not None:
            decay = np.exp(-(n - cold[3]) / cold[2])
            extra = {"liner_temp_C": -cold[0] * decay, "billet_temp_C": -cold[1] * decay}
        cond = process.step(n, extra_offsets=extra)
        # truth_rng must advance every cycle, even ones not yielded, to keep rows identical
        row, stroke = simulate_cycle(cond, model, ctx, stroke_seed(scn.seed, n), truth_rng,
                                     full_stroke=keep_strokes and n >= start)
        if n >= start:
            row["scenario"] = scn.name
            row["event_cold_start"] = float(extra["liner_temp_C"]) if extra else 0.0
            row["extrusion_ratio"] = ctx.extrusion_ratio
            yield row, stroke
