"""Live side of the feature layer: assemble tag samples into ``CycleInputs`` (Task 2.3).

The assembler is source-agnostic. It is fed ``(signal, t, value)`` samples, one
at a time or in blocks, named as in ``contracts/signals.yaml``, from an OPC-UA
subscription or a replayed recording. It emits one ``CycleInputs`` per cycle when
the phase leaves ``extrusion``. Out-of-range samples are dropped and counted,
using the contract's validator.

Aggregation rules (per cycle):

* stroke arrays: ram position, cap and rod pressure recorded while phase == extrusion
* liner temps, oil temp: mean over the extrusion phase
* supply pressure: minimum over the extrusion phase
* pump energy: time integral of pump power over the extrusion phase
* phase durations: time from each phase-change event to the next
* billet temp, billet length, alloy, die: last event value before extrusion

A dead-cycle phase is attributed to the cycle it occurs in. The shear stroke that
cuts billet n's discard runs at the start of cycle n+1 (DCTO phase order).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from skinflow_discard_optimizer.contracts.validator import check_array
from skinflow_discard_optimizer.core.features import CycleInputs

STROKE_SIGNALS = ("ram_position", "ram_cap_pressure", "ram_rod_pressure")
MEAN_SIGNALS = ("container_liner_temp_1", "container_liner_temp_2", "container_liner_temp_3",
                "container_liner_temp_4", "oil_temp")
EVENT_SIGNALS = ("billet_temp", "billet_length", "alloy_id", "die_id", "cycle_id")


@dataclass
class _Buffer:
    series: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    times: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))

    def add(self, sig: str, t: np.ndarray, v: np.ndarray) -> None:
        self.times[sig].append(t)
        self.series[sig].append(v)

    def get(self, sig: str) -> tuple[np.ndarray, np.ndarray]:
        if sig not in self.series:
            return np.array([]), np.array([])
        return np.concatenate(self.times[sig]), np.concatenate(self.series[sig])


class CycleAssembler:
    def __init__(self):
        self.phase: str = "idle"
        self.phase_start: float | None = None
        self.phase_durations: dict[str, float] = {}
        self.events: dict[str, object] = {}
        self.buf = _Buffer()
        self.rejected: dict[str, int] = defaultdict(int)
        self.completed: list[CycleInputs] = []

    # ------------------------------------------------------------------ input
    def push(self, signal: str, t: float, value) -> None:
        if signal == "cycle_phase":
            self._on_phase(str(value), float(t))
        elif signal in EVENT_SIGNALS:
            self.events[signal] = value
        else:
            self.push_block(signal, np.array([t], float), np.array([value], float))

    def push_block(self, signal: str, t: np.ndarray, v: np.ndarray) -> None:
        """Numeric samples of one signal. Only samples during extrusion are kept."""
        if self.phase != "extrusion":
            return
        t = np.asarray(t, float)
        v = np.asarray(v, float)
        ok = check_array(signal, v)
        if not ok.all():
            self.rejected[signal] += int((~ok).sum())
            t, v = t[ok], v[ok]
        self.buf.add(signal, t, v)

    def _on_phase(self, phase: str, t: float) -> None:
        if self.phase_start is not None and self.phase not in ("idle", "extrusion"):
            self.phase_durations[self.phase] = t - self.phase_start
        if self.phase == "extrusion" and phase != "extrusion":
            self._emit()
        self.phase, self.phase_start = phase, t

    # ------------------------------------------------------------------ output
    def _emit(self) -> None:
        tx, x = self.buf.get("ram_position")
        tc, pc = self.buf.get("ram_cap_pressure")
        tr, pr = self.buf.get("ram_rod_pressure")
        if len(tx) < 100:
            self._reset()
            return
        # align all stroke signals on the position timestamps
        pc_i = np.interp(tx, tc, pc) if len(tc) else np.full_like(tx, np.nan)
        pr_i = np.interp(tx, tr, pr) if len(tr) else np.full_like(tx, np.nan)
        means = {s: float(np.mean(self.buf.get(s)[1])) if len(self.buf.get(s)[1]) else float("nan")
                 for s in MEAN_SIGNALS}
        ts, sp = self.buf.get("pump_supply_pressure")
        tp, pw = self.buf.get("pump_power")
        energy = float(np.sum(0.5 * (pw[1:] + pw[:-1]) * np.diff(tp)) / 3600.0) if len(tp) > 1 else float("nan")
        e = self.events
        self.completed.append(CycleInputs(
            cycle_id=int(e.get("cycle_id", len(self.completed))),
            alloy_id=str(e.get("alloy_id", "")),
            die_id=str(e.get("die_id", "")),
            t_s=tx - tx[0],
            x_mm=x,
            p_cap_bar=pc_i,
            p_rod_bar=pr_i,
            billet_temp_C=float(e.get("billet_temp", np.nan)),
            billet_length_mm=float(e.get("billet_length", np.nan)),
            liner_temps_C=tuple(means[f"container_liner_temp_{i}"] for i in range(1, 5)),  # type: ignore[arg-type]
            oil_temp_C=means["oil_temp"],
            supply_pressure_min_bar=float(np.min(sp)) if len(sp) else float("nan"),
            pump_energy_kwh=energy,
            phase_durations_s=dict(self.phase_durations),
        ))
        self._reset()

    def _reset(self) -> None:
        self.buf = _Buffer()
        self.phase_durations = {}

    def pop_completed(self) -> list[CycleInputs]:
        out, self.completed = self.completed, []
        return out
