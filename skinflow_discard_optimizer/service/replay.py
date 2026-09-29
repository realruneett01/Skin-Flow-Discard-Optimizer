"""Tag-level recordings: write simulated cycles as an OPC-UA-style tag stream and replay them.

A recording is a long table ``(t, signal, value, text)``, one row per tag sample,
with signal names from ``contracts/signals.yaml``. That is the shape a historian
export or an OPC-UA subscription log has. ``replay`` feeds a recording into any
consumer with the ``push``/``push_block`` interface (``CycleAssembler``) in time
order, which is how Task 2.3 checks that live and batch features match.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from skinflow_discard_optimizer.sim.cycle import DCTO_PHASES
from skinflow_discard_optimizer.sim.force_model import StrokeData

TEXT_SIGNALS = {"cycle_phase", "alloy_id", "die_id"}


class TagConsumer(Protocol):
    def push(self, signal: str, t: float, value) -> None: ...
    def push_block(self, signal: str, t: np.ndarray, v: np.ndarray) -> None: ...


def cycle_to_records(row: dict, stroke: StrokeData, t0: float = 0.0) -> tuple[pd.DataFrame, float]:
    """Tag samples for one simulated cycle starting at ``t0``. Returns ``(records, t_end)``."""
    num: list[tuple[np.ndarray, str, np.ndarray]] = []
    txt: list[tuple[float, str, object]] = []
    t = t0
    for p in DCTO_PHASES:
        txt.append((t, "cycle_phase", p))
        if p == "billet_load":
            txt += [(t, "cycle_id", int(row["cycle"])), (t, "billet_temp", float(row["billet_temp_C"])),
                    (t, "billet_length", float(row["billet_length_mm"])), (t, "alloy_id", row["alloy_id"]),
                    (t, "die_id", row["die_id"])]
        t += float(row[f"phase_{p}_s"])
    ts = t
    txt.append((ts, "cycle_phase", "extrusion"))
    dur = float(stroke.t_s[-1])
    num += [(ts + stroke.t_s, "ram_position", stroke.x_mm),
            (ts + stroke.t_s, "ram_cap_pressure", stroke.p_cap_bar),
            (ts + stroke.t_s, "ram_rod_pressure", stroke.p_rod_bar)]
    t1 = ts + np.arange(0.0, dur, 1.0)
    for i in range(1, 5):
        num.append((t1, f"container_liner_temp_{i}", np.full(t1.size, float(row[f"container_liner_temp_{i}"]))))
    num.append((t1, "oil_temp", np.full(t1.size, float(row["oil_temp_C"]))))
    t100 = ts + np.arange(0.0, dur, 0.01)
    sp = np.full(t100.size, float(row["pump_supply_pressure_min_bar"]) + 1.0)
    sp[int(0.9 * sp.size)] = float(row["pump_supply_pressure_min_bar"])   # the recorded minimum
    num.append((t100, "pump_supply_pressure", sp))
    t10 = ts + np.linspace(0.0, dur, max(int(dur * 10), 2))
    kw = float(row["pump_energy_kwh"]) * 3600.0 / (t10[-1] - t10[0])      # integrates to the recorded energy
    num.append((t10, "pump_power", np.full(t10.size, kw)))
    t_end = ts + dur + 0.001
    txt.append((t_end, "cycle_phase", "dwell"))

    frames = [pd.DataFrame({"t": tt, "signal": s, "value": vv, "text": None}) for tt, s, vv in num]
    frames.append(pd.DataFrame(
        {"t": [a for a, _, _ in txt], "signal": [b for _, b, _ in txt],
         "value": [float(c) if b not in TEXT_SIGNALS else np.nan for _, b, c in txt],
         "text": [str(c) if b in TEXT_SIGNALS else None for _, b, c in txt]}))
    return _time_order(pd.concat(frames, ignore_index=True)), t_end


def _time_order(rec: pd.DataFrame) -> pd.DataFrame:
    """Sort by time; at equal times, events (phase changes, billet data) come before samples."""
    is_event = rec["text"].notna() | rec["signal"].isin(["cycle_id", "billet_temp", "billet_length"])
    return (rec.assign(_ev=~is_event).sort_values(["t", "_ev"], kind="stable")
            .drop(columns="_ev").reset_index(drop=True))


def build_recording(rows: list[dict], strokes: list[StrokeData]) -> pd.DataFrame:
    t, parts = 0.0, []
    for r, s in zip(rows, strokes):
        rec, t = cycle_to_records(r, s, t)
        parts.append(rec)
    return pd.concat(parts, ignore_index=True)


def save_recording(rec: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(path, index=False)


def load_recording(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def replay(rec: pd.DataFrame, consumer: TagConsumer) -> None:
    """Feed a recording in time order. Numeric samples between two text events go in blocks."""
    rec = _time_order(rec)
    is_text = rec["text"].notna().to_numpy() | rec["signal"].isin(["cycle_id", "billet_temp", "billet_length"]).to_numpy()
    idx = np.flatnonzero(is_text)
    text_rows = set(idx.tolist())
    bounds = [0, *idx.tolist(), len(rec)]
    t_arr, sig_arr = rec["t"].to_numpy(), rec["signal"].to_numpy()
    val_arr, txt_arr = rec["value"].to_numpy(), rec["text"].to_numpy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        start = a
        if a in text_rows:
            sig = sig_arr[a]
            text = txt_arr[a]
            consumer.push(sig, float(t_arr[a]), text if isinstance(text, str) else val_arr[a])
            start = a + 1
        if b > start:
            seg = slice(start, b)
            sigs = sig_arr[seg]
            for s in pd.unique(sigs):
                m = sigs == s
                consumer.push_block(s, t_arr[seg][m], val_arr[seg][m])
