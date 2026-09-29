"""Build the labelled simulation dataset from the scenario library (Task 1.4).

    python -m skinflow_discard_optimizer.sim.build_dataset            # full, ~200k cycles
    python -m skinflow_discard_optimizer.sim.build_dataset --scale 0.05 --out data/small

Output directory:

* ``cycles.parquet``: one row per cycle with per-cycle signals, every ground-truth
  label, the stroke parameters and ``stroke_seed``, plus a ``split`` column.
  ``sim.cycle.regenerate_stroke(row)`` rebuilds any cycle's exact 1 kHz stroke.
* ``manifest.json``: scenarios, seeds, row counts, split sizes, a fingerprint of
  every config file, and a content hash of the table.

Splits are by scenario and by time, never by random cycle (plan Task 1.4):

* Two whole scenarios are held out as ``test_scenario`` (a combination of faults
  and an operating event the models never train on).
* Every other scenario is cut in time order into train (first 60%), val (next
  20%) and test (last 20%), with an embargo of ``EMBARGO`` cycles marked ``gap``
  at each boundary. Latent states are autocorrelated over about 100 cycles, so
  neighbouring cycles would otherwise leak across the split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from skinflow_discard_optimizer.paths import CONFIG_DIR, DATA_DIR
from skinflow_discard_optimizer.sim.scenario import SCENARIO_DIR, Scenario, list_scenarios, run_scenario

HELD_OUT_SCENARIOS = ("combined_wear_and_scale", "die_change")
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
EMBARGO = 200


def assign_split(scenario: str, cycle: np.ndarray, n_cycles: int) -> np.ndarray:
    """Split label for each cycle of one scenario (see module docstring)."""
    cycle = np.asarray(cycle)
    if scenario in HELD_OUT_SCENARIOS:
        return np.full(cycle.shape, "test_scenario", dtype=object)
    b1 = int(n_cycles * SPLIT_FRACTIONS[0])
    b2 = int(n_cycles * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))
    out = np.where(cycle < b1, "train", np.where(cycle < b2, "val", "test")).astype(object)
    half = EMBARGO // 2
    near = ((cycle >= b1 - half) & (cycle < b1 + half)) | ((cycle >= b2 - half) & (cycle < b2 + half))
    out[near] = "gap"
    return out


def _run_one(args: tuple[str, float]) -> pd.DataFrame:
    name, scale = args
    scn = Scenario.load(name)
    n = max(int(round(scn.n_cycles * scale)), 50)
    if scale != 1.0:
        scn = _scaled(scn, scale, n)
    df = pd.DataFrame([row for row, _ in run_scenario(scn)])
    df["split"] = assign_split(name, df["cycle"].to_numpy(), n)
    df["scenario_n_cycles"] = n
    return df


def _scaled(scn: Scenario, scale: float, n: int) -> Scenario:
    """Shrink a scenario in time, keeping onsets, ramps and events at the same relative position."""
    from dataclasses import replace

    faults = tuple(replace(f, onset_cycle=int(f.onset_cycle * scale),
                           ramp_cycles=max(int(f.ramp_cycles * scale), 1),
                           end_cycle=None if f.end_cycle is None else int(f.end_cycle * scale))
                   for f in scn.faults)
    events = tuple(replace(e, cycle=int(e.cycle * scale)) for e in scn.events)
    return replace(scn, n_cycles=n, faults=faults, events=events)


def config_fingerprint() -> dict[str, str]:
    files = sorted(CONFIG_DIR.glob("*.yaml")) + sorted(SCENARIO_DIR.glob("*.yaml"))
    return {f"{p.parent.name}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files}


def table_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def build(out_dir: Path, scale: float = 1.0, scenarios: list[str] | None = None,
          workers: int | None = None) -> dict:
    names = scenarios or list_scenarios()
    t0 = time.perf_counter()
    if workers == 1:
        frames = [_run_one((n, scale)) for n in names]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            frames = list(ex.map(_run_one, [(n, scale) for n in names]))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["scenario", "cycle"], kind="stable").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "cycles.parquet", index=False)
    manifest = {
        "rows": int(len(df)),
        "scale": scale,
        "scenarios": {n: {"seed": Scenario.load(n).seed, "rows": int((df["scenario"] == n).sum())}
                      for n in names},
        "held_out_scenarios": list(HELD_OUT_SCENARIOS),
        "split_fractions": list(SPLIT_FRACTIONS),
        "embargo_cycles": EMBARGO,
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().items()},
        "config_fingerprint": config_fingerprint(),
        "table_hash": table_hash(df),
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_cycles(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet((path or DATA_DIR / "dataset") / "cycles.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the simulation dataset.")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "dataset")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of each scenario's cycles")
    ap.add_argument("--scenarios", nargs="*", help="subset of scenario names")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    m = build(a.out, a.scale, a.scenarios, a.workers)
    print(json.dumps({k: m[k] for k in ("rows", "split_counts", "table_hash", "build_seconds")}, indent=2))


if __name__ == "__main__":
    main()
