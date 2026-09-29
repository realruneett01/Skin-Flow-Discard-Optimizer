"""Batch feature build: every dataset cycle -> feature row -> time-series store (Task 2.3).

    python -m skinflow_discard_optimizer.core.build_features [--stride 1] [--workers N]

Regenerates each cycle's 1 kHz stroke from its seed and runs the same
``FeatureExtractor`` the live service uses. Ground-truth columns (``h_crit_mm``,
fault labels, split, ...) are joined onto the feature rows under a ``y_`` prefix
so they can never be mistaken for inputs.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from skinflow_discard_optimizer.core.features import FeatureExtractor, inputs_from_row
from skinflow_discard_optimizer.core.fit_fpca import FPCA_PATH
from skinflow_discard_optimizer.core.functional import FPCA
from skinflow_discard_optimizer.core.store import ParquetStore
from skinflow_discard_optimizer.paths import default_workers
from skinflow_discard_optimizer.sim.build_dataset import load_cycles
from skinflow_discard_optimizer.sim.cycle import LABEL_COLUMNS, regenerate_stroke
from skinflow_discard_optimizer.sim.force_model import Press

TRUTH_COLUMNS = ("h_crit_mm", "h_crit_mean_mm", "oracle_cut_mm", "static_cut_mm", "static_defect_prob",
                 "static_cost_eur", "oracle_cost_eur", "spec_h_onset_mm", "spec_mu", "spec_F_tool_N",
                 "spec_sigma_scale", "spec_lam_mm", "true_liner_scale_mm", "true_die_wear", "true_billet_temp_C",
                 "true_liner_temp_C", "true_mu_base", "true_dT_K", "eff_cap_gain", "eff_encoder_offset_mm",
                 "eff_supply_sag_bar", "eff_flash_amp_frac", "split", "scenario", *LABEL_COLUMNS)

_fx: FeatureExtractor | None = None


def _init() -> None:
    global _fx
    _fx = FeatureExtractor(Press.load(), FPCA.load(FPCA_PATH))


def _one(row: dict) -> dict:
    assert _fx is not None
    f = _fx.extract(inputs_from_row(row, regenerate_stroke(row, _fx.press)))
    f.update({f"y_{c}": row[c] for c in TRUTH_COLUMNS if c in row})
    return f


def build(stride: int = 1, workers: int | None = None, scenarios: list[str] | None = None,
          store: ParquetStore | None = None) -> dict:
    df = load_cycles()
    if scenarios:
        df = df[df.scenario.isin(scenarios)]
    df = df[df.cycle % stride == 0]
    store = store or ParquetStore()
    workers = workers or default_workers()
    t0 = time.perf_counter()
    counts = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        for name, sub in df.groupby("scenario", sort=True):
            rows = list(ex.map(_one, sub.to_dict("records"), chunksize=32))
            store.write(name, pd.DataFrame(rows))
            counts[name] = len(rows)
            print(f"{name}: {len(rows)} cycles ({time.perf_counter() - t0:.0f}s)", flush=True)
    return {"rows": int(sum(counts.values())), "per_scenario": counts, "seconds": round(time.perf_counter() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="use every n-th cycle")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--scenarios", nargs="*")
    a = ap.parse_args()
    print(build(a.stride, a.workers, a.scenarios))
