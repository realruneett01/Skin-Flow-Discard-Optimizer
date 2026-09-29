"""Time-series feature store (Task 2.3).

Phase 0 did not name a time-series database, so this is a small Parquet-backed
store behind an interface an InfluxDB or TimescaleDB adapter could implement
(docs/assumptions.md A-17). Rows are keyed by ``(stream, cycle)``, where a stream
is one press or one simulated scenario, and kept in cycle order per stream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from skinflow_discard_optimizer.paths import DATA_DIR


class TimeSeriesStore(Protocol):
    def write(self, stream: str, rows: pd.DataFrame) -> None: ...
    def read(self, stream: str, columns: Iterable[str] | None = None,
             cycle_from: int | None = None, cycle_to: int | None = None) -> pd.DataFrame: ...
    def streams(self) -> list[str]: ...


class ParquetStore:
    """One Parquet file per stream. ``write`` upserts by cycle, so re-runs are idempotent."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else DATA_DIR / "features"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, stream: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stream)
        return self.root / f"{safe}.parquet"

    def write(self, stream: str, rows: pd.DataFrame) -> None:
        if "cycle" not in rows.columns:
            raise ValueError("feature rows need a 'cycle' column")
        p = self._path(stream)
        new = rows.assign(stream=stream)
        if p.exists():
            old = pd.read_parquet(p)
            old = old[~old["cycle"].isin(new["cycle"])]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values("cycle", kind="stable").reset_index(drop=True).to_parquet(p, index=False)

    def read(self, stream: str, columns: Iterable[str] | None = None,
             cycle_from: int | None = None, cycle_to: int | None = None) -> pd.DataFrame:
        p = self._path(stream)
        if not p.exists():
            return pd.DataFrame()
        cols = None if columns is None else list(dict.fromkeys(["cycle", *columns]))
        filters = []
        if cycle_from is not None:
            filters.append(("cycle", ">=", cycle_from))
        if cycle_to is not None:
            filters.append(("cycle", "<", cycle_to))
        return pd.read_parquet(p, columns=cols, filters=filters or None)

    def streams(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.parquet"))

    def read_all(self, columns: Iterable[str] | None = None) -> pd.DataFrame:
        frames = [self.read(s, None if columns is None else [*columns, "stream"]) for s in self.streams()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
