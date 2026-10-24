"""First-run model build, resumable and invalidated by config changes.

Runs the offline pipeline (dataset -> FPCA -> features -> UKF noise -> decision models -> monitor)
sized by the host's ``DeployProfile``. Each stage runs in its own interpreter so a memory spike
cannot take down the whole bootstrap, and is skipped when its outputs exist and its inputs
(config/*.yaml, scenarios, stage parameters, upstream stages) are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from skinflow_discard_optimizer.deploy.profile import DeployProfile
from skinflow_discard_optimizer.paths import ARTIFACTS_DIR, CONFIG_DIR, DATA_DIR, REPO_ROOT

PKG = "skinflow_discard_optimizer"
STATE_FILE = ARTIFACTS_DIR / "bootstrap_state.json"
SCENARIO_DIR = REPO_ROOT / PKG / "sim" / "scenarios"

# Files the running service needs. Anything missing here means the service would fall back.
REQUIRED_ARTIFACTS = ("fpca.npz", "hcrit_models.npz", "ukf_params.json", "monitor_config.npz")


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    args: tuple[str, ...]
    outputs: tuple[Path, ...]


def stages(p: DeployProfile) -> list[Stage]:
    w = str(p.workers)
    return [
        Stage("dataset", f"{PKG}.sim.build_dataset", ("--scale", str(p.dataset_scale), "--workers", w),
              (DATA_DIR / "dataset" / "cycles.parquet",)),
        Stage("fpca", f"{PKG}.core.fit_fpca", ("--n", str(p.fpca_n), "--n-val", str(p.fpca_n_val)),
              (ARTIFACTS_DIR / "fpca.npz",)),
        Stage("features", f"{PKG}.core.build_features", ("--stride", str(p.feature_stride), "--workers", w),
              (DATA_DIR / "features",)),
        Stage("ukf", f"{PKG}.core.observer.tune_ukf", ("--n-tune", str(p.ukf_n_tune), "--n-eval", str(p.ukf_n_eval)),
              (ARTIFACTS_DIR / "ukf_params.json",)),
        Stage("decision", f"{PKG}.core.estimator.fit_decision",
              ("--n-train", str(p.n_train), "--n-cal", str(p.n_cal), "--stride", str(p.fit_stride),
               "--device", p.device),
              (ARTIFACTS_DIR / "hcrit_models.npz", ARTIFACTS_DIR / "conformal_scores.npy")),
        Stage("monitor", f"{PKG}.aware.eval_monitor", ("--fit-only",), (ARTIFACTS_DIR / "monitor_config.npz",)),
    ]


def config_fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(CONFIG_DIR.glob("*.yaml")) + sorted(SCENARIO_DIR.glob("*.yaml")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


# Execution-only flags: they change how fast a stage runs, not what it produces, so they stay out of the
# cache key (otherwise models built on a big host would look stale on a smaller one).
_VOLATILE_FLAGS = ("--workers", "--device")


def _key_args(args: tuple[str, ...]) -> list[str]:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in _VOLATILE_FLAGS:
            skip = True
        else:
            out.append(a)
    return out


def stage_keys(p: DeployProfile) -> dict[str, str]:
    """Per-stage cache key; chained so changing an upstream stage invalidates everything after it."""
    key = config_fingerprint()
    out = {}
    for s in stages(p):
        key = hashlib.sha256((key + s.module + " ".join(_key_args(s.args))).encode()).hexdigest()
        out[s.name] = key
    return out


def _load_state() -> dict[str, str]:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _exists(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir()) if path.suffix == "" else path.exists()


def status(p: DeployProfile) -> dict[str, str]:
    """``done`` / ``stale`` / ``missing`` per stage."""
    state, keys = _load_state(), stage_keys(p)
    res = {}
    for s in stages(p):
        if not all(_exists(o) for o in s.outputs):
            res[s.name] = "missing"
        else:
            res[s.name] = "done" if state.get(s.name) == keys[s.name] else "stale"
    return res


def run_stage(s: Stage, log=print) -> None:
    cmd = [sys.executable, "-m", s.module, *s.args]
    log(f"[bootstrap] {s.name}: {' '.join(cmd[1:])}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
        raise RuntimeError(f"stage '{s.name}' failed (exit {proc.returncode}):\n{tail}")
    log(f"[bootstrap] {s.name}: ok in {time.perf_counter() - t0:.0f}s")


def bootstrap(p: DeployProfile, force: bool = False, log=print) -> dict[str, str]:
    """Build whatever is missing or stale. Returns the final per-stage status."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    keys = stage_keys(p)
    state = {} if force else _load_state()
    dirty = force
    for s in stages(p):
        current = state.get(s.name) == keys[s.name] and all(_exists(o) for o in s.outputs)
        if current and not dirty:
            log(f"[bootstrap] {s.name}: up to date")
            continue
        dirty = True  # everything downstream of a rebuilt stage is rebuilt too
        run_stage(s, log)
        state[s.name] = keys[s.name]
        STATE_FILE.write_text(json.dumps(state, indent=2))
    return status(p)


def smoke_test(cycles: int = 3) -> dict:
    """Push a few simulated cycles through the real service; proves models load and the safety gate runs."""
    from skinflow_discard_optimizer.service.run import StreamingOptimizerService

    svc = StreamingOptimizerService()
    entries = svc.run_mock(n_cycles=cycles)
    health = svc.health_check().to_dict()
    health["cuts_mm"] = [round(e.cut_recommendation_mm, 1) for e in entries]
    return health
