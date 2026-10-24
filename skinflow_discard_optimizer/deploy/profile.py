"""Turn host facts into run parameters. One tier per class of machine, plus explicit overrides.

The tiers trade fitting time and model fidelity for what the host can carry; the decision
algorithm itself and the safety gate are identical in every tier.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace

from skinflow_discard_optimizer.deploy.probe import HostFacts
from skinflow_discard_optimizer.paths import default_workers


@dataclass(frozen=True)
class DeployProfile:
    tier: str                 # edge | standard | full
    workers: int              # process-pool size for dataset/feature/fit stages
    device: str               # cpu | cuda (fit_decision and GPU inference)
    dataset_scale: float      # fraction of each scenario's cycles to simulate
    feature_stride: int       # build_features: use every n-th cycle
    fpca_n: int
    fpca_n_val: int
    ukf_n_tune: int
    ukf_n_eval: int
    n_train: int
    n_cal: int
    fit_stride: int
    latency_budget_ms: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


# (dataset_scale, feature_stride, fpca_n, fpca_n_val, ukf_tune, ukf_eval, n_train, n_cal, fit_stride)
_TIERS = {
    "edge":     (0.10, 10, 400, 150, 4, 20, 1500, 200, 10),
    "standard": (0.25, 4, 800, 250, 6, 40, 3000, 500, 6),
    "full":     (1.00, 1, 1500, 400, 8, 60, 6000, 1500, 5),
}


def choose_tier(h: HostFacts) -> tuple[str, str]:
    if h.cpu_count <= 2 or h.ram_gb < 4:
        return "edge", f"{h.cpu_count} usable core(s), {h.ram_gb} GB RAM"
    if h.cpu_count >= 8 and h.ram_gb >= 16:
        return "full", f"{h.cpu_count} cores, {h.ram_gb} GB RAM"
    return "standard", f"{h.cpu_count} cores, {h.ram_gb} GB RAM"


def build_profile(h: HostFacts, env: dict[str, str] | None = None) -> DeployProfile:
    """Auto profile for ``h``; ``SKDO_*`` environment variables override individual fields."""
    env = dict(os.environ if env is None else env)
    tier, why = choose_tier(h)
    if env.get("SKDO_TIER") in _TIERS:
        tier, why = env["SKDO_TIER"], "SKDO_TIER override"
    scale, fstride, fn, fnv, kt, ke, ntr, nca, fs = _TIERS[tier]

    reasons = [f"tier={tier} ({why})"]
    # Each worker loads SciPy (~150 MB); keep two spare cores for the service itself on small hosts.
    workers = min(default_workers(), max(1, int(h.ram_gb // 1.0) - 1)) if h.ram_gb else default_workers()
    workers = max(1, min(workers, h.cpu_count - 1 if h.cpu_count > 1 else 1))

    device = "cuda" if (h.torch_cuda and tier != "edge") else "cpu"
    reasons.append("GPU inference/fit enabled" if device == "cuda" else
                   ("GPU present but torch has no CUDA; install the gpu extra" if h.gpu_names else "CPU only"))

    prof = DeployProfile(tier, workers, device, scale, fstride, fn, fnv, kt, ke, ntr, nca, fs, 200.0,
                         tuple(reasons))

    overrides = {
        "SKDO_WORKERS": ("workers", int), "SKDO_DEVICE": ("device", str),
        "SKDO_DATASET_SCALE": ("dataset_scale", float), "SKDO_LATENCY_BUDGET_MS": ("latency_budget_ms", float),
    }
    for key, (fld, cast) in overrides.items():
        if env.get(key):
            prof = replace(prof, **{fld: cast(env[key])}, reasons=prof.reasons + (f"{key} override",))
    if prof.device not in ("cpu", "cuda"):
        raise ValueError(f"SKDO_DEVICE must be cpu or cuda, got {prof.device!r}")
    return prof
