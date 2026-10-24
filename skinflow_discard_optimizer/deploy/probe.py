"""Facts about the machine the service is landing on. Pure observation, no side effects."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HostFacts:
    os: str
    arch: str
    python: str
    cpu_count: int
    ram_gb: float
    in_container: bool
    gpu_names: tuple[str, ...] = field(default_factory=tuple)
    torch_cuda: bool = False
    disk_free_gb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _cpu_count() -> int:
    """Cores this process may actually use (honours cgroup/affinity limits, unlike ``os.cpu_count``)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _cgroup_cpu_limit() -> float | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        return None


def _ram_gb() -> float:
    limit = None
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(p).read_text().strip()
            if raw.isdigit() and int(raw) < 1 << 50:
                limit = int(raw)
                break
        except OSError:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
                return round(min(total, limit or total) / 2**30, 1)
    except OSError:
        pass
    try:  # Windows / macOS fallback
        import psutil  # type: ignore[import-not-found]

        return round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        return 0.0


def _in_container() -> bool:
    if Path("/.dockerenv").exists() or os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    try:
        return "docker" in Path("/proc/1/cgroup").read_text() or "kubepods" in Path("/proc/1/cgroup").read_text()
    except OSError:
        return False


def _nvidia_gpus() -> tuple[str, ...]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ()
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(line.strip() for line in out.splitlines() if line.strip())


def probe(state_dir: Path | None = None) -> HostFacts:
    from skinflow_discard_optimizer.accel import gpu_available

    cpus = _cpu_count()
    limit = _cgroup_cpu_limit()
    if limit is not None:
        cpus = max(1, min(cpus, int(limit)))
    where = state_dir or Path.cwd()
    where = where if where.exists() else Path.cwd()
    return HostFacts(
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        python=platform.python_version(),
        cpu_count=cpus,
        ram_gb=_ram_gb(),
        in_container=_in_container(),
        gpu_names=_nvidia_gpus(),
        torch_cuda=gpu_available(),
        disk_free_gb=round(shutil.disk_usage(where).free / 2**30, 1),
    )
