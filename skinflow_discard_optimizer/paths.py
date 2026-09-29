"""Filesystem locations shared by every subpackage."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "contracts"
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
DOCS_DIR = REPO_ROOT / "docs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

SIGNALS_YAML = CONTRACTS_DIR / "signals.yaml"


def default_workers(cap: int = 8) -> int:
    """Process-pool size: one core spare, capped, because every worker loads SciPy
    (~150 MB), and 23 of them exhausted the Windows page file on the dev machine."""
    import os

    return max(1, min(cap, (os.cpu_count() or 2) - 1))
