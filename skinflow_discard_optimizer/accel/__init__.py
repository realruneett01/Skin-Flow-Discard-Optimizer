"""GPU acceleration (PyTorch). Everything here has a CPU reference it is tested against."""
from __future__ import annotations


def get_device(prefer: str = "cuda"):
    """CUDA if available (and requested), else CPU. Imports torch lazily so CPU-only installs still work."""
    import torch

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()
