"""Utility functions for device detection, progress tracking, and helpers."""

import torch
import sys
from typing import Optional


def get_device() -> str:
    """Auto-detect and return the best available device (cuda/cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_faiss_gpu() -> bool:
    """Check if FAISS GPU support is available."""
    try:
        import faiss
        return hasattr(faiss, "StandardGpuResources")
    except ImportError:
        return False


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 3600 % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def print_progress(message: str, verbose: bool = True):
    """Print progress message if verbose mode is enabled."""
    if verbose:
        print(message, file=sys.stderr)

