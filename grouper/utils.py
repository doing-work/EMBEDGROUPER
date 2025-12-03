"""Utility functions for device detection, progress tracking, and helpers."""

import torch
import sys
from typing import Optional, Dict


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


def calculate_adaptive_topk(
    n_samples: int,
    base_topk: int,
    threshold: float,
    dataset_density: Optional[float] = None
) -> int:
    """
    Calculate adaptive top-K based on dataset characteristics.
    
    Args:
        n_samples: Number of samples in dataset
        base_topk: Base top-K value
        threshold: Similarity threshold
        dataset_density: Optional dataset density estimate (0-1)
        
    Returns:
        Adaptive top-K value
    """
    # Base scaling by dataset size
    if n_samples > 5000000:
        # For 5M+ records, use higher top-k
        adaptive_topk = max(300, base_topk * 3)
    elif n_samples > 2000000:
        # For 2M-5M records
        adaptive_topk = max(200, base_topk * 2)
    elif n_samples > 1000000:
        # For 1M-2M records
        adaptive_topk = max(150, int(base_topk * 1.5))
    elif n_samples > 500000:
        # For 500K-1M records
        adaptive_topk = max(100, int(base_topk * 1.2))
    elif n_samples > 100000:
        # For 100K-500K records
        adaptive_topk = max(50, base_topk)
    else:
        adaptive_topk = base_topk
    
    # Adjust based on threshold
    # Lower thresholds need more neighbors (more candidates to filter)
    if threshold < 0.75:
        adaptive_topk = int(adaptive_topk * 1.5)
    elif threshold > 0.90:
        # High threshold means fewer matches, can use fewer neighbors
        adaptive_topk = max(base_topk, int(adaptive_topk * 0.8))
    
    # Adjust based on dataset density if provided
    if dataset_density is not None:
        # Higher density = more similar items = need more neighbors
        if dataset_density > 0.7:
            adaptive_topk = int(adaptive_topk * 1.3)
        elif dataset_density < 0.3:
            # Low density = fewer similar items = can use fewer neighbors
            adaptive_topk = max(base_topk, int(adaptive_topk * 0.9))
    
    # Clamp to reasonable bounds
    adaptive_topk = max(10, min(adaptive_topk, 500))
    
    return adaptive_topk


def optimize_threshold(
    initial_threshold: float,
    cluster_stats: Dict,
    target_clusters: Optional[int] = None
) -> float:
    """
    Optimize similarity threshold based on cluster statistics.
    
    Args:
        initial_threshold: Initial similarity threshold
        cluster_stats: Dictionary with cluster statistics (num_clusters, avg_size, etc.)
        target_clusters: Optional target number of clusters
        
    Returns:
        Optimized threshold value
    """
    num_clusters = cluster_stats.get('num_clusters', 0)
    avg_cluster_size = cluster_stats.get('avg_cluster_size', 1.0)
    max_cluster_size = cluster_stats.get('max_cluster_size', 1)
    
    # If too many clusters (likely threshold too high)
    if target_clusters and num_clusters > target_clusters * 2:
        # Lower threshold to merge more clusters
        optimized = max(0.70, initial_threshold - 0.05)
        return optimized
    
    # If too few clusters (likely threshold too low)
    if target_clusters and num_clusters < target_clusters / 2:
        # Raise threshold to split clusters
        optimized = min(0.95, initial_threshold + 0.05)
        return optimized
    
    # If average cluster size is very large, threshold might be too low
    if avg_cluster_size > 1000:
        optimized = min(0.95, initial_threshold + 0.03)
        return optimized
    
    # If max cluster size is extremely large, threshold is definitely too low
    if max_cluster_size > 10000:
        optimized = min(0.95, initial_threshold + 0.05)
        return optimized
    
    # Default: return original threshold
    return initial_threshold
