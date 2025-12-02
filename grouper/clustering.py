"""Clustering algorithms for grouping similar company names."""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from tqdm import tqdm
from .utils import print_progress

# Try to import RAPIDS cuGraph, fallback to Union-Find if not available
try:
    import cudf
    import cugraph
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False


class UnionFind:
    """Disjoint-set (Union-Find) structure for efficient clustering without storing all edges."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def select_canonical_name(
    cluster_indices: List[int],
    original_names: List[str],
    normalized_names: List[str],
    embeddings: np.ndarray,
    method: str = "longest"
) -> Tuple[str, int, float]:
    """
    Select a canonical name for a cluster.
    
    Args:
        cluster_indices: List of indices in this cluster
        original_names: List of all original company names
        normalized_names: List of all normalized company names
        embeddings: All embeddings (n_samples, embedding_dim)
        method: Selection method ('longest', 'most_frequent', 'centroid')
        
    Returns:
        Tuple of (canonical_name, canonical_index, avg_similarity)
    """
    if not cluster_indices:
        return "", -1, 0.0
    
    if len(cluster_indices) == 1:
        idx = cluster_indices[0]
        return original_names[idx], idx, 1.0
    
    if method == "longest":
        # Select the longest name
        canonical_idx = max(cluster_indices, key=lambda i: len(original_names[i]))
        canonical_name = original_names[canonical_idx]
    
    elif method == "most_frequent":
        # Select the most frequent name in the dataset
        cluster_names = [original_names[i] for i in cluster_indices]
        name_counts = Counter(cluster_names)
        most_common_name = name_counts.most_common(1)[0][0]
        # Find index of first occurrence of most common name
        canonical_idx = next(
            i for i in cluster_indices if original_names[i] == most_common_name
        )
        canonical_name = most_common_name
    
    elif method == "centroid":
        # Select name closest to cluster centroid
        # For very large clusters, sample to speed up computation
        if len(cluster_indices) > 10000:
            # Sample up to 10K members for centroid computation
            import random
            sampled_indices = random.sample(cluster_indices, min(10000, len(cluster_indices)))
            cluster_embeddings = embeddings[sampled_indices]
        else:
            cluster_embeddings = embeddings[cluster_indices]
        
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        
        # Compute similarities to centroid (use full cluster for final selection)
        full_cluster_embeddings = embeddings[cluster_indices]
        similarities = np.dot(full_cluster_embeddings, centroid)
        best_idx_in_cluster = np.argmax(similarities)
        canonical_idx = cluster_indices[best_idx_in_cluster]
        canonical_name = original_names[canonical_idx]
    
    else:
        raise ValueError(f"Unknown canonical selection method: {method}")
    
    # Compute average similarity to canonical
    canonical_embedding = embeddings[canonical_idx]
    cluster_embeddings = embeddings[cluster_indices]
    similarities = np.dot(cluster_embeddings, canonical_embedding)
    avg_similarity = float(np.mean(similarities))
    
    return canonical_name, canonical_idx, avg_similarity


def _cluster_with_cugraph(n_samples: int, edges_list: List[Tuple[int, int]], verbose: bool) -> Dict[int, int]:
    """Build clusters using RAPIDS cuGraph connected components."""
    # Create cuDF DataFrame from edges - use numpy arrays for efficiency
    # Convert to numpy first, then to cuDF to avoid Python list overhead
    edges_array = np.array(edges_list, dtype=np.int32)
    src = edges_array[:, 0]
    dst = edges_array[:, 1]
    
    # Create cuDF DataFrame directly from numpy arrays (more memory efficient)
    df = cudf.DataFrame({'src': src, 'dst': dst})
    
    # Create graph and find connected components
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst', renumber=False)
    
    # Get connected components
    components = cugraph.connected_components(G)
    
    # Convert to dictionary mapping node -> cluster_id
    cluster_assignments = {}
    # cuGraph returns labels, we need to map them to sequential cluster IDs
    labels = components['labels'].to_pandas().values
    vertices = components['vertex'].to_pandas().values
    
    # Map labels to sequential cluster IDs
    unique_labels = {}
    next_cluster_id = 0
    for vertex, label in zip(vertices, labels):
        if label not in unique_labels:
            unique_labels[label] = next_cluster_id
            next_cluster_id += 1
        cluster_assignments[int(vertex)] = unique_labels[label]
    
    # Assign cluster IDs to all nodes (including isolated ones)
    for i in range(n_samples):
        if i not in cluster_assignments:
            cluster_assignments[i] = next_cluster_id
            next_cluster_id += 1
    
    print_progress(f"Found {len(unique_labels)} clusters using cuGraph", verbose)
    return cluster_assignments


def _build_cluster_assignments_from_unionfind(uf: UnionFind, n_samples: int, verbose: bool) -> Dict[int, int]:
    """Build cluster assignments from an existing Union-Find structure."""
    # Assign cluster IDs based on UnionFind roots
    cluster_assignments: Dict[int, int] = {}
    root_to_cluster: Dict[int, int] = {}
    next_cluster_id = 0
    
    for idx in range(n_samples):
        root = uf.find(idx)
        if root not in root_to_cluster:
            root_to_cluster[root] = next_cluster_id
            next_cluster_id += 1
        cluster_assignments[idx] = root_to_cluster[root]
    
    print_progress(f"Found {len(root_to_cluster)} clusters using Union-Find", verbose)
    return cluster_assignments


def _cluster_with_unionfind(n_samples: int, edges_list: List[Tuple[int, int]], verbose: bool) -> Dict[int, int]:
    """Build clusters using Union-Find from edge list (for RAPIDS fallback)."""
    uf = UnionFind(n_samples)
    
    # Process edges
    for src, dst in edges_list:
        uf.union(src, dst)
    
    return _build_cluster_assignments_from_unionfind(uf, n_samples, verbose)


def cluster_companies(
    n_samples: int,
    faiss_index,
    embeddings: np.ndarray,
    original_names: List[str],
    normalized_names: List[str],
    threshold: float = 0.85,
    top_k: int = 50,
    clustering_method: str = "connected_components",
    canonical_method: str = "longest",
    search_batch_size: int = 1000,
    verbose: bool = True
) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, float], Dict[int, int], Dict[int, int]]:
    """
    Cluster companies based on similarity search results.
    
    Args:
        n_samples: Total number of companies
        faiss_index: FAISSIndex instance
        embeddings: All embeddings (n_samples, embedding_dim)
        original_names: List of original company names
        normalized_names: List of normalized company names
        threshold: Similarity threshold for clustering
        top_k: Number of neighbors to retrieve per company
        clustering_method: Clustering algorithm ('connected_components')
        canonical_method: Method to select canonical name
        verbose: Whether to print progress
        
    Returns:
        Tuple of:
        - cluster_assignments: Dict mapping index -> cluster_id
        - canonical_names: Dict mapping cluster_id -> canonical_name
        - similarity_scores: Dict mapping index -> similarity to canonical
        - neighbor_counts: Dict mapping index -> number of neighbors found
        - cluster_sizes: Dict mapping cluster_id -> cluster size
    """
    print_progress(f"Finding similar pairs (threshold={threshold})...", verbose)
    
    # Find all similar pairs - batch processing for efficiency
    # Use streaming Union-Find to avoid RAM issues (no edge list storage)
    neighbor_counts: Dict[int, int] = {}
    uf = UnionFind(n_samples)  # Stream edges directly into Union-Find
    edge_count = 0
    
    # Process in batches for better performance
    # Use larger batches for very large datasets
    if search_batch_size is None:
        search_batch_size = min(1000, max(100, n_samples // 1000))
    total_batches = (n_samples + search_batch_size - 1) // search_batch_size
    
    # Use tqdm for progress tracking
    batch_range = range(total_batches)
    if verbose:
        batch_range = tqdm(batch_range, desc="Finding similar pairs", unit="batch")
    
    for batch_idx in batch_range:
        start_idx = batch_idx * search_batch_size
        end_idx = min(start_idx + search_batch_size, n_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        # Batch search for all companies in this batch
        batch_embeddings = embeddings[start_idx:end_idx]
        # Use lower threshold for search to find more candidates
        # The actual threshold will be applied when building the graph
        # This ensures we don't miss similar companies that are just outside the strict threshold
        search_threshold = max(0.70, threshold - 0.10)  # Lower threshold for search
        distances_batch, indices_batch = faiss_index.search(
            batch_embeddings,
            k=top_k,
            threshold=search_threshold  # More permissive search to find candidates
        )
        
        # Process results for each company in the batch
        for local_idx, global_idx in enumerate(batch_indices):
            distances = distances_batch[local_idx]
            indices = indices_batch[local_idx]
            
            # Count valid neighbors (excluding self and -1 padding)
            valid_neighbors = [
                (int(indices[j]), float(distances[j]))
                for j in range(len(indices))
                if indices[j] != -1 and indices[j] != global_idx
            ]
            
            neighbor_counts[global_idx] = len(valid_neighbors)
            
            # Stream edges directly into Union-Find (memory efficient - no edge storage)
            for neighbor_idx, similarity in valid_neighbors:
                if similarity >= threshold and global_idx != neighbor_idx:
                    uf.union(global_idx, neighbor_idx)
                    edge_count += 1
    
    print_progress(f"Processed {edge_count:,} edges, building clusters...", verbose)
    
    # Build cluster assignments from Union-Find structure
    cluster_assignments = _build_cluster_assignments_from_unionfind(uf, n_samples, verbose)
    
    # Group indices by cluster
    clusters_dict: Dict[int, List[int]] = {}
    for idx, cluster_id in cluster_assignments.items():
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = []
        clusters_dict[cluster_id].append(idx)
    
    # Select canonical names for each cluster
    print_progress("Selecting canonical names...", verbose)
    canonical_names = {}
    similarity_scores = {}
    cluster_sizes = {}
    
    # Add progress tracking for large numbers of clusters
    cluster_items = clusters_dict.items()
    if verbose and len(clusters_dict) > 1000:
        cluster_items = tqdm(cluster_items, desc="Selecting canonicals", unit="cluster")
    
    for cluster_id, cluster_indices in cluster_items:
        cluster_size = len(cluster_indices)
        cluster_sizes[cluster_id] = cluster_size
        
        # Fast path for single-member clusters
        if cluster_size == 1:
            idx = cluster_indices[0]
            canonical_names[cluster_id] = original_names[idx]
            similarity_scores[idx] = 1.0
            continue
        
        # Multi-member clusters
        canonical_name, canonical_idx, avg_sim = select_canonical_name(
            cluster_indices,
            original_names,
            normalized_names,
            embeddings,
            method=canonical_method
        )
        canonical_names[cluster_id] = canonical_name
        
        # Compute similarity to canonical for each member - BATCHED for performance
        canonical_embedding = embeddings[canonical_idx]
        cluster_embeddings = embeddings[cluster_indices]
        # Compute all similarities in one vectorized operation
        similarities = np.dot(cluster_embeddings, canonical_embedding)
        # Store all similarities at once
        for idx, similarity in zip(cluster_indices, similarities):
            similarity_scores[idx] = float(similarity)
    
    print_progress(f"Created {len(clusters_dict)} clusters", verbose)
    
    return cluster_assignments, canonical_names, similarity_scores, neighbor_counts, cluster_sizes

