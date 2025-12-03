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


def _validate_and_split_cluster(
    cluster_indices: List[int],
    embeddings: np.ndarray,
    threshold: float,
    verbose: bool
) -> Dict[int, int]:
    """
    Validate a cluster using average similarity and percentage of pairs meeting threshold.
    Split if validation fails (prevents transitive closure).
    Uses a more lenient approach than requiring ALL pairs to meet threshold.
    
    Args:
        cluster_indices: Indices of companies in the cluster
        embeddings: All embeddings (n_samples, embedding_dim)
        threshold: Similarity threshold
        verbose: Whether to print progress
        
    Returns:
        Dict mapping original index -> new sub-cluster ID (0 if valid, split IDs if invalid)
    """
    if len(cluster_indices) <= 1:
        return {idx: 0 for idx in cluster_indices}
    
    # Extract embeddings for this cluster
    cluster_embeddings = embeddings[cluster_indices]
    n_cluster = len(cluster_indices)
    
    # Use a more lenient threshold for validation (default: threshold - 0.05)
    validation_threshold = max(0.70, threshold - 0.05)
    
    # For very large clusters, use centroid filtering (faster)
    if n_cluster > 1000:
        return _filter_cluster_by_centroid(cluster_indices, embeddings, validation_threshold, verbose)
    
    # For smaller clusters, check average similarity and percentage of pairs
    # Compute pairwise similarities
    similarities = np.dot(cluster_embeddings, cluster_embeddings.T)
    np.fill_diagonal(similarities, 1.0)  # Ignore self-similarity
    
    # Calculate statistics
    min_similarity = float(np.min(similarities))
    avg_similarity = float(np.mean(similarities))
    
    # Count pairs above threshold
    pairs_above_threshold = np.sum(similarities >= threshold)
    total_pairs = n_cluster * (n_cluster - 1) / 2
    pct_above_threshold = pairs_above_threshold / total_pairs if total_pairs > 0 else 1.0
    
    # Cluster is valid if:
    # - Average similarity is above validation threshold (more lenient), OR
    # - At least 80% of pairs meet the original threshold
    if avg_similarity >= validation_threshold or pct_above_threshold >= 0.80:
        # Cluster is valid
        return {idx: 0 for idx in cluster_indices}
    
    # Cluster failed validation - split it using stricter threshold
    split_threshold = min(0.95, threshold + 0.05)
    
    # Build Union-Find with stricter threshold
    uf = UnionFind(n_cluster)
    
    # Add edges only for pairs meeting stricter threshold
    for i in range(n_cluster):
        for j in range(i + 1, n_cluster):
            similarity = similarities[i, j]
            if similarity >= split_threshold:
                uf.union(i, j)
    
    # Build sub-cluster assignments
    sub_cluster_assignments: Dict[int, int] = {}
    root_to_sub_cluster: Dict[int, int] = {}
    next_sub_cluster_id = 0
    
    for local_idx in range(n_cluster):
        root = uf.find(local_idx)
        if root not in root_to_sub_cluster:
            root_to_sub_cluster[root] = next_sub_cluster_id
            next_sub_cluster_id += 1
        sub_cluster_assignments[local_idx] = root_to_sub_cluster[root]
    
    # Map back to original indices
    result = {cluster_indices[local_idx]: sub_cluster_id 
              for local_idx, sub_cluster_id in sub_cluster_assignments.items()}
    
    if verbose and len(root_to_sub_cluster) > 1:
        print_progress(f"Split invalid cluster of {n_cluster} members into {len(root_to_sub_cluster)} sub-clusters (avg similarity: {avg_similarity:.3f}, min: {min_similarity:.3f} < {validation_threshold:.3f})", verbose)
    
    return result


def _filter_cluster_by_centroid(
    cluster_indices: List[int],
    embeddings: np.ndarray,
    threshold: float,
    verbose: bool
) -> Dict[int, int]:
    """
    Filter cluster members by similarity to centroid (for very large clusters).
    Removes members that are too dissimilar from the cluster center.
    Uses a more lenient threshold for very large clusters.
    """
    if len(cluster_indices) <= 1:
        return {idx: 0 for idx in cluster_indices}
    
    cluster_embeddings = embeddings[cluster_indices]
    centroid = np.mean(cluster_embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)  # Normalize
    
    # Use a more lenient threshold for centroid filtering of large clusters
    # For very large clusters (>10K), allow more variation
    if len(cluster_indices) > 10000:
        centroid_threshold = max(0.70, threshold - 0.10)
    else:
        centroid_threshold = max(0.75, threshold - 0.05)
    
    # Compute similarities to centroid
    similarities = np.dot(cluster_embeddings, centroid)
    
    # Keep only members similar to centroid (using lenient threshold)
    valid_mask = similarities >= centroid_threshold
    valid_indices = [cluster_indices[i] for i in range(len(cluster_indices)) if valid_mask[i]]
    
    if len(valid_indices) == len(cluster_indices):
        # All members are valid
        return {idx: 0 for idx in cluster_indices}
    
    # Split: valid members stay together, invalid members become singletons
    result = {}
    next_sub_cluster_id = 0
    
    # Valid members get cluster 0
    for idx in valid_indices:
        result[idx] = 0
    
    # Invalid members become singletons
    invalid_indices = [idx for idx in cluster_indices if idx not in valid_indices]
    for idx in invalid_indices:
        result[idx] = next_sub_cluster_id + 1
        next_sub_cluster_id += 1
    
    if verbose:
        print_progress(f"Filtered cluster: {len(valid_indices)}/{len(cluster_indices)} members remain (centroid threshold: {centroid_threshold:.3f})", verbose)
    
    return result


def _split_large_cluster(
    cluster_indices: List[int],
    embeddings: np.ndarray,
    threshold: float,
    max_cluster_size: int,
    verbose: bool
) -> Dict[int, int]:
    """
    Split a large cluster into smaller sub-clusters by re-clustering with stricter criteria.
    
    Args:
        cluster_indices: Indices of companies in the large cluster
        embeddings: All embeddings (n_samples, embedding_dim)
        threshold: Similarity threshold (use stricter threshold for splitting)
        max_cluster_size: Maximum allowed cluster size
        verbose: Whether to print progress
        
    Returns:
        Dict mapping original cluster index -> new sub-cluster ID
    """
    if len(cluster_indices) <= max_cluster_size:
        # Cluster is already small enough
        return {idx: 0 for idx in cluster_indices}
    
    # Use stricter threshold for splitting (add 0.05 to original threshold)
    split_threshold = min(0.95, threshold + 0.05)
    
    # Extract embeddings for this cluster
    cluster_embeddings = embeddings[cluster_indices]
    
    # Build a mini Union-Find for this cluster
    n_cluster = len(cluster_indices)
    uf = UnionFind(n_cluster)
    
    # Compute pairwise similarities and add edges
    # Use batch processing to avoid memory issues
    batch_size = min(1000, n_cluster)
    edges_added = 0
    
    for i in range(0, n_cluster, batch_size):
        end_i = min(i + batch_size, n_cluster)
        batch_embeddings = cluster_embeddings[i:end_i]
        
        # Compute similarities to all other cluster members
        similarities = np.dot(batch_embeddings, cluster_embeddings.T)
        
        for local_i, global_i in enumerate(range(i, end_i)):
            for local_j, global_j in enumerate(range(n_cluster)):
                if global_i < global_j:  # Only add each edge once
                    similarity = similarities[local_i, local_j]
                    if similarity >= split_threshold:
                        uf.union(global_i, global_j)
                        edges_added += 1
    
    # Build sub-cluster assignments
    sub_cluster_assignments: Dict[int, int] = {}
    root_to_sub_cluster: Dict[int, int] = {}
    next_sub_cluster_id = 0
    
    for local_idx in range(n_cluster):
        root = uf.find(local_idx)
        if root not in root_to_sub_cluster:
            root_to_sub_cluster[root] = next_sub_cluster_id
            next_sub_cluster_id += 1
        sub_cluster_assignments[local_idx] = root_to_sub_cluster[root]
    
    # Map back to original indices
    result = {cluster_indices[local_idx]: sub_cluster_id 
              for local_idx, sub_cluster_id in sub_cluster_assignments.items()}
    
    if verbose and len(root_to_sub_cluster) > 1:
        print_progress(f"Split cluster of {n_cluster} members into {len(root_to_sub_cluster)} sub-clusters (threshold={split_threshold:.2f})", verbose)
    
    return result


def _cluster_with_unionfind(n_samples: int, edges_list: List[Tuple[int, int]], verbose: bool) -> Dict[int, int]:
    """Build clusters using Union-Find from edge list (for RAPIDS fallback)."""
    uf = UnionFind(n_samples)
    
    # Process edges
    for src, dst in edges_list:
        uf.union(src, dst)
    
    return _build_cluster_assignments_from_unionfind(uf, n_samples, verbose)


def cluster_with_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    metric: str = "cosine",
    verbose: bool = True
) -> Dict[int, int]:
    """
    Cluster embeddings using HDBSCAN algorithm.
    
    Args:
        embeddings: Normalized embedding vectors (n_samples, embedding_dim)
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples in neighborhood (auto if None)
        metric: Distance metric ('cosine', 'euclidean', etc.)
        verbose: Whether to print progress
        
    Returns:
        Dict mapping index -> cluster_id (-1 for noise points)
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan is required for HDBSCAN clustering. Install with: pip install hdbscan")
    
    print_progress(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...", verbose)
    
    # Auto-set min_samples if not provided (typically same as min_cluster_size)
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Create HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1  # Use all available cores
    )
    
    # Fit the clusterer
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Convert to dictionary format
    cluster_assignments = {i: int(label) for i, label in enumerate(cluster_labels)}
    
    # Count clusters (excluding noise points labeled as -1)
    unique_clusters = set(cluster_labels)
    num_clusters = len([c for c in unique_clusters if c >= 0])
    num_noise = sum(1 for c in cluster_labels if c == -1)
    
    print_progress(f"HDBSCAN found {num_clusters} clusters ({num_noise} noise points)", verbose)
    
    return cluster_assignments


def cluster_with_agglomerative(
    embeddings: np.ndarray,
    threshold: float = 0.85,
    n_clusters: Optional[int] = None,
    linkage: str = "average",
    metric: str = "cosine",
    verbose: bool = True
) -> Dict[int, int]:
    """
    Cluster embeddings using Agglomerative Clustering.
    
    Args:
        embeddings: Normalized embedding vectors (n_samples, embedding_dim)
        threshold: Distance threshold for clustering (when n_clusters is None)
        n_clusters: Number of clusters (if None, uses distance_threshold)
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        metric: Distance metric ('cosine', 'euclidean', etc.)
        verbose: Whether to print progress
        
    Returns:
        Dict mapping index -> cluster_id
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        raise ImportError("scikit-learn is required for Agglomerative clustering. Install with: pip install scikit-learn")
    
    print_progress(f"Clustering with Agglomerative Clustering (threshold={threshold}, linkage={linkage})...", verbose)
    
    # Convert cosine similarity threshold to distance
    # For cosine similarity: distance = 1 - similarity
    # For normalized vectors, cosine distance = 1 - cosine similarity
    distance_threshold = 1.0 - threshold if metric == "cosine" else threshold
    
    # Create clusterer
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold if n_clusters is None else None,
        linkage=linkage,
        metric=metric,
        compute_full_tree=True if n_clusters is None else 'auto'
    )
    
    # Fit the clusterer
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Convert to dictionary format
    cluster_assignments = {i: int(label) for i, label in enumerate(cluster_labels)}
    
    num_clusters = len(set(cluster_labels))
    print_progress(f"Agglomerative Clustering found {num_clusters} clusters", verbose)
    
    return cluster_assignments


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
    max_cluster_size: int = 1000,
    verbose: bool = True
) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, float], Dict[int, int], Dict[int, int]]:
    """
    Cluster companies based on similarity search results.
    
    Args:
        n_samples: Total number of companies
        faiss_index: FAISSIndex instance (not used for hdbscan/agglomerative)
        embeddings: All embeddings (n_samples, embedding_dim)
        original_names: List of original company names
        normalized_names: List of normalized company names
        threshold: Similarity threshold for clustering
        top_k: Number of neighbors to retrieve per company (not used for hdbscan/agglomerative)
        clustering_method: Clustering algorithm ('connected_components', 'hdbscan', 'agglomerative')
        canonical_method: Method to select canonical name
        max_cluster_size: Maximum cluster size before splitting (default: 1000, not used for hdbscan/agglomerative)
        verbose: Whether to print progress
        
    Returns:
        Tuple of:
        - cluster_assignments: Dict mapping index -> cluster_id
        - canonical_names: Dict mapping cluster_id -> canonical_name
        - similarity_scores: Dict mapping index -> similarity to canonical
        - neighbor_counts: Dict mapping index -> number of neighbors found
        - cluster_sizes: Dict mapping cluster_id -> cluster size
    """
    # Route to appropriate clustering method
    if clustering_method == "hdbscan":
        # HDBSCAN clustering - works directly on embeddings
        print_progress("Using HDBSCAN clustering method...", verbose)
        # Auto-determine min_cluster_size based on dataset size
        min_cluster_size = max(5, min(50, n_samples // 10000))
        cluster_assignments = cluster_with_hdbscan(
            embeddings=embeddings,
            min_cluster_size=min_cluster_size,
            verbose=verbose
        )
        # Set dummy neighbor counts (not applicable for HDBSCAN)
        neighbor_counts = {i: 0 for i in range(n_samples)}
    elif clustering_method == "agglomerative":
        # Agglomerative clustering - works directly on embeddings
        print_progress("Using Agglomerative Clustering method...", verbose)
        cluster_assignments = cluster_with_agglomerative(
            embeddings=embeddings,
            threshold=threshold,
            verbose=verbose
        )
        # Set dummy neighbor counts (not applicable for Agglomerative)
        neighbor_counts = {i: 0 for i in range(n_samples)}
    elif clustering_method == "connected_components":
        # Original connected components method using FAISS
        print_progress(f"Finding similar pairs (threshold={threshold})...", verbose)
        cluster_assignments, neighbor_counts = _cluster_with_connected_components(
            n_samples=n_samples,
            faiss_index=faiss_index,
            embeddings=embeddings,
            threshold=threshold,
            top_k=top_k,
            search_batch_size=search_batch_size,
            max_cluster_size=max_cluster_size,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}. Choose from: 'connected_components', 'hdbscan', 'agglomerative'")
    
    # Handle noise points from HDBSCAN (label -1) - assign them unique cluster IDs
    if clustering_method == "hdbscan":
        max_cluster_id = max(cluster_assignments.values()) if cluster_assignments else -1
        next_noise_id = max_cluster_id + 1
        for idx, cluster_id in list(cluster_assignments.items()):
            if cluster_id == -1:
                cluster_assignments[idx] = next_noise_id
                next_noise_id += 1
    
    # Common post-processing: select canonical names and compute similarity scores
    return _post_process_clusters(
        cluster_assignments=cluster_assignments,
        embeddings=embeddings,
        original_names=original_names,
        normalized_names=normalized_names,
        canonical_method=canonical_method,
        neighbor_counts=neighbor_counts,
        verbose=verbose
    )


def _post_process_clusters(
    cluster_assignments: Dict[int, int],
    embeddings: np.ndarray,
    original_names: List[str],
    normalized_names: List[str],
    canonical_method: str,
    neighbor_counts: Dict[int, int],
    verbose: bool
) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, float], Dict[int, int], Dict[int, int]]:
    """
    Post-process clusters: select canonical names and compute similarity scores.
    
    Returns:
        Tuple of (cluster_assignments, canonical_names, similarity_scores, neighbor_counts, cluster_sizes)
    """
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
    
    # Validate clustering quality and provide feedback
    _validate_clustering_quality(clusters_dict, similarity_scores, neighbor_counts, verbose)
    
    return cluster_assignments, canonical_names, similarity_scores, neighbor_counts, cluster_sizes


def _validate_clustering_quality(
    clusters_dict: Dict[int, List[int]],
    similarity_scores: Dict[int, float],
    neighbor_counts: Dict[int, int],
    verbose: bool
):
    """
    Validate clustering quality and provide actionable feedback.
    
    Args:
        clusters_dict: Dictionary mapping cluster_id -> list of indices
        similarity_scores: Dictionary mapping index -> similarity to canonical
        neighbor_counts: Dictionary mapping index -> number of neighbors found
        verbose: Whether to print feedback
    """
    if not verbose or not clusters_dict:
        return
    
    # Calculate quality metrics
    cluster_sizes = [len(indices) for indices in clusters_dict.values()]
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    num_singletons = sum(1 for size in cluster_sizes if size == 1)
    pct_singletons = (num_singletons / len(clusters_dict)) * 100 if clusters_dict else 0
    
    # Calculate average similarity scores
    avg_similarities = [np.mean([similarity_scores.get(idx, 0.0) for idx in indices]) 
                       for indices in clusters_dict.values() if len(indices) > 1]
    overall_avg_similarity = np.mean(avg_similarities) if avg_similarities else 0.0
    
    # Calculate average neighbor counts
    avg_neighbors = np.mean(list(neighbor_counts.values())) if neighbor_counts else 0
    
    # Provide quality feedback
    print_progress("\n" + "="*60, verbose)
    print_progress("CLUSTERING QUALITY METRICS", verbose)
    print_progress("="*60, verbose)
    print_progress(f"Total clusters: {len(clusters_dict):,}", verbose)
    print_progress(f"Average cluster size: {avg_cluster_size:.2f}", verbose)
    print_progress(f"Largest cluster size: {max_cluster_size:,}", verbose)
    print_progress(f"Singleton clusters: {num_singletons:,} ({pct_singletons:.1f}%)", verbose)
    print_progress(f"Average similarity to canonical: {overall_avg_similarity:.3f}", verbose)
    print_progress(f"Average neighbors found: {avg_neighbors:.1f}", verbose)
    
    # Quality warnings and recommendations
    if pct_singletons > 80:
        print_progress(f"\nWARNING: {pct_singletons:.1f}% of clusters are singletons.", verbose)
        print_progress("  -> RECOMMENDATION: Lower threshold to find more matches", verbose)
    
    if max_cluster_size > 10000:
        print_progress(f"\nWARNING: Very large cluster detected ({max_cluster_size:,} members).", verbose)
        print_progress("  -> RECOMMENDATION: Increase threshold to split large clusters", verbose)
    
    if overall_avg_similarity < 0.80 and avg_similarities:
        print_progress(f"\nWARNING: Low average similarity ({overall_avg_similarity:.3f}).", verbose)
        print_progress("  -> RECOMMENDATION: Increase threshold for tighter clusters", verbose)
    
    if avg_neighbors < 1.0:
        print_progress(f"\nWARNING: Low average neighbors ({avg_neighbors:.1f}).", verbose)
        print_progress("  -> RECOMMENDATION: Lower threshold or increase top_k", verbose)
    
    print_progress("="*60 + "\n", verbose)


def _cluster_with_connected_components(
    n_samples: int,
    faiss_index,
    embeddings: np.ndarray,
    threshold: float,
    top_k: int,
    search_batch_size: int,
    max_cluster_size: int,
    verbose: bool
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Cluster using connected components method (original implementation).
    
    Returns:
        Tuple of (cluster_assignments, neighbor_counts)
    """
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
        # For very large datasets, use more permissive search threshold to improve recall
        if n_samples > 1000000:
            # More permissive for 1M+ records to account for approximate search limitations
            search_threshold = max(0.65, threshold - 0.15)
        elif n_samples > 500000:
            # Moderately permissive for 500K-1M records
            search_threshold = max(0.68, threshold - 0.12)
        else:
            # Default for smaller datasets
            search_threshold = max(0.70, threshold - 0.10)
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
            # Filter by search_threshold first (FAISS already filtered, but double-check)
            # Add explicit checks to prevent self-matches
            valid_neighbors = [
                (int(indices[j]), float(distances[j]))
                for j in range(len(indices))
                if indices[j] != -1 
                and int(indices[j]) != int(global_idx)  # Explicit self-match prevention
                and 0 <= int(indices[j]) < n_samples  # Ensure valid index range
                and distances[j] >= search_threshold  # Ensure above search threshold
            ]
            
            neighbor_counts[global_idx] = len(valid_neighbors)
            
            # Stream edges directly into Union-Find (memory efficient - no edge storage)
            # Only add edges where global_idx < neighbor_idx to ensure each edge is added exactly once
            # Apply the actual clustering threshold (stricter than search threshold)
            edges_added_this_batch = 0
            for neighbor_idx, similarity in valid_neighbors:
                # Only add if similarity meets the actual threshold AND we haven't processed this pair yet
                if similarity >= threshold and global_idx < neighbor_idx:
                    uf.union(global_idx, neighbor_idx)
                    edge_count += 1
                    edges_added_this_batch += 1
            
            # Enhanced warning for excessive edges per company
            if verbose and edges_added_this_batch > 200:
                print_progress(f"WARNING: Added {edges_added_this_batch} edges for company {global_idx}", verbose)
                print_progress(f"  -> This indicates threshold ({threshold:.2f}) may be too low", verbose)
                print_progress(f"  -> RECOMMENDATION: Increase threshold to {min(0.95, threshold + 0.05):.2f}", verbose)
            elif verbose and edges_added_this_batch > 100:
                print_progress(f"Warning: Added {edges_added_this_batch} edges for company {global_idx} (consider increasing threshold)", verbose)
    
    print_progress(f"Processed {edge_count:,} edges above threshold {threshold}, building clusters...", verbose)
    
    # Enhanced error handling and warnings
    avg_edges_per_node = edge_count / n_samples if n_samples > 0 else 0
    
    if avg_edges_per_node > 10:
        print_progress(f"WARNING: Very high average edges per node ({avg_edges_per_node:.2f}).", verbose)
        print_progress(f"  -> RECOMMENDATION: Increase threshold from {threshold:.2f} to {min(0.95, threshold + 0.10):.2f}", verbose)
        print_progress(f"  -> OR reduce top_k from {top_k} to {max(10, top_k // 2)}", verbose)
        print_progress(f"  -> This will reduce cluster sizes and improve quality.", verbose)
    elif avg_edges_per_node > 5:
        print_progress(f"Warning: High average edges per node ({avg_edges_per_node:.2f}). This may cause large clusters.", verbose)
        print_progress(f"  -> Consider increasing threshold from {threshold:.2f} to {min(0.95, threshold + 0.05):.2f}", verbose)
        print_progress(f"  -> OR reducing top_k from {top_k} to {max(20, int(top_k * 0.75))}", verbose)
    
    if edge_count == 0:
        print_progress("WARNING: No edges found above threshold. All companies will be in separate clusters.", verbose)
        print_progress(f"  -> RECOMMENDATION: Lower threshold from {threshold:.2f} to {max(0.70, threshold - 0.10):.2f}", verbose)
        print_progress(f"  -> OR increase top_k from {top_k} to {top_k * 2}", verbose)
    
    # Build cluster assignments from Union-Find structure
    cluster_assignments = _build_cluster_assignments_from_unionfind(uf, n_samples, verbose)
    
    # Group indices by cluster
    clusters_dict: Dict[int, List[int]] = {}
    for idx, cluster_id in cluster_assignments.items():
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = []
        clusters_dict[cluster_id].append(idx)
    
    # Validate clusters to prevent transitive closure issues
    # This ensures all pairs in each cluster meet the threshold
    if max_cluster_size > 0:
        print_progress("Validating clusters to prevent transitive closure...", verbose)
        clusters_to_validate = list(clusters_dict.items())
        next_cluster_id = max(cluster_assignments.values()) + 1
        validation_count = 0
        split_count = 0
        
        for original_cluster_id, cluster_indices in clusters_to_validate:
            # Skip single-member clusters
            if len(cluster_indices) <= 1:
                continue
            
            # Validate cluster (checks if all pairs meet threshold)
            validation_result = _validate_and_split_cluster(
                cluster_indices,
                embeddings,
                threshold,
                verbose
            )
            
            validation_count += 1
            
            # Check if cluster was split
            unique_sub_clusters = len(set(validation_result.values()))
            
            if unique_sub_clusters > 1:
                # Cluster was split - update assignments
                split_count += 1
                max_sub_cluster_id = max(validation_result.values())
                
                for idx, sub_cluster_id in validation_result.items():
                    new_cluster_id = next_cluster_id + sub_cluster_id
                    cluster_assignments[idx] = new_cluster_id
                
                next_cluster_id += max_sub_cluster_id + 1
        
        if split_count > 0:
            # Rebuild clusters_dict with validated/split clusters
            clusters_dict = {}
            for idx, cluster_id in cluster_assignments.items():
                if cluster_id not in clusters_dict:
                    clusters_dict[cluster_id] = []
                clusters_dict[cluster_id].append(idx)
            
            print_progress(f"Validated {validation_count} clusters, split {split_count} invalid clusters", verbose)
            print_progress(f"After validation: {len(clusters_dict)} clusters", verbose)
        
        # Also split clusters that are too large (size-based splitting)
        large_clusters = {cid: indices for cid, indices in clusters_dict.items() 
                         if len(indices) > max_cluster_size}
        
        if large_clusters:
            print_progress(f"Splitting {len(large_clusters)} large clusters (max size: {max_cluster_size})...", verbose)
            next_cluster_id = max(cluster_assignments.values()) + 1
            
            for original_cluster_id, cluster_indices in large_clusters.items():
                # Split the large cluster
                split_assignments = _split_large_cluster(
                    cluster_indices,
                    embeddings,
                    threshold,
                    max_cluster_size,
                    verbose
                )
                
                # Get the maximum sub-cluster ID from this split
                max_sub_cluster_id = max(split_assignments.values()) if split_assignments else 0
                
                # Update cluster assignments with new sub-cluster IDs
                for idx, sub_cluster_id in split_assignments.items():
                    # Assign new cluster ID (offset by next_cluster_id)
                    new_cluster_id = next_cluster_id + sub_cluster_id
                    cluster_assignments[idx] = new_cluster_id
                
                # Update next_cluster_id to avoid collisions
                next_cluster_id += max_sub_cluster_id + 1
            
            # Rebuild clusters_dict with updated assignments
            clusters_dict = {}
            for idx, cluster_id in cluster_assignments.items():
                if cluster_id not in clusters_dict:
                    clusters_dict[cluster_id] = []
                clusters_dict[cluster_id].append(idx)
            
            print_progress(f"After size-based splitting: {len(clusters_dict)} clusters", verbose)
    
    return cluster_assignments, neighbor_counts

