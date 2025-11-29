"""Clustering algorithms for grouping similar company names."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import Counter
from .utils import print_progress


def build_similarity_graph(
    n_samples: int,
    similarity_pairs: List[Tuple[int, int, float]],
    threshold: float = 0.85
) -> nx.Graph:
    """
    Build a graph where nodes are companies and edges connect similar pairs.
    
    Args:
        n_samples: Total number of samples
        similarity_pairs: List of (idx1, idx2, similarity_score) tuples
        threshold: Minimum similarity to create an edge
        
    Returns:
        NetworkX graph with edges for similar pairs
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n_samples))
    
    # Add edges for pairs above threshold
    edges_added = 0
    for idx1, idx2, similarity in similarity_pairs:
        if similarity >= threshold and idx1 != idx2:
            graph.add_edge(idx1, idx2, weight=similarity)
            edges_added += 1
    
    print_progress(f"Built graph with {edges_added} edges", True)
    return graph


def connected_components_clustering(graph: nx.Graph) -> Dict[int, int]:
    """
    Cluster nodes using connected components.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Dictionary mapping node index to cluster_id
    """
    clusters = {}
    cluster_id = 0
    
    for component in nx.connected_components(graph):
        for node in component:
            clusters[node] = cluster_id
        cluster_id += 1
    
    return clusters


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
        cluster_embeddings = embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        
        # Compute similarities to centroid
        similarities = np.dot(cluster_embeddings, centroid)
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
    
    # Find all similar pairs
    similarity_pairs = []
    neighbor_counts = {}
    
    for i in range(n_samples):
        if verbose and i % 10000 == 0:
            print_progress(f"Processed {i}/{n_samples} companies...", verbose)
        
        # Search for neighbors
        distances, indices = faiss_index.search_single(
            embeddings[i],
            k=top_k,
            threshold=threshold
        )
        
        # Count valid neighbors (excluding self and -1 padding)
        valid_neighbors = [
            (int(indices[j]), float(distances[j]))
            for j in range(len(indices))
            if indices[j] != -1 and indices[j] != i
        ]
        
        neighbor_counts[i] = len(valid_neighbors)
        
        # Add to similarity pairs
        for neighbor_idx, similarity in valid_neighbors:
            # Add both directions, but we'll deduplicate in graph building
            if i < neighbor_idx:  # Only add once per pair
                similarity_pairs.append((i, neighbor_idx, similarity))
    
    print_progress(f"Found {len(similarity_pairs)} similar pairs", verbose)
    
    # Build similarity graph
    print_progress("Building similarity graph...", verbose)
    graph = build_similarity_graph(n_samples, similarity_pairs, threshold)
    
    # Perform clustering
    print_progress(f"Clustering using {clustering_method}...", verbose)
    if clustering_method == "connected_components":
        cluster_assignments = connected_components_clustering(graph)
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    
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
    
    for cluster_id, cluster_indices in clusters_dict.items():
        cluster_sizes[cluster_id] = len(cluster_indices)
        canonical_name, canonical_idx, avg_sim = select_canonical_name(
            cluster_indices,
            original_names,
            normalized_names,
            embeddings,
            method=canonical_method
        )
        canonical_names[cluster_id] = canonical_name
        
        # Compute similarity to canonical for each member
        canonical_embedding = embeddings[canonical_idx]
        for idx in cluster_indices:
            similarity = float(np.dot(embeddings[idx], canonical_embedding))
            similarity_scores[idx] = similarity
    
    print_progress(f"Created {len(clusters_dict)} clusters", verbose)
    
    return cluster_assignments, canonical_names, similarity_scores, neighbor_counts, cluster_sizes

