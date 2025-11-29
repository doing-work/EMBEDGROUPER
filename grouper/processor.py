"""Main processing pipeline for company name grouping."""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from grouper.normalizer import normalize_company_names
from grouper.embeddings import EmbeddingGenerator
from grouper.faiss_index import FAISSIndex
from grouper.clustering import cluster_companies
from grouper.utils import print_progress, format_time


class CompanyGrouper:
    """Main processor for grouping company names."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        threshold: float = 0.85,
        top_k: int = 50,
        index_type: str = "auto",
        clustering_method: str = "connected_components",
        canonical_method: str = "longest",
        verbose: bool = True
    ):
        """
        Initialize the company grouper.
        
        Args:
            model_name: Embedding model name
            batch_size: Batch size for embeddings
            threshold: Similarity threshold for clustering
            top_k: Number of neighbors to retrieve
            index_type: FAISS index type
            clustering_method: Clustering algorithm
            canonical_method: Method to select canonical names
            verbose: Whether to print progress
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.threshold = threshold
        self.top_k = top_k
        self.index_type = index_type
        self.clustering_method = clustering_method
        self.canonical_method = canonical_method
        self.verbose = verbose
        
        self.timing_stats = {}
    
    def process(
        self,
        input_file: str,
        output_file: str,
        column_name: str = "company_name"
    ) -> Dict:
        """
        Process the input CSV and generate grouped output.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            column_name: Name of column containing company names
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        
        # Step 1: Load data
        print_progress(f"Loading data from {input_file}...", self.verbose)
        load_start = time.time()
        df = pd.read_csv(input_file)
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        company_names = df[column_name].fillna("").astype(str).tolist()
        n_samples = len(company_names)
        print_progress(f"Loaded {n_samples} company names", self.verbose)
        self.timing_stats['load'] = time.time() - load_start
        
        # Step 2: Normalize names
        print_progress("Normalizing company names...", self.verbose)
        norm_start = time.time()
        normalized_names, original_mapping = normalize_company_names(company_names)
        self.timing_stats['normalize'] = time.time() - norm_start
        
        # Step 3: Generate embeddings
        print_progress("Generating embeddings...", self.verbose)
        embed_start = time.time()
        embedding_gen = EmbeddingGenerator(
            model_name=self.model_name,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        embeddings = embedding_gen.generate_embeddings(normalized_names)
        self.timing_stats['embeddings'] = time.time() - embed_start
        
        # Step 4: Build FAISS index
        print_progress("Building FAISS index...", self.verbose)
        index_start = time.time()
        faiss_index = FAISSIndex(
            embeddings,
            index_type=self.index_type,
            verbose=self.verbose
        )
        self.timing_stats['index_build'] = time.time() - index_start
        
        # Step 5: Find similar pairs and cluster
        print_progress("Clustering companies...", self.verbose)
        cluster_start = time.time()
        cluster_assignments, canonical_names, similarity_scores, neighbor_counts, cluster_sizes = cluster_companies(
            n_samples=n_samples,
            faiss_index=faiss_index,
            embeddings=embeddings,
            original_names=company_names,
            normalized_names=normalized_names,
            threshold=self.threshold,
            top_k=self.top_k,
            clustering_method=self.clustering_method,
            canonical_method=self.canonical_method,
            verbose=self.verbose
        )
        self.timing_stats['clustering'] = time.time() - cluster_start
        
        # Step 6: Prepare output
        print_progress("Preparing output...", self.verbose)
        output_start = time.time()
        
        # Create output dataframe
        output_data = {
            'original_name': company_names,
            'cluster_id': [cluster_assignments.get(i, i) for i in range(n_samples)],
            'canonical_name': [canonical_names.get(cluster_assignments.get(i, i), company_names[i]) for i in range(n_samples)],
            'similarity_score_to_canonical': [similarity_scores.get(i, 1.0) for i in range(n_samples)],
            'neighbor_count': [neighbor_counts.get(i, 0) for i in range(n_samples)],
            'cluster_size': [cluster_sizes.get(cluster_assignments.get(i, i), 1) for i in range(n_samples)]
        }
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file, index=False)
        self.timing_stats['output'] = time.time() - output_start
        
        # Calculate statistics
        total_time = time.time() - start_time
        self.timing_stats['total'] = total_time
        
        stats = {
            'total_records': n_samples,
            'unique_clusters': len(canonical_names),
            'avg_cluster_size': np.mean(list(cluster_sizes.values())),
            'max_cluster_size': max(cluster_sizes.values()) if cluster_sizes else 1,
            'min_cluster_size': min(cluster_sizes.values()) if cluster_sizes else 1,
            'timing': self.timing_stats.copy()
        }
        
        # Print summary
        self._print_summary(stats)
        
        return stats
    
    def _print_summary(self, stats: Dict):
        """Print processing summary statistics."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total records processed: {stats['total_records']:,}")
        print(f"Unique clusters found: {stats['unique_clusters']:,}")
        print(f"Average cluster size: {stats['avg_cluster_size']:.2f}")
        print(f"Largest cluster size: {stats['max_cluster_size']}")
        print(f"Smallest cluster size: {stats['min_cluster_size']}")
        print("\nTiming Breakdown:")
        for stage, duration in stats['timing'].items():
            print(f"  {stage:20s}: {format_time(duration)}")
        print(f"\nTotal time: {format_time(stats['timing']['total'])}")
        print("="*60 + "\n")

