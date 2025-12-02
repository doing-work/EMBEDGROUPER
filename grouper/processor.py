"""Main processing pipeline for company name grouping."""

import time
import os
import gc
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from .normalizer import normalize_company_names
from .embeddings import EmbeddingGenerator
from .faiss_index import FAISSIndex
from .clustering import cluster_companies
from .utils import print_progress, format_time, check_faiss_gpu


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
        
        # Store original dataframe to preserve all columns
        self.original_df = df.copy()
        
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
        
        # Checkpoint: Save embeddings and index to disk, then free memory
        # This is critical for large datasets to avoid RAM issues
        checkpoint_dir = output_file.replace('.csv', '_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        embeddings_path = os.path.join(checkpoint_dir, 'embeddings.npy')
        index_path = os.path.join(checkpoint_dir, 'faiss_index.bin')
        
        print_progress("Saving checkpoints to disk to free RAM...", self.verbose)
        print_progress(f"Saving embeddings ({embeddings.nbytes / 1024**3:.2f} GB) to {embeddings_path}...", self.verbose)
        np.save(embeddings_path, embeddings)
        
        print_progress(f"Saving FAISS index to {index_path}...", self.verbose)
        faiss_index.save_to_disk(index_path)
        
        # Store metadata for reloading
        embedding_dim = embeddings.shape[1]
        index_type_used = faiss_index.index_type
        use_gpu = faiss_index.use_gpu
        
        # Free memory: delete large objects
        print_progress("Freeing memory...", self.verbose)
        del embedding_gen
        del faiss_index
        del embeddings
        gc.collect()
        
        # Clear GPU cache if using GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        print_progress("Memory freed. Loading from disk for clustering...", self.verbose)
        
        # Step 5: Load from disk and cluster
        print_progress("Loading embeddings and index from disk...", self.verbose)
        # Load embeddings - use memory mapping for large files to save RAM
        # We'll load in chunks during clustering if needed
        embeddings = np.load(embeddings_path)
        faiss_index = FAISSIndex.load_from_disk(
            index_path,
            dimension=embedding_dim,
            n_samples=n_samples,
            index_type=index_type_used,
            use_gpu=use_gpu,
            verbose=self.verbose
        )
        
        print_progress("Clustering companies...", self.verbose)
        cluster_start = time.time()
        # Use larger batch size for search on very large datasets
        search_batch_size = min(5000, max(1000, n_samples // 500)) if n_samples > 100000 else 1000
        
        # For very large datasets, increase top_k to find more neighbors
        # This helps ensure similar companies aren't missed due to ranking
        effective_top_k = self.top_k
        if n_samples > 1000000 and self.top_k < 100:
            effective_top_k = max(100, self.top_k)
            if self.verbose:
                print_progress(f"Increased top_k from {self.top_k} to {effective_top_k} for large dataset", self.verbose)
        
        cluster_assignments, canonical_names, similarity_scores, neighbor_counts, cluster_sizes = cluster_companies(
            n_samples=n_samples,
            faiss_index=faiss_index,
            embeddings=embeddings,
            original_names=company_names,
            normalized_names=normalized_names,
            threshold=self.threshold,
            top_k=effective_top_k,  # Use increased top_k for large datasets
            clustering_method=self.clustering_method,
            canonical_method=self.canonical_method,
            search_batch_size=search_batch_size,
            verbose=self.verbose
        )
        
        # Clean up checkpoints after clustering
        print_progress("Cleaning up checkpoints...", self.verbose)
        try:
            os.remove(embeddings_path)
            os.remove(index_path)
            os.rmdir(checkpoint_dir)
        except Exception as e:
            print_progress(f"Warning: Could not clean up checkpoints: {e}", self.verbose)
        self.timing_stats['clustering'] = time.time() - cluster_start
        
        # Step 6: Prepare output
        print_progress("Preparing output...", self.verbose)
        output_start = time.time()
        
        # Start with original dataframe to preserve all input columns
        output_df = self.original_df.copy()
        
        # Add clustering columns
        output_df['cluster_id'] = [cluster_assignments.get(i, i) for i in range(n_samples)]
        output_df['canonical_name'] = [canonical_names.get(cluster_assignments.get(i, i), company_names[i]) for i in range(n_samples)]
        output_df['similarity_score_to_canonical'] = [similarity_scores.get(i, 1.0) for i in range(n_samples)]
        output_df['neighbor_count'] = [neighbor_counts.get(i, 0) for i in range(n_samples)]
        output_df['cluster_size'] = [cluster_sizes.get(cluster_assignments.get(i, i), 1) for i in range(n_samples)]
        
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

