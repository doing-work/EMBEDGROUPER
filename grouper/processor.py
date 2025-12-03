"""Main processing pipeline for company name grouping."""

import time
import os
import gc
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from .normalizer import normalize_company_names
from .embeddings import EmbeddingGenerator
from .faiss_index import FAISSIndex
from .clustering import cluster_companies
from .utils import print_progress, format_time, check_faiss_gpu, calculate_adaptive_topk, optimize_threshold


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
        max_cluster_size: int = 1000,
        use_memmap: bool = False,
        cache_embeddings: bool = True,
        force_regenerate: bool = False,
        reduce_dimensions: Optional[int] = None,
        preserve_variance: float = 0.95,
        verbose: bool = True
    ):
        """
        Initialize the company grouper.
        
        Args:
            model_name: Embedding model name
            batch_size: Batch size for embeddings (0 for auto-tuning)
            threshold: Similarity threshold for clustering
            top_k: Number of neighbors to retrieve
            index_type: FAISS index type ('auto', 'flat', 'hnsw', 'ivf', 'ivfpq')
            clustering_method: Clustering algorithm ('connected_components', 'hdbscan', 'agglomerative')
            canonical_method: Method to select canonical names
            max_cluster_size: Maximum cluster size before splitting (default: 1000)
            use_memmap: Whether to use memory-mapped files for embeddings
            cache_embeddings: Whether to cache embeddings to disk for reuse (default: True)
            force_regenerate: Force regeneration of embeddings even if cache exists (default: False)
            reduce_dimensions: Target dimension for PCA reduction (None to disable, e.g., 256, 128)
            preserve_variance: Variance to preserve when using PCA (0.0-1.0, default: 0.95)
            verbose: Whether to print progress
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.threshold = threshold
        self.top_k = top_k
        self.index_type = index_type
        self.clustering_method = clustering_method
        self.canonical_method = canonical_method
        self.max_cluster_size = max_cluster_size
        self.use_memmap = use_memmap
        self.cache_embeddings = cache_embeddings
        self.force_regenerate = force_regenerate
        self.reduce_dimensions = reduce_dimensions
        self.preserve_variance = preserve_variance
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
        
        # Step 3: Generate or load cached embeddings
        embed_start = time.time()
        embeddings = self._get_embeddings(
            normalized_names=normalized_names,
            input_file=input_file,
            output_file=output_file,
            n_samples=n_samples
        )
        self.timing_stats['embeddings'] = time.time() - embed_start
        
        # Step 3.5: Apply dimension reduction if requested
        if self.reduce_dimensions is not None:
            reduce_start = time.time()
            embeddings = self._reduce_embeddings(
                embeddings=embeddings,
                input_file=input_file,
                output_file=output_file,
                n_samples=n_samples
            )
            self.timing_stats['dimension_reduction'] = time.time() - reduce_start
        
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
        embedding_dim = embeddings.shape[1]
        
        # Use memory-mapped file for embeddings (more memory efficient)
        print_progress(f"Saving embeddings ({embeddings.nbytes / 1024**3:.2f} GB) as memory-mapped file: {embeddings_path}...", self.verbose)
        # Remove existing file if it exists
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)
        
        # Create memory-mapped file and copy data
        memmap_embeddings = np.memmap(
            embeddings_path,
            dtype='float32',
            mode='w+',
            shape=(n_samples, embedding_dim)
        )
        memmap_embeddings[:] = embeddings[:]
        memmap_embeddings.flush()
        del memmap_embeddings  # Close the memory-mapped file
        
        print_progress(f"Saving FAISS index to {index_path}...", self.verbose)
        faiss_index.save_to_disk(index_path)
        
        # Store metadata for reloading
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
        print_progress("Loading embeddings and index from disk (using memory mapping)...", self.verbose)
        # Load embeddings as memory-mapped file (doesn't load into RAM)
        embeddings = np.memmap(
            embeddings_path,
            dtype='float32',
            mode='r',
            shape=(n_samples, embedding_dim)
        )
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
        
        # Calculate adaptive top_k based on dataset characteristics
        effective_top_k = calculate_adaptive_topk(
            n_samples=n_samples,
            base_topk=self.top_k,
            threshold=self.threshold
        )
        if effective_top_k != self.top_k and self.verbose:
            print_progress(f"Adaptive top_k: {self.top_k} -> {effective_top_k} (dataset size: {n_samples:,})", self.verbose)
        
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
            max_cluster_size=self.max_cluster_size,
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
    
    def _get_cache_key(self, input_file: str, model_name: str) -> str:
        """
        Generate cache key based on file hash and model name.
        
        Args:
            input_file: Path to input CSV file
            model_name: Name of the embedding model
            
        Returns:
            MD5 hash string for cache key
        """
        # Hash the input file (first 1MB + file stats for speed and accuracy)
        try:
            with open(input_file, 'rb') as f:
                file_hash = hashlib.md5(f.read(1024*1024)).hexdigest()
            
            # Include file size and modification time for better cache invalidation
            stat = os.stat(input_file)
            cache_string = f"{model_name}_{file_hash}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(cache_string.encode()).hexdigest()
        except Exception as e:
            # Fallback: use model name and current time if file access fails
            cache_string = f"{model_name}_{time.time()}"
            return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _generate_embeddings(
        self, 
        normalized_names: List[str],
        cache_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embeddings and optionally save to cache.
        
        Args:
            normalized_names: List of normalized company names
            cache_path: Optional path to save embeddings cache
            
        Returns:
            Numpy array of embeddings
        """
        print_progress("Generating embeddings...", self.verbose)
        
        # Determine memmap path if using memory-mapped files
        memmap_path = cache_path if (self.use_memmap and cache_path) else None
        
        embedding_gen = EmbeddingGenerator(
            model_name=self.model_name,
            batch_size=self.batch_size,
            use_memmap=self.use_memmap,
            memmap_path=memmap_path,
            verbose=self.verbose
        )
        embeddings = embedding_gen.generate_embeddings(
            normalized_names, 
            memmap_path=memmap_path if self.use_memmap else None
        )
        
        # Save to cache if not using memmap and caching is enabled
        if (not self.use_memmap and cache_path and self.cache_embeddings):
            print_progress(f"Saving embeddings to cache: {cache_path}...", self.verbose)
            np.save(cache_path, embeddings)
        
        return embeddings
    
    def _get_embeddings(
        self,
        normalized_names: List[str],
        input_file: str,
        output_file: str,
        n_samples: int
    ) -> np.ndarray:
        """
        Get embeddings from cache or generate new ones.
        
        Args:
            normalized_names: List of normalized company names
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            n_samples: Number of samples
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = None
        
        # Check for cached embeddings if caching is enabled
        if self.cache_embeddings and not self.force_regenerate:
            cache_key = self._get_cache_key(input_file, self.model_name)
            cache_dir = os.path.join(os.path.dirname(output_file) or '.', '.embedding_cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
            cache_meta_path = os.path.join(cache_dir, f"{cache_key}.json")
            
            # Check if cached embeddings exist
            if os.path.exists(cache_path) and os.path.exists(cache_meta_path):
                try:
                    with open(cache_meta_path, 'r') as f:
                        cache_meta = json.load(f)
                    
                    # Verify cache is valid (same model, same number of samples)
                    if (cache_meta.get('model_name') == self.model_name and 
                        cache_meta.get('n_samples') == n_samples):
                        print_progress(f"Loading cached embeddings from {cache_path}...", self.verbose)
                        
                        # Load embeddings (memory-mapped if large file)
                        if self.use_memmap:
                            embedding_dim = cache_meta.get('embedding_dim', 384)
                            embeddings = np.memmap(
                                cache_path,
                                dtype='float32',
                                mode='r',
                                shape=(n_samples, embedding_dim)
                            )
                        else:
                            embeddings = np.load(cache_path, mmap_mode='r' if n_samples > 500000 else None)
                        
                        cache_age = time.time() - cache_meta.get('created', 0)
                        print_progress(f"Cache loaded successfully (age: {cache_age/3600:.1f} hours)", self.verbose)
                    else:
                        print_progress("Cache invalid (model or data changed), regenerating...", self.verbose)
                        embeddings = None
                except Exception as e:
                    print_progress(f"Error loading cache: {e}, regenerating...", self.verbose)
                    embeddings = None
        
        # Generate embeddings if not loaded from cache
        if embeddings is None:
            # Determine cache path for saving
            cache_path = None
            if self.cache_embeddings:
                cache_key = self._get_cache_key(input_file, self.model_name)
                cache_dir = os.path.join(os.path.dirname(output_file) or '.', '.embedding_cache')
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
                cache_meta_path = os.path.join(cache_dir, f"{cache_key}.json")
            
            embeddings = self._generate_embeddings(normalized_names, cache_path)
            
            # Save cache metadata
            if cache_path and self.cache_embeddings:
                embedding_dim = embeddings.shape[1]
                cache_meta = {
                    'model_name': self.model_name,
                    'n_samples': n_samples,
                    'embedding_dim': embedding_dim,
                    'created': time.time()
                }
                try:
                    with open(cache_meta_path, 'w') as f:
                        json.dump(cache_meta, f)
                except Exception as e:
                    print_progress(f"Warning: Could not save cache metadata: {e}", self.verbose)
        
        return embeddings
    
    def _reduce_embeddings(
        self,
        embeddings: np.ndarray,
        input_file: str,
        output_file: str,
        n_samples: int
    ) -> np.ndarray:
        """
        Apply PCA dimension reduction to embeddings.
        
        Args:
            embeddings: Original embeddings (n_samples, original_dim)
            input_file: Path to input CSV file (for cache key)
            output_file: Path to output CSV file (for cache location)
            n_samples: Number of samples
            
        Returns:
            Reduced embeddings (n_samples, reduced_dim)
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "scikit-learn is required for dimension reduction. "
                "Install with: pip install scikit-learn"
            )
        
        original_dim = embeddings.shape[1]
        target_dim = self.reduce_dimensions
        
        # Validate target dimension
        if target_dim >= original_dim:
            print_progress(
                f"Warning: Target dimension ({target_dim}) >= original ({original_dim}), skipping reduction",
                self.verbose
            )
            return embeddings
        
        if target_dim < 32:
            print_progress(
                f"Warning: Target dimension ({target_dim}) is very low, may affect quality",
                self.verbose
            )
        
        # Check for cached reduced embeddings
        cache_key = None
        cache_path = None
        cache_meta_path = None
        
        if self.cache_embeddings:
            # Create cache key based on original embeddings hash
            cache_key_base = self._get_cache_key(input_file, self.model_name)
            cache_key = f"{cache_key_base}_reduced_{target_dim}_{self.preserve_variance}"
            cache_dir = os.path.join(os.path.dirname(output_file) or '.', '.embedding_cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
            cache_meta_path = os.path.join(cache_dir, f"{cache_key}.json")
            
            # Try to load cached reduced embeddings
            if os.path.exists(cache_path) and os.path.exists(cache_meta_path) and not self.force_regenerate:
                try:
                    with open(cache_meta_path, 'r') as f:
                        cache_meta = json.load(f)
                    
                    if (cache_meta.get('original_dim') == original_dim and
                        cache_meta.get('reduced_dim') == target_dim and
                        cache_meta.get('n_samples') == n_samples):
                        print_progress(f"Loading cached reduced embeddings from {cache_path}...", self.verbose)
                        
                        if self.use_memmap:
                            reduced_embeddings = np.memmap(
                                cache_path,
                                dtype='float32',
                                mode='r',
                                shape=(n_samples, target_dim)
                            )
                        else:
                            reduced_embeddings = np.load(cache_path, mmap_mode='r' if n_samples > 500000 else None)
                        
                        variance_explained = cache_meta.get('variance_explained', 0.0)
                        print_progress(
                            f"Reduced embeddings loaded: {original_dim}D -> {target_dim}D "
                            f"(variance explained: {variance_explained:.2%})",
                            self.verbose
                        )
                        return reduced_embeddings
                except Exception as e:
                    print_progress(f"Error loading reduced cache: {e}, regenerating...", self.verbose)
        
        # Apply PCA reduction
        print_progress(
            f"Reducing embeddings: {original_dim}D -> {target_dim}D using PCA...",
            self.verbose
        )
        
        # Determine PCA parameters
        if self.preserve_variance > 0 and self.preserve_variance < 1.0:
            # Use variance-based selection
            pca = PCA(n_components=self.preserve_variance)
            pca.fit(embeddings)
            
            # Find number of components needed
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= self.preserve_variance) + 1
            
            # Use the smaller of target_dim or n_components needed for variance
            actual_dim = min(target_dim, n_components)
            
            print_progress(
                f"PCA analysis: {actual_dim} components preserve {cumsum_variance[actual_dim-1]:.2%} variance",
                self.verbose
            )
            
            pca = PCA(n_components=actual_dim)
        else:
            # Use fixed dimension
            pca = PCA(n_components=target_dim)
            actual_dim = target_dim
        
        # Fit and transform
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Calculate variance explained
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        print_progress(
            f"Dimension reduction complete: {original_dim}D -> {actual_dim}D "
            f"(variance explained: {variance_explained:.2%}, "
            f"memory saved: {(1 - actual_dim/original_dim)*100:.1f}%)",
            self.verbose
        )
        
        # Save to cache if enabled
        if cache_path and self.cache_embeddings:
            print_progress(f"Saving reduced embeddings to cache: {cache_path}...", self.verbose)
            np.save(cache_path, reduced_embeddings)
            
            cache_meta = {
                'original_dim': original_dim,
                'reduced_dim': actual_dim,
                'n_samples': n_samples,
                'variance_explained': float(variance_explained),
                'preserve_variance': self.preserve_variance,
                'created': time.time()
            }
            try:
                with open(cache_meta_path, 'w') as f:
                    json.dump(cache_meta, f)
            except Exception as e:
                print_progress(f"Warning: Could not save reduction cache metadata: {e}", self.verbose)
        
        return reduced_embeddings

