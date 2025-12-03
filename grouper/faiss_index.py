"""FAISS index building and approximate nearest neighbor search."""

import numpy as np
import faiss
from typing import Tuple, Optional
from .utils import check_faiss_gpu, print_progress


class FAISSIndex:
    """FAISS-based approximate nearest neighbor index for similarity search."""
    
    def __init__(
        self,
        embeddings: np.ndarray,
        index_type: str = "auto",
        use_gpu: Optional[bool] = None,
        verbose: bool = True
    ):
        """
        Initialize and build FAISS index.
        
        Args:
            embeddings: Normalized embedding vectors (n_samples, embedding_dim)
            index_type: Type of index ('auto', 'flat', 'hnsw', 'ivf', 'ivfpq')
            use_gpu: Whether to use GPU (auto-detect if None)
            verbose: Whether to print progress messages
        """
        self.embeddings = embeddings
        self.n_samples, self.dimension = embeddings.shape
        self.verbose = verbose
        self.index_type = index_type
        
        # Auto-detect GPU
        if use_gpu is None:
            use_gpu = check_faiss_gpu() and self.n_samples > 100000
        
        self.use_gpu = use_gpu
        self.index = None
        self.gpu_index = None
        self.gpu_resources = None
        
        self._build_index()
    
    def _build_index(self):
        """Build the FAISS index based on dataset size and type selection."""
        print_progress(f"Building FAISS index for {self.n_samples} vectors...", self.verbose)
        
        # Auto-select index type if needed
        if self.index_type == "auto":
            if self.n_samples < 500000:
                # Use flat index for smaller datasets (exact search)
                index_type_actual = "flat"
            elif self.n_samples < 2000000:
                # Use HNSW for medium-large datasets (better recall than IVF)
                index_type_actual = "hnsw"
            elif self.n_samples < 5000000:
                # Use IVF for very large datasets (2M-5M) for memory efficiency
                index_type_actual = "ivf"
            else:
                # Use IVFPQ for extremely large datasets (>5M) for maximum memory efficiency
                index_type_actual = "ivfpq"
        else:
            index_type_actual = self.index_type
        
        print_progress(f"Selected index type: {index_type_actual}", self.verbose)
        
        # Build index based on type
        if index_type_actual == "flat":
            # Flat index with inner product (cosine similarity on normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type_actual == "hnsw":
            # HNSW index for approximate nearest neighbor search
            # M=32 is a good default for balance between accuracy and speed
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            # Set ef_construction for better recall
            # For very large datasets, increase ef_construction for better quality
            if self.n_samples > 1000000:
                self.index.hnsw.efConstruction = 300  # Higher quality for large datasets
            else:
                self.index.hnsw.efConstruction = 200  # Default
            # Set ef_search for query time (will be set per query, but set default)
            if hasattr(self.index.hnsw, 'efSearch'):
                self.index.hnsw.efSearch = min(200, self.n_samples // 10000) if self.n_samples > 100000 else 100
        elif index_type_actual == "ivf":
            # IVF index for very large datasets
            nlist = min(4096, int(np.sqrt(self.n_samples)))  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            # Increase nprobe for very large datasets to improve recall
            # For 1M+ records, search more clusters to find similar companies
            if self.n_samples > 1000000:
                # Search up to 50% of clusters for better recall
                self.index.nprobe = min(128, max(64, nlist // 2))
            elif self.n_samples > 500000:
                # Search more clusters for medium-large datasets
                self.index.nprobe = min(64, max(32, nlist // 3))
            else:
                self.index.nprobe = min(32, nlist // 4)  # Default
            print_progress(f"IVF nprobe set to {self.index.nprobe} (searches {self.index.nprobe}/{nlist} clusters)", self.verbose)
        elif index_type_actual == "ivfpq":
            # IVFPQ index with Product Quantization for extremely large datasets
            self.index = self._build_pq_index()
        else:
            raise ValueError(f"Unknown index type: {index_type_actual}")
        
        # Train index if needed (IVF and IVFPQ require training)
        if isinstance(self.index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            print_progress("Training index...", self.verbose)
            self.index.train(self.embeddings)
        
        # Add vectors to index
        print_progress("Adding vectors to index...", self.verbose)
        self.index.add(self.embeddings)
        
        # Move to GPU if requested and available
        if self.use_gpu and check_faiss_gpu():
            print_progress("Moving index to GPU...", self.verbose)
            self.gpu_resources = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
            print_progress("Index moved to GPU", self.verbose)
        else:
            print_progress("Using CPU index", self.verbose)
    
    def _build_pq_index(self):
        """
        Build IndexIVFPQ with Product Quantization for memory-efficient indexing.
        
        Returns:
            FAISS IndexIVFPQ instance
        """
        # Calculate optimal parameters for Product Quantization
        # nlist: number of clusters (similar to IVF)
        nlist = min(4096, int(np.sqrt(self.n_samples)))
        
        # m: number of subquantizers (must divide dimension)
        # Common choices: 8, 16, 32, 64
        # For memory efficiency, use m=8 or m=16 for most cases
        if self.dimension >= 64:
            # For dimensions >= 64, use m=8 or m=16
            # Choose m such that dimension is divisible by m
            if self.dimension % 8 == 0:
                m = 8
            elif self.dimension % 16 == 0:
                m = 16
            elif self.dimension % 32 == 0:
                m = 32
            else:
                # Find largest divisor <= 32
                m = max([d for d in [8, 16, 32] if self.dimension % d == 0], default=8)
        else:
            # For smaller dimensions, use smaller m
            if self.dimension % 4 == 0:
                m = 4
            else:
                m = self.dimension  # Use full dimension if not divisible
        
        # nbits: bits per subquantizer (typically 8)
        # Higher nbits = better quality but more memory
        # For large datasets, 8 bits is a good balance
        nbits = 8
        
        print_progress(f"Building IVFPQ index: nlist={nlist}, m={m}, nbits={nbits}", self.verbose)
        
        # Create quantizer
        quantizer = faiss.IndexFlatIP(self.dimension)
        
        # Create IVFPQ index
        index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
        
        # Set nprobe dynamically based on dataset size
        # For PQ, we can use slightly lower nprobe since PQ already provides compression
        if self.n_samples > 5000000:
            # For 5M+ records, search more clusters
            index.nprobe = min(128, max(64, nlist // 2))
        elif self.n_samples > 2000000:
            # For 2M-5M records
            index.nprobe = min(64, max(32, nlist // 3))
        else:
            # For smaller datasets using PQ
            index.nprobe = min(32, max(16, nlist // 4))
        
        print_progress(f"IVFPQ nprobe set to {index.nprobe} (searches {index.nprobe}/{nlist} clusters)", self.verbose)
        
        return index
    
    def _optimize_nprobe(self, k: int, threshold: float) -> int:
        """
        Dynamically optimize nprobe based on query parameters and dataset characteristics.
        
        Args:
            k: Number of neighbors to retrieve
            threshold: Similarity threshold
            
        Returns:
            Optimized nprobe value
        """
        # Only applicable for IVF and IVFPQ indices
        if not isinstance(self.index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            return None
        
        current_nprobe = self.index.nprobe
        nlist = self.index.nlist if hasattr(self.index, 'nlist') else 4096
        
        # Base optimization: adjust nprobe based on k
        # Higher k requires searching more clusters
        if k > 200:
            # For very high k, search more clusters
            optimal_nprobe = min(nlist, max(current_nprobe, int(nlist * 0.5)))
        elif k > 100:
            # For high k, search moderate number of clusters
            optimal_nprobe = min(nlist, max(current_nprobe, int(nlist * 0.3)))
        elif k < 20:
            # For low k, can search fewer clusters
            optimal_nprobe = max(1, min(current_nprobe, int(nlist * 0.1)))
        else:
            # For moderate k, use current nprobe
            optimal_nprobe = current_nprobe
        
        # Adjust based on threshold
        # Lower thresholds need more clusters searched (more candidates)
        if threshold < 0.70:
            optimal_nprobe = min(nlist, int(optimal_nprobe * 1.5))
        elif threshold > 0.90:
            # High threshold means fewer matches, can search fewer clusters
            optimal_nprobe = max(1, int(optimal_nprobe * 0.8))
        
        # Clamp to reasonable bounds
        optimal_nprobe = max(1, min(nlist, optimal_nprobe))
        
        return optimal_nprobe
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 50,
        threshold: float = 0.70,
        dynamic_nprobe: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embeddings: Query vectors (n_queries, embedding_dim)
            k: Number of neighbors to retrieve
            threshold: Minimum similarity threshold (filter results)
            dynamic_nprobe: Whether to dynamically optimize nprobe (default: True)
            
        Returns:
            Tuple of (distances, indices) where:
            - distances: Similarity scores (n_queries, k)
            - indices: Indices of nearest neighbors (n_queries, k)
        """
        # Use GPU index if available, otherwise CPU
        index_to_use = self.gpu_index if self.gpu_index is not None else self.index
        
        # Optimize nprobe dynamically if enabled and applicable
        original_nprobe = None
        if dynamic_nprobe and isinstance(self.index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            optimal_nprobe = self._optimize_nprobe(k, threshold)
            if optimal_nprobe is not None and optimal_nprobe != self.index.nprobe:
                original_nprobe = self.index.nprobe
                self.index.nprobe = optimal_nprobe
                # Also update GPU index if it exists
                if self.gpu_index is not None:
                    self.gpu_index.nprobe = optimal_nprobe
        
        # Search
        k = min(k, self.n_samples)  # Can't retrieve more than total samples
        distances, indices = index_to_use.search(query_embeddings, k)
        
        # Restore original nprobe if it was changed
        if original_nprobe is not None:
            self.index.nprobe = original_nprobe
            if self.gpu_index is not None:
                self.gpu_index.nprobe = original_nprobe
        
        # Filter by threshold (distances are inner products, so >= threshold)
        # For cosine similarity on normalized vectors, inner product = cosine similarity
        mask = distances >= threshold
        filtered_distances = np.where(mask, distances, 0.0)
        filtered_indices = np.where(mask, indices, -1)
        
        return filtered_distances, filtered_indices
    
    def search_single(
        self,
        query_embedding: np.ndarray,
        k: int = 50,
        threshold: float = 0.70,
        dynamic_nprobe: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors of a single query vector.
        
        Args:
            query_embedding: Single query vector (embedding_dim,)
            k: Number of neighbors to retrieve
            threshold: Minimum similarity threshold
            dynamic_nprobe: Whether to dynamically optimize nprobe (default: True)
            
        Returns:
            Tuple of (distances, indices) - 1D arrays
        """
        # Reshape to 2D for search
        query_2d = query_embedding.reshape(1, -1)
        distances, indices = self.search(query_2d, k, threshold, dynamic_nprobe)
        
        # Return 1D arrays
        return distances[0], indices[0]
    
    def save_to_disk(self, index_path: str):
        """Save FAISS index to disk."""
        # Save the CPU index (GPU index can be reconstructed)
        index_to_save = self.index
        faiss.write_index(index_to_save, index_path)
        print_progress(f"Saved FAISS index to {index_path}", self.verbose)
    
    @classmethod
    def load_from_disk(
        cls, 
        index_path: str, 
        dimension: int,
        n_samples: int,
        index_type: str = "auto",
        use_gpu: Optional[bool] = None,
        verbose: bool = True
    ):
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to saved FAISS index file
            dimension: Embedding dimension
            n_samples: Number of samples in the index
            index_type: Type of index (for compatibility)
            use_gpu: Whether to use GPU (auto-detect if None)
            verbose: Whether to print progress messages
            
        Returns:
            FAISSIndex instance
        """
        print_progress(f"Loading FAISS index from {index_path}...", verbose)
        index = faiss.read_index(index_path)
        
        # Auto-detect GPU
        if use_gpu is None:
            use_gpu = check_faiss_gpu() and n_samples > 100000
        
        instance = cls.__new__(cls)
        instance.index = index
        instance.dimension = dimension
        instance.n_samples = n_samples
        instance.verbose = verbose
        instance.index_type = index_type
        instance.use_gpu = use_gpu
        instance.embeddings = None  # Not stored, will be loaded separately
        
        # Move to GPU if requested and available
        if use_gpu and check_faiss_gpu():
            print_progress("Moving index to GPU...", verbose)
            instance.gpu_resources = faiss.StandardGpuResources()
            instance.gpu_index = faiss.index_cpu_to_gpu(instance.gpu_resources, 0, index)
            print_progress("Index moved to GPU", verbose)
        else:
            instance.gpu_index = None
            instance.gpu_resources = None
            print_progress("Using CPU index", verbose)
        
        return instance

