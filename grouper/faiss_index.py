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
            index_type: Type of index ('auto', 'flat', 'hnsw', 'ivf')
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
            if self.n_samples < 2000000:
                # Use flat index for smaller datasets (exact search)
                index_type_actual = "flat"
            else:
                # Use HNSW for larger datasets (approximate search)
                index_type_actual = "hnsw"
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
            # Set ef_construction for better recall (default is 200)
            self.index.hnsw.efConstruction = 200
        elif index_type_actual == "ivf":
            # IVF index for very large datasets
            nlist = min(4096, int(np.sqrt(self.n_samples)))  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.nprobe = min(32, nlist // 4)  # Number of clusters to search
        else:
            raise ValueError(f"Unknown index type: {index_type_actual}")
        
        # Train index if needed (IVF requires training)
        if isinstance(self.index, faiss.IndexIVFFlat):
            print_progress("Training IVF index...", self.verbose)
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
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 50,
        threshold: float = 0.70
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embeddings: Query vectors (n_queries, embedding_dim)
            k: Number of neighbors to retrieve
            threshold: Minimum similarity threshold (filter results)
            
        Returns:
            Tuple of (distances, indices) where:
            - distances: Similarity scores (n_queries, k)
            - indices: Indices of nearest neighbors (n_queries, k)
        """
        # Use GPU index if available, otherwise CPU
        index_to_use = self.gpu_index if self.gpu_index is not None else self.index
        
        # Search
        k = min(k, self.n_samples)  # Can't retrieve more than total samples
        distances, indices = index_to_use.search(query_embeddings, k)
        
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
        threshold: float = 0.70
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors of a single query vector.
        
        Args:
            query_embedding: Single query vector (embedding_dim,)
            k: Number of neighbors to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (distances, indices) - 1D arrays
        """
        # Reshape to 2D for search
        query_2d = query_embedding.reshape(1, -1)
        distances, indices = self.search(query_2d, k, threshold)
        
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

