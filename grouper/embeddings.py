"""Embedding generation using sentence transformers with batching."""

import os
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .utils import get_device, print_progress


class EmbeddingGenerator:
    """Generate embeddings for company names using sentence transformers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        use_memmap: bool = False,
        memmap_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cuda' or 'cpu'), auto-detected if None
            batch_size: Batch size for embedding generation (0 for auto-tuning)
            use_memmap: Whether to use memory-mapped files for embeddings
            memmap_path: Path for memory-mapped file (auto-generated if None)
            verbose: Whether to print progress messages
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.batch_size = batch_size
        self.use_memmap = use_memmap
        self.memmap_path = memmap_path
        self.verbose = verbose
        
        print_progress(f"Loading model: {model_name}", self.verbose)
        print_progress(f"Using device: {self.device}", self.verbose)
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        
        # Get embedding dimension for memory-mapped file creation
        self._embedding_dim = None
    
    def _calculate_optimal_batch_size(self, num_texts: int, embedding_dim: int) -> int:
        """
        Calculate optimal batch size based on available memory and dataset size.
        
        Args:
            num_texts: Number of texts to process
            embedding_dim: Dimension of embeddings
            
        Returns:
            Optimal batch size
        """
        if self.batch_size > 0:
            return self.batch_size
        
        # Estimate memory requirements
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            # Fallback if psutil not available
            available_memory_gb = 4.0  # Conservative default
        
        # Estimate memory per batch (embeddings + model overhead)
        # Each float32 embedding: 4 bytes * embedding_dim
        # Model overhead: ~500MB for typical sentence transformer
        model_overhead_gb = 0.5
        memory_per_sample_mb = (embedding_dim * 4) / (1024**2)  # MB per sample
        
        # Reserve 2GB for system and other operations
        usable_memory_gb = max(1.0, available_memory_gb - model_overhead_gb - 2.0)
        usable_memory_mb = usable_memory_gb * 1024
        
        # Calculate batch size: use at most 50% of available memory per batch
        max_batch_size = int((usable_memory_mb * 0.5) / memory_per_sample_mb)
        
        # Clamp to reasonable bounds
        optimal_batch_size = max(8, min(max_batch_size, 256))
        
        # For very large datasets, use larger batches for efficiency
        if num_texts > 1000000:
            optimal_batch_size = max(optimal_batch_size, 64)
        elif num_texts > 500000:
            optimal_batch_size = max(optimal_batch_size, 32)
        
        if self.verbose:
            print_progress(f"Auto-tuned batch size: {optimal_batch_size} (available memory: {available_memory_gb:.2f} GB)", self.verbose)
        
        return optimal_batch_size
    
    def generate_embeddings(
        self, 
        texts: List[str],
        memmap_path: Optional[str] = None
    ) -> Union[np.ndarray, np.memmap]:
        """
        Generate embeddings for a list of texts in batches.
        
        Args:
            texts: List of text strings to embed
            memmap_path: Optional path for memory-mapped file (overrides use_memmap setting)
            
        Returns:
            Numpy array or memory-mapped array of shape (n_texts, embedding_dim) with normalized embeddings
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        non_empty_texts = [text if text else " " for text in texts]
        num_texts = len(non_empty_texts)
        
        print_progress(f"Generating embeddings for {num_texts} texts...", self.verbose)
        
        # Get embedding dimension
        if self._embedding_dim is None:
            self._embedding_dim = self.get_embedding_dim()
        
        embedding_dim = self._embedding_dim
        
        # Calculate optimal batch size
        effective_batch_size = self._calculate_optimal_batch_size(num_texts, embedding_dim)
        
        # Determine if we should use memory-mapped files
        use_memmap = self.use_memmap or (memmap_path is not None)
        if use_memmap and memmap_path is None:
            # Generate default path if not provided
            memmap_path = self.memmap_path or 'embeddings_memmap.npy'
        
        # Create memory-mapped file if needed
        if use_memmap:
            print_progress(f"Using memory-mapped file: {memmap_path}", self.verbose)
            # Remove existing file if it exists
            if os.path.exists(memmap_path):
                os.remove(memmap_path)
            
            # Create memory-mapped array
            embeddings = np.memmap(
                memmap_path,
                dtype='float32',
                mode='w+',
                shape=(num_texts, embedding_dim)
            )
            all_embeddings = None
        else:
            # Use regular numpy array - accumulate batches
            all_embeddings = []
            embeddings = None
        
        # Generate embeddings in batches
        for i in tqdm(
            range(0, num_texts, effective_batch_size),
            desc="Embedding batches",
            disable=not self.verbose
        ):
            end_idx = min(i + effective_batch_size, num_texts)
            batch = non_empty_texts[i:end_idx]
            
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            if use_memmap:
                # Write directly to memory-mapped file
                embeddings[i:end_idx] = batch_embeddings
            else:
                # Accumulate in list for later concatenation
                all_embeddings.append(batch_embeddings)
        
        if not use_memmap:
            # Concatenate all batches
            embeddings = np.vstack(all_embeddings)
        
        print_progress(f"Generated embeddings shape: {embeddings.shape}", self.verbose)
        
        if use_memmap:
            # Flush to disk
            embeddings.flush()
            print_progress(f"Embeddings saved to memory-mapped file: {memmap_path}", self.verbose)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        # Generate a dummy embedding to get dimension
        dummy_embedding = self.model.encode(["dummy"], convert_to_numpy=True)
        self._embedding_dim = dummy_embedding.shape[1]
        return self._embedding_dim

