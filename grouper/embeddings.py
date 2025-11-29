"""Embedding generation using sentence transformers with batching."""

import numpy as np
from typing import List, Optional
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
        verbose: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cuda' or 'cpu'), auto-detected if None
            batch_size: Batch size for embedding generation
            verbose: Whether to print progress messages
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.batch_size = batch_size
        self.verbose = verbose
        
        print_progress(f"Loading model: {model_name}", self.verbose)
        print_progress(f"Using device: {self.device}", self.verbose)
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts in batches.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim) with normalized embeddings
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        non_empty_texts = [text if text else " " for text in texts]
        
        print_progress(f"Generating embeddings for {len(non_empty_texts)} texts...", self.verbose)
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in tqdm(
            range(0, len(non_empty_texts), self.batch_size),
            desc="Embedding batches",
            disable=not self.verbose
        ):
            batch = non_empty_texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        print_progress(f"Generated embeddings shape: {embeddings.shape}", self.verbose)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Generate a dummy embedding to get dimension
        dummy_embedding = self.model.encode(["dummy"], convert_to_numpy=True)
        return dummy_embedding.shape[1]

