"""
Embedding service for generating vector embeddings from text.
Uses sentence-transformers for local, offline embeddings.
"""

import os
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")


class EmbeddingService:
    """
    Service for generating text embeddings.
    Uses sentence-transformers model for local embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' (fast, good quality).
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embeddings
        """
        self._load_model()
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        self._load_model()
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        self._load_model()
        # Get dimension by encoding a dummy string
        dummy_embedding = self.model.encode("dummy", convert_to_numpy=True)
        return dummy_embedding.shape[0]


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """
    Get or create the singleton embedding service instance.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name=model_name)
    return _embedding_service


