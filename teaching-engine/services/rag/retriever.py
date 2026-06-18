"""
Retriever service for RAG - retrieves relevant context from the vector store.
"""

from typing import List, Dict, Optional, Any
import numpy as np

from .embedding_service import get_embedding_service
from .vector_store import get_vector_store


class Retriever:
    """
    Retrieves relevant documents from the vector store based on queries.
    """
    
    def __init__(
        self,
        embedding_service=None,
        vector_store=None,
        top_k: int = 5,
        min_score: float = 0.0
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_service: EmbeddingService instance (auto-created if None)
            vector_store: VectorStore instance (auto-created if None)
            top_k: Number of results to retrieve
            min_score: Minimum similarity score (0-1) for results
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k
        self.min_score = min_score
    
    def retrieve(
        self,
        query: str,
        era: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        exclude_method_books: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query text
            era: Optional era filter (TRUMPET_YOGA, SUPERCHOPS, TCE)
            top_k: Number of results (overrides instance default)
            min_score: Minimum score threshold (overrides instance default)
            exclude_method_books: If True, exclude method books (Arban, Clarke, etc.)
                                 from results. Use this for Callet embouchure teaching.
            
        Returns:
            List of dictionaries with 'text', 'metadata', 'score', etc.
        """
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)
        
        # Build metadata filter
        where = None
        if era:
            where = {"era": era}
        
        # If excluding method books, we'll filter post-query
        # since we need to check is_method_book field
        # ChromaDB doesn't support empty dict for where clause
        
        # Query vector store - request more results if we need to filter
        query_top_k = top_k * 2 if exclude_method_books else top_k
        
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=query_top_k,
            where=where
        )
        
        # Format results
        retrieved = []
        if results.get('documents') and len(results['documents']) > 0:
            documents = results['documents'][0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            ids = results.get('ids', [[]])[0]
            
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                # Skip method books if exclude_method_books is True
                if exclude_method_books:
                    is_method_book = metadata.get('is_method_book', False)
                    # Handle both bool and string representations
                    if is_method_book is True or (isinstance(is_method_book, str) and is_method_book.lower() == 'true'):
                        continue
                
                # Convert distance to similarity score (assuming cosine distance)
                # Distance is 0 for identical, 2 for opposite
                # Similarity is 1 - (distance / 2)
                distance = distances[i] if i < len(distances) else 1.0
                score = 1.0 - (distance / 2.0) if distance <= 2.0 else 0.0
                
                if score >= min_score:
                    retrieved.append({
                        'text': doc,
                        'metadata': metadata,
                        'score': score,
                        'distance': distance,
                        'id': ids[i] if i < len(ids) else None
                    })
                    
                    # Stop once we have enough results
                    if len(retrieved) >= top_k:
                        break
        
        return retrieved
    
    def format_context(self, retrieved: List[Dict[str, Any]], max_chars: int = 2000) -> str:
        """
        Format retrieved documents into context string for the LLM.
        
        Args:
            retrieved: List of retrieved documents (from retrieve())
            max_chars: Maximum characters in the context
            
        Returns:
            Formatted context string
        """
        if not retrieved:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved):
            text = doc['text']
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'unknown')
            url = metadata.get('url', '')
            filename = metadata.get('filename', '')
            topic_title = metadata.get('topic_title', '')
            page_number = metadata.get('page_number') or metadata.get('page')
            chunk_index = metadata.get('chunk_index')
            era = metadata.get('era')
            
            # Build citation with URL for reference
            citation_parts = []
            if source == 'media':
                media_title = metadata.get('media_title', 'unknown media')
                speaker_name = metadata.get('speaker_name', '')
                citation_parts.append(media_title)
                if speaker_name and speaker_name != 'UNKNOWN':
                    citation_parts.append(f"{speaker_name} speaking")
                if era:
                    citation_parts.append(f"({era} era)")
            elif filename:
                citation_parts.append(filename)
                if page_number:
                    citation_parts.append(f"page {page_number}")
                if era:
                    citation_parts.append(f"({era} era)")
            elif url:
                if topic_title and source == 'forum':
                    citation_parts.append(f"{topic_title}")
                    citation_parts.append(url)
                else:
                    citation_parts.append(url)
                if era:
                    citation_parts.append(f"({era} era)")
            else:
                citation_parts.append(source)
                if era:
                    citation_parts.append(f"({era} era)")
            
            citation = f"[Source: {' - '.join(citation_parts)}]"
            
            chunk_text = f"{citation}\n{text}\n"
            
            if current_length + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n---\n\n".join(context_parts)


# Singleton instance
_retriever: Optional[Retriever] = None


def get_retriever(top_k: int = 5, min_score: float = 0.0) -> Retriever:
    """
    Get or create the singleton retriever instance.
    
    Args:
        top_k: Number of results to retrieve
        min_score: Minimum similarity score
        
    Returns:
        Retriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = Retriever(top_k=top_k, min_score=min_score)
    return _retriever

