"""
Vector store for storing and retrieving document embeddings.
Uses ChromaDB for local vector database.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not available. Install with: pip install chromadb")


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a metadata dict to ChromaDB's scalar-only schema.

    ChromaDB only accepts ``str``/``int``/``float``/``bool``/``None`` metadata
    values. This is the single place that rule is enforced, so ingest callers
    pass domain metadata and stay ignorant of the store's constraint.

    - scalars (``str``/``int``/``float``/``bool``) and ``None`` pass through
    - lists are normalized to a comma-joined string of their elements
    - dicts are dropped (they have no scalar representation)
    - anything else is coerced with ``str()``
    """
    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list):
            cleaned[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            # No scalar representation - drop rather than crash the store.
            continue
        else:
            cleaned[key] = str(value)
    return cleaned


class VectorStore:
    """
    Vector store using ChromaDB for document embeddings.
    Stores documents with metadata for RAG retrieval.
    """
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "callet_knowledge"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database. If None, uses in-memory.
            collection_name: Name of the ChromaDB collection to use.
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        if persist_directory is None:
            # Default to data/vector_db/ relative to project root
            _project_root = Path(__file__).resolve().parent.parent.parent
            persist_directory = str(_project_root / "data" / "vector_db")
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Callet teaching knowledge base"}
        )
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks to store
            embeddings: List of embedding vectors (must match texts length)
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document. Auto-generated if not provided.
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        
        if ids is None:
            ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        else:
            # The store owns the scalar-only schema: callers pass domain
            # metadata and this layer normalizes it (see clean_metadata).
            metadatas = [clean_metadata(md) for md in metadatas]

        # Convert embeddings to lists if they're numpy arrays
        embeddings_list = []
        for emb in embeddings:
            if hasattr(emb, 'tolist'):
                embeddings_list.append(emb.tolist())
            else:
                embeddings_list.append(emb)
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Embedding vector of the query
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"era": "TRUMPET_YOGA"})
            include: What to include in results (documents, metadatas, distances, embeddings)
            
        Returns:
            Dictionary with 'ids', 'documents', 'metadatas', 'distances', etc.
        """
        # Convert embedding to list if it's a numpy array
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=include
        )
        
        return results
    
    def get_collection_size(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def topic_exists(self, topic_url: str) -> bool:
        """
        Check if a topic URL already exists in the database.
        
        Args:
            topic_url: URL of the topic to check
            
        Returns:
            True if topic exists, False otherwise
        """
        # Normalize URL (remove session ID for comparison)
        normalized_url = self._normalize_topic_url(topic_url)
        
        # Query for documents with this URL
        results = self.collection.get(
            where={"url": normalized_url},
            limit=1
        )
        return len(results.get('ids', [])) > 0
    
    def get_topic_chunks(self, topic_url: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a given topic URL.
        
        Args:
            topic_url: URL of the topic
            
        Returns:
            List of dictionaries with 'id', 'document', 'metadata' for each chunk
        """
        normalized_url = self._normalize_topic_url(topic_url)
        
        results = self.collection.get(
            where={"url": normalized_url},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        ids = results.get('ids', [])
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        
        for i, doc_id in enumerate(ids):
            chunks.append({
                'id': doc_id,
                'document': documents[i] if i < len(documents) else None,
                'metadata': metadatas[i] if i < len(metadatas) else {}
            })
        
        return chunks
    
    def delete_topic(self, topic_url: str) -> int:
        """
        Delete all chunks for a given topic URL.
        
        Args:
            topic_url: URL of the topic to delete
            
        Returns:
            Number of documents deleted
        """
        chunks = self.get_topic_chunks(topic_url)
        if not chunks:
            return 0
        
        ids_to_delete = [chunk['id'] for chunk in chunks]
        self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    
    def _normalize_topic_url(self, url: str) -> str:
        """Normalize topic URL by removing session ID for comparison."""
        from urllib.parse import urlparse, parse_qs, urlunparse
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        # Remove sid parameter
        if 'sid' in query:
            del query['sid']
        # Rebuild URL
        new_query = '&'.join([f"{k}={v[0]}" for k, v in query.items()])
        new_parsed = parsed._replace(query=new_query)
        return urlunparse(new_parsed)
    
    def find_duplicates(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find duplicate topics in the database (same URL with multiple entries).
        
        Returns:
            Dictionary mapping normalized URLs to lists of chunk information
        """
        # Get all documents with their metadata
        all_docs = self.collection.get(include=["metadatas"])
        
        # Group by URL
        url_groups: Dict[str, List[Dict[str, Any]]] = {}
        
        ids = all_docs.get('ids', [])
        metadatas = all_docs.get('metadatas', [])
        
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            url = metadata.get('url', '')
            
            if url:
                normalized = self._normalize_topic_url(url)
                if normalized not in url_groups:
                    url_groups[normalized] = []
                
                url_groups[normalized].append({
                    'id': doc_id,
                    'metadata': metadata,
                    'url': url
                })
        
        # Return only URLs that have multiple chunks (potential duplicates)
        duplicates = {url: chunks for url, chunks in url_groups.items() if len(chunks) > 1}
        return duplicates
    
    def delete_collection(self):
        """Delete the entire collection (use with caution!)."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Callet teaching knowledge base"}
        )


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store(persist_directory: Optional[str] = None, collection_name: str = "callet_knowledge") -> VectorStore:
    """
    Get or create the singleton vector store instance.
    
    Args:
        persist_directory: Directory to persist the database
        collection_name: Name of the collection
        
    Returns:
        VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(persist_directory=persist_directory, collection_name=collection_name)
    return _vector_store


