"""
Embedding Computation Cache
Caches computed embeddings for identical queries to avoid redundant computation.
"""

import hashlib
from typing import Dict, Optional
import numpy as np
from loguru import logger

# In-memory cache for embeddings (query hash -> embedding)
_embedding_cache: Dict[str, np.ndarray] = {}
EMBEDDING_CACHE_SIZE = 10000  # Maximum cached embeddings


def get_embedding_cache_key(query: str) -> str:
    """
    Generate cache key for query.
    
    Args:
        query: Normalized query string
        
    Returns:
        Cache key (hash of query)
    """
    return hashlib.md5(query.encode('utf-8')).hexdigest()


def get_cached_embedding(query: str) -> Optional[np.ndarray]:
    """
    Get cached embedding for a query if it exists.
    
    Args:
        query: Normalized query string
        
    Returns:
        Cached embedding or None if not found
    """
    cache_key = get_embedding_cache_key(query)
    return _embedding_cache.get(cache_key)


def cache_embedding(query: str, embedding: np.ndarray) -> None:
    """
    Cache an embedding for a query.
    
    Args:
        query: Normalized query string
        embedding: Computed embedding vector
    """
    cache_key = get_embedding_cache_key(query)
    
    # Limit cache size (simple FIFO eviction)
    if len(_embedding_cache) >= EMBEDDING_CACHE_SIZE:
        # Remove oldest entry (first key)
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]
        logger.debug(f"Evicted embedding cache entry (cache size limit)")
    
    _embedding_cache[cache_key] = embedding


def clear_embedding_cache() -> None:
    """Clear all cached embeddings."""
    global _embedding_cache
    _embedding_cache.clear()
    logger.info("Cleared embedding cache")


def get_embedding_cache_size() -> int:
    """Get current embedding cache size."""
    return len(_embedding_cache)

