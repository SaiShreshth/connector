"""
Qdrant Client Singleton with Connection Pooling
Provides a shared Qdrant client instance with optimized connection settings.
"""

import os
from typing import Optional
from qdrant_client import QdrantClient
from loguru import logger

_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> Optional[QdrantClient]:
    """
    Get or create singleton Qdrant client with connection pooling.
    Qdrant client internally manages connection pooling for HTTP/gRPC.
    
    Returns:
        QdrantClient instance or None if connection fails
    """
    global _qdrant_client
    
    if _qdrant_client is not None:
        return _qdrant_client
    
    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Qdrant client uses connection pooling internally
        # Configure with reasonable timeouts and connection settings
        client_kwargs = {
            "url": qdrant_url,
            "timeout": 10,
            "prefer_grpc": False,  # HTTP is fine for most cases, gRPC can be enabled if needed
        }
        
        if qdrant_api_key:
            client_kwargs["api_key"] = qdrant_api_key
        
        _qdrant_client = QdrantClient(**client_kwargs)
        
        # Test connection
        _qdrant_client.get_collections()
        logger.info(f"Qdrant client initialized: {qdrant_url} (with connection pooling)")
        return _qdrant_client
        
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        return None

