"""
Qdrant Cache Utility
This module provides helper functions to connect to Qdrant and store
classified query embeddings along with metadata (intent, confidence, etc.)
for analytics and debugging.
"""

import time
import os
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from loguru import logger

# -------------------------------------------------------------
# Connect to Qdrant
# -------------------------------------------------------------
def get_qdrant_client():
    """Initialize and return a Qdrant client connection."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)
    return client


# -------------------------------------------------------------
# Ensure collection exists
# -------------------------------------------------------------
def ensure_collection_exists(client: QdrantClient, collection_name: str = "chatnshop_products"):
    """Creates the Qdrant collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        logger.info(f"Creating Qdrant collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{collection_name}' created successfully.")
    else:
        logger.debug(f"Qdrant collection '{collection_name}' already exists.")


# -------------------------------------------------------------
# Store vector in Qdrant
# -------------------------------------------------------------
def store_vector(intent: str, vector: list, confidence: float, status: str, variant: str, query: str):
    """
    Stores an embedding vector and its metadata in the Qdrant collection.
    Automatically ensures the collection exists.
    """
    try:
        client = get_qdrant_client()
        collection_name = "chatnshop_products"
        ensure_collection_exists(client, collection_name)

        # Create unique ID for this record
        point_id = int(time.time() * 1000000)

        # Metadata payload
        payload = {
            "intent": intent,
            "confidence": confidence,
            "status": status,
            "variant": variant,
            "query": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Prepare the Qdrant point
        point = PointStruct(id=point_id, vector=vector, payload=payload)

        # Store in Qdrant
        client.upsert(collection_name=collection_name, points=[point])

        logger.info(f"Stored vector in Qdrant ({collection_name})", extra={"payload": payload})
        return True

    except Exception as e:
        logger.error(f"Failed to store vector in Qdrant: {e}", exc_info=True)
        return False


# -------------------------------------------------------------
# Optional: Fetch recent records (for debugging)
# -------------------------------------------------------------
def fetch_recent_records(limit: int = 5, collection_name: str = "chatnshop_products"):
    """
    Retrieve the most recent stored queries from Qdrant.
    Useful for debugging or verifying that storage works.
    """
    try:
        client = get_qdrant_client()
        ensure_collection_exists(client, collection_name)

        result = client.scroll(collection_name=collection_name, limit=limit)
        points = result[0] if result else []

        logger.info(f"Retrieved {len(points)} recent records from Qdrant")
        for p in points:
            logger.debug(f"Record payload: {p.payload}")
        return points

    except Exception as e:
        logger.error(f"Error fetching records from Qdrant: {e}", exc_info=True)
        return []
