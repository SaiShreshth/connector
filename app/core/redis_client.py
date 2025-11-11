"""
Redis connection helper with connection pooling
Handles connecting to Redis with graceful fallback and connection pooling.
"""

import os
import redis
from typing import Optional
from loguru import logger

# Default Redis URL (you can override via environment variable)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))

_redis_client: Optional[redis.Redis] = None
_redis_pool: Optional[redis.ConnectionPool] = None


def get_redis_client() -> redis.Redis:
    """
    Returns a singleton Redis client with connection pooling.
    Raises an exception if the connection fails.
    """
    global _redis_client, _redis_pool

    if _redis_client is not None:
        return _redis_client

    try:
        # Create connection pool for better performance
        _redis_pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            decode_responses=True,
            max_connections=REDIS_MAX_CONNECTIONS,
            retry_on_timeout=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        # Create client from pool
        _redis_client = redis.Redis(connection_pool=_redis_pool)
        
        # Test the connection
        _redis_client.ping()
        logger.info(f"Redis connected to {REDIS_URL} (pool size: {REDIS_MAX_CONNECTIONS})")
        return _redis_client
    except Exception as e:
        logger.error(f"Could not connect to Redis at {REDIS_URL}: {e}")
        raise

if __name__ == "__main__":
    try:
        client = get_redis_client()
        logger.info("Redis connection test successful")
        logger.debug(f"Ping response: {client.ping()}")
    except Exception as e:
        logger.error(f"Redis connection test failed: {e}")

