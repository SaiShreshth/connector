"""
Async Redis Connection Helper
Provides async Redis client with connection pooling for FastAPI async endpoints.
"""

import os
from typing import Optional
from loguru import logger

# Redis 5.0+ supports redis.asyncio
aioredis = None
ASYNC_REDIS_AVAILABLE = False

try:
    import redis.asyncio as aioredis
    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    try:
        import aioredis
        ASYNC_REDIS_AVAILABLE = True
    except ImportError:
        logger.warning("Async Redis not available - redis.asyncio not found. Using sync Redis fallback.")

if aioredis is not None:
    _async_redis_client: Optional[aioredis.Redis] = None
    _async_redis_pool: Optional[aioredis.ConnectionPool] = None
else:
    _async_redis_client = None
    _async_redis_pool = None


async def get_async_redis_client():
    """
    Get or create singleton async Redis client with connection pool.
    
    Returns:
        Async Redis client instance or None if connection fails
    """
    global _async_redis_client, _async_redis_pool
    
    if not ASYNC_REDIS_AVAILABLE or aioredis is None:
        return None
    
    if _async_redis_client is not None:
        return _async_redis_client
    
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        
        # Create connection pool
        _async_redis_pool = aioredis.ConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=max_connections,
            retry_on_timeout=True
        )
        
        # Create client from pool
        _async_redis_client = aioredis.Redis(connection_pool=_async_redis_pool)
        
        # Test connection
        await _async_redis_client.ping()
        logger.info(f"Async Redis client initialized with connection pool (max_connections={max_connections})")
        
        return _async_redis_client
        
    except Exception as e:
        logger.error(f"Failed to initialize async Redis client: {e}")
        _async_redis_client = None
        _async_redis_pool = None
        return None


async def close_async_redis() -> None:
    """Close async Redis client and connection pool."""
    global _async_redis_client, _async_redis_pool
    
    if _async_redis_client:
        try:
            await _async_redis_client.aclose()
        except Exception:
            pass
        _async_redis_client = None
    
    if _async_redis_pool:
        try:
            await _async_redis_pool.aclose()
        except Exception:
            pass
        _async_redis_pool = None

