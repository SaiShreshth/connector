"""
Cache Management API Endpoints

Provides REST API endpoints for managing and monitoring the LLM response cache.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging

from app.ai.llm_intent.response_cache import get_response_cache
from app.ai.llm_intent.cache_metrics import get_cache_metrics

logger = logging.getLogger("api.cache")

router = APIRouter(prefix="/cache", tags=["cache"])


# Response models
class CacheMetricsResponse(BaseModel):
    """Cache performance metrics."""
    hit_rate_percent: float = Field(..., description="Cache hit rate as percentage (0-100)")
    miss_rate_percent: float = Field(..., description="Cache miss rate as percentage (0-100)")
    total_requests: int = Field(..., description="Total cache lookup requests")
    cache_hits: int = Field(..., description="Number of cache hits")
    cache_misses: int = Field(..., description="Number of cache misses")
    api_calls_saved: int = Field(..., description="Number of API calls saved by cache")
    avg_latency_ms: float = Field(..., description="Average cache lookup latency in milliseconds")
    p95_latency_ms: float = Field(..., description="95th percentile cache lookup latency in milliseconds")
    cache_size: int = Field(..., description="Current number of cached entries")
    uptime_seconds: float = Field(..., description="Cache uptime in seconds")
    uptime_hours: float = Field(..., description="Cache uptime in hours")
    requests_per_hour: float = Field(..., description="Average requests per hour")


class CacheHealthResponse(BaseModel):
    """Cache health status."""
    enabled: bool = Field(..., description="Whether cache is enabled")
    redis_available: bool = Field(..., description="Whether Redis is available")
    embedding_model_loaded: bool = Field(..., description="Whether embedding model is loaded")
    cache_size: int = Field(..., description="Current number of cached entries")
    max_cache_size: int = Field(..., description="Maximum cache size")
    similarity_threshold: float = Field(..., description="Similarity threshold for cache hit")
    ttl_seconds: int = Field(..., description="Cache TTL in seconds")
    status: str = Field(..., description="Overall health status (healthy/degraded/unhealthy)")


class TopQueryResponse(BaseModel):
    """Top cached query entry."""
    query: str = Field(..., description="Cached query")
    hit_count: int = Field(..., description="Number of times this query was hit")


class TopQueriesResponse(BaseModel):
    """List of top cached queries."""
    top_queries: List[TopQueryResponse] = Field(..., description="Top N cached queries by hit count")
    total_tracked: int = Field(..., description="Total number of queries being tracked")


class InvalidateCacheRequest(BaseModel):
    """Request to invalidate cache entry."""
    query: str = Field(..., description="Query to invalidate from cache")


class ClearCacheResponse(BaseModel):
    """Response for cache clear operation."""
    success: bool = Field(..., description="Whether clear was successful")
    message: str = Field(..., description="Status message")


class InvalidateCacheResponse(BaseModel):
    """Response for cache invalidation operation."""
    success: bool = Field(..., description="Whether invalidation was successful")
    message: str = Field(..., description="Status message")
    query: str = Field(..., description="Query that was invalidated")


# Endpoints

@router.get("/metrics", response_model=CacheMetricsResponse)
async def get_metrics():
    """
    Get cache performance metrics.
    
    Returns comprehensive metrics including:
    - Hit/miss rates
    - API calls saved
    - Latency statistics
    - Cache size
    - Uptime
    """
    try:
        cache = get_response_cache()
        metrics = get_cache_metrics()
        
        metrics_summary = metrics.get_summary()
        metrics_summary["cache_size"] = cache.size()
        
        return CacheMetricsResponse(**metrics_summary)
    
    except Exception as e:
        logger.error(f"Error getting cache metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cache metrics: {str(e)}"
        )


@router.get("/health", response_model=CacheHealthResponse)
async def get_health():
    """
    Check cache health status.
    
    Returns health information including:
    - Whether cache is enabled
    - Redis availability
    - Embedding model status
    - Configuration parameters
    """
    try:
        cache = get_response_cache()
        
        # Determine overall health status
        if not cache.enabled:
            health_status = "unhealthy"
        elif cache.redis_available and cache.embedding_model is not None:
            health_status = "healthy"
        elif cache.embedding_model is not None:
            health_status = "degraded"  # Running on memory cache only
        else:
            health_status = "unhealthy"
        
        return CacheHealthResponse(
            enabled=cache.enabled,
            redis_available=cache.redis_available,
            embedding_model_loaded=(cache.embedding_model is not None),
            cache_size=cache.size(),
            max_cache_size=cache.max_cache_size,
            similarity_threshold=cache.similarity_threshold,
            ttl_seconds=cache.ttl_seconds,
            status=health_status
        )
    
    except Exception as e:
        logger.error(f"Error checking cache health: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check cache health: {str(e)}"
        )


@router.get("/top", response_model=TopQueriesResponse)
async def get_top_queries(limit: int = 10):
    """
    Get top N most frequently cached queries.
    
    Args:
        limit: Number of top queries to return (default: 10, max: 100)
    
    Returns list of queries with hit counts, sorted by frequency.
    """
    try:
        # Validate limit
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 100"
            )
        
        metrics = get_cache_metrics()
        top_queries = metrics.get_top_queries(limit=limit)
        
        return TopQueriesResponse(
            top_queries=[TopQueryResponse(**q) for q in top_queries],
            total_tracked=len(top_queries)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting top queries: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve top queries: {str(e)}"
        )


@router.post("/clear", response_model=ClearCacheResponse)
async def clear_cache():
    """
    Clear all cached entries.
    
    This is an administrative operation that removes all cached responses.
    Use with caution as this will reset cache metrics and require rebuilding the cache.
    
    **Note:** This endpoint should be protected with authentication in production.
    """
    try:
        cache = get_response_cache()
        success = cache.clear()
        
        if success:
            # Also reset metrics
            metrics = get_cache_metrics()
            metrics.reset()
            
            logger.info("✅ Cache cleared successfully")
            return ClearCacheResponse(
                success=True,
                message="Cache cleared successfully"
            )
        else:
            logger.warning("⚠️ Cache clear operation failed")
            return ClearCacheResponse(
                success=False,
                message="Failed to clear cache"
            )
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post("/invalidate", response_model=InvalidateCacheResponse)
async def invalidate_cache_entry(request: InvalidateCacheRequest):
    """
    Invalidate a specific cached query.
    
    Removes the cached entry for a specific query, forcing it to be
    re-fetched from the LLM API on the next request.
    
    Args:
        request: Request containing the query to invalidate
    """
    try:
        cache = get_response_cache()
        success = cache.invalidate(request.query)
        
        if success:
            logger.info(f"✅ Invalidated cache for query: '{request.query[:50]}...'")
            return InvalidateCacheResponse(
                success=True,
                message="Cache entry invalidated successfully",
                query=request.query
            )
        else:
            logger.warning(f"⚠️ Failed to invalidate cache for query: '{request.query[:50]}...'")
            return InvalidateCacheResponse(
                success=False,
                message="Failed to invalidate cache entry",
                query=request.query
            )
    
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to invalidate cache entry: {str(e)}"
        )


@router.post("/invalidate/{query_hash}")
async def invalidate_cache_by_hash(query_hash: str):
    """
    Invalidate a cached entry by its hash.
    
    This is a low-level operation for direct cache manipulation.
    Use the `/invalidate` endpoint with the actual query for standard operations.
    
    Args:
        query_hash: Hash of the cached query to invalidate
        
    **Note:** This endpoint is for advanced use cases and may be removed in future versions.
    """
    try:
        cache = get_response_cache()
        
        # For hash-based invalidation, we need to access Redis directly
        if not cache.redis_available or not cache.redis_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Redis not available for hash-based invalidation"
            )
        
        # Construct cache key from hash
        cache_key = f"{cache.cache_prefix}:{query_hash}"
        
        # Remove from Redis
        deleted = cache.redis_client.delete(cache_key)
        cache.redis_client.zrem(cache.index_key, cache_key)
        
        if deleted > 0:
            logger.info(f"✅ Invalidated cache entry with hash: {query_hash}")
            return {
                "success": True,
                "message": "Cache entry invalidated successfully",
                "query_hash": query_hash
            }
        else:
            logger.warning(f"⚠️ No cache entry found for hash: {query_hash}")
            return {
                "success": False,
                "message": "No cache entry found for this hash",
                "query_hash": query_hash
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invalidating cache by hash: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to invalidate cache entry: {str(e)}"
        )


# Export router
__all__ = ["router"]

