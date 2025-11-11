"""
LLM Response Cache with Semantic Similarity Matching

Implements a hybrid Qdrant + Redis cache for LLM responses with semantic similarity-based lookup.
- Qdrant: Optimized vector similarity search (O(log n))
- Redis: Cache entry storage with TTL
This reduces API costs by caching similar queries and improves response times.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.ai.llm_intent.query_normalizer import get_query_normalizer
from app.ai.llm_intent.cache_metrics import get_cache_metrics
from app.core.embedding_cache import get_cached_embedding, cache_embedding

try:
    from app.core.redis_client import get_redis_client
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from app.core.async_redis_client import get_async_redis_client
    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False

try:
    from app.core.qdrant_client import get_qdrant_client
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger("response_cache")


class LLMResponseCache:
    """
    Semantic similarity-based cache for LLM responses using Qdrant + Redis hybrid approach.
    
    Features:
    - Qdrant: Optimized vector similarity search (O(log n) instead of O(n))
    - Redis: Cache entry storage with TTL and LRU eviction
    - Embedding-based similarity search for cache lookup
    - Automatic query normalization
    - Configurable similarity threshold
    - In-memory fallback if services unavailable
    - Thread-safe operations
    """
    
    def __init__(
        self,
        redis_client=None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = None,
        ttl_seconds: int = None,
        max_cache_size: int = None,
        min_query_length: int = None,
        enabled: bool = None
    ):
        """
        Initialize LLM response cache.
        
        Args:
            redis_client: Redis client instance (optional, will create if None)
            embedding_model_name: SentenceTransformer model name (default: all-MiniLM-L6-v2)
            similarity_threshold: Cosine similarity threshold for cache hit (default: 0.95)
            ttl_seconds: Cache TTL in seconds (default: 86400 = 24 hours)
            max_cache_size: Maximum number of cached responses (default: 10000)
            min_query_length: Minimum query length to cache (default: 3)
            enabled: Enable/disable caching (default: True)
        """
        # Configuration from environment or parameters
        self.enabled = enabled if enabled is not None else (
            os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true"
        )
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else (
            float(os.getenv("LLM_CACHE_SIMILARITY_THRESHOLD", "0.95"))
        )
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else (
            int(os.getenv("LLM_CACHE_TTL", "86400"))  # 24 hours
        )
        self.max_cache_size = max_cache_size if max_cache_size is not None else (
            int(os.getenv("LLM_CACHE_MAX_SIZE", "10000"))
        )
        self.min_query_length = min_query_length if min_query_length is not None else (
            int(os.getenv("LLM_CACHE_MIN_QUERY_LENGTH", "3"))
        )
        
        # Initialize components
        self.normalizer = get_query_normalizer()
        self.metrics = get_cache_metrics()
        
        # Initialize Redis client (sync)
        self.redis_client = None
        self.redis_available = False
        if REDIS_AVAILABLE and self.enabled:
            try:
                self.redis_client = redis_client or get_redis_client()
                self.redis_client.ping()
                self.redis_available = True
                logger.info("✅ LLM Response Cache connected to Redis (sync)")
            except Exception as e:
                logger.warning(f"⚠️ Redis not available for LLM cache, using in-memory fallback: {e}")
                self.redis_available = False
        
        # Initialize async Redis client (for async endpoints)
        self.async_redis_client = None
        self.async_redis_available = False
        # Note: Async client initialized lazily on first async call to avoid blocking during init
        
        # In-memory fallback cache (if Redis unavailable)
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize embedding model (singleton pattern - loaded once)
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self._model_loaded = False
        
        if self.enabled:
            logger.info(f"🔄 Loading embedding model '{embedding_model_name}' for cache...")
            try:
                self._load_embedding_model()
                logger.info(f"✅ Embedding model loaded for cache")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                self.enabled = False
                self.embedding_model = None
    
    def _load_embedding_model(self) -> None:
        """Load embedding model (only once, singleton pattern)."""
        if self._model_loaded and self.embedding_model is not None:
            return
        
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self._model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Cache key prefix
        self.cache_prefix = "chatns:llm_cache"
        self.index_key = f"{self.cache_prefix}:index"
        
        # Initialize Qdrant client for vector search
        self.qdrant_client = None
        self.qdrant_available = False
        self.qdrant_collection = os.getenv("QDRANT_CACHE_COLLECTION", "chatns_cache_embeddings")
        self.vector_size = 384  # Must match embedding model dimension
        
        if QDRANT_AVAILABLE and self.enabled:
            try:
                self.qdrant_client = get_qdrant_client()
                if self.qdrant_client:
                    # Ensure Qdrant collection exists
                    self._ensure_qdrant_collection()
                    self.qdrant_available = True
                    logger.info(f"✅ LLM Response Cache connected to Qdrant for vector search")
                else:
                    logger.warning("⚠️ Qdrant client not available, falling back to Redis scanning")
            except Exception as e:
                logger.warning(f"⚠️ Qdrant not available for LLM cache: {e}. Using Redis fallback.")
                self.qdrant_available = False
        
        if self.enabled:
            logger.info(
                f"✅ LLM Response Cache initialized: "
                f"threshold={self.similarity_threshold}, "
                f"ttl={self.ttl_seconds}s, "
                f"qdrant={'available' if self.qdrant_available else 'unavailable (using Redis scan)'}, "
                f"redis={'available' if self.redis_available else 'unavailable (using memory)'}"
            )
        else:
            logger.warning("⚠️ LLM Response Cache is DISABLED")
    
    async def _ensure_async_redis(self):
        """Lazy initialization of async Redis client."""
        if self.async_redis_available:
            return self.async_redis_client
        
        if ASYNC_REDIS_AVAILABLE and self.enabled:
            try:
                self.async_redis_client = await get_async_redis_client()
                if self.async_redis_client:
                    await self.async_redis_client.ping()
                    self.async_redis_available = True
                    logger.debug("Async Redis client initialized for cache")
            except Exception as e:
                logger.debug(f"Async Redis not available: {e}")
                self.async_redis_available = False
        
        return self.async_redis_client
    
    def get(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a query using semantic similarity.
        
        Args:
            query: User query
            context: Optional context (not used in cache lookup currently)
            
        Returns:
            Cached response dict if found, None otherwise
        """
        if not self.enabled:
            return None
        
        start_time = time.time()
        
        try:
            # Normalize query
            normalized_query = self.normalizer.normalize(query)
            
            # Check if query is cacheable
            if not self.normalizer.is_cacheable(normalized_query, self.min_query_length):
                logger.debug(f"Query not cacheable (too short): '{query}'")
                return None
            
            # Compute query embedding
            query_embedding = self._compute_embedding(normalized_query)
            if query_embedding is None:
                return None
            
            # Search for similar cached queries (uses Qdrant if available, falls back to Redis)
            cached_response = self._search_similar(normalized_query, query_embedding)
            
            latency_ms = (time.time() - start_time) * 1000
            
            if cached_response:
                logger.info(f"✅ Cache HIT for query: '{query[:50]}...' (latency: {latency_ms:.2f}ms)")
                self.metrics.record_hit(query=normalized_query, latency_ms=latency_ms)
                
                # Update hit count
                cached_response['_cache_hit_count'] = cached_response.get('_cache_hit_count', 0) + 1
                
                # Return the cached LLM response
                return cached_response.get('response')
            else:
                logger.debug(f"❌ Cache MISS for query: '{query[:50]}...' (latency: {latency_ms:.2f}ms)")
                self.metrics.record_miss(latency_ms=latency_ms)
                return None
                
        except Exception as e:
            logger.error(f"Error during cache lookup: {e}", exc_info=True)
            return None
    
    async def aget(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Async version of get() for use in async FastAPI endpoints.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Cached response dict if found, None otherwise
        """
        if not self.enabled:
            return None
        
        start_time = time.time()
        
        try:
            # Normalize query
            normalized_query = self.normalizer.normalize(query)
            
            # Check if query is cacheable
            if not self.normalizer.is_cacheable(normalized_query, self.min_query_length):
                logger.debug(f"Query not cacheable (too short): '{query}'")
                return None
            
            # Compute query embedding
            query_embedding = self._compute_embedding(normalized_query)
            if query_embedding is None:
                return None
            
            # Search for similar cached queries (async)
            cached_response = await self._asearch_similar(normalized_query, query_embedding)
            
            latency_ms = (time.time() - start_time) * 1000
            
            if cached_response:
                logger.info(f"✅ Cache HIT (async) for query: '{query[:50]}...' (latency: {latency_ms:.2f}ms)")
                self.metrics.record_hit(query=normalized_query, latency_ms=latency_ms)
                
                # Update hit count (async)
                cached_response['_cache_hit_count'] = cached_response.get('_cache_hit_count', 0) + 1
                await self._aupdate_hit_count(cached_response)
                
                # Return the cached LLM response
                return cached_response.get('response')
            else:
                logger.debug(f"❌ Cache MISS (async) for query: '{query[:50]}...' (latency: {latency_ms:.2f}ms)")
                self.metrics.record_miss(latency_ms=latency_ms)
                return None
                
        except Exception as e:
            logger.error(f"Error during async cache lookup: {e}", exc_info=True)
            return None
    
    async def _aupdate_hit_count(self, cached_entry: Dict[str, Any]) -> None:
        """Update hit count in Redis (async)."""
        if not self.async_redis_available:
            return
        
        try:
            cache_key = self._generate_cache_key(
                np.array(cached_entry.get('embedding', []))
            )
            cached_entry['_cache_hit_count'] = cached_entry.get('_cache_hit_count', 0)
            await self.async_redis_client.setex(
                cache_key,
                self.ttl_seconds,
                json.dumps(cached_entry)
            )
        except Exception as e:
            logger.debug(f"Failed to update hit count async: {e}")
    
    def set(
        self,
        query: str,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache an LLM response for a query.
        
        Args:
            query: User query
            response: LLM response to cache
            context: Optional context (not used in cache storage currently)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Normalize query
            normalized_query = self.normalizer.normalize(query)
            
            # Check if query is cacheable
            if not self.normalizer.is_cacheable(normalized_query, self.min_query_length):
                logger.debug(f"Query not cacheable (too short): '{query}'")
                return False
            
            # Compute query embedding
            query_embedding = self._compute_embedding(normalized_query)
            if query_embedding is None:
                return False
            
            # Create cache entry
            cache_entry = {
                'query': normalized_query,
                'response': response,
                'embedding': query_embedding.tolist(),
                'timestamp': time.time(),
                '_cache_hit_count': 0
            }
            
            # Generate cache key from embedding hash
            cache_key = self._generate_cache_key(query_embedding)
            
            # Store in Redis (for cache data) and Qdrant (for vector search)
            if self.redis_available and self.redis_client:
                try:
                    # Store cache entry in Redis with TTL
                    self.redis_client.setex(
                        cache_key,
                        self.ttl_seconds,
                        json.dumps(cache_entry)
                    )
                    
                    # Add to Redis index for LRU eviction (timestamp-based)
                    self.redis_client.zadd(
                        self.index_key,
                        {cache_key: time.time()}
                    )
                    
                    # Store embedding in Qdrant for efficient vector search
                    if self.qdrant_available and self.qdrant_client:
                        try:
                            point_id = int(hashlib.md5(cache_key.encode()).hexdigest()[:8], 16)
                            point = PointStruct(
                                id=point_id,
                                vector=query_embedding.tolist(),
                                payload={
                                    "cache_key": cache_key,
                                    "query": normalized_query,
                                    "timestamp": time.time()
                                }
                            )
                            self.qdrant_client.upsert(
                                collection_name=self.qdrant_collection,
                                points=[point]
                            )
                        except Exception as e:
                            logger.warning(f"Failed to store embedding in Qdrant: {e}. Continuing with Redis only.")
                    
                    # Limit cache size (remove oldest entries if over limit)
                    cache_size = self.redis_client.zcard(self.index_key)
                    if cache_size > self.max_cache_size:
                        # Remove oldest entries
                        to_remove = cache_size - self.max_cache_size
                        oldest_keys = self.redis_client.zrange(self.index_key, 0, to_remove - 1)
                        if oldest_keys:
                            # Remove from Redis
                            self.redis_client.zrem(self.index_key, *oldest_keys)
                            self.redis_client.delete(*oldest_keys)
                            
                            # Remove from Qdrant if available
                            if self.qdrant_available and self.qdrant_client:
                                try:
                                    point_ids = [int(hashlib.md5(k.encode()).hexdigest()[:8], 16) for k in oldest_keys]
                                    self.qdrant_client.delete(
                                        collection_name=self.qdrant_collection,
                                        points_selector=point_ids
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to delete from Qdrant during eviction: {e}")
                            
                            logger.debug(f"Evicted {len(oldest_keys)} old cache entries (LRU)")
                    
                    logger.debug(f"✅ Cached response for query: '{normalized_query[:50]}...'")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to cache in Redis, falling back to memory: {e}")
                    # Fall through to memory cache
            
            # In-memory fallback
            self._memory_cache[cache_key] = cache_entry
            
            # Limit memory cache size
            if len(self._memory_cache) > self.max_cache_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]
                logger.debug(f"Evicted old memory cache entry (FIFO)")
            
            logger.debug(f"✅ Cached response in memory for query: '{normalized_query[:50]}...'")
            return True
            
        except Exception as e:
            logger.error(f"Error caching response: {e}", exc_info=True)
            return False
    
    async def aset(
        self,
        query: str,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Async version of set() for use in async FastAPI endpoints.
        
        Args:
            query: User query
            response: LLM response to cache
            context: Optional context
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Normalize query
            normalized_query = self.normalizer.normalize(query)
            
            # Check if query is cacheable
            if not self.normalizer.is_cacheable(normalized_query, self.min_query_length):
                logger.debug(f"Query not cacheable (too short): '{query}'")
                return False
            
            # Compute query embedding
            query_embedding = self._compute_embedding(normalized_query)
            if query_embedding is None:
                return False
            
            # Create cache entry
            cache_entry = {
                'query': normalized_query,
                'response': response,
                'embedding': query_embedding.tolist(),
                'timestamp': time.time(),
                '_cache_hit_count': 0
            }
            
            # Generate cache key
            cache_key = self._generate_cache_key(query_embedding)
            
            # Ensure async Redis is available
            await self._ensure_async_redis()
            
            # Store in async Redis and Qdrant
            if self.async_redis_available and self.async_redis_client:
                try:
                    # Store cache entry in Redis with TTL (async)
                    await self.async_redis_client.setex(
                        cache_key,
                        self.ttl_seconds,
                        json.dumps(cache_entry)
                    )
                    
                    # Add to Redis index for LRU eviction (async)
                    await self.async_redis_client.zadd(
                        self.index_key,
                        {cache_key: time.time()}
                    )
                    
                    # Store embedding in Qdrant for efficient vector search
                    if self.qdrant_available and self.qdrant_client:
                        try:
                            point_id = int(hashlib.md5(cache_key.encode()).hexdigest()[:8], 16)
                            point = PointStruct(
                                id=point_id,
                                vector=query_embedding.tolist(),
                                payload={
                                    "cache_key": cache_key,
                                    "query": normalized_query,
                                    "timestamp": time.time()
                                }
                            )
                            self.qdrant_client.upsert(
                                collection_name=self.qdrant_collection,
                                points=[point]
                            )
                        except Exception as e:
                            logger.warning(f"Failed to store embedding in Qdrant: {e}. Continuing with Redis only.")
                    
                    # Limit cache size (async)
                    cache_size = await self.async_redis_client.zcard(self.index_key)
                    if cache_size > self.max_cache_size:
                        to_remove = cache_size - self.max_cache_size
                        oldest_keys = await self.async_redis_client.zrange(self.index_key, 0, to_remove - 1)
                        if oldest_keys:
                            await self.async_redis_client.zrem(self.index_key, *oldest_keys)
                            await self.async_redis_client.delete(*oldest_keys)
                            
                            # Remove from Qdrant if available
                            if self.qdrant_available and self.qdrant_client:
                                try:
                                    point_ids = [int(hashlib.md5(k.encode()).hexdigest()[:8], 16) for k in oldest_keys]
                                    self.qdrant_client.delete(
                                        collection_name=self.qdrant_collection,
                                        points_selector=point_ids
                                    )
                                except Exception:
                                    pass
                            
                            logger.debug(f"Evicted {len(oldest_keys)} old cache entries (async, LRU)")
                    
                    logger.debug(f"✅ Cached response (async) for query: '{normalized_query[:50]}...'")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to cache in async Redis: {e}")
                    # Fall back to sync Redis
                    if self.redis_available and self.redis_client:
                        return self.set(query, response, context)
                    return False
            
            # Fall back to sync or memory
            if self.redis_available:
                return self.set(query, response, context)
            
            # In-memory fallback
            self._memory_cache[cache_key] = cache_entry
            if len(self._memory_cache) > self.max_cache_size:
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]
            
            logger.debug(f"✅ Cached response in memory (async) for query: '{normalized_query[:50]}...'")
            return True
            
        except Exception as e:
            logger.error(f"Error caching response (async): {e}", exc_info=True)
            return False
    
    def _compute_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Compute embedding for a query with caching to avoid redundant computation.
        
        Args:
            query: Normalized query string
            
        Returns:
            Embedding vector as numpy array, or None on error
        """
        if self.embedding_model is None:
            if not self._model_loaded:
                self._load_embedding_model()
            if self.embedding_model is None:
                return None
        
        # Check embedding cache first
        cached_embedding = get_cached_embedding(query)
        if cached_embedding is not None:
            logger.debug(f"Embedding cache HIT for query: '{query[:50]}...'")
            return cached_embedding
        
        try:
            # Compute embedding (ensure model is loaded)
            if not self._model_loaded:
                self._load_embedding_model()
            
            embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            embedding_array = np.array(embedding)
            
            # Cache the computed embedding
            cache_embedding(query, embedding_array)
            
            return embedding_array
        except Exception as e:
            logger.error(f"Failed to compute embedding: {e}")
            return None
    
    def _generate_cache_key(self, embedding: np.ndarray) -> str:
        """
        Generate cache key from embedding vector hash.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Cache key string
        """
        # Hash the embedding vector to create a stable key
        embedding_bytes = embedding.tobytes()
        embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()[:16]
        return f"{self.cache_prefix}:{embedding_hash}"
    
    def _search_similar(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Search for similar cached queries using Qdrant vector search (fast) or Redis fallback.
        
        Args:
            query: Normalized query string
            query_embedding: Query embedding vector
            
        Returns:
            Cached entry if similar query found, None otherwise
        """
        try:
            # Prefer Qdrant for vector search (O(log n)), fall back to Redis scanning
            if self.qdrant_available and self.qdrant_client:
                return self._search_qdrant(query, query_embedding)
            elif self.redis_available and self.redis_client:
                return self._search_redis(query, query_embedding)
            else:
                return self._search_memory(query, query_embedding)
        except Exception as e:
            logger.error(f"Error searching cache: {e}", exc_info=True)
            return None
    
    async def _asearch_similar(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Async search for similar cached queries using Qdrant vector search or async Redis fallback.
        
        Args:
            query: Normalized query string
            query_embedding: Query embedding vector
            
        Returns:
            Cached entry if similar query found, None otherwise
        """
        try:
            # Ensure async Redis is available
            await self._ensure_async_redis()
            
            # Prefer Qdrant for vector search (O(log n)), fall back to async Redis scanning
            if self.qdrant_available and self.qdrant_client:
                return await self._asearch_qdrant(query, query_embedding)
            elif self.async_redis_available and self.async_redis_client:
                return await self._asearch_redis(query, query_embedding)
            elif self.redis_available and self.redis_client:
                # Fall back to sync Redis
                return self._search_redis(query, query_embedding)
            else:
                return self._search_memory(query, query_embedding)
        except Exception as e:
            logger.error(f"Error searching cache (async): {e}", exc_info=True)
            return None
    
    async def _asearch_qdrant(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Async search Qdrant for similar cached queries using optimized vector search.
        
        Args:
            query: Normalized query string
            query_embedding: Query embedding vector
            
        Returns:
            Cached entry if similar query found, None otherwise
        """
        try:
            # Use Qdrant's efficient vector search (O(log n))
            search_results = self.qdrant_client.search(
                collection_name=self.qdrant_collection,
                query_vector=query_embedding.tolist(),
                limit=1,
                score_threshold=self.similarity_threshold
            )
            
            if not search_results or len(search_results) == 0:
                return None
            
            best_match = search_results[0]
            similarity = best_match.score
            
            if similarity < self.similarity_threshold:
                return None
            
            # Extract cache_key from payload
            cache_key = best_match.payload.get("cache_key")
            if not cache_key:
                logger.warning("Qdrant result missing cache_key in payload")
                return None
            
            # Fetch actual cache entry from async Redis
            if not self.async_redis_available or not self.async_redis_client:
                # Try sync Redis as fallback
                if self.redis_available and self.redis_client:
                    cached_data = self.redis_client.get(cache_key)
                else:
                    logger.warning("Qdrant found match but Redis unavailable for cache data")
                    return None
            else:
                cached_data = await self.async_redis_client.get(cache_key)
            
            if not cached_data:
                # Entry expired, remove from Qdrant
                try:
                    point_id = best_match.id
                    self.qdrant_client.delete(
                        collection_name=self.qdrant_collection,
                        points_selector=[point_id]
                    )
                except Exception:
                    pass
                return None
            
            cached_entry = json.loads(cached_data)
            
            logger.debug(
                f"Qdrant cache match found (async): similarity={similarity:.4f}, "
                f"query='{cached_entry.get('query', '')[:50]}...'"
            )
            
            return cached_entry
            
        except Exception as e:
            logger.error(f"Error searching Qdrant cache (async): {e}", exc_info=True)
            # Fall back to async Redis scanning
            if self.async_redis_available and self.async_redis_client:
                logger.debug("Falling back to async Redis scanning due to Qdrant error")
                return await self._asearch_redis(query, query_embedding)
            return None
    
    async def _asearch_redis(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Async search Redis cache for similar queries (fallback method - O(n) linear scan).
        This is used when Qdrant is unavailable. Prefer _asearch_qdrant for better performance.
        
        Args:
            query: Normalized query string
            query_embedding: Query embedding vector
            
        Returns:
            Cached entry if similar query found, None otherwise
        """
        try:
            # Get all cache keys from index (O(n) operation - inefficient)
            cache_keys = await self.async_redis_client.zrange(self.index_key, 0, -1)
            
            if not cache_keys:
                return None
            
            best_match = None
            best_similarity = 0.0
            
            # Compare with all cached embeddings (linear scan)
            for cache_key in cache_keys:
                try:
                    # Get cached entry (async)
                    cached_data = await self.async_redis_client.get(cache_key)
                    if not cached_data:
                        # Entry expired, remove from index
                        await self.async_redis_client.zrem(self.index_key, cache_key)
                        continue
                    
                    cached_entry = json.loads(cached_data)
                    cached_embedding = np.array(cached_entry['embedding'], dtype=np.float32)
                    query_emb = query_embedding.astype(np.float32) if query_embedding.dtype != np.float32 else query_embedding
                    
                    # Compute cosine similarity
                    similarity = float(util.cos_sim(query_emb, cached_embedding).item())
                    
                    if similarity >= self.similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = cached_entry
                        
                        if similarity > 0.99:
                            logger.debug(f"Perfect cache match found (async Redis scan): similarity={similarity:.4f}")
                            return best_match
                
                except Exception as e:
                    logger.debug(f"Error checking cache entry {cache_key}: {e}")
                    continue
            
            if best_match:
                logger.debug(
                    f"Cache match found (async Redis scan): similarity={best_similarity:.4f}, "
                    f"query='{best_match['query'][:50]}...'"
                )
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error searching Redis cache (async): {e}")
            return None
    
    def _ensure_qdrant_collection(self) -> None:
        """Ensure Qdrant collection exists for cache embeddings."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.qdrant_collection not in collection_names:
                logger.info(f"Creating Qdrant collection '{self.qdrant_collection}' for cache...")
                self.qdrant_client.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Qdrant collection '{self.qdrant_collection}' created")
            else:
                logger.debug(f"Qdrant collection '{self.qdrant_collection}' already exists")
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}", exc_info=True)
            raise
    
    def _search_qdrant(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Search Qdrant for similar cached queries using optimized vector search.
        
        Args:
            query: Normalized query string
            query_embedding: Query embedding vector
            
        Returns:
            Cached entry if similar query found, None otherwise
        """
        try:
            # Use Qdrant's efficient vector search (O(log n))
            search_results = self.qdrant_client.search(
                collection_name=self.qdrant_collection,
                query_vector=query_embedding.tolist(),
                limit=1,  # Only need top match
                score_threshold=self.similarity_threshold
            )
            
            if not search_results or len(search_results) == 0:
                return None
            
            # Get best match
            best_match = search_results[0]
            similarity = best_match.score
            
            if similarity < self.similarity_threshold:
                return None
            
            # Extract cache_key from payload
            cache_key = best_match.payload.get("cache_key")
            if not cache_key:
                logger.warning("Qdrant result missing cache_key in payload")
                return None
            
            # Fetch actual cache entry from Redis
            if not self.redis_available or not self.redis_client:
                logger.warning("Qdrant found match but Redis unavailable for cache data")
                return None
            
            cached_data = self.redis_client.get(cache_key)
            if not cached_data:
                # Entry expired in Redis, remove from Qdrant
                try:
                    point_id = best_match.id
                    self.qdrant_client.delete(
                        collection_name=self.qdrant_collection,
                        points_selector=[point_id]
                    )
                except Exception:
                    pass
                return None
            
            cached_entry = json.loads(cached_data)
            
            logger.debug(
                f"Qdrant cache match found: similarity={similarity:.4f}, "
                f"query='{cached_entry.get('query', '')[:50]}...'"
            )
            
            return cached_entry
            
        except Exception as e:
            logger.error(f"Error searching Qdrant cache: {e}", exc_info=True)
            # Fall back to Redis scanning
            if self.redis_available and self.redis_client:
                logger.debug("Falling back to Redis scanning due to Qdrant error")
                return self._search_redis(query, query_embedding)
            return None
    
    def _search_redis(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Search Redis cache for similar queries (fallback method - O(n) linear scan).
        This is used when Qdrant is unavailable. Prefer _search_qdrant for better performance.
        
        Args:
            query: Normalized query string
            query_embedding: Query embedding vector
            
        Returns:
            Cached entry if similar query found, None otherwise
        """
        try:
            # Get all cache keys from index (O(n) operation - inefficient)
            cache_keys = self.redis_client.zrange(self.index_key, 0, -1)
            
            if not cache_keys:
                return None
            
            best_match = None
            best_similarity = 0.0
            
            # Compare with all cached embeddings (linear scan)
            for cache_key in cache_keys:
                try:
                    # Get cached entry
                    cached_data = self.redis_client.get(cache_key)
                    if not cached_data:
                        # Entry expired, remove from index
                        self.redis_client.zrem(self.index_key, cache_key)
                        continue
                    
                    cached_entry = json.loads(cached_data)
                    # Ensure consistent dtype (float32) to avoid dtype mismatch errors
                    cached_embedding = np.array(cached_entry['embedding'], dtype=np.float32)
                    
                    # Ensure query embedding is also float32
                    query_emb = query_embedding.astype(np.float32) if query_embedding.dtype != np.float32 else query_embedding
                    
                    # Compute cosine similarity
                    similarity = float(util.cos_sim(query_emb, cached_embedding).item())
                    
                    # Check if above threshold and better than current best
                    if similarity >= self.similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = cached_entry
                        
                        # If perfect match (similarity ~1.0), return immediately
                        if similarity > 0.99:
                            logger.debug(f"Perfect cache match found (Redis scan): similarity={similarity:.4f}")
                            return best_match
                
                except Exception as e:
                    logger.debug(f"Error checking cache entry {cache_key}: {e}")
                    continue
            
            if best_match:
                logger.debug(
                    f"Cache match found (Redis scan): similarity={best_similarity:.4f}, "
                    f"query='{best_match['query'][:50]}...'"
                )
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error searching Redis cache: {e}")
            return None
    
    def _search_memory(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Search in-memory cache for similar queries.
        
        Args:
            query: Normalized query string
            query_embedding: Query embedding vector
            
        Returns:
            Cached entry if similar query found, None otherwise
        """
        best_match = None
        best_similarity = 0.0
        
        for cache_key, cached_entry in self._memory_cache.items():
            try:
                # Ensure consistent dtype (float32) to avoid dtype mismatch errors
                cached_embedding = np.array(cached_entry['embedding'], dtype=np.float32)
                
                # Ensure query embedding is also float32
                query_emb = query_embedding.astype(np.float32) if query_embedding.dtype != np.float32 else query_embedding
                
                # Compute cosine similarity
                similarity = float(util.cos_sim(query_emb, cached_embedding).item())
                
                # Check if above threshold and better than current best
                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_entry
                    
                    # If perfect match, return immediately
                    if similarity > 0.99:
                        return best_match
            
            except Exception as e:
                logger.debug(f"Error checking memory cache entry: {e}")
                continue
        
        if best_match:
            logger.debug(f"Memory cache match found: similarity={best_similarity:.4f}")
        
        return best_match
    
    def size(self) -> int:
        """
        Get current cache size (number of cached entries).
        
        Returns:
            Number of cached entries
        """
        try:
            if self.redis_available and self.redis_client:
                return int(self.redis_client.zcard(self.index_key))
            else:
                return len(self._memory_cache)
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return 0
    
    def clear(self) -> bool:
        """
        Clear all cached entries from Redis, Qdrant, and memory.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            if self.redis_available and self.redis_client:
                # Get all cache keys
                cache_keys = self.redis_client.zrange(self.index_key, 0, -1)
                if cache_keys:
                    # Delete all cache entries
                    self.redis_client.delete(*cache_keys)
                # Clear index
                self.redis_client.delete(self.index_key)
                logger.info("✅ Cleared Redis cache")
            
            # Clear Qdrant collection if available
            if self.qdrant_available and self.qdrant_client:
                try:
                    # Delete all points in collection
                    self.qdrant_client.delete(
                        collection_name=self.qdrant_collection,
                        points_selector={"filter": {}}  # Select all points
                    )
                    logger.info("✅ Cleared Qdrant cache embeddings")
                except Exception as e:
                    logger.warning(f"Failed to clear Qdrant cache: {e}")
            
            # Clear memory cache
            self._memory_cache.clear()
            logger.info("✅ Cleared memory cache")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def invalidate(self, query: str) -> bool:
        """
        Invalidate cached entry for a specific query from Redis, Qdrant, and memory.
        
        Args:
            query: Query to invalidate
            
        Returns:
            True if invalidated successfully, False otherwise
        """
        try:
            # Normalize query
            normalized_query = self.normalizer.normalize(query)
            
            # Compute embedding
            query_embedding = self._compute_embedding(normalized_query)
            if query_embedding is None:
                return False
            
            # Generate cache key
            cache_key = self._generate_cache_key(query_embedding)
            
            if self.redis_available and self.redis_client:
                # Remove from Redis
                self.redis_client.delete(cache_key)
                self.redis_client.zrem(self.index_key, cache_key)
            
            # Remove from Qdrant if available
            if self.qdrant_available and self.qdrant_client:
                try:
                    point_id = int(hashlib.md5(cache_key.encode()).hexdigest()[:8], 16)
                    self.qdrant_client.delete(
                        collection_name=self.qdrant_collection,
                        points_selector=[point_id]
                    )
                except Exception as e:
                    logger.warning(f"Failed to remove from Qdrant during invalidation: {e}")
            
            # Remove from memory
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            
            logger.info(f"✅ Invalidated cache for query: '{normalized_query[:50]}...'")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False


# Singleton instance
_cache_instance: Optional[LLMResponseCache] = None


def get_response_cache() -> LLMResponseCache:
    """
    Get or create singleton LLMResponseCache instance.
    
    Returns:
        Singleton LLMResponseCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LLMResponseCache()
    return _cache_instance


__all__ = ["LLMResponseCache", "get_response_cache"]

