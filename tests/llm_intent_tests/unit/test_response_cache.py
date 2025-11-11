"""
Unit tests for LLMResponseCache
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.ai.llm_intent.response_cache import LLMResponseCache


class TestLLMResponseCache:
    """Test suite for LLMResponseCache."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create cache with in-memory mode (no Redis)
        with patch('app.ai.llm_intent.response_cache.REDIS_AVAILABLE', False):
            self.cache = LLMResponseCache(
                similarity_threshold=0.95,
                ttl_seconds=3600,
                max_cache_size=100,
                min_query_length=3,
                enabled=True
            )
    
    def test_initialization(self):
        """Test cache initialization."""
        assert self.cache.enabled is True
        assert self.cache.similarity_threshold == 0.95
        assert self.cache.ttl_seconds == 3600
        assert self.cache.max_cache_size == 100
        assert self.cache.min_query_length == 3
        assert self.cache.embedding_model is not None
    
    def test_disabled_cache(self):
        """Test that disabled cache returns None."""
        with patch('app.ai.llm_intent.response_cache.REDIS_AVAILABLE', False):
            cache = LLMResponseCache(enabled=False)
            
            result = cache.get("test query")
            assert result is None
            
            success = cache.set("test query", {"response": "test"})
            assert success is False
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        query = "show me red shoes"
        response = {"intent": "SEARCH_PRODUCT", "confidence": 0.95}
        
        # Set cache entry
        success = self.cache.set(query, response)
        assert success is True
        
        # Get cache entry
        cached = self.cache.get(query)
        assert cached is not None
        assert cached["intent"] == "SEARCH_PRODUCT"
        assert cached["confidence"] == 0.95
    
    def test_cache_miss(self):
        """Test cache miss for non-existent query."""
        result = self.cache.get("query that was never cached")
        assert result is None
    
    def test_similar_query_hit(self):
        """Test cache hit for similar queries."""
        # Cache original query
        original_query = "show me red shoes"
        response = {"intent": "SEARCH_PRODUCT", "confidence": 0.95}
        self.cache.set(original_query, response)
        
        # Similar query should hit cache
        similar_query = "Show me RED shoes!"  # Different case and punctuation
        cached = self.cache.get(similar_query)
        
        # Should get cache hit due to normalization
        assert cached is not None
    
    def test_query_normalization(self):
        """Test that queries are normalized before caching."""
        query1 = "Show me RED shoes!!!"
        query2 = "show me red shoes"
        
        response = {"intent": "SEARCH_PRODUCT"}
        self.cache.set(query1, response)
        
        # Should hit cache because normalized forms are identical
        cached = self.cache.get(query2)
        assert cached is not None
    
    def test_min_query_length_filter(self):
        """Test that short queries are not cached."""
        short_query = "hi"
        response = {"intent": "HELP"}
        
        # Should not cache short query
        success = self.cache.set(short_query, response)
        assert success is False
        
        # Should not be retrievable
        cached = self.cache.get(short_query)
        assert cached is None
    
    def test_cache_size(self):
        """Test cache size tracking."""
        initial_size = self.cache.size()
        
        # Add entry
        self.cache.set("test query 1", {"intent": "TEST"})
        assert self.cache.size() == initial_size + 1
        
        # Add another entry
        self.cache.set("test query 2", {"intent": "TEST"})
        assert self.cache.size() == initial_size + 2
    
    def test_cache_clear(self):
        """Test cache clear operation."""
        # Add entries
        self.cache.set("test query 1", {"intent": "TEST"})
        self.cache.set("test query 2", {"intent": "TEST"})
        
        assert self.cache.size() > 0
        
        # Clear cache
        success = self.cache.clear()
        assert success is True
        assert self.cache.size() == 0
    
    def test_cache_invalidate(self):
        """Test cache invalidation for specific query."""
        query = "test query"
        self.cache.set(query, {"intent": "TEST"})
        
        # Verify it's cached
        assert self.cache.get(query) is not None
        
        # Invalidate
        success = self.cache.invalidate(query)
        assert success is True
        
        # Should no longer be cached
        assert self.cache.get(query) is None
    
    def test_max_cache_size_limit(self):
        """Test that cache respects max size limit."""
        # Create small cache
        with patch('app.ai.llm_intent.response_cache.REDIS_AVAILABLE', False):
            small_cache = LLMResponseCache(
                max_cache_size=5,
                enabled=True
            )
        
        # Add more entries than max size
        for i in range(10):
            small_cache.set(f"query number {i}", {"intent": "TEST"})
        
        # Cache size should not exceed max
        assert small_cache.size() <= 5
    
    def test_semantic_similarity_threshold(self):
        """Test that similarity threshold is respected."""
        # Cache a query
        self.cache.set("show me red shoes", {"intent": "SEARCH"})
        
        # Very different query should NOT hit cache
        different_query = "help me with returns"
        cached = self.cache.get(different_query)
        assert cached is None
    
    def test_context_parameter(self):
        """Test that context parameter is accepted but not used."""
        query = "test query"
        response = {"intent": "TEST"}
        context = {"user_id": "123", "session_id": "abc"}
        
        # Set with context (should not affect caching)
        success = self.cache.set(query, response, context=context)
        assert success is True
        
        # Get with context (should not affect lookup)
        cached = self.cache.get(query, context=context)
        assert cached is not None
    
    def test_compute_embedding(self):
        """Test embedding computation."""
        embedding = self.cache._compute_embedding("test query")
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        embedding = np.array([0.1, 0.2, 0.3])
        key1 = self.cache._generate_cache_key(embedding)
        key2 = self.cache._generate_cache_key(embedding)
        
        # Same embedding should generate same key
        assert key1 == key2
        assert key1.startswith(self.cache.cache_prefix)
        
        # Different embedding should generate different key
        different_embedding = np.array([0.4, 0.5, 0.6])
        key3 = self.cache._generate_cache_key(different_embedding)
        assert key3 != key1


class TestLLMResponseCacheWithRedis:
    """Test suite for LLMResponseCache with Redis."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.setex.return_value = True
        mock_client.get.return_value = None
        mock_client.delete.return_value = 1
        mock_client.zadd.return_value = 1
        mock_client.zrem.return_value = 1
        mock_client.zcard.return_value = 0
        mock_client.zrange.return_value = []
        return mock_client
    
    def test_redis_initialization(self, mock_redis):
        """Test cache initialization with Redis."""
        with patch('app.ai.llm_intent.response_cache.REDIS_AVAILABLE', True):
            with patch('app.ai.llm_intent.response_cache.get_redis_client', return_value=mock_redis):
                cache = LLMResponseCache(enabled=True)
                
                assert cache.redis_available is True
                assert cache.redis_client is not None
                mock_redis.ping.assert_called_once()
    
    def test_redis_set(self, mock_redis):
        """Test cache set with Redis."""
        with patch('app.ai.llm_intent.response_cache.REDIS_AVAILABLE', True):
            with patch('app.ai.llm_intent.response_cache.get_redis_client', return_value=mock_redis):
                cache = LLMResponseCache(enabled=True)
                
                success = cache.set("test query", {"intent": "TEST"})
                assert success is True
                
                # Should call Redis setex
                assert mock_redis.setex.called
    
    def test_redis_clear(self, mock_redis):
        """Test cache clear with Redis."""
        mock_redis.zrange.return_value = ["key1", "key2"]
        
        with patch('app.ai.llm_intent.response_cache.REDIS_AVAILABLE', True):
            with patch('app.ai.llm_intent.response_cache.get_redis_client', return_value=mock_redis):
                cache = LLMResponseCache(enabled=True)
                
                success = cache.clear()
                assert success is True
                
                # Should delete keys and index
                assert mock_redis.delete.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

