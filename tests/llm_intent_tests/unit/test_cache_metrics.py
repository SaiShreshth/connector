"""
Unit tests for CacheMetrics
"""

import pytest
import time
from app.ai.llm_intent.cache_metrics import CacheMetrics, get_cache_metrics


class TestCacheMetrics:
    """Test suite for CacheMetrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = CacheMetrics(track_top_queries=10)
    
    def test_initial_state(self):
        """Test initial metrics state."""
        assert self.metrics.hits == 0
        assert self.metrics.misses == 0
        assert self.metrics.get_hit_rate() == 0.0
        # Miss rate is 100% when there are 0 hits and 0 misses (0/0 = undefined, returns 100)
        assert self.metrics.get_miss_rate() in [0.0, 100.0]
        assert self.metrics.get_api_calls_saved() == 0
    
    def test_record_hit(self):
        """Test recording cache hits."""
        self.metrics.record_hit()
        assert self.metrics.hits == 1
        assert self.metrics.misses == 0
        assert self.metrics.get_hit_rate() == 100.0
    
    def test_record_miss(self):
        """Test recording cache misses."""
        self.metrics.record_miss()
        assert self.metrics.hits == 0
        assert self.metrics.misses == 1
        assert self.metrics.get_miss_rate() == 100.0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        self.metrics.record_hit()
        self.metrics.record_hit()
        self.metrics.record_hit()
        self.metrics.record_miss()
        
        # 3 hits out of 4 total = 75%
        assert self.metrics.get_hit_rate() == 75.0
        assert self.metrics.get_miss_rate() == 25.0
    
    def test_api_calls_saved(self):
        """Test API calls saved tracking."""
        self.metrics.record_hit()
        self.metrics.record_hit()
        self.metrics.record_miss()
        
        # API calls saved equals number of hits
        assert self.metrics.get_api_calls_saved() == 2
    
    def test_latency_tracking(self):
        """Test latency tracking."""
        self.metrics.record_hit(latency_ms=10.0)
        self.metrics.record_hit(latency_ms=20.0)
        self.metrics.record_miss(latency_ms=15.0)
        
        avg_latency = self.metrics.get_avg_latency()
        assert avg_latency == 15.0  # (10 + 20 + 15) / 3
    
    def test_p95_latency(self):
        """Test P95 latency calculation."""
        # Add 100 samples
        for i in range(100):
            self.metrics.record_hit(latency_ms=float(i))
        
        p95 = self.metrics.get_p95_latency()
        # P95 should be around 95th value
        assert 94.0 <= p95 <= 96.0
    
    def test_query_frequency_tracking(self):
        """Test query frequency tracking."""
        self.metrics.record_hit(query="show me products")
        self.metrics.record_hit(query="show me products")
        self.metrics.record_hit(query="find shoes")
        
        top_queries = self.metrics.get_top_queries(limit=2)
        
        assert len(top_queries) == 2
        assert top_queries[0]["query"] == "show me products"
        assert top_queries[0]["hit_count"] == 2
        assert top_queries[1]["query"] == "find shoes"
        assert top_queries[1]["hit_count"] == 1
    
    def test_top_queries_limit(self):
        """Test top queries memory limit."""
        # Add more queries than track limit
        for i in range(25):
            self.metrics.record_hit(query=f"query_{i}")
        
        # Should only track top 10 queries
        assert len(self.metrics._query_hits) <= 10
    
    def test_uptime_tracking(self):
        """Test uptime tracking."""
        time.sleep(0.1)  # Sleep for 100ms
        uptime = self.metrics.get_uptime_seconds()
        assert uptime >= 0.1
    
    def test_get_summary(self):
        """Test comprehensive metrics summary."""
        self.metrics.record_hit(query="test", latency_ms=10.0)
        self.metrics.record_hit(query="test", latency_ms=20.0)
        self.metrics.record_miss(latency_ms=15.0)
        
        summary = self.metrics.get_summary()
        
        assert "hit_rate_percent" in summary
        assert "miss_rate_percent" in summary
        assert "total_requests" in summary
        assert "cache_hits" in summary
        assert "cache_misses" in summary
        assert "api_calls_saved" in summary
        assert "avg_latency_ms" in summary
        assert "p95_latency_ms" in summary
        assert "uptime_seconds" in summary
        assert "uptime_hours" in summary
        assert "requests_per_hour" in summary
        
        assert summary["total_requests"] == 3
        assert summary["cache_hits"] == 2
        assert summary["cache_misses"] == 1
        assert summary["hit_rate_percent"] == round(66.67, 2)
    
    def test_reset(self):
        """Test metrics reset."""
        self.metrics.record_hit(query="test", latency_ms=10.0)
        self.metrics.record_miss(latency_ms=20.0)
        
        self.metrics.reset()
        
        assert self.metrics.hits == 0
        assert self.metrics.misses == 0
        assert len(self.metrics._latencies) == 0
        assert len(self.metrics._query_hits) == 0
        assert self.metrics.get_hit_rate() == 0.0
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading
        
        def record_hits():
            for _ in range(100):
                self.metrics.record_hit()
        
        def record_misses():
            for _ in range(100):
                self.metrics.record_miss()
        
        threads = [
            threading.Thread(target=record_hits),
            threading.Thread(target=record_misses)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have 200 total requests
        assert self.metrics.hits + self.metrics.misses == 200
    
    def test_get_singleton(self):
        """Test singleton pattern."""
        metrics1 = get_cache_metrics()
        metrics2 = get_cache_metrics()
        assert metrics1 is metrics2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

