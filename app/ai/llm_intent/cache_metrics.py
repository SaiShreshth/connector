"""
Cache Metrics Tracking for LLM Response Cache

Tracks cache performance metrics including hit rate, miss rate, API calls saved,
and latency statistics.
"""

import time
from typing import Dict, Any, Optional
from collections import deque
import threading


class CacheMetrics:
    """
    Thread-safe metrics tracker for LLM response cache.
    
    Tracks:
    - Cache hits and misses
    - API calls saved
    - Cache lookup latency
    - Cache size and memory usage
    - Top cached queries
    """
    
    def __init__(self, track_top_queries: int = 100):
        """
        Initialize cache metrics tracker.
        
        Args:
            track_top_queries: Number of top queries to track (default: 100)
        """
        self._lock = threading.RLock()  # Use RLock for reentrant locking (needed for get_summary)
        
        # Hit/miss counters
        self.hits = 0
        self.misses = 0
        
        # Latency tracking (keep last 1000 measurements)
        self._latencies = deque(maxlen=1000)
        
        # Query frequency tracking (query -> hit_count)
        self._query_hits: Dict[str, int] = {}
        self._track_top_queries = track_top_queries
        
        # Start time for uptime calculation
        self._start_time = time.time()
    
    def record_hit(self, query: Optional[str] = None, latency_ms: Optional[float] = None):
        """
        Record a cache hit.
        
        Args:
            query: The query that hit (optional, for tracking popular queries)
            latency_ms: Cache lookup latency in milliseconds (optional)
        """
        with self._lock:
            self.hits += 1
            
            if latency_ms is not None:
                self._latencies.append(latency_ms)
            
            if query:
                # Track query frequency
                self._query_hits[query] = self._query_hits.get(query, 0) + 1
                
                # Limit memory usage by keeping only top queries
                # Clean up whenever we exceed the limit (not just at 2x)
                if len(self._query_hits) > self._track_top_queries:
                    # Keep top N queries
                    sorted_queries = sorted(
                        self._query_hits.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    self._query_hits = dict(sorted_queries[:self._track_top_queries])
    
    def record_miss(self, latency_ms: Optional[float] = None):
        """
        Record a cache miss.
        
        Args:
            latency_ms: Cache lookup latency in milliseconds (optional)
        """
        with self._lock:
            self.misses += 1
            
            if latency_ms is not None:
                self._latencies.append(latency_ms)
    
    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate as percentage.
        
        Returns:
            Hit rate percentage (0-100)
        """
        with self._lock:
            total = self.hits + self.misses
            return (self.hits / total * 100) if total > 0 else 0.0
    
    def get_miss_rate(self) -> float:
        """
        Calculate cache miss rate as percentage.
        
        Returns:
            Miss rate percentage (0-100)
        """
        return 100.0 - self.get_hit_rate()
    
    def get_api_calls_saved(self) -> int:
        """
        Get number of API calls saved by cache hits.
        
        Returns:
            Number of API calls saved (equal to number of hits)
        """
        with self._lock:
            return self.hits
    
    def get_avg_latency(self) -> float:
        """
        Get average cache lookup latency in milliseconds.
        
        Returns:
            Average latency in ms, or 0.0 if no data
        """
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)
    
    def get_p95_latency(self) -> float:
        """
        Get 95th percentile cache lookup latency in milliseconds.
        
        Returns:
            P95 latency in ms, or 0.0 if no data
        """
        with self._lock:
            if not self._latencies:
                return 0.0
            sorted_latencies = sorted(self._latencies)
            index = int(len(sorted_latencies) * 0.95)
            return sorted_latencies[index] if index < len(sorted_latencies) else 0.0
    
    def get_top_queries(self, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Get top N most frequently cached queries.
        
        Args:
            limit: Number of top queries to return (default: 10)
            
        Returns:
            List of dicts with 'query' and 'hit_count' keys
        """
        with self._lock:
            sorted_queries = sorted(
                self._query_hits.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [
                {"query": query, "hit_count": count}
                for query, count in sorted_queries[:limit]
            ]
    
    def get_uptime_seconds(self) -> float:
        """
        Get cache uptime in seconds.
        
        Returns:
            Uptime in seconds since metrics started
        """
        return time.time() - self._start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete metrics summary.
        
        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            total_requests = self.hits + self.misses
            uptime = self.get_uptime_seconds()  # This doesn't need lock, so it's safe
            
            # Calculate hit rate inline to avoid nested lock acquisition
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
            miss_rate = 100.0 - hit_rate
            
            # Calculate avg latency inline
            avg_latency = (sum(self._latencies) / len(self._latencies)) if self._latencies else 0.0
            
            # Calculate P95 latency inline
            p95_latency = 0.0
            if self._latencies:
                sorted_latencies = sorted(self._latencies)
                index = int(len(sorted_latencies) * 0.95)
                p95_latency = sorted_latencies[index] if index < len(sorted_latencies) else 0.0
            
            return {
                "hit_rate_percent": round(hit_rate, 2),
                "miss_rate_percent": round(miss_rate, 2),
                "total_requests": total_requests,
                "cache_hits": self.hits,
                "cache_misses": self.misses,
                "api_calls_saved": self.hits,
                "avg_latency_ms": round(avg_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "uptime_seconds": round(uptime, 2),
                "uptime_hours": round(uptime / 3600, 2),
                "requests_per_hour": round((total_requests / uptime * 3600), 2) if uptime > 0 else 0.0,
            }
    
    def reset(self):
        """Reset all metrics to initial state."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self._latencies.clear()
            self._query_hits.clear()
            self._start_time = time.time()


# Singleton instance
_metrics_instance: Optional[CacheMetrics] = None


def get_cache_metrics() -> CacheMetrics:
    """
    Get or create singleton CacheMetrics instance.
    
    Returns:
        Singleton CacheMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = CacheMetrics()
    return _metrics_instance


__all__ = ["CacheMetrics", "get_cache_metrics"]

