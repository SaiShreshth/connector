"""
Queue Monitoring

Provides metrics and monitoring for queue infrastructure.
"""

import time
from typing import Dict, Any
from loguru import logger

from .queue_manager import queue_manager
from .config import queue_config


class QueueMonitor:
    """Monitors queue health and performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            "messages_enqueued": 0,
            "messages_dequeued": 0,
            "messages_failed": 0,
            "messages_retried": 0,
            "total_processing_time": 0.0
        }
    
    def record_enqueue(self) -> None:
        """Record message enqueued"""
        self.metrics["messages_enqueued"] += 1
    
    def record_dequeue(self) -> None:
        """Record message dequeued"""
        self.metrics["messages_dequeued"] += 1
    
    def record_failure(self) -> None:
        """Record message failure"""
        self.metrics["messages_failed"] += 1
    
    def record_retry(self) -> None:
        """Record message retry"""
        self.metrics["messages_retried"] += 1
    
    def record_processing_time(self, duration: float) -> None:
        """Record processing time"""
        self.metrics["total_processing_time"] += duration
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            metrics: Performance and health metrics
        """
        queue_stats = {}
        if queue_manager:
            queue_stats = queue_manager.get_queue_stats()
        else:
            queue_stats = {
                "status": "unavailable",
                "ambiguous_queue_size": 0,
                "result_queue_size": 0,
                "dead_letter_queue_size": 0
            }
        
        uptime = time.time() - self.start_time
        
        avg_processing_time = 0.0
        if self.metrics["messages_dequeued"] > 0:
            avg_processing_time = (
                self.metrics["total_processing_time"] / 
                self.metrics["messages_dequeued"]
            )
        
        return {
            **queue_stats,
            "uptime_seconds": round(uptime, 2),
            "messages_enqueued": self.metrics["messages_enqueued"],
            "messages_dequeued": self.metrics["messages_dequeued"],
            "messages_failed": self.metrics["messages_failed"],
            "messages_retried": self.metrics["messages_retried"],
            "avg_processing_time": round(avg_processing_time, 3),
            "throughput": round(
                self.metrics["messages_dequeued"] / uptime if uptime > 0 else 0, 
                2
            )
        }
    
    def log_metrics(self) -> None:
        """Log current metrics"""
        metrics = self.get_metrics()
        logger.info(f"📊 Queue Metrics: {metrics}")


# Global monitor instance
queue_monitor = QueueMonitor()
