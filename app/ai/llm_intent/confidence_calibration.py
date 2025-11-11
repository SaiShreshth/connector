"""
Confidence Calibration System

Tracks historical accuracy vs. reported confidence and adjusts scores accordingly.
Implements calibration logic, threshold adjustment, and fallback for low-confidence responses.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from loguru import logger

# Try to import Redis for historical tracking
try:
    import redis
    from app.queue.config import queue_config
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory calibration tracking")


class ConfidenceCalibrationTracker:
    """
    Tracks historical accuracy vs. reported confidence for calibration.
    
    Stores:
    - Actual outcomes (correct/incorrect) per confidence bin
    - Accuracy metrics per action_code
    - Confidence distribution statistics
    """
    
    def __init__(self, redis_client=None):
        """Initialize calibration tracker."""
        self.redis_client = None
        self._use_redis = False
        
        if REDIS_AVAILABLE:
            try:
                if redis_client:
                    self.redis_client = redis_client
                else:
                    # Try to use queue_manager's Redis connection
                    try:
                        from app.queue.queue_manager import queue_manager
                        if queue_manager and queue_manager.is_available() and hasattr(queue_manager, 'redis_client'):
                            self.redis_client = queue_manager.redis_client
                        else:
                            raise AttributeError("Queue manager not available")
                    except (ImportError, AttributeError):
                        # Create new connection
                        self.redis_client = redis.Redis(
                            host=queue_config.REDIS_HOST,
                            port=queue_config.REDIS_PORT,
                            db=queue_config.REDIS_DB,
                            password=queue_config.REDIS_PASSWORD,
                            decode_responses=True
                        )
                        self.redis_client.ping()
                
                self._use_redis = True
                logger.info("✅ ConfidenceCalibrationTracker using Redis")
            except Exception as e:
                logger.warning(f"⚠️ Redis not available for calibration tracking: {e}. Using in-memory.")
                self._use_redis = False
        
        # In-memory fallback
        if not self._use_redis:
            self._in_memory_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def _get_key(self, action_code: Optional[str] = None) -> str:
        """Get Redis key for calibration data."""
        if action_code:
            return f"chatns:calibration:{action_code}"
        return "chatns:calibration:global"
    
    def record_prediction(
        self,
        action_code: Optional[str],
        reported_confidence: float,
        actual_outcome: bool,
        request_id: Optional[str] = None
    ):
        """
        Record a prediction and its actual outcome for calibration.
        
        Args:
            action_code: Predicted action code
            reported_confidence: Confidence score reported by LLM (0-1)
            actual_outcome: True if prediction was correct, False otherwise
            request_id: Optional request identifier
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        record = {
            "timestamp": timestamp,
            "reported_confidence": reported_confidence,
            "actual_outcome": actual_outcome,
            "request_id": request_id,
        }
        
        try:
            if self._use_redis and self.redis_client:
                # Store in Redis as a list (using Redis LIST)
                key = self._get_key(action_code)
                self.redis_client.rpush(key, json.dumps(record))
                
                # Set TTL to 90 days (retain calibration data for ~3 months)
                self.redis_client.expire(key, 90 * 24 * 3600)
                
                # Also store in global tracking
                global_key = self._get_key(None)
                self.redis_client.rpush(global_key, json.dumps({**record, "action_code": action_code}))
                self.redis_client.expire(global_key, 90 * 24 * 3600)
            else:
                # In-memory storage
                key = action_code or "global"
                self._in_memory_store[key].append(record)
                
                # Keep only last 10000 records per action_code
                if len(self._in_memory_store[key]) > 10000:
                    self._in_memory_store[key] = self._in_memory_store[key][-10000:]
            
            logger.debug(f"Recorded calibration: {action_code} | confidence={reported_confidence:.2f} | correct={actual_outcome}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record calibration data: {e}")
    
    def get_calibration_stats(
        self,
        action_code: Optional[str] = None,
        min_samples: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Get calibration statistics for an action_code or globally.
        
        Args:
            action_code: Action code to analyze, or None for global stats
            min_samples: Minimum samples required for reliable calibration
        
        Returns:
            Dict with calibration metrics or None if insufficient data
        """
        try:
            if self._use_redis and self.redis_client:
                key = self._get_key(action_code)
                records_json = self.redis_client.lrange(key, 0, -1)
                records = [json.loads(r) for r in records_json]
            else:
                key = action_code or "global"
                records = self._in_memory_store.get(key, [])
            
            if len(records) < min_samples:
                return None
            
            # Calculate calibration metrics
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            bin_counts = defaultdict(int)
            bin_correct = defaultdict(int)
            
            for record in records:
                conf = record["reported_confidence"]
                outcome = record["actual_outcome"]
                
                # Find bin
                for i in range(len(bins) - 1):
                    if bins[i] <= conf < bins[i + 1]:
                        bin_key = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                        bin_counts[bin_key] += 1
                        if outcome:
                            bin_correct[bin_key] += 1
                        break
            
            # Calculate accuracy per bin
            bin_accuracy = {}
            for bin_key in bin_counts:
                bin_accuracy[bin_key] = bin_correct[bin_key] / bin_counts[bin_key] if bin_counts[bin_key] > 0 else 0.0
            
            # Overall accuracy
            total_correct = sum(record["actual_outcome"] for record in records)
            overall_accuracy = total_correct / len(records) if records else 0.0
            
            # Average confidence
            avg_confidence = sum(r["reported_confidence"] for r in records) / len(records) if records else 0.0
            
            return {
                "action_code": action_code or "global",
                "total_samples": len(records),
                "overall_accuracy": overall_accuracy,
                "average_confidence": avg_confidence,
                "calibration_error": abs(overall_accuracy - avg_confidence),
                "bin_accuracy": bin_accuracy,
                "bin_counts": dict(bin_counts),
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get calibration stats: {e}")
            return None


class ConfidenceCalibrator:
    """
    Calibrates confidence scores based on historical accuracy.
    Adjusts reported confidence to match observed accuracy.
    """
    
    def __init__(self, tracker: Optional[ConfidenceCalibrationTracker] = None):
        """Initialize calibrator with tracker."""
        self.tracker = tracker or ConfidenceCalibrationTracker()
        self._calibration_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 3600  # Cache calibration stats for 1 hour
        
    def calibrate_confidence(
        self,
        action_code: Optional[str],
        reported_confidence: float,
        use_cache: bool = True
    ) -> float:
        """
        Calibrate a reported confidence score based on historical accuracy.
        
        Args:
            action_code: Action code for the prediction
            reported_confidence: Original confidence score (0-1)
            use_cache: Whether to use cached calibration stats
        
        Returns:
            Calibrated confidence score (0-1)
        """
        # Get calibration stats
        cache_key = action_code or "global"
        stats = None
        
        if use_cache and cache_key in self._calibration_cache:
            cache_entry = self._calibration_cache[cache_key]
            if (datetime.now(timezone.utc).timestamp() - cache_entry["timestamp"]) < self._cache_ttl:
                stats = cache_entry["stats"]
        
        if stats is None:
            stats = self.tracker.get_calibration_stats(action_code)
            if stats:
                self._calibration_cache[cache_key] = {
                    "stats": stats,
                    "timestamp": datetime.now(timezone.utc).timestamp()
                }
        
        # If no calibration data available, return original confidence
        if not stats:
            return reported_confidence
        
        # Find the bin for this confidence level
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_accuracy_map = stats.get("bin_accuracy", {})
        
        # Find matching bin and use its observed accuracy
        calibrated = reported_confidence
        for i in range(len(bins) - 1):
            if bins[i] <= reported_confidence < bins[i + 1]:
                bin_key = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                if bin_key in bin_accuracy_map:
                    # Use observed accuracy as calibrated confidence
                    calibrated = bin_accuracy_map[bin_key]
                break
        
        # Apply global adjustment if calibration error is significant
        calibration_error = stats.get("calibration_error", 0.0)
        if calibration_error > 0.1:  # Significant miscalibration
            # Adjust towards observed accuracy
            adjustment_factor = stats.get("overall_accuracy", reported_confidence) / stats.get("average_confidence", 1.0)
            calibrated = min(1.0, max(0.0, reported_confidence * adjustment_factor))
        
        return round(calibrated, 3)
    
    def should_trigger_fallback(
        self,
        action_code: Optional[str],
        calibrated_confidence: float,
        fallback_threshold: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Determine if low confidence should trigger fallback behavior.
        
        Args:
            action_code: Action code for the prediction
            calibrated_confidence: Calibrated confidence score
            fallback_threshold: Minimum confidence threshold (default 0.5)
        
        Returns:
            Tuple of (should_fallback, reason)
        """
        if calibrated_confidence < fallback_threshold:
            return True, f"low_confidence_{calibrated_confidence:.2f}"
        
        # Check if action_code has high historical accuracy requirements
        stats = self.tracker.get_calibration_stats(action_code)
        if stats:
            if stats["overall_accuracy"] < 0.6 and calibrated_confidence < 0.7:
                return True, "low_historical_accuracy"
        
        return False, "sufficient_confidence"


# Global instances
_calibration_tracker: Optional[ConfidenceCalibrationTracker] = None
_calibrator: Optional[ConfidenceCalibrator] = None


def get_calibrator() -> ConfidenceCalibrator:
    """Get global confidence calibrator instance."""
    global _calibrator, _calibration_tracker
    if _calibrator is None:
        if _calibration_tracker is None:
            _calibration_tracker = ConfidenceCalibrationTracker()
        _calibrator = ConfidenceCalibrator(_calibration_tracker)
    return _calibrator


def get_tracker() -> ConfidenceCalibrationTracker:
    """Get global calibration tracker instance."""
    global _calibration_tracker
    if _calibration_tracker is None:
        _calibration_tracker = ConfidenceCalibrationTracker()
    return _calibration_tracker

