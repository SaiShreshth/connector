"""
Status Store - In-memory and Redis-based request status tracking.
"""

from datetime import datetime, timezone
from typing import Optional, Dict
from app.schemas.request_status import RequestStatus, ResultSchema

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from loguru import logger

from app.schemas.request_status import RequestStatus, ResultSchema

# Try to import Redis
try:
    import redis
    from app.queue.config import queue_config
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, StatusStore will use in-memory fallback")


class InMemoryStatusStore:
    """
    Simple in-memory fallback for request status tracking.
    Used when Redis is unavailable or for testing.
    """

    def __init__(self):
        self._store: Dict[str, RequestStatus] = {}

    def save(self, status: RequestStatus):
        """Save or update a RequestStatus in memory."""
        self._store[status.request_id] = status

    def get(self, request_id: str) -> Optional[RequestStatus]:
        """Retrieve a RequestStatus by ID."""
        return self._store.get(request_id)

    def update_status(self, request_id: str, new_status: str):
        """Update the status of a request."""
        if request_id in self._store:
            entry = self._store[request_id]
            entry.status = new_status
            if new_status == "PROCESSING":
                entry.started_at = datetime.now(timezone.utc)
            elif new_status in ["COMPLETED", "FAILED"]:
                entry.completed_at = datetime.now(timezone.utc)
            self._store[request_id] = entry

    def complete_request(self, request_id: str, result: Optional[ResultSchema] = None):
        """Mark request as completed and store result."""
        if request_id in self._store:
            entry = self._store[request_id]
            entry.status = "COMPLETED"
            entry.completed_at = datetime.now(timezone.utc)
            entry.result = result
            self._store[request_id] = entry

    def clear(self):
        """Clear all entries (for testing)."""
        self._store.clear()
    def complete_request(self, request_id: str, result: Optional[ResultSchema] = None, error: Optional[Dict[str, Any]] = None):
        """Mark request as completed and store result."""
        if request_id in self._store:
            entry = self._store[request_id]
            entry.status = "FAILED" if error else "COMPLETED"
            entry.completed_at = datetime.now(timezone.utc)
            entry.result = result
            entry.error = error
            self._store[request_id] = entry

    def fail_request(self, request_id: str, error: Dict[str, Any]):
        """Mark request as failed with error details."""
        self.complete_request(request_id, result=None, error=error)

    def clear(self):
        """Clear all entries (for testing)."""
        self._store.clear()


class RedisStatusStore:
    """
    Redis-based status store for request tracking.
    Provides fast lookups (<10ms p95) and automatic expiration.
    """

    def __init__(self, redis_client=None, fail_silently: bool = False):
        """
        Initialize Redis status store.
        
        Args:
            redis_client: Optional Redis client (will create if not provided)
            fail_silently: If True, fall back to in-memory on Redis failure
        """
        self.redis_client = None
        self.fail_silently = fail_silently
        self._fallback_store = InMemoryStatusStore()
        self._use_fallback = False
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory fallback")
            self._use_fallback = True
            return
            
        try:
            if redis_client:
                self.redis_client = redis_client
            else:
                # Try to use existing queue_manager connection first
                try:
                    from app.queue.queue_manager import queue_manager
                    if queue_manager and queue_manager.is_available() and hasattr(queue_manager, 'redis_client'):
                        self.redis_client = queue_manager.redis_client
                    else:
                        raise AttributeError("Queue manager not available")
                except (ImportError, AttributeError):
                    # Create new connection using queue config
                    self.redis_client = redis.Redis(
                        host=queue_config.REDIS_HOST,
                        port=queue_config.REDIS_PORT,
                        db=queue_config.REDIS_DB,
                        password=queue_config.REDIS_PASSWORD,
                        decode_responses=True
                    )
                    self.redis_client.ping()
            
            logger.info("✅ RedisStatusStore initialized successfully")
        except Exception as e:
            if fail_silently:
                logger.warning(f"⚠️ Redis not available for status store: {e}. Using in-memory fallback.")
                self._use_fallback = True
            else:
                logger.error(f"❌ Failed to initialize RedisStatusStore: {e}")
                raise

    def _get_key(self, request_id: str) -> str:
        """Get Redis key for request status."""
        return f"chatns:status:{request_id}"

    def _is_available(self) -> bool:
        """Check if Redis is available."""
        if self._use_fallback:
            return False
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception:
            if self.fail_silently:
                self._use_fallback = True
            return False

    def save(self, status: RequestStatus):
        """
        Save or update a RequestStatus in Redis.
        
        Sets TTL based on status:
        - QUEUED/PROCESSING: 24 hours
        - COMPLETED/FAILED: 1 hour
        """
        if not self._is_available():
            return self._fallback_store.save(status)
        
        try:
            key = self._get_key(status.request_id)
            status_dict = status.model_dump(mode='json')
            # Convert datetime to ISO string for JSON
            for field in ['queued_at', 'started_at', 'completed_at']:
                if status_dict.get(field) and isinstance(status_dict[field], datetime):
                    status_dict[field] = status_dict[field].isoformat() + 'Z'
            
            # Serialize result if present
            if status_dict.get('result'):
                status_dict['result'] = status_dict['result']
            
            self.redis_client.set(key, json.dumps(status_dict))
            
            # Set TTL based on status (1 hour for completed/failed, 24 hours for others)
            if status.status in ["COMPLETED", "FAILED"]:
                ttl = 3600  # 1 hour
            else:
                ttl = 86400  # 24 hours for queued/processing
            
            self.redis_client.expire(key, ttl)
            logger.debug(f"✅ Saved status for {status.request_id}: {status.status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save status to Redis: {e}")
            if self.fail_silently:
                self._fallback_store.save(status)

    def get(self, request_id: str) -> Optional[RequestStatus]:
        """
        Retrieve a RequestStatus by ID from Redis.
        Fast lookup (<10ms p95).
        """
        if not self._is_available():
            return self._fallback_store.get(request_id)
        
        try:
            key = self._get_key(request_id)
            data = self.redis_client.get(key)
            
            if not data:
                return None
            
            status_dict = json.loads(data)
            
            # Parse datetime strings back to datetime objects
            for field in ['queued_at', 'started_at', 'completed_at']:
                if status_dict.get(field):
                    status_dict[field] = datetime.fromisoformat(status_dict[field].replace('Z', '+00:00'))
            
            # Parse result if present
            if status_dict.get('result'):
                status_dict['result'] = ResultSchema(**status_dict['result'])
            
            return RequestStatus(**status_dict)
            
        except Exception as e:
            logger.error(f"❌ Failed to get status from Redis: {e}")
            if self.fail_silently:
                return self._fallback_store.get(request_id)
            return None

    def update_status(self, request_id: str, new_status: str):
        """
        Update the status of a request.
        Automatically sets started_at for PROCESSING, completed_at for COMPLETED/FAILED.
        """
        if not self._is_available():
            return self._fallback_store.update_status(request_id, new_status)
        
        try:
            status = self.get(request_id)
            if not status:
                logger.warning(f"⚠️ Cannot update status for unknown request_id: {request_id}")
                return
            
            status.status = new_status
            
            # Set timestamps based on status
            now = datetime.now(timezone.utc)
            if new_status == "PROCESSING" and not status.started_at:
                status.started_at = now
            elif new_status in ["COMPLETED", "FAILED"] and not status.completed_at:
                status.completed_at = now
            
            self.save(status)
            logger.debug(f"✅ Updated status for {request_id}: {new_status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update status: {e}")
            if self.fail_silently:
                self._fallback_store.update_status(request_id, new_status)

    def complete_request(self, request_id: str, result: Optional[ResultSchema] = None, error: Optional[Dict[str, Any]] = None):
        """
        Mark request as completed and store result.
        Sets status to COMPLETED or FAILED based on error presence.
        """
        if not self._is_available():
            return self._fallback_store.complete_request(request_id, result, error)
        
        try:
            status = self.get(request_id)
            if not status:
                logger.warning(f"⚠️ Cannot complete unknown request_id: {request_id}")
                return
            
            status.status = "FAILED" if error else "COMPLETED"
            status.completed_at = datetime.now(timezone.utc)
            status.result = result
            status.error = error
            
            self.save(status)
            logger.debug(f"✅ Completed request {request_id}: {status.status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to complete request: {e}")
            if self.fail_silently:
                self._fallback_store.complete_request(request_id, result, error)

    def fail_request(self, request_id: str, error: Dict[str, Any]):
        """Mark request as failed with error details."""
        self.complete_request(request_id, result=None, error=error)


# Global instance - use Redis if available, fallback to in-memory
try:
    status_store = RedisStatusStore(fail_silently=True)
except Exception:
    status_store = InMemoryStatusStore()
    logger.warning("⚠️ Using InMemoryStatusStore fallback")

