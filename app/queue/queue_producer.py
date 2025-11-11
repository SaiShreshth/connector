"""
Queue Producer for Ambiguous Queries

Publishes ambiguous queries from rule-based module to the intent classification queue.
Uses Redis for queue management.
"""

import json
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, TypedDict, Union

from loguru import logger

# Import queue manager and status store
try:
    from app.queue.queue_manager import queue_manager
    from app.core.status_store import status_store
    from app.schemas.request_status import RequestStatus
    QUEUE_AVAILABLE = True
    STATUS_STORE_AVAILABLE = True
except ImportError:
    queue_manager = None
    status_store = None
    QUEUE_AVAILABLE = False
    STATUS_STORE_AVAILABLE = False

from app.queue.config import queue_config


# -----------------------
# Type definitions
# -----------------------
class IntentScore(TypedDict):
    intent: str
    score: float


class RuleBasedResult(TypedDict, total=False):
    action_code: str
    confidence: float
    matched_keywords: List[str]


class QueueMessage(TypedDict):
    """Standard message format for queue communication."""
    request_id: str  # UUID v4
    user_query: str
    session_id: Optional[str]
    user_id: Optional[str]
    conversation_history: List[Dict[str, Any]]
    rule_based_result: Optional[RuleBasedResult]
    timestamp: str  # ISO8601Z format
    priority: str  # "normal" | "high"
    metadata: Dict[str, Any]


# -----------------------
# Custom exception
# -----------------------
class QueuePublishError(Exception):
    """Custom exception for queue publishing errors."""


# -----------------------
# Producer class
# -----------------------
class IntentQueueProducer:
    """
    Handles publishing of ambiguous queries to the intent classification queue via Redis.
    
    Features:
    - Standard message format for queue communication
    - UUID v4 request_id generation
    - Batching support
    - Idempotency checking
    - Error handling
    """

    def __init__(self, queue_config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize queue producer.
        
        Args:
            queue_config_dict: Optional config dict (uses queue_config from app.queue.config by default)
        """
        self.queue_config_dict = queue_config_dict or {}
        self.batch_size = int(self.queue_config_dict.get("batch_size", 1))
        self.batch_timeout = float(self.queue_config_dict.get("batch_timeout", 60.0))  # seconds
        self.enable_idempotency = bool(self.queue_config_dict.get("enable_idempotency", False))
        self.idempotency_ttl = int(self.queue_config_dict.get("idempotency_ttl", 300))  # 5 minutes
        
        # Idempotency store (in-memory for simplicity, could use Redis)
        self._idempotency_store: Dict[str, float] = {}
        self._idempotency_lock = threading.Lock()
        
        logger.info(
            f"IntentQueueProducer initialized | batch_size={self.batch_size} | "
            f"idempotency={self.enable_idempotency}"
        )

    def publish_ambiguous_query(
        self,
        query: str,
        rule_based_result: Optional[RuleBasedResult] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        priority: Union[bool, str] = False,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Publish an ambiguous query to the intent classification queue.
        
        Args:
            query: User's ambiguous query
            rule_based_result: Result from rule-based classifier
            session_id: Session identifier
            user_id: User identifier
            conversation_history: Previous conversation context
            priority: Priority ("high", "normal", or bool)
            metadata: Additional metadata
            request_id: Optional request ID (generates UUID v4 if not provided)
        
        Returns:
            str: request_id for tracking
        
        Raises:
            QueuePublishError: If publishing fails
        """
        # Generate UUID v4 request_id
        request_id = request_id or str(uuid.uuid4())
        
        # Idempotency check
        if self.enable_idempotency:
            now_ts = datetime.now(timezone.utc).timestamp()
            with self._idempotency_lock:
                expired = [k for k, v in self._idempotency_store.items() if v < now_ts]
                for k in expired:
                    del self._idempotency_store[k]
                if request_id in self._idempotency_store:
                    logger.info(f"Duplicate request_id {request_id} detected; skipping enqueue")
                    return request_id
                self._idempotency_store[request_id] = now_ts + self.idempotency_ttl
        
        # Priority normalization
        if isinstance(priority, bool):
            priority_str = "high" if priority else "normal"
        else:
            priority_str = str(priority) if priority in ["high", "normal"] else "normal"
        
        # Build message with standard format
        message: QueueMessage = {
            "request_id": request_id,
            "user_query": query,
            "session_id": session_id,
            "user_id": user_id,
            "conversation_history": conversation_history or [],
            "rule_based_result": rule_based_result,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z",
            "priority": priority_str,
            "metadata": metadata or {},
        }
        
        try:
            # Check queue availability
            if not QUEUE_AVAILABLE or queue_manager is None or not queue_manager.is_available():
                raise QueuePublishError("Queue manager not available")
            
            # Convert priority string to integer for queue_manager
            priority_int = queue_config.PRIORITY_HIGH if priority_str == "high" else queue_config.PRIORITY_NORMAL
            
            # Create status in status store
            if STATUS_STORE_AVAILABLE and status_store:
                try:
                    status = RequestStatus(
                        request_id=request_id,
                        status="QUEUED",
                        queued_at=datetime.now(timezone.utc),
                        retry_count=0
                    )
                    status_store.save(status)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create status entry: {e}")
            
            # Extract context from message for queue_manager (backward compatibility)
            # queue_manager expects: query, context dict, priority int
            context = {
                "request_id": request_id,
                "session_id": session_id,
                "user_id": user_id,
                "conversation_history": conversation_history or [],
                "rule_based_result": rule_based_result,
                "priority": priority_str,
                "metadata": metadata or {},
            }
            
            # Use existing queue_manager to enqueue
            # Store full message in context for worker to access
            context["_full_message"] = message
            
            message_id = queue_manager.enqueue_ambiguous_query(
                query=query,
                context=context,
                priority=priority_int
            )
            
            if not message_id:
                raise QueuePublishError("Failed to enqueue message")
            
            logger.info(
                f"✅ Published ambiguous query to queue | "
                f"request_id={request_id} | query='{query[:50]}...' | priority={priority_str}"
            )
            
            return request_id
            
        except Exception as e:
            logger.error(f"❌ Failed to publish ambiguous query: {e}")
            
            # Update status to FAILED if status store is available
            if STATUS_STORE_AVAILABLE and status_store:
                try:
                    error = {"type": "QUEUE_PUBLISH_ERROR", "message": str(e)}
                    status_store.fail_request(request_id, error)
                except Exception:
                    pass  # Ignore status update failures
            
            raise QueuePublishError(f"Failed to publish message: {e}") from e

    def publish_batch(
        self,
        queries: List[Dict[str, Any]],
        default_session_id: Optional[str] = None,
        default_user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Publish multiple ambiguous queries in batch.
        
        Args:
            queries: List of query dicts with keys: query, rule_based_result, session_id, etc.
            default_session_id: Default session_id if not provided in each query
            default_user_id: Default user_id if not provided in each query
        
        Returns:
            List of request_ids
        """
        request_ids = []
        for q in queries:
            try:
                request_id = self.publish_ambiguous_query(
                    query=q.get("query", ""),
                    rule_based_result=q.get("rule_based_result"),
                    session_id=q.get("session_id", default_session_id),
                    user_id=q.get("user_id", default_user_id),
                    conversation_history=q.get("conversation_history", []),
                    priority=q.get("priority", "normal"),
                    metadata=q.get("metadata", {}),
                    request_id=q.get("request_id"),
                )
                request_ids.append(request_id)
            except Exception as e:
                logger.error(f"❌ Failed to publish query in batch: {e}")
                # Continue with other queries
        return request_ids


# Global producer instance
_producer_instance: Optional[IntentQueueProducer] = None


def get_queue_producer() -> IntentQueueProducer:
    """Get global queue producer instance."""
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = IntentQueueProducer()
    return _producer_instance

