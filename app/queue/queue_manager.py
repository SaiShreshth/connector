"""
Queue Manager

Manages message queue operations for LLM intent classification.
Handles input/output queues, dead letter queue, and retry logic.
"""

import json
import redis
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from .config import queue_config


class QueueManager:
    """
    Manages Redis-based message queue for asynchronous LLM processing.
    
    Queue Flow:
    1. Ambiguous queries from rule-based classifier → AMBIGUOUS_QUERY_QUEUE
    2. LLM processes queries → CLASSIFICATION_RESULT_QUEUE
    3. Failed messages → DEAD_LETTER_QUEUE
    """
    
    def __init__(self, fail_silently: bool = False):
        """
        Initialize Redis connection with connection pooling.
        
        Args:
            fail_silently: If True, don't raise exception on connection failure (for graceful degradation)
        """
        self.redis_client = None
        self.pool = None
        self._is_available = False
        
        try:
            # Connection pool for better performance
            self.pool = redis.ConnectionPool(
                host=queue_config.REDIS_HOST,
                port=queue_config.REDIS_PORT,
                db=queue_config.REDIS_DB,
                password=queue_config.REDIS_PASSWORD,
                max_connections=queue_config.REDIS_MAX_CONNECTIONS,
                socket_timeout=queue_config.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=queue_config.REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=True
            )
            
            self.redis_client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.redis_client.ping()
            self._is_available = True
            logger.info("✅ Queue Manager initialized successfully")
            
        except redis.ConnectionError as e:
            self._is_available = False
            if fail_silently:
                logger.warning(f"⚠️ Redis not available: {e}. Queue operations will be disabled.")
            else:
                logger.error(f"❌ Failed to connect to Redis: {e}")
                raise
        except Exception as e:
            self._is_available = False
            if fail_silently:
                logger.warning(f"⚠️ Queue Manager initialization failed: {e}. Queue operations will be disabled.")
            else:
                logger.error(f"❌ Queue Manager initialization failed: {e}")
                raise
    
    def is_available(self) -> bool:
        """Check if queue is available."""
        if not self._is_available or self.redis_client is None:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception:
            self._is_available = False
            return False
    
    def enqueue_ambiguous_query(
        self, 
        query: str, 
        context: Dict[str, Any],
        priority: int = queue_config.PRIORITY_NORMAL
    ) -> Optional[str]:
        """
        Add ambiguous query to input queue for LLM processing.
        
        Args:
            query: User's ambiguous query text
            context: Additional context (possible_intents, user_id, etc.)
            priority: Queue priority (1=high, 5=normal, 10=low)
        
        Returns:
            message_id: Unique message identifier, or None if queue unavailable
        """
        if not self.is_available():
            logger.warning("⚠️ Queue unavailable, cannot enqueue message")
            return None

        message_id = f"msg_{int(time.time() * 1000)}"

        message = {
            "message_id": message_id,
            "query": query,
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
            "retry_count": 0,
            "priority": priority
        }
        # Check if context has a full message from producer
        # This allows preserving the standard message format
        if "_full_message" in context:
            full_message = context["_full_message"]
            message_id = full_message.get("request_id", f"msg_{int(time.time() * 1000)}")
            # Use the full message format from producer
            message = full_message
            # Add internal fields needed by queue_manager
            message["message_id"] = message_id
            message["retry_count"] = 0
            # Store original context separately
            message["_context"] = {k: v for k, v in context.items() if k != "_full_message"}
        else:
            # Legacy format (backward compatibility)
            message_id = f"msg_{int(time.time() * 1000)}"
            message = {
                "message_id": message_id,
                "query": query,
                "context": context,
                "timestamp": datetime.utcnow().isoformat(),
                "retry_count": 0,
                "priority": priority
            }
        
        try:
            # Use sorted set for priority queue
            if queue_config.ENABLE_PRIORITY_QUEUE:
                self.redis_client.zadd(
                    queue_config.AMBIGUOUS_QUERY_QUEUE,
                    {json.dumps(message): priority}
                )
            else:
                # Regular FIFO queue
                self.redis_client.rpush(
                    queue_config.AMBIGUOUS_QUERY_QUEUE,
                    json.dumps(message)
                )
            
            # Set TTL on message
            self.redis_client.expire(
                queue_config.AMBIGUOUS_QUERY_QUEUE,
                queue_config.MESSAGE_TTL
            )
            
            logger.info(f"✅ Enqueued ambiguous query: {message_id} (priority: {priority})")
            return message_id
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue message {message_id}: {e}")
            self._is_available = False  # Mark as unavailable
            return None
    
    def dequeue_ambiguous_query(self, timeout: int = 0) -> Optional[Dict[str, Any]]:
        """
        Get next ambiguous query from input queue (blocking).
        
        Args:
            timeout: Seconds to wait for message (0 = non-blocking)
        
        Returns:
            message: Query message dict or None
        """
        if not self.is_available():
            return None
            
        try:
            if queue_config.ENABLE_PRIORITY_QUEUE:
                # Get highest priority (lowest score) message
                result = self.redis_client.zpopmin(
                    queue_config.AMBIGUOUS_QUERY_QUEUE,
                    count=1
                )
                if result:
                    message_json, priority = result[0]
                    message = json.loads(message_json)
                    logger.info(f"✅ Dequeued message: {message['message_id']}")
                    return message
            else:
                # Regular FIFO
                if timeout > 0:
                    result = self.redis_client.blpop(
                        queue_config.AMBIGUOUS_QUERY_QUEUE,
                        timeout=timeout
                    )
                else:
                    result = self.redis_client.lpop(
                        queue_config.AMBIGUOUS_QUERY_QUEUE
                    )
                
                if result:
                    if isinstance(result, tuple):
                        _, message_json = result
                    else:
                        message_json = result
                    
                    message = json.loads(message_json)
                    logger.info(f"✅ Dequeued message: {message['message_id']}")
                    return message
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to dequeue message: {e}")
            return None
    
    def enqueue_classification_result(
        self,
        message_id: str,
        result: Dict[str, Any],
        processing_time: float
    ) -> bool:
        """
        Add classification result to output queue.
        
        Args:
            message_id: Original message ID
            result: LLM classification result
            processing_time: Processing time in seconds
        
        Returns:
            bool: True if successful, False if queue unavailable
        """
        if not self.is_available():
            logger.warning(f"⚠️ Queue unavailable, cannot enqueue result for {message_id}")
            return False
            
        output_message = {
            "message_id": message_id,
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            self.redis_client.rpush(
                queue_config.CLASSIFICATION_RESULT_QUEUE,
                json.dumps(output_message)
            )
            
            # Set TTL
            self.redis_client.expire(
                queue_config.CLASSIFICATION_RESULT_QUEUE,
                queue_config.MESSAGE_TTL
            )
            
            logger.info(f"✅ Enqueued result for message: {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue result for {message_id}: {e}")
            self._is_available = False
            return False
    
    def move_to_dead_letter_queue(
        self,
        message: Dict[str, Any],
        error: str
    ) -> bool:
        """
        Move failed message to dead letter queue.
        
        Args:
            message: Original message
            error: Error description
        
        Returns:
            bool: True if successful, False if queue unavailable
        """
        if not self.is_available():
            logger.warning(f"⚠️ Queue unavailable, cannot move message {message.get('message_id')} to DLQ")
            return False
            
        dlq_message = {
            **message,
            "error": error,
            "failed_at": datetime.utcnow().isoformat(),
            "final_retry_count": message.get("retry_count", 0)
        }
        
        try:
            self.redis_client.rpush(
                queue_config.DEAD_LETTER_QUEUE,
                json.dumps(dlq_message)
            )
            
            logger.warning(f"⚠️ Moved message {message['message_id']} to DLQ: {error}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to move message to DLQ: {e}")
            self._is_available = False
            return False
    
    def retry_message(self, message: Dict[str, Any]) -> bool:
        """
        Retry failed message with exponential backoff.
        
        Args:
            message: Failed message
        
        Returns:
            bool: True if retrying, False if max retries exceeded
        """
        retry_count = message.get("retry_count", 0)
        
        if retry_count >= queue_config.MAX_RETRY_ATTEMPTS:
            logger.warning(f"⚠️ Max retries exceeded for {message['message_id']}")
            return False
        
        # Increment retry count
        message["retry_count"] = retry_count + 1
        message["last_retry_at"] = datetime.utcnow().isoformat()
        
        # Exponential backoff: 5s, 10s, 20s
        delay = queue_config.RETRY_DELAY * (2 ** retry_count)
        
        try:
            # Re-enqueue with delay
            time.sleep(delay)
            
            if queue_config.ENABLE_PRIORITY_QUEUE:
                # Lower priority for retries
                priority = message.get("priority", queue_config.PRIORITY_NORMAL) + 1
                self.redis_client.zadd(
                    queue_config.AMBIGUOUS_QUERY_QUEUE,
                    {json.dumps(message): priority}
                )
            else:
                self.redis_client.rpush(
                    queue_config.AMBIGUOUS_QUERY_QUEUE,
                    json.dumps(message)
                )
            
            logger.info(f"🔄 Retrying message {message['message_id']} (attempt {message['retry_count']})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to retry message: {e}")
            return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics for monitoring.
        
        Returns:
            stats: Queue metrics (sizes, processing rates, etc.)
        """
        if not self.is_available():
            return {
                "status": "unavailable",
                "ambiguous_queue_size": 0,
                "result_queue_size": 0,
                "dead_letter_queue_size": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        try:
            if queue_config.ENABLE_PRIORITY_QUEUE:
                ambiguous_count = self.redis_client.zcard(
                    queue_config.AMBIGUOUS_QUERY_QUEUE
                )
            else:
                ambiguous_count = self.redis_client.llen(
                    queue_config.AMBIGUOUS_QUERY_QUEUE
                )
            
            result_count = self.redis_client.llen(
                queue_config.CLASSIFICATION_RESULT_QUEUE
            )
            dlq_count = self.redis_client.llen(
                queue_config.DEAD_LETTER_QUEUE
            )
            
            stats = {
                "status": "available",
                "ambiguous_queue_size": ambiguous_count,
                "result_queue_size": result_count,
                "dead_letter_queue_size": dlq_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.debug(f"📊 Queue stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get queue stats: {e}")
            self._is_available = False
            return {
                "status": "error",
                "error": str(e),
                "ambiguous_queue_size": 0,
                "result_queue_size": 0,
                "dead_letter_queue_size": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def clear_queue(self, queue_name: str) -> None:
        """
        Clear specific queue (for testing/admin use).
        
        Args:
            queue_name: Name of queue to clear
        """
        try:
            self.redis_client.delete(queue_name)
            logger.info(f"🗑️ Cleared queue: {queue_name}")
        except Exception as e:
            logger.error(f"❌ Failed to clear queue {queue_name}: {e}")
    
    def health_check(self) -> bool:
        """
        Check if queue system is healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        if not self.is_available():
            return False
        try:
            self.redis_client.ping()
            self._is_available = True
            return True
        except Exception as e:
            logger.error(f"❌ Queue health check failed: {e}")
            self._is_available = False
            return False
    
    def close(self) -> None:
        """Close Redis connections"""
        try:
            self.redis_client.close()
            logger.info("✅ Queue Manager closed")
        except Exception as e:
            logger.error(f"❌ Error closing Queue Manager: {e}")


# Global queue manager instance (initialized with graceful failure)
try:
    queue_manager = QueueManager(fail_silently=True)
except Exception:
    # If initialization fails completely, create a dummy instance
    queue_manager = None
