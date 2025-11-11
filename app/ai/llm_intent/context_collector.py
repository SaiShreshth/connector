"""
Context Collector

Collects conversation history and user session context for LLM prompt enhancement.
"""

import os
from typing import Dict, List, Optional, Any
from loguru import logger

# Try to import Redis for session storage
try:
    import redis
    from app.queue.config import queue_config
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    logger.warning("Redis not available for context collection")


class ContextCollector:
    """
    Collects conversation history and user session context.
    
    Gathers:
    - Conversation history (previous N messages)
    - User session context (browsing history, cart items, viewed products)
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize context collector.
        
        Args:
            redis_client: Optional Redis client (will create if not provided)
        """
        self.redis_client = None
        
        if REDIS_AVAILABLE:
            try:
                if redis_client:
                    self.redis_client = redis_client
                else:
                    # Try to use existing queue_manager's Redis connection
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
                
                logger.info("✅ ContextCollector initialized with Redis")
            except Exception as e:
                logger.warning(f"⚠️ Redis not available for context collection: {e}")
                self.redis_client = None
    
    def _get_session_key(self, session_id: str, key_type: str) -> str:
        """Get Redis key for session data."""
        return f"chatns:session:{session_id}:{key_type}"
    
    def collect_conversation_history(
        self,
        conversation_history: List[Dict[str, Any]],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Collect conversation history from provided list.
        
        Args:
            conversation_history: List of conversation messages
            limit: Maximum number of previous messages to include (default: 5)
        
        Returns:
            List of conversation messages (most recent first, limited to N)
        """
        if not conversation_history:
            return []
        
        # Return last N messages (most recent first)
        # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        limited_history = conversation_history[-limit:] if len(conversation_history) > limit else conversation_history
        
        logger.debug(f"Collected {len(limited_history)} conversation messages (limit: {limit})")
        return limited_history
    
    def collect_session_context(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collect user session context from session store.
        
        Collects:
        - Browsing history (recently viewed products)
        - Cart items
        - Recently viewed products
        
        Args:
            user_id: User identifier (optional)
            session_id: Session identifier (optional)
        
        Returns:
            Dict with session context:
            {
                "browsing_history": [...],
                "cart_items": [...],
                "viewed_products": [...]
            }
        """
        if not session_id:
            return {
                "browsing_history": [],
                "cart_items": [],
                "viewed_products": []
            }
        
        context = {
            "browsing_history": [],
            "cart_items": [],
            "viewed_products": []
        }
        
        try:
            # Use SessionStore to get session context
            from app.core.session_store import get_session_store
            session_store = get_session_store()
            
            context["browsing_history"] = session_store.get_browsing_history(session_id)
            context["cart_items"] = session_store.get_cart_items(session_id)
            context["viewed_products"] = session_store.get_viewed_products(session_id)
            
            logger.debug(f"Collected session context for session: {session_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to collect session context: {e}")
        
        return context
    
    def collect_all_context(
        self,
        conversation_history: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        history_limit: int = 5
    ) -> Dict[str, Any]:
        """
        Collect all context (conversation history + session context).
        
        Args:
            conversation_history: List of conversation messages
            user_id: User identifier (optional)
            session_id: Session identifier (optional)
            history_limit: Maximum number of previous messages (default: 5)
        
        Returns:
            Dict with all collected context:
            {
                "conversation_history": [...],
                "session_context": {
                    "browsing_history": [...],
                    "cart_items": [...],
                    "viewed_products": [...]
                }
            }
        """
        # Collect conversation history
        conv_history = self.collect_conversation_history(conversation_history, limit=history_limit)
        
        # Collect session context
        session_context = self.collect_session_context(user_id=user_id, session_id=session_id)
        
        return {
            "conversation_history": conv_history,
            "session_context": session_context
        }


# Global instance
_context_collector: Optional[ContextCollector] = None


def get_context_collector() -> ContextCollector:
    """Get global context collector instance."""
    global _context_collector
    if _context_collector is None:
        _context_collector = ContextCollector()
    return _context_collector

