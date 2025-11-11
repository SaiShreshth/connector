"""
Session Store

Stores and retrieves user session context:
- Browsing history (recently viewed products)
- Cart items
- Viewed products

Uses Redis for persistence with TTL.
"""

import json
import os
from typing import List, Dict, Optional, Any
from loguru import logger

# Try to import Redis
try:
    import redis
    from app.queue.config import queue_config
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, session store will use in-memory fallback")


class SessionStore:
    """
    Stores and retrieves user session context.
    
    Supports:
    - Browsing history
    - Cart items
    - Viewed products
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize session store.
        
        Args:
            redis_client: Optional Redis client (will create if not provided)
        """
        self.redis_client = None
        self._use_redis = False
        self._in_memory_store: Dict[str, Dict[str, Any]] = {}
        
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
                
                self._use_redis = True
                logger.info("✅ SessionStore initialized with Redis")
            except Exception as e:
                logger.warning(f"⚠️ Redis not available for session store: {e}. Using in-memory fallback.")
                self._use_redis = False
    
    def _get_key(self, session_id: str, data_type: str) -> str:
        """Get Redis key for session data."""
        return f"chatns:session:{session_id}:{data_type}"
    
    def _get_ttl(self) -> int:
        """Get TTL for session data (24 hours)."""
        return 24 * 60 * 60
    
    def store_browsing_history(
        self,
        session_id: str,
        products: List[Dict[str, Any]],
        max_items: int = 20
    ):
        """
        Store browsing history for a session.
        
        Args:
            session_id: Session identifier
            products: List of product dictionaries
            max_items: Maximum items to keep (default: 20)
        """
        if not session_id:
            return
        
        # Keep only most recent items
        products = products[-max_items:] if len(products) > max_items else products
        
        try:
            if self._use_redis and self.redis_client:
                key = self._get_key(session_id, "browsing_history")
                self.redis_client.set(key, json.dumps(products))
                self.redis_client.expire(key, self._get_ttl())
            else:
                # In-memory storage
                if session_id not in self._in_memory_store:
                    self._in_memory_store[session_id] = {}
                self._in_memory_store[session_id]["browsing_history"] = products
        except Exception as e:
            logger.warning(f"⚠️ Failed to store browsing history: {e}")
    
    def get_browsing_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get browsing history for a session."""
        if not session_id:
            return []
        
        try:
            if self._use_redis and self.redis_client:
                key = self._get_key(session_id, "browsing_history")
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                return self._in_memory_store.get(session_id, {}).get("browsing_history", [])
        except Exception as e:
            logger.warning(f"⚠️ Failed to get browsing history: {e}")
        
        return []
    
    def store_cart_items(
        self,
        session_id: str,
        cart_items: List[Dict[str, Any]]
    ):
        """
        Store cart items for a session.
        
        Args:
            session_id: Session identifier
            cart_items: List of cart item dictionaries
        """
        if not session_id:
            return
        
        try:
            if self._use_redis and self.redis_client:
                key = self._get_key(session_id, "cart_items")
                self.redis_client.set(key, json.dumps(cart_items))
                self.redis_client.expire(key, self._get_ttl())
            else:
                # In-memory storage
                if session_id not in self._in_memory_store:
                    self._in_memory_store[session_id] = {}
                self._in_memory_store[session_id]["cart_items"] = cart_items
        except Exception as e:
            logger.warning(f"⚠️ Failed to store cart items: {e}")
    
    def get_cart_items(self, session_id: str) -> List[Dict[str, Any]]:
        """Get cart items for a session."""
        if not session_id:
            return []
        
        try:
            if self._use_redis and self.redis_client:
                key = self._get_key(session_id, "cart_items")
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                return self._in_memory_store.get(session_id, {}).get("cart_items", [])
        except Exception as e:
            logger.warning(f"⚠️ Failed to get cart items: {e}")
        
        return []
    
    def store_viewed_products(
        self,
        session_id: str,
        products: List[Dict[str, Any]],
        max_items: int = 20
    ):
        """
        Store viewed products for a session.
        
        Args:
            session_id: Session identifier
            products: List of product dictionaries
            max_items: Maximum items to keep (default: 20)
        """
        if not session_id:
            return
        
        # Keep only most recent items
        products = products[-max_items:] if len(products) > max_items else products
        
        try:
            if self._use_redis and self.redis_client:
                key = self._get_key(session_id, "viewed_products")
                self.redis_client.set(key, json.dumps(products))
                self.redis_client.expire(key, self._get_ttl())
            else:
                # In-memory storage
                if session_id not in self._in_memory_store:
                    self._in_memory_store[session_id] = {}
                self._in_memory_store[session_id]["viewed_products"] = products
        except Exception as e:
            logger.warning(f"⚠️ Failed to store viewed products: {e}")
    
    def get_viewed_products(self, session_id: str) -> List[Dict[str, Any]]:
        """Get viewed products for a session."""
        if not session_id:
            return []
        
        try:
            if self._use_redis and self.redis_client:
                key = self._get_key(session_id, "viewed_products")
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                return self._in_memory_store.get(session_id, {}).get("viewed_products", [])
        except Exception as e:
            logger.warning(f"⚠️ Failed to get viewed products: {e}")
        
        return []


# Global instance
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Get global session store instance."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store

