"""
Fallback logic used when the LLM cannot determine an intent.

Integrates with Task 19 (LLM Caching) to use cached responses as fallback
before returning default responses.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# ✅ Default constants
# -------------------------------------------------------------------
DEFAULT_FALLBACK_INTENT: str = "clarify_intent"
DEFAULT_FALLBACK_CATEGORY: str = "SUPPORT_HELP"
DEFAULT_FALLBACK_ACTION: str = "HELP_GENERAL"
DEFAULT_CLARIFICATION_MESSAGE: str = (
    "I'm not sure I understood that. Could you rephrase or provide more details?"
)


def try_cached_fallback(user_query: str, similarity_threshold: float = 0.90) -> Optional[Dict]:
    """
    Try to get fallback from cache before returning generic response.
    
    Uses the LLM cache (Task 19) with a lower similarity threshold (0.90 instead of 0.95)
    to find similar queries that were successfully processed before.
    
    Args:
        user_query: User's input query
        similarity_threshold: Similarity threshold for cache lookup (default: 0.90)
        
    Returns:
        Cached response dict if found, None otherwise
    """
    try:
        from app.ai.llm_intent.response_cache import get_response_cache
        cache = get_response_cache()
        
        # Check cache with lower threshold for fallback
        # This is more permissive than normal cache lookup (0.95) to increase fallback success
        cached = cache.get(user_query, similarity_threshold=similarity_threshold)
        if cached:
            logger.info(f"Using cached response as fallback for: '{user_query[:50]}...'")
            return cached
    except ImportError:
        logger.debug("Response cache not available for fallback (import failed)")
    except Exception as e:
        logger.warning(f"Cache fallback failed: {e}")
    
    return None

# -------------------------------------------------------------------
# ✅ Primary fallback response with cache integration
# -------------------------------------------------------------------
def build_fallback_response(reason: str, user_query: Optional[str] = None) -> Dict[str, object]:
    """
    Return a standardized payload when both rule-based and LLM methods fail.
    This ensures the system never crashes on ambiguous or failed queries.
    
    Fallback priority:
    1. Try cached similar response (if user_query provided)
    2. Return default clarification response
    
    Args:
        reason: Reason for fallback (e.g., "llm_failure", "timeout")
        user_query: Optional user query to try cache fallback
        
    Returns:
        Fallback response dict
    """
    # Try cache fallback first if query provided
    if user_query:
        cached_fallback = try_cached_fallback(user_query)
        if cached_fallback:
            # Found a similar cached response, use it as fallback
            return {
                **cached_fallback,
                "metadata": {
                    **cached_fallback.get("metadata", {}),
                    "fallback_reason": reason,
                    "fallback_source": "cache"
                }
            }
    
    # No cache hit, return default fallback
    return {
        "intent": DEFAULT_FALLBACK_INTENT,
        "intent_category": DEFAULT_FALLBACK_CATEGORY,
        "action_code": DEFAULT_FALLBACK_ACTION,
        "confidence": 0.0,
        "requires_clarification": True,
        "clarification_message": DEFAULT_CLARIFICATION_MESSAGE,
        "metadata": {
            "fallback_reason": reason,
            "fallback_source": "default"
        },
    }

# -------------------------------------------------------------------
# ✅ Optional "UNCLEAR" intent response (for low-confidence cases)
# -------------------------------------------------------------------
def build_unclear_response(suggested_questions: List[str]) -> Dict[str, object]:
    """
    Return an UNCLEAR response with suggested clarifying questions.
    Used when LLM confidence is low or multiple conflicting intents detected.
    """
    return {
        "intent": "unclear",
        "intent_category": DEFAULT_FALLBACK_CATEGORY,
        "action_code": "UNCLEAR",
        "confidence": 0.0,
        "requires_clarification": True,
        "clarification_message": DEFAULT_CLARIFICATION_MESSAGE,
        "clarifying_questions": suggested_questions,
        "metadata": {"fallback_reason": "unclear_intent"},
    }

# -------------------------------------------------------------------
# ✅ Public exports
# -------------------------------------------------------------------
__all__ = [
    "DEFAULT_FALLBACK_INTENT",
    "DEFAULT_FALLBACK_CATEGORY",
    "DEFAULT_FALLBACK_ACTION",
    "DEFAULT_CLARIFICATION_MESSAGE",
    "build_fallback_response",
    "build_unclear_response",
    "try_cached_fallback",
]
