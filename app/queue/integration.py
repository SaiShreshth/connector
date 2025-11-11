"""
Queue Integration

Integrates queue infrastructure with Ambiguity Resolver.
Routes ambiguous/unclear queries to LLM processing queue.
"""

from typing import Dict, Any, Optional
from loguru import logger

from app.queue.queue_manager import queue_manager
from app.queue.monitor import queue_monitor
from app.queue.config import queue_config


def send_to_llm_queue(
    query: str,
    ambiguity_result: Dict[str, Any],
    user_id: str = "anonymous",
    is_premium: bool = False
) -> Optional[str]:
    """
    Send ambiguous/unclear query to LLM processing queue.
    
    This integrates with the ambiguity resolver output.
    
    Args:
        query: User's original query
        ambiguity_result: Result from detect_intent() in ambiguity_resolver.py
        user_id: User identifier
        is_premium: Whether user has premium subscription
    
    Returns:
        message_id: Queue message identifier, or None if queue unavailable
    
    Example:
        from app.ai.intent_classification.ambiguity_resolver import detect_intent
        
        result = detect_intent("add shoes and track order")
        if result["action"] == "AMBIGUOUS":
            message_id = send_to_llm_queue(
                query="add shoes and track order",
                ambiguity_result=result,
                user_id="user_123"
            )
    """
    # Check if queue is available
    if queue_manager is None or not queue_manager.is_available():
        logger.warning("⚠️ Queue unavailable, cannot send message to LLM queue")
        return None
    
    # Determine priority
    priority = (
        queue_config.PRIORITY_HIGH if is_premium 
        else queue_config.PRIORITY_NORMAL
    )
    
    # Build context for LLM
    context = {
        "ambiguity_type": ambiguity_result.get("action"),  # AMBIGUOUS or UNCLEAR
        "possible_intents": ambiguity_result.get("possible_intents", {}),
        "confidence_scores": ambiguity_result.get("possible_intents", {}),
        "user_id": user_id,
        "is_premium": is_premium,
        "source": "ambiguity_resolver"
    }
    
    try:
        # Enqueue for LLM processing
        message_id = queue_manager.enqueue_ambiguous_query(
            query=query,
            context=context,
            priority=priority
        )
        
        if message_id:
            # Record metrics
            queue_monitor.record_enqueue()
            
            logger.info(
                f"✅ Sent to LLM queue: {message_id} | "
                f"Query: '{query}' | Type: {context['ambiguity_type']}"
            )
        
        return message_id
        
    except Exception as e:
        logger.error(f"❌ Failed to send to LLM queue: {e}")
        queue_monitor.record_failure()
        return None


def get_classification_result(message_id: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Get LLM classification result from status store.
    
    Uses status store instead of polling output queue for faster lookups.
    
    Args:
        message_id: Request ID to look up
        timeout: Seconds to wait (not used, status store provides immediate lookup)
    
    Returns:
        result: Request status with result if completed, or pending status
    """
    try:
        from app.core.status_store import status_store
        
        if status_store is None:
            return {"status": "error", "message": "Status store not available"}
        
        status = status_store.get(message_id)
        
        if not status:
            return {"status": "not_found", "message_id": message_id}
        
        # Return status with result
        result_dict = {
            "status": status.status,
            "request_id": status.request_id,
            "queued_at": status.queued_at.isoformat() + "Z" if status.queued_at else None,
            "started_at": status.started_at.isoformat() + "Z" if status.started_at else None,
            "completed_at": status.completed_at.isoformat() + "Z" if status.completed_at else None,
        }
        
        if status.result:
            result_dict["result"] = {
                "action_code": status.result.action_code,
                "confidence": status.result.confidence,
                "entities": status.result.entities,
            }
        
        if status.error:
            result_dict["error"] = status.error
        
        return result_dict
        
    except Exception as e:
        logger.error(f"❌ Failed to get classification result: {e}")
        return {"status": "error", "message": str(e), "message_id": message_id}
