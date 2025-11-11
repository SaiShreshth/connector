"""
LLM Worker (CNS-21)
LLM Worker

Processes ambiguous queries from the queue using LLM intent classification.
This worker should be run as a separate process/container for async processing.
"""

import time
import logging
from typing import Optional, Dict, Any
from loguru import logger

from app.queue.queue_manager import queue_manager
from app.queue.monitor import queue_monitor
from app.ai.llm_intent.request_handler import RequestHandler
from app.schemas.llm_intent import LLMIntentRequest

# Import status store for request tracking
try:
    from app.core.status_store import status_store
    from app.schemas.request_status import ResultSchema
    STATUS_STORE_AVAILABLE = True
except ImportError:
    status_store = None
    STATUS_STORE_AVAILABLE = False

# Import context enhancement components
try:
    from app.ai.llm_intent.context_collector import get_context_collector
    from app.ai.llm_intent.context_summarizer import get_context_summarizer
    CONTEXT_ENHANCEMENT_AVAILABLE = True
except ImportError:
    CONTEXT_ENHANCEMENT_AVAILABLE = False
    logger.warning("⚠️ Context enhancement not available")

# Initialize LLM handler
try:
    from app.ai.llm_intent.openai_client import OpenAIClient
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        llm_client = OpenAIClient(api_key=api_key)
        llm_handler = RequestHandler(client=llm_client)
        LLM_AVAILABLE = True
    else:
        logger.warning("⚠️ OPENAI_API_KEY not set, LLM processing disabled")
        llm_handler = None
        LLM_AVAILABLE = False
except Exception as e:
    logger.warning(f"⚠️ LLM handler initialization failed: {e}")
    llm_handler = None
    LLM_AVAILABLE = False


def process_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single ambiguous query message using LLM.
    
    Args:
        message: Queue message with query and context
    
    Returns:
        Classification result or None if processing failed
    """
    if not LLM_AVAILABLE or llm_handler is None:
        logger.error("❌ LLM handler not available, cannot process message")
        return None

    query = message.get("query", "")
    context = message.get("context", {})
    
    try:
        # Build LLM request
        llm_request = LLMIntentRequest(
            user_input=query,
            rule_intent=context.get("possible_intents", {}),
            action_code=None,
            top_confidence=0.0,
            next_best_confidence=0.0,
            is_fallback=True
    # Handle both message formats: user_query (current) and query (legacy)
    query = message.get("user_query") or message.get("query", "")
    context = message.get("_context") or message.get("context", {})
    
    try:
        # Initialize variables before use
        rule_intent = None
        top_confidence = 0.0
        next_best_confidence = 0.0
        
        # Extract rule_based_result if available
        rule_based_result = message.get("rule_based_result") or context.get("rule_based_result")
        if rule_based_result:
            rule_intent = rule_based_result.get("action_code")
            top_confidence = rule_based_result.get("confidence", 0.0)
            # Extract next_best_confidence if available
            matched_keywords = rule_based_result.get("matched_keywords", [])
            # If there's a secondary confidence in context, use it
            possible_intents = context.get("possible_intents", {})
            if isinstance(possible_intents, dict) and len(possible_intents) > 1:
                confidences = sorted(possible_intents.values(), reverse=True)
                next_best_confidence = confidences[1] if len(confidences) > 1 else 0.0
        
        # Extract context for LLM
        # Get conversation history from message
        conversation_history = message.get("conversation_history", []) or context.get("conversation_history", [])
        user_id = message.get("user_id") or context.get("user_id")
        session_id = message.get("session_id") or context.get("session_id")
        
        # Use context enhancement if available
        context_snippets = []
        if CONTEXT_ENHANCEMENT_AVAILABLE:
            try:
                # Collect and summarize context to stay within token limits
                context_collector = get_context_collector()
                context_summarizer = get_context_summarizer(max_tokens=2000)
                
                # Collect all context (conversation + session)
                full_context = context_collector.collect_all_context(
                    conversation_history=conversation_history,
                    user_id=user_id,
                    session_id=session_id,
                    history_limit=5  # Keep last 5 messages
                )
                
                # Truncate context to fit token limits
                truncated_context = context_summarizer.truncate_context(
                    full_context,
                    max_tokens=2000
                )
                
                # Extract conversation history for context_snippets
                context_snippets = truncated_context.get("conversation_history", [])
                
                # Add session context to metadata if present
                session_context = truncated_context.get("session_context", {})
                if session_context:
                    context["session_context"] = session_context
                
                logger.debug(f"Enhanced context: {len(context_snippets)} conversation messages, session_context available")
                
            except Exception as e:
                logger.warning(f"⚠️ Context enhancement failed: {e}. Using basic context.")
                # Fallback to basic context processing
                if conversation_history and isinstance(conversation_history, list):
                    context_snippets = conversation_history[-5:]  # Keep last 5 messages
        else:
            # Fallback: Convert conversation_history to context_snippets format
            if conversation_history:
                if isinstance(conversation_history, list):
                    for msg in conversation_history[-5:]:  # Keep last 5
                        if isinstance(msg, dict):
                            context_snippets.append(msg)
                        elif isinstance(msg, str):
                            context_snippets.append({"role": "user", "content": msg})
                        else:
                            context_snippets.append(str(msg))
                else:
                    context_snippets = [str(conversation_history)]
        
        # Build LLM request with context
        llm_request = LLMIntentRequest(
            user_input=query,
            rule_intent=rule_intent or context.get("possible_intents", {}),
            action_code=rule_intent,
            top_confidence=float(top_confidence),
            next_best_confidence=float(next_best_confidence),
            is_fallback=True,
            context_snippets=context_snippets,
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                **(context if isinstance(context, dict) else {}),
                **(message.get("metadata", {}) if isinstance(message.get("metadata"), dict) else {})
            }
        )
        
        # Process with LLM
        start_time = time.time()
        result = llm_handler.handle(llm_request)
        processing_time = time.time() - start_time
        
        # Record metrics
        queue_monitor.record_dequeue()
        queue_monitor.record_processing_time(processing_time)
        
        # Log classification and track metrics for feedback system
        try:
            from app.ai.feedback.classification_logger import get_classification_logger
            from app.ai.monitoring.intent_distribution_tracker import get_intent_distribution_tracker
            from app.ai.monitoring.confidence_tracker import get_confidence_tracker
            
            classification_logger = get_classification_logger()
            intent_tracker = get_intent_distribution_tracker()
            confidence_tracker = get_confidence_tracker()
            
            action_code = result.get("action_code") or result.get("intent", "UNKNOWN")
            confidence = result.get("confidence", 0.0)
            
            # Log handoff: rule-based → LLM queue → LLM result
            logger.info(f"Handoff complete: queue → LLM (message_id: {message_id}, latency: {processing_time*1000:.2f}ms)")
            
            classification_logger.log_llm_classification(
                query=query,
                action_code=action_code,
                confidence=confidence,
                request_id=message_id,
                latency_ms=processing_time * 1000,
                context_snippets=context_snippets,
            )
            
            intent_tracker.record_intent(action_code)
            confidence_tracker.record_confidence(confidence)
        except Exception as e:
            logger.warning(f"Failed to log classification from queue: {e}")
        
        logger.info(
            f"✅ Processed message {message['message_id']} | "
            f"Intent: {result.get('action_code')} | "
            f"Confidence: {result.get('confidence', 0.0):.2f} | "
            f"Time: {processing_time:.2f}s"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to process message {message.get('message_id')}: {e}")
        queue_monitor.record_failure()
        return None


def llm_worker(poll_interval: int = 5, max_iterations: Optional[int] = None):
    """
    Main worker loop that processes messages from the queue.
    
    Args:
        poll_interval: Seconds to wait between queue polls
        max_iterations: Maximum number of iterations (None = infinite)
    """
    if queue_manager is None or not queue_manager.is_available():
        logger.error("❌ Queue manager not available, worker cannot start")
        return
    
    if not LLM_AVAILABLE:
        logger.error("❌ LLM handler not available, worker cannot start")
        return
    
    logger.info("🚀 LLM worker started...")
    iteration = 0
    
    try:
        while True:
            if max_iterations and iteration >= max_iterations:
                logger.info(f"✅ Worker completed {max_iterations} iterations")
                break
            
            iteration += 1
            
            # Get next message from queue
            message = queue_manager.dequeue_ambiguous_query(timeout=poll_interval)
            
            if message:
                logger.info(f"📨 Processing message: {message['message_id']} | Query: '{message['query']}'")
                message_id = message.get("message_id", "unknown")
                logger.info(f"📨 Processing message: {message_id} | Query: '{message.get('query', '')}'")
                
                # Update status to PROCESSING
                if STATUS_STORE_AVAILABLE and status_store:
                    try:
                        status_store.update_status(message_id, "PROCESSING")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to update status to PROCESSING: {e}")
                
                # Process message
                result = process_message(message)
                
                if result:
                    # Enqueue result
                    success = queue_manager.enqueue_classification_result(
                        message_id=message["message_id"],
                    # Apply confidence calibration
                    original_confidence = result.get("confidence", 0.0)
                    calibrated_confidence = original_confidence
                    
                    try:
                        from app.ai.llm_intent.confidence_calibration import get_calibrator
                        calibrator = get_calibrator()
                        calibrated_confidence = calibrator.calibrate_confidence(
                            action_code=result.get("action_code"),
                            reported_confidence=original_confidence
                        )
                        result["confidence"] = calibrated_confidence
                        if "metadata" not in result:
                            result["metadata"] = {}
                        result["metadata"]["original_confidence"] = original_confidence
                        result["metadata"]["calibrated_confidence"] = calibrated_confidence
                        
                        if abs(calibrated_confidence - original_confidence) > 0.05:
                            logger.info(
                                f"📊 Calibrated confidence: {original_confidence:.2f} → {calibrated_confidence:.2f} | "
                                f"action_code={result.get('action_code')}"
                            )
                    except Exception as e:
                        logger.warning(f"⚠️ Confidence calibration failed: {e}")
                    
                    # Update status to COMPLETED
                    if STATUS_STORE_AVAILABLE and status_store:
                        try:
                            result_schema = ResultSchema(
                                action_code=result.get("action_code"),
                                confidence=calibrated_confidence,  # Use calibrated confidence
                                entities=result.get("metadata", {}).get("entities_extracted", {})
                            )
                            status_store.complete_request(message_id, result=result_schema)
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to update status to COMPLETED: {e}")
                    
                    # Enqueue result
                    success = queue_manager.enqueue_classification_result(
                        message_id=message_id,
                        result=result,
                        processing_time=result.get("metadata", {}).get("processing_time_ms", 0) / 1000.0
                    )
                    
                    if not success:
                        logger.warning(f"⚠️ Failed to enqueue result for {message['message_id']}")
                else:
                    # Processing failed, retry or move to DLQ
                        logger.warning(f"⚠️ Failed to enqueue result for {message_id}")
                else:
                    # Processing failed, update status to FAILED
                    if STATUS_STORE_AVAILABLE and status_store:
                        try:
                            error = {"type": "LLM_PROCESSING_ERROR", "message": "LLM processing failed"}
                            status_store.fail_request(message_id, error)
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to update status to FAILED: {e}")
                    
                    # Retry or move to DLQ
                    if not queue_manager.retry_message(message):
                        queue_manager.move_to_dead_letter_queue(
                            message=message,
                            error="LLM processing failed"
                        )
            else:
                # No message, continue polling
                logger.debug(f"No messages in queue, continuing to poll...")
                
    except KeyboardInterrupt:
        logger.info("🛑 Worker stopped by user")
    except Exception as e:
        logger.error(f"❌ Worker error: {e}")
        raise


if __name__ == "__main__":
    # Run worker with default settings
    llm_worker()
