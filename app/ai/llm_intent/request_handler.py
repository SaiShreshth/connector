"""Handles LLM-based intent classification with robust fallback, logging, and error handling."""

from __future__ import annotations

import os
import time
import random
import logging
from typing import Dict, Any, List, Optional
from app.schemas.llm_intent import LLMIntentRequest

# ---------------------------------------------------------------------
# External Integrations
# ---------------------------------------------------------------------
from .confidence_calibrator import TriggerContext, should_trigger_llm
from .entity_extractor import INTENT_CATEGORIES
from .fallback_manager import build_fallback_response
from .openai_client import OpenAIClient
from .response_parser import LLMIntentResponse as ParsedLLMResponse, parse_llm_response
from .prompt_loader import PromptLoader, PromptLoadError, get_prompt_loader

# Error handling components
from .resilient_openai_client import ResilientOpenAIClient, LLMRequestError
from .error_handler import (
    handle_timeout_error,
    handle_rate_limit_error,
    handle_service_unavailable_error,
    handle_auth_error,
    handle_context_length_error,
    log_error_with_full_context,
    build_user_error_response
)
from app.core.alert_notifier import send_alert

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "llm_intent.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
from .context_collector import get_context_collector
from .context_summarizer import get_context_summarizer
from .response_cache import get_response_cache
import logging
import os

logger = logging.getLogger("request_handler")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # exponential backoff in seconds


class RequestHandler:
    """Manages LLM intent classification with fallback, retry resilience, and prompt-based coordination."""

    def __init__(
        self,
        client: Optional[OpenAIClient] = None,
        prompt_loader: Optional[PromptLoader] = None
    ) -> None:
        """
        Initialize request handler.

        Args:
            client: OpenAI client instance (optional, will use default if None)
            prompt_loader: Prompt loader instance (optional, will use singleton if None)
        """
        self.simulate_mode = True  # True = simulation when no API key
        self.client = client or OpenAIClient()
        try:
            self.prompt_loader = prompt_loader or get_prompt_loader()
            logger.info("RequestHandler initialized with prompt loader")
        except PromptLoadError as e:
            logger.warning(
                f"Failed to load prompts: {e}. "
                f"LLM requests will use simple user messages without prompts."
            )
            self.prompt_loader = None
        
        # Initialize context collector and summarizer
        self.context_collector = get_context_collector()
        
        # Get token limit from environment or use default
        context_token_limit = int(os.getenv("CONTEXT_TOKEN_LIMIT", "2000"))
        self.context_summarizer = get_context_summarizer(max_tokens=context_token_limit)
        
        # Initialize response cache
        self.response_cache = get_response_cache()
        
        # Configuration
        self.enable_context = os.getenv("ENABLE_SESSION_CONTEXT", "true").lower() == "true"
        self.context_history_limit = int(os.getenv("CONTEXT_HISTORY_LIMIT", "5"))

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    def handle(self, request: LLMIntentRequest) -> Dict[str, Any]:
        """
        Handle an intent classification request with retries and fallbacks.
        
        Flow:
        1. Check cache first (if enabled and high-confidence not met)
        2. High-confidence rule-based → skip LLM
        3. Try LLM with retries and comprehensive error handling
        4. All attempts failed → escalate and fallback (with cache fallback)
        """
        logging.info(f"🔍 Received input: '{request.user_input}'")

        # 0️⃣ Check cache first (before rule-based decision) if low confidence
        if self.response_cache and request.top_confidence < 0.85:
            try:
                cached = self.response_cache.get(request.user_input)
                if cached:
                    logger.info("✅ Cache hit, returning cached response")
                    return {
                        **cached,
                        "metadata": {
                            **cached.get("metadata", {}),
                            "source": "cache",
                            "cache_hit": True
                        }
                    }
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")

        # 1️⃣ High-confidence rule-based intent: no LLM required
        if request.top_confidence >= 0.85 and not request.is_fallback:
            logging.info("✅ Rule-based classifier confident, skipping LLM call.")
            return self._retain_rule_result(request)

        # 2️⃣ Otherwise: attempt LLM classification with retries and error handling
        last_error = None
        last_error_code = "unknown_error"
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logging.info(f"🧠 Attempt {attempt}/{MAX_RETRIES} to query LLM...")
                result = self._query_llm(request)
                if result:
                    logging.info(f"✅ LLM succeeded on attempt {attempt}.")
                    
                    # Cache successful result (if high confidence)
                    if self.response_cache and result.get("confidence", 0) > 0.7:
                        try:
                            self.response_cache.set(request.user_input, result)
                            logger.debug("Cached successful LLM response")
                        except Exception as e:
                            logger.warning(f"Failed to cache result: {e}")
                    
                    return result
                    
            except LLMRequestError as exc:
                # Structured error from ResilientOpenAIClient
                last_error = exc
                last_error_code = exc.code
                
                # Build error context
                error_context = {
                    "user_input": request.user_input,
                    "error_code": exc.code,
                    "error_details": exc.details,
                    "attempt": attempt,
                    "timestamp": time.time(),
                }
                
                # Log with full context
                log_error_with_full_context(exc, error_context)
                
                # Send alerts based on error type
                if exc.code == "rate_limit":
                    send_alert("RATE_LIMIT_EXCEEDED", error_context, severity="warning")
                elif exc.code == "timeout":
                    send_alert("LLM_TIMEOUT", error_context, severity="warning")
                elif exc.code == "server_error":
                    send_alert("LLM_SERVER_ERROR", error_context, severity="error")
                elif exc.code == "auth_error":
                    send_alert("LLM_AUTH_ERROR", error_context, severity="critical")
                
                logging.warning(f"⚠️ LLM error on attempt {attempt}: {exc.code}")
                
            except TimeoutError as exc:
                last_error = exc
                last_error_code = "timeout"
                logging.warning(f"⚠️ Timeout occurred on attempt {attempt}.")
                
            except Exception as exc:
                last_error = exc
                last_error_code = "unknown_error"
                logging.error(f"❌ Unexpected error on attempt {attempt}: {str(exc)}")

            # Retry delay (if not last attempt)
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF[attempt - 1]
                logging.info(f"⏳ Retrying in {delay}s...")
                time.sleep(delay)

        # 3️⃣ All attempts failed — escalate and return fallback
        logging.critical("🚨 All LLM attempts failed. Triggering escalation path.")
        self._escalate_failure(request)
        
        # Return fallback (will try cache fallback first, then UNCLEAR)
        return self._fallback_unclear_response(request)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _query_llm(self, request: LLMIntentRequest) -> Dict[str, Any]:
        """Simulate or query an external LLM service."""
        if self.simulate_mode:
            text = request.user_input.lower()

            # Simulated intent predictions
            if "remove" in text and "cart" in text:
                return self._build_response("REMOVE_FROM_CART", 0.93, request)
            elif "add" in text and "cart" in text:
                return self._build_response("ADD_TO_CART", 0.91, request)
            elif "wishlist" in text or "save for later" in text:
                return self._build_response("ADD_TO_WISHLIST", 0.89, request)
            elif "checkout" in text:
                return self._build_response("VIEW_CART", 0.86, request)
            elif "total" in text or "price" in text:
                return self._build_response("CART_TOTAL", 0.88, request)
            elif "empty" in text and "cart" in text:
                return self._build_response("EMPTY_CART", 0.87, request)
            else:
                return self._fallback_unclear_response(request)

        # Real LLM Invocation (if simulate_mode = False)
        try:
            messages = self.build_messages(request)
            result = self.client.complete(
                {"messages": messages, "temperature": 0.3, "max_tokens": 400}
            )

            if "error" in result:
                raise RuntimeError(f"OpenAI API error: {result['error']}")

            parsed: ParsedLLMResponse = parse_llm_response(result.get("response", ""))
            
            # Extract and merge entities (LLM + rule-based fallback)
            entities = self._extract_and_merge_entities(request.user_input, parsed.entities)
            
            logger.info(
                f"LLM intent result: intent={parsed.intent}, action_code={parsed.action_code}, "
                f"confidence={parsed.confidence}, entities={'present' if entities else 'none'}"
            )
            
            # Log classification for feedback system
            try:
                from app.ai.feedback.classification_logger import get_classification_logger
                from app.ai.monitoring.intent_distribution_tracker import get_intent_distribution_tracker
                from app.ai.monitoring.confidence_tracker import get_confidence_tracker
                
                classification_logger = get_classification_logger()
                intent_tracker = get_intent_distribution_tracker()
                confidence_tracker = get_confidence_tracker()
                
                # Get request_id from metadata if available
                request_id = request.metadata.get("request_id") if request.metadata else None
                
                # Log LLM classification
                classification_logger.log_llm_classification(
                    query=request.user_input,
                    action_code=parsed.action_code,
                    confidence=parsed.confidence,
                    request_id=request_id,
                    prompt_version=self.prompt_loader.version if self.prompt_loader else None,
                    tokens_used=result.get("usage", {}).get("total_tokens") if "usage" in result else None,
                    latency_ms=result.get("latency_ms"),
                    context_snippets=request.context_snippets,
                    entities=entities,
                    reasoning=parsed.reasoning,
                )
                
                # Track intent distribution
                intent_tracker.record_intent(parsed.action_code)
                
                # Track confidence distribution
                confidence_tracker.record_confidence(parsed.confidence)
            except Exception as e:
                logger.warning(f"Failed to log classification: {e}")
            
            return {
                "triggered": True,
                "intent": parsed.intent,
                "intent_category": parsed.intent_category,
                "action_code": parsed.action_code,
                "confidence": parsed.confidence,
                "requires_clarification": False,
                "clarification_message": None,
                "entities": entities,  # NEW: Include entities in response
                "metadata": {
                    "source": "llm",
                    "processing_time_ms": result.get("latency_ms"),
                    "reasoning": parsed.reasoning,
                    "prompt_version": self.prompt_loader.version if self.prompt_loader else None,
                },
                "status": "LLM_RESULT",
                "variant": "A"
            }
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            fallback = build_fallback_response(reason="llm_failure")
            fallback["metadata"].update({"error": str(exc), "source": "llm"})
            return {"triggered": True, **fallback}

    # -----------------------------------------------------------------
    # Supporting methods
    # -----------------------------------------------------------------
    def _retain_rule_result(self, request: LLMIntentRequest) -> Dict[str, Any]:
        """Return rule-based classification directly."""
        return {
            "triggered": False,
            "intent": request.rule_intent or "UNKNOWN",
            "intent_category": "rule_based",
            "action_code": request.action_code or "NONE",
            "confidence": request.top_confidence,
            "requires_clarification": False,
            "clarification_message": None,
            "metadata": {
                "source": "rule_pipeline",
                "reason": "high_confidence",
                "fallback": False
            },
            "status": "RULE_CONFIDENT",
            "variant": "A"
        }

    def _fallback_unclear_response(self, request: LLMIntentRequest) -> Dict[str, Any]:
        """
        Return fallback intent when LLM is uncertain or fails.
        
        Now integrates with cache fallback (Task 19) - will try to find
        a similar cached response before returning default UNCLEAR.
        """
        # Use enhanced fallback with cache integration
        fallback_data = build_fallback_response(
            reason="llm_failure_or_unclear",
            user_query=request.user_input
        )
        
        # If fallback came from cache, return it wrapped with our standard structure
        if fallback_data.get("metadata", {}).get("fallback_source") == "cache":
            logger.info(f"Using cached response as fallback for unclear query")
            return {
                "triggered": True,
                **fallback_data
            }
        
        # Otherwise, return UNCLEAR with suggested questions
        clarification_msg = "I couldn't determine your intent clearly. Could you clarify what you meant?"

        fallback = {
            "triggered": True,
            "intent": "UNCLEAR",
            "intent_category": "uncertain",
            "action_code": "REQUEST_CLARIFICATION",
            "confidence": 0.0,
            "requires_clarification": True,
            "clarification_message": clarification_msg,
            "suggested_questions": [
                "Do you want to remove something from your cart?",
                "Are you trying to add a product?",
                "Would you like to view or clear your cart?",
            ],
            "metadata": {
                "source": "LLM",
                "reason": "fallback_unclear",
                "attempts": MAX_RETRIES,
                "variant": "A",
            },
            "status": "UNCLEAR",
            "variant": "A"
        }
        
        return fallback

    def _build_response(self, intent: str, confidence: float, request: LLMIntentRequest) -> Dict[str, Any]:
        """Helper to construct standardized LLM response structure."""
        logging.info(f"🧩 Intent classified as {intent} (confidence={confidence:.2f})")

        return {
            "triggered": True,
            "intent": intent,
            "intent_category": "llm",
            "action_code": intent,
            "confidence": confidence,
            "requires_clarification": False,
            "clarification_message": None,
            "metadata": {
                "source": "LLM",
                "reason": "model_prediction",
                "input": request.user_input,
                "variant": "A"
            },
            "status": "LLM_RESULT",
            "variant": "A"
        }

    def _escalate_failure(self, request: LLMIntentRequest):
        """
        Escalate after all retries fail.
        
        Sends critical alert to monitoring/alert system with full context.
        """
        # Build escalation context
        context = {
            "user_input": request.user_input,
            "rule_intent": request.rule_intent,
            "top_confidence": request.top_confidence,
            "action_code": request.action_code,
            "attempts": MAX_RETRIES,
            "timestamp": time.time(),
            "error_code": "all_retries_failed",
        }
        
        # Log critical error
        logging.critical(
            f"🚨 ESCALATION: LLM completely failed for input '{request.user_input}' "
            f"(rule_intent={request.rule_intent}, attempts={MAX_RETRIES})"
        )
        
        # Send critical alert (always escalates, not subject to frequency throttling)
        send_alert(
            event_type="LLM_COMPLETE_FAILURE",
            context=context,
            severity="critical"
        )

    # -----------------------------------------------------------------
    # Message Builder (from other branch)
    # -----------------------------------------------------------------
    def build_messages(self, payload: LLMIntentRequest) -> List[Dict[str, str]]:
        """
        Build message array for OpenAI ChatCompletion API.
        
        Uses system prompt and few-shot examples if available, otherwise
        falls back to simple user message.
        Enhances with context if available.
        
        Args:
            payload: LLM intent request payload
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        # Collect and enhance context
        context = None
        if self.enable_context:
            try:
                # Extract user_id and session_id from metadata or context_snippets
                user_id = payload.metadata.get("user_id") if payload.metadata else None
                session_id = payload.metadata.get("session_id") if payload.metadata else None
                
                # Collect conversation history from context_snippets or metadata
                conversation_history = []
                if payload.context_snippets:
                    # Convert context_snippets to conversation history format
                    conversation_history = [
                        {"role": "user", "content": snippet} if isinstance(snippet, str)
                        else snippet
                        for snippet in payload.context_snippets
                    ]
                
                # Collect all context
                raw_context = self.context_collector.collect_all_context(
                    conversation_history=conversation_history,
                    user_id=user_id,
                    session_id=session_id,
                    history_limit=self.context_history_limit
                )
                
                # Summarize to fit token limit
                context = self.context_summarizer.truncate_context(raw_context)
                
                logger.debug(f"Enhanced prompt with context: {len(context.get('conversation_history', []))} messages")
            except Exception as e:
                logger.warning(f"⚠️ Failed to collect/enhance context: {e}")
                context = None
        
        if self.prompt_loader is not None:
            try:
                messages = self.prompt_loader.build_messages(payload.user_input)
                logger.debug(
                    f"Built messages with system prompt and few-shot examples "
                    f"(version: {self.prompt_loader.version})"
                )
                return messages
            except (PromptLoadError, Exception) as e:
                logger.warning(f"Failed to use prompts, fallback to simple message: {e}")
                return [{"role": "user", "content": payload.user_input}]
        else:
            return [{"role": "user", "content": payload.user_input}]

    def _extract_and_merge_entities(self, user_query: str, llm_entities: Any) -> Optional[Dict[str, Any]]:
        """
        Extract and merge entities from LLM and rule-based sources.
        
        Strategy:
        1. If LLM entities are complete, use them
        2. Otherwise, extract via rule-based fallback
        3. Merge both sources (LLM takes precedence)
        
        Args:
            user_query: User input query
            llm_entities: Entities extracted by LLM (Entities object or None)
            
        Returns:
            Merged entities dict or None
        """
        try:
            # Import entity extractor
            from app.ai.entity_extraction.extractor import EntityExtractor
            from app.ai.entity_extraction.schema import Entities
            
            # Check if LLM entities are sufficient
            llm_has_entities = False
            if llm_entities and hasattr(llm_entities, 'product_type'):
                # Check if at least one meaningful entity is present
                llm_has_entities = any([
                    llm_entities.product_type,
                    llm_entities.brand,
                    llm_entities.category,
                    llm_entities.color,
                    llm_entities.size,
                    (llm_entities.price_range and (
                        llm_entities.price_range.min or llm_entities.price_range.max
                    ))
                ])
            
            # If LLM has complete entities, use them directly
            if llm_has_entities:
                logger.debug("Using entities from LLM (complete)")
                if hasattr(llm_entities, 'dict'):
                    return llm_entities.dict()
                else:
                    return llm_entities.__dict__ if hasattr(llm_entities, '__dict__') else None
            
            # Otherwise, fall back to rule-based extraction
            logger.debug("Falling back to rule-based entity extraction")
            extractor = EntityExtractor()
            rule_entities = extractor.extract_entities(user_query)
            
            # If no LLM entities, return rule-based directly
            if not llm_entities:
                logger.debug("Using entities from rule-based extraction")
                return rule_entities
            
            # Merge: LLM takes precedence, rule-based fills gaps
            merged = rule_entities.copy() if rule_entities else {}
            
            if hasattr(llm_entities, 'dict'):
                llm_dict = llm_entities.dict()
            elif hasattr(llm_entities, '__dict__'):
                llm_dict = llm_entities.__dict__
            else:
                llm_dict = {}
            
            # Override with LLM values where present
            for key, value in llm_dict.items():
                if value is not None:
                    if key == 'price_range' and hasattr(value, 'dict'):
                        merged[key] = value.dict()
                    elif key == 'price_range' and isinstance(value, dict):
                        merged[key] = value
                    else:
                        merged[key] = value
            
            logger.debug(f"Merged entities: LLM + rule-based")
            return merged if merged else None
            
        except ImportError:
            logger.warning("Entity extraction module not available")
            # Return LLM entities as-is if available
            if llm_entities:
                if hasattr(llm_entities, 'dict'):
                    return llm_entities.dict()
                elif hasattr(llm_entities, '__dict__'):
                    return llm_entities.__dict__
            return None
        except Exception as e:
            logger.error(f"Error extracting/merging entities: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _infer_category(action_code: Optional[str]) -> str:
        """Best-effort lookup of the category for a given action code."""
        if not action_code:
            return "SUPPORT_HELP"
        for category, codes in INTENT_CATEGORIES.items():
            if action_code in codes:
                return category
        return "SUPPORT_HELP"


__all__ = ["RequestHandler"]
