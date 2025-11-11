"""
Decision Engine
Handles hybrid rule-based + embedding + LLM fallback intent classification.
Now includes resilience, caching, structured logging, and queue-based escalation.
"""
import os
import re
import uuid
import traceback
import logging
from typing import List, Dict, Any

# --- CONFIG MANAGER INTEGRATION ---
from app.core.config_manager import CONFIG_CACHE, ACTIVE_VARIANT, switch_variant
from app.ai.config import PRIORITY_THRESHOLD, WEIGHTS
from app.ai.intent_classification.keyword_matcher import KeywordMatcher
from app.ai.intent_classification.embedding_matcher import EmbeddingMatcher
from app.ai.intent_classification.hybrid_classifier import HybridClassifier
from app.ai.intent_classification import confidence_threshold

# --- QDRANT INTEGRATION ---
try:
    from app.ai.llm_intent.qdrant_cache import store_vector
except Exception:
    store_vector = None

# --- LLM HANDLER / FALLBACK INTEGRATION ---
try:
    from app.ai.llm_intent.request_handler import RequestHandler as _LLMHandler
    from app.ai.llm_intent.openai_client import OpenAIClient as _OpenAIClient
    from app.schemas.llm_intent import LLMIntentRequest as _LLMReq
except Exception:
    _LLMHandler = None  # type: ignore
    _OpenAIClient = None  # type: ignore
    _LLMReq = None  # type: ignore
# --- END CONFIG MANAGER INTEGRATION ---
    _LLMHandler = None
    _OpenAIClient = None
    _LLMReq = None

# Note: Real ResilientOpenAIClient is in app.ai.llm_intent.resilient_openai_client
# and is used via RequestHandler for proper LLM calls

# --- ALERT NOTIFIER (NEW ADDITION) ---
try:
    from app.core.alert_notifier import send_alert
except Exception:
    send_alert = None

# --- LOGGER ---
from loguru import logger

# --- TEXT NORMALIZER ---
def _clean_text(query: str) -> str:
    """Normalize common typos, spacing issues, and casing."""
    query = query.lower().strip()
    query = re.sub(r"ing\s+", "ing", query)  # Fix 'show ing' → 'showing'
    query = re.sub(r"\s+", " ", query)       # Remove extra spaces
    query = re.sub(r"[^a-z0-9\s]", "", query)  # Clean punctuation
    return query


class DecisionEngine:
    """Orchestrates hybrid search with resilience, caching, and escalation."""

    def __init__(self):
        logger.info("Initializing DecisionEngine...")
        self.keyword_matcher = KeywordMatcher()
        self.embedding_matcher = None
        self.hybrid_classifier = HybridClassifier()
        self._cache: Dict[str, Dict[str, Any]] = {}  # ✅ Local read-through cache
        self._load_config_from_manager()
        logger.info(f"✅ DecisionEngine Initialized: variant={ACTIVE_VARIANT}")

    # ------------------------------------------------------------------
    def _load_config_from_manager(self):
        """Loads config dynamically or uses fallback."""
        try:
            rules_root = CONFIG_CACHE.get("rules", {})
            rules = rules_root.get("rules", rules_root)
            rule_sets = rules.get("rule_sets", {})
            current_rules = rule_sets.get(ACTIVE_VARIANT, {})

            if current_rules:
                self.use_embedding = current_rules.get("use_embedding", True)
                self.use_keywords = current_rules.get("use_keywords", True)
                self.kw_weight = current_rules.get("kw_weight", WEIGHTS.get("keyword", 0.6))
                self.emb_weight = current_rules.get("emb_weight", WEIGHTS.get("embedding", 0.4))
                self.priority_threshold = current_rules.get("priority_threshold", PRIORITY_THRESHOLD)
                self.hybrid_classifier.update_weights(self.kw_weight, self.emb_weight)
                logger.info(f"📋 Using dynamic rules for {ACTIVE_VARIANT}")
            else:
                raise KeyError("Missing dynamic config")

        except Exception as e:
            logger.warning(f"⚠️ Config manager error: {e} — Using fallback from app/ai/config.py")
            # Use centralized fallback values from app/ai/config.py
            self.use_embedding = True
            self.use_keywords = True
            self.priority_threshold = PRIORITY_THRESHOLD  # From fallback file
            self.kw_weight = WEIGHTS["keyword"]           # From fallback file
            self.emb_weight = WEIGHTS["embedding"]        # From fallback file
            self.hybrid_classifier.update_weights(self.kw_weight, self.emb_weight)

    # ------------------------------------------------------------------
    def _send_escalation_alert(self, reason: str, query: str):
        """Concrete escalation path integrated with alert_notifier."""
        alert_id = str(uuid.uuid4())
        logger.warning(f"[ALERT:{alert_id}] Escalation: {reason} | Query='{query}'")
        if send_alert:
            try:
                send_alert(
                    event_type=reason,
                    context={
                        "query": query,
                        "alert_id": alert_id,
                        "source": "DecisionEngine",
                    },
                )
            except Exception as e:
                logger.error(f"Failed to send escalation alert: {e}")

    # ------------------------------------------------------------------
    def search(self, query: str) -> Dict[str, Any]:
        """Executes hybrid search + LLM + cache fallback with resilience."""
        import time
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        logger.info(f"[{correlation_id}] 🔍 Starting intent classification for: '{query}'")

        # 🧹 Clean query text to handle typos like "show ing"
        query = _clean_text(query)

        # ✅ Cache read-through fallback
        if query in self._cache:
            logger.info(f"[{correlation_id}] Cache hit for query '{query}'")
            cache_result = self._cache[query]
            # Log cache hit and track latency
            total_latency = (time.time() - start_time) * 1000
            try:
                from app.ai.feedback.classification_logger import get_classification_logger
                from app.ai.monitoring.intent_distribution_tracker import get_intent_distribution_tracker
                from app.ai.monitoring.confidence_tracker import get_confidence_tracker
                
                classification_logger = get_classification_logger()
                intent_tracker = get_intent_distribution_tracker()
                confidence_tracker = get_confidence_tracker()
                
                action_code = cache_result.get("intent", {}).get("id") or cache_result.get("action_code", "UNKNOWN")
                confidence = cache_result.get("intent", {}).get("score") or cache_result.get("confidence_score", 0.0)
                
                classification_logger.log_cache_hit(
                    query=query,
                    action_code=action_code,
                    confidence=confidence,
                    request_id=correlation_id,
                )
                intent_tracker.record_intent(action_code)
                confidence_tracker.record_confidence(confidence)
            except Exception as e:
                logger.warning(f"Failed to log cache hit: {e}")
            
            if total_latency > 3000:
                logger.warning(f"⚠️ Total pipeline latency {total_latency:.2f}ms exceeds 3s threshold")
            
            return cache_result

        try:
            # Run hybrid classification
            rule_based_start = time.time()
            result = self._run_hybrid_search(query)
            total_latency = (time.time() - start_time) * 1000
            rule_based_latency = (time.time() - rule_based_start) * 1000
            
            # Log handoff and track latency
            source = result.get("status", "UNKNOWN")
            action_code = result.get("intent", {}).get("id") if isinstance(result.get("intent"), dict) else result.get("intent", "UNKNOWN")
            confidence = result.get("intent", {}).get("score") if isinstance(result.get("intent"), dict) else result.get("confidence_score", 0.0)
            
            try:
                from app.ai.feedback.classification_logger import get_classification_logger
                from app.ai.monitoring.intent_distribution_tracker import get_intent_distribution_tracker
                from app.ai.monitoring.confidence_tracker import get_confidence_tracker
                
                classification_logger = get_classification_logger()
                intent_tracker = get_intent_distribution_tracker()
                confidence_tracker = get_confidence_tracker()
                
                # Log handoff to LLM queue if applicable
                if source == "QUEUED_FOR_LLM":
                    logger.info(f"[{correlation_id}] Handoff: rule-based → LLM queue (latency: {rule_based_latency:.2f}ms)")
                    # Handoff logging will be done by queue worker
                elif "CONFIDENT" in source:
                    # Get entities and match info from result
                    entities = result.get("entities")
                    intent_obj = result.get("intent", {})
                    match_type = intent_obj.get("match_type")
                    matched_text = intent_obj.get("matched_text")
                    
                    classification_logger.log_rule_based_classification(
                        query=query,
                        action_code=action_code,
                        confidence=confidence,
                        request_id=correlation_id,
                        entities=entities,
                        match_type=match_type,
                        matched_text=matched_text,
                    )
                    intent_tracker.record_intent(action_code)
                    confidence_tracker.record_confidence(confidence)
                
                # Track total pipeline latency
                if total_latency > 3000:
                    logger.warning(f"⚠️ Total pipeline latency {total_latency:.2f}ms exceeds 3s threshold")
                else:
                    logger.info(f"[{correlation_id}] Total pipeline latency: {total_latency:.2f}ms")
            except Exception as e:
                logger.warning(f"Failed to log classification/handoff: {e}")
            
            self._cache[query] = result  # Cache successful result
            return result

        except Exception as e:
            total_latency = (time.time() - start_time) * 1000
            logger.error(f"[{correlation_id}] Hybrid classification failed: {e} (latency: {total_latency:.2f}ms)", exc_info=True)
            traceback.print_exc()

            # Note: LLM fallback should be handled via RequestHandler and queue system
            # The stub resilient_client has been removed; proper LLM handling goes through
            # app.ai.llm_intent.request_handler.RequestHandler which has full error handling

            # 🧩 Last-good fallback
            fallback = {
                "status": "FALLBACK_LAST_GOOD",
                "intent": {
                    "id": "SEARCH_PRODUCT",
                    "intent": "SEARCH_PRODUCT",
                    "score": 0.3,
                    "source": "cached_default",
                },
                "correlation_id": correlation_id,
            }
            self._send_escalation_alert("CACHE_FALLBACK", query)
            self._cache[query] = fallback
            return fallback

    # ------------------------------------------------------------------
    def _run_hybrid_search(self, query: str) -> Dict[str, Any]:
        """Main hybrid + queue-enhanced logic."""
        self._load_config_from_manager()

        keyword_results = []
        embedding_results = []

        if self.use_keywords:
            keyword_results = self.keyword_matcher.search(query)
            if keyword_results and keyword_results[0]["score"] >= self.priority_threshold:
                logger.info(f"✅ High-confidence keyword match: {keyword_results[0]['id']}")
                
                # Extract entities for keyword-based results
                entities = None
                try:
                    from app.ai.entity_extraction.extractor import EntityExtractor
                    extractor = EntityExtractor()
                    entities = extractor.extract_entities(query)
                    # Convert to proper format (product_name -> product_type, capitalize brand, etc.)
                    if entities:
                        # Map product_name to product_type for consistency
                        if "product_name" in entities and entities["product_name"]:
                            entities["product_type"] = entities.pop("product_name")
                        
                        # Capitalize brand name (e.g., "nike" -> "Nike")
                        if "brand" in entities and entities["brand"]:
                            entities["brand"] = entities["brand"].capitalize()
                        
                        # Ensure price_range is in correct format if present
                        if "price_range" in entities:
                            if isinstance(entities["price_range"], dict):
                                # Already in correct format, ensure currency is set if price found
                                if (entities["price_range"].get("max") or entities["price_range"].get("min")) and not entities["price_range"].get("currency"):
                                    entities["price_range"]["currency"] = "USD" if "$" in query else "INR"
                            elif isinstance(entities["price_range"], str):
                                # Convert string format to dict if needed
                                price_str = entities["price_range"]
                                if "-" in price_str:
                                    parts = price_str.split("-")
                                    entities["price_range"] = {
                                        "min": int(parts[0]) if parts[0] else None,
                                        "max": int(parts[1]) if len(parts) > 1 and parts[1] else None,
                                        "currency": "USD" if "$" in query else "INR"
                                    }
                        
                        # Remove None values to clean up the response, but keep price_range dict even if min/max are None
                        price_range_backup = entities.get("price_range")
                        entities = {k: v for k, v in entities.items() if v is not None}
                        # Restore price_range if it was a dict (even with None values, it's still useful)
                        if price_range_backup and isinstance(price_range_backup, dict):
                            entities["price_range"] = price_range_backup
                        
                        logger.debug(f"Extracted entities for keyword match: {entities}")
                except Exception as e:
                    logger.warning(f"Failed to extract entities for keyword match: {e}")
                    entities = None
                
                return {
                    "status": "CONFIDENT_KEYWORD",
                    "intent": keyword_results[0],
                    "config_variant": ACTIVE_VARIANT,
                    "entities": entities,
                }

        if self.use_embedding:
            if not self.embedding_matcher:
                self.embedding_matcher = EmbeddingMatcher()
            embedding_results = self.embedding_matcher.search(query)

        if self.use_keywords and self.use_embedding:
            blended_results = self.hybrid_classifier.blend(keyword_results, embedding_results)
        elif self.use_keywords:
            blended_results = keyword_results
        elif self.use_embedding:
            blended_results = embedding_results
        else:
            blended_results = []

        if not blended_results:
            logger.warning(f"⚠ No match found for query '{query}'. Falling back.")
            return self._fallback_generic(query)

        # 🚦 Confidence threshold guard
        if blended_results and blended_results[0].get("score", 0.0) < 0.3:
            logger.warning(f"⚠ Low confidence score ({blended_results[0].get('score')}). Forcing fallback.")
            return self._fallback_generic(query)

        is_confident, reason = confidence_threshold.is_confident(blended_results)
        if is_confident:
            logger.info(f"✅ Blended result is confident. Reason: {reason}")
            return {
                "status": reason,
                "intent": blended_results[0],
                "config_variant": ACTIVE_VARIANT,
            }

        # 🚀 Queue-based fallback + LLM fallback
        logger.warning(f"⚠ Blended result is NOT confident. Reason: {reason}")

        enable_queue = os.getenv("ENABLE_LLM_QUEUE", "true").lower() == "true"
        if enable_queue:
            try:
                from app.queue.integration import send_to_llm_queue
                from app.ai.intent_classification.ambiguity_resolver import detect_intent

                ambiguity_result = detect_intent(query)
                action = ambiguity_result.get("action")
                
                # Send to queue if:
                # 1. Explicitly AMBIGUOUS or UNCLEAR, OR
                # 2. Low confidence (< 0.6) from rule-based matching (needs LLM processing), OR
                # 3. We're in fallback mode (this path means rule-based wasn't confident enough)
                confidence = ambiguity_result.get("confidence", 0.0)
                blended_score = blended_results[0].get("score", 0.0) if blended_results else 0.0
                should_queue = (
                    action in ["AMBIGUOUS", "UNCLEAR"] or
                    confidence < 0.6 or
                    blended_score < 0.6 or
                    not blended_results  # No matches at all - definitely needs LLM
                )
                
                if should_queue:
                    message_id = send_to_llm_queue(
                        query=query,
                        ambiguity_result=ambiguity_result,
                        user_id="anonymous",
                        is_premium=False,
                    )
                    if message_id:
                        logger.info(f"✅ Sent query to LLM queue: {message_id} (action={action}, confidence={confidence})")
                        return {
                            "status": "QUEUED_FOR_LLM",
                            "intent": {
                                "id": "PROCESSING",
                                "intent": "PROCESSING",
                                "score": 0.0,
                                "source": "queue",
                                "message_id": message_id,
                            },
                            "message": "Query sent to LLM queue for processing",
                            "config_variant": ACTIVE_VARIANT,
                        }
                    else:
                        logger.warning("⚠ Queue returned None message_id, falling back to sync LLM")
                else:
                    logger.info(f"⏭ Skipping queue for high-confidence action: {action} (confidence={confidence})")
            except Exception as e:
                logger.warning(f"⚠ Queue integration error: {e}. Falling back to sync LLM processing.", exc_info=True)

        # Fallback: Sync LLM call
        if _LLMHandler and _LLMReq and _OpenAIClient:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("⚠ OPENAI_API_KEY not set, skipping LLM fallback")
                else:
                    llm_client = _OpenAIClient()
                    handler = _LLMHandler(client=llm_client)
                    top_kw = keyword_results[0]["score"] if keyword_results else 0.0
                    next_kw = keyword_results[1]["score"] if len(keyword_results) > 1 else 0.0
                    llm_req = _LLMReq(
                        user_input=query,
                        rule_intent=blended_results[0]["id"] if blended_results else None,
                        action_code=blended_results[0]["id"] if blended_results else None,
                        top_confidence=float(top_kw),
                        next_best_confidence=float(next_kw),
                        is_fallback=True,
                    )
                    llm_out = handler.handle(llm_req)
                    if llm_out and isinstance(llm_out, dict):
                        resolved = {
                            "id": llm_out.get("action_code"),
                            "intent": llm_out.get("action_code"),
                            "score": llm_out.get("confidence", 0.0),
                            "source": "llm",
                            "reason": "llm_fallback",
                        }
                        return {
                            "status": "LLM_FALLBACK",
                            "intent": resolved,
                            "config_variant": ACTIVE_VARIANT,
                        }
            except Exception as e:
                logger.error(f"⚠ LLM fallback error: {e}", exc_info=True)

        # Deterministic fallback
        if blended_results:
            blended_results.sort(key=lambda x: (x.get("score", 0.0), x.get("embedding_score", 0.0)), reverse=True)
            top = blended_results[0]
            logger.info("✅ Selecting top blended action deterministically")
            return {
                "status": f"BLENDED_TOP_{reason}",
                "intent": top,
                "config_variant": ACTIVE_VARIANT,
            }
        else:
            logger.warning("⚠ No suitable results, returning generic search")
            return self._fallback_generic(query)

    # ------------------------------------------------------------------
    def _fallback_generic(self, query: str) -> Dict[str, Any]:
        """Generic search fallback intent."""
        return {
            "status": "FALLBACK_GENERIC",
            "intent": {
                "id": "SEARCH_PRODUCT",
                "intent": "SEARCH_PRODUCT",
                "score": 0.1,
                "source": "fallback",
            },
            "config_variant": ACTIVE_VARIANT,
        }


# ----------------------------------------------------------------------
# Singleton for app.main
# ----------------------------------------------------------------------
_decision_engine = None


def get_intent_classification(query: str) -> Dict[str, Any]:
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine()
    return _decision_engine.search(query)
