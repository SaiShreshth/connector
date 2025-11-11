"""
Ambiguity Resolver for Intent Classification

Handles ambiguous and multi-intent user queries by:
1. Detecting when user input matches multiple intents (AMBIGUOUS)
2. Detecting when user input has low confidence (UNCLEAR)
3. Publishing ambiguous queries to classification queue
4. Providing fallback behavior

Dependencies:
- Uses action codes and intent structure from IntentTaxonomy
- Uses keyword dictionaries aligned with action codes

Confidence Thresholds:
- UNCLEAR_THRESHOLD (0.4): Below this = unclear intent
- MIN_CONFIDENCE (0.6): Above this = valid intent
- AMBIGUOUS: Multiple intents above this = AMBIGUOUS

This module now integrates with the queue producer to handle ambiguous cases
that need further processing by the ML-based classifier.
"""

import os
import json
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, cast

# Add parent directory to path for imports when running standalone
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import keyword dictionaries aligned with action codes
try:
    from .keywords.loader import load_keywords
    from app.queue.queue_producer import IntentQueueProducer, IntentScore, RuleBasedResult
except ImportError:
    from keywords.loader import load_keywords
    from app.queue.queue_producer import IntentQueueProducer, IntentScore, RuleBasedResult

# Initialize queue producer
_queue_producer = IntentQueueProducer()

# ------------------ CONFIGURATION ------------------
# Load config from config manager (dynamic) or fallback to static defaults

# --- CONFIG MANAGER INTEGRATION ---
try:
    from app.core.config_manager import CONFIG_CACHE, ACTIVE_VARIANT
    _USE_CONFIG_MANAGER = True
except Exception:
    _USE_CONFIG_MANAGER = False

# Fallback imports (only used if config manager fails)
# If app/ai/config.py is missing, system should fail (critical error)
from app.ai.config import UNCLEAR_THRESHOLD as _FALLBACK_UNCLEAR
from app.ai.config import MIN_CONFIDENCE as _FALLBACK_MIN_CONF


def _get_config_value(key: str, fallback: float) -> float:
    """Load config value from config manager, or use fallback."""
    if not _USE_CONFIG_MANAGER:
        return fallback
    
    try:
        rules = CONFIG_CACHE.get("rules", {}).get("rules", {})
        rule_sets = rules.get("rule_sets", {})
        current_rules = rule_sets.get(ACTIVE_VARIANT, {})
        return current_rules.get(key, fallback)
    except Exception:
        return fallback


# Load config values dynamically
UNCLEAR_THRESHOLD = _get_config_value("unclear_threshold", _FALLBACK_UNCLEAR)
MIN_CONFIDENCE = _get_config_value("min_confidence", _FALLBACK_MIN_CONF)
AMBIGUOUS_THRESHOLD = MIN_CONFIDENCE

# Persist logs as JSON Lines for easy processing
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs")
LOG_DIR = os.path.abspath(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "ambiguous.jsonl")

# ------------------ LOAD KEYWORD DATA ------------------
# Load keyword dictionaries
# Action codes match IntentTaxonomy definitions
# (e.g., SEARCH_PRODUCT, ADD_TO_CART)
loaded_keywords = load_keywords()

INTENT_KEYWORDS = {
    action_code: details.get("keywords", [])
    for action_code, details in loaded_keywords.items()
}

# ------------------ FUNCTIONS ------------------

def log_ambiguous_case(user_input, intent_scores):
    """Append ambiguous/unclear cases as JSONL for later review."""
    entry = {"user_input": user_input, "intent_scores": intent_scores}
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def calculate_confidence(user_input, keywords):
    """
    Calculate confidence score using keywords.
    
    Keywords align with action codes from the intent taxonomy.
    
    Args:
        user_input: User's query (lowercase)
        keywords: List of keyword phrases for this action code
        
    Returns:
        float: Confidence score (0.0 to 1.0)
    """
    if not keywords:
        return 0.0
    
    user_input_lower = user_input.lower()
    
    # Count exact phrase matches
    matches = sum(1 for keyword in keywords if keyword.lower() in user_input_lower)
    
    if matches == 0:
        return 0.0
    
    # Weighted scoring: more matches = higher confidence
    # 1 match = 0.7, 2 matches = 0.85, 3+ matches = 0.95
    if matches == 1:
        return 0.7
    elif matches == 2:
        return 0.85
    else:
        return 0.95


def find_matched_keywords_for_action(user_input_lower: str, keywords: List[str]) -> List[str]:
    """Return the list of keywords that matched in the user_input (case-insensitive)."""
    matched = [k for k in keywords if k.lower() in user_input_lower]
    return matched


def fallback_behavior(user_input):
    """Handle unclear or low-confidence user inputs."""
    logger.warning(f"Fallback: Input unclear or low confidence: '{user_input}'")
    logger.info("Sending this input to LLM or asking user for clarification...")


def detect_intent(user_input: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Detect user intent with ambiguity resolution.
    
    Uses keywords that align with action codes from the intent taxonomy.
    Returns action codes defined in IntentTaxonomy.
    
    Uses standardized confidence thresholds:
    - < 0.4 (UNCLEAR_THRESHOLD): Returns UNCLEAR
    - >= 0.6 (MIN_CONFIDENCE): Valid intent
    - Multiple >= 0.6: Returns AMBIGUOUS
    
    For ambiguous or unclear intents, publishes to the classification queue
    for further processing.
    
    Args:
        user_input: The user's query text
        metadata: Optional additional context about the request
        
    Returns:
        dict: Contains action, confidence, possible_intents and request_id
        - action: "UNCLEAR", "AMBIGUOUS", or specific action code
        - confidence: confidence score (if single intent)
        - possible_intents: all detected action codes with scores
        - request_id: tracking ID for queued messages (if applicable)
    """
    user_input_lower = user_input.lower()
    intent_confidences = {}
    request_id = None

    # Calculate confidence for each action code using keywords
    for action_code, keywords in INTENT_KEYWORDS.items():
        confidence = calculate_confidence(user_input_lower, keywords)
        if confidence > 0:
            intent_confidences[action_code] = round(confidence, 2)

    # Filter only high-confidence intents (>= MIN_CONFIDENCE)
    high_conf_intents = {
        action_code: conf for action_code, conf in intent_confidences.items()
        if conf >= MIN_CONFIDENCE
    }

    def _format_intent_scores(intents_dict: Dict[str, float]) -> List[IntentScore]:
        return [{"intent": action_code, "score": float(score)}
                for action_code, score in intents_dict.items()]

    # Case 1: No high-confidence intents → UNCLEAR
    if not high_conf_intents:
        log_ambiguous_case(user_input, {"type": "unclear", "intents": intent_confidences})
        
        try:
            # Queue for further processing
            # Build rule_based_result shape per spec
            matched_keywords = []
            for ac, kws in INTENT_KEYWORDS.items():
                matched_keywords.extend(find_matched_keywords_for_action(user_input_lower, kws))

            rb_result = cast(dict, {
                "action_code": "UNCLEAR",
                "confidence": round(max(intent_confidences.values()) if intent_confidences else 0.0, 2),
                "matched_keywords": matched_keywords
            })

            request_id = _queue_producer.publish_ambiguous_query(
                query=user_input,
                intent_scores=_format_intent_scores(intent_confidences),
                priority=False,  # Unclear cases are not typically priority
                metadata={"type": "unclear", **(metadata or {})},
                session_id=(metadata or {}).get('session_id'),
                user_id=(metadata or {}).get('user_id'),
                conversation_history=(metadata or {}).get('conversation_history'),
                rule_based_result=cast(RuleBasedResult, rb_result)
            )
            logger.info(f"Queued unclear intent for processing with request_id={request_id}")
            
        except Exception as e:
            logger.error(f"Failed to queue unclear intent: {str(e)}", exc_info=True)
            fallback_behavior(user_input)
            
        return {
            "action": "UNCLEAR",
            "possible_intents": intent_confidences,
            "request_id": request_id
        }

    # Case 2: Multiple high-confidence intents → AMBIGUOUS
    if len(high_conf_intents) > 1:
        log_ambiguous_case(user_input, {"type": "multiple", "intents": high_conf_intents})
        
        try:
            # Build rule_based_result for ambiguous case
            matched_keywords = []
            for ac in high_conf_intents.keys():
                matched_keywords.extend(find_matched_keywords_for_action(user_input_lower, INTENT_KEYWORDS.get(ac, [])))

            rb_result = cast(dict, {
                "action_code": "AMBIGUOUS",
                "confidence": round(max(high_conf_intents.values()), 2),
                "matched_keywords": matched_keywords
            })

            # Queue ambiguous case with priority
            request_id = _queue_producer.publish_ambiguous_query(
                query=user_input,
                intent_scores=_format_intent_scores(high_conf_intents),
                priority=True,  # Ambiguous high-confidence cases are priority
                metadata={"type": "ambiguous", **(metadata or {})},
                session_id=(metadata or {}).get('session_id'),
                user_id=(metadata or {}).get('user_id'),
                conversation_history=(metadata or {}).get('conversation_history'),
                rule_based_result=cast(RuleBasedResult, rb_result)
            )
            logger.info(f"Queued ambiguous intent for processing with request_id={request_id}")
            
        except Exception as e:
            logger.error(f"Failed to queue ambiguous intent: {str(e)}", exc_info=True)
            
        return {
            "action": "AMBIGUOUS",
            "possible_intents": high_conf_intents,
            "request_id": request_id
        }

    # Case 3: Single clear intent → Return action code
    final_action_code = list(high_conf_intents.keys())[0]
    return {
        "action": final_action_code,
        "confidence": high_conf_intents[final_action_code],
        "possible_intents": high_conf_intents
    }


# ------------------ MAIN (for testing) ------------------

def run_demo():
    """Run an interactive demo of the ambiguity resolver with queue integration."""
    print("Welcome! Type your message to detect intent (type 'exit' to quit).")

    # Print summary of loaded action codes
    print(f"\nLoaded {len(INTENT_KEYWORDS)} action codes from intent taxonomy:")
    for action_code, keywords in list(INTENT_KEYWORDS.items())[:10]:
        print(f"  - {action_code}: {len(keywords)} keywords")
    if len(INTENT_KEYWORDS) > 10:
        print(f"  ... and {len(INTENT_KEYWORDS) - 10} more action codes")
        
    print("\nQueue producer initialized and ready to process ambiguous queries.")
    print("Ambiguous or unclear intents will be queued for ML processing.")
    print("\nTest queries to try:")
    print("1. 'show red shoes' (single clear intent)")
    print("2. 'show red shoes and add to cart' (ambiguous - multiple intents)")
    print("3. 'help me please' (unclear intent)")

if __name__ == "__main__":
    run_demo()

    while True:
        try:
            user_input = input("\nYour input: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
                
            # Add some test metadata
            metadata = {
                "session_id": "test-session",
                "timestamp": datetime.now().isoformat()
            }
            
            result = detect_intent(user_input, metadata)
            print("\nDetected Result:", json.dumps(result, indent=4))
            
            if result.get("request_id"):
                print(f"\n🔄 Query queued for ML processing with ID: {result['request_id']}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.error("Error processing input", exc_info=True)
