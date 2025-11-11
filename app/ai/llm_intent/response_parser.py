"""Utilities for validating and normalizing responses from the LLM."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

try:
    from app.ai.entity_extraction.schema import Entities, PriceRange
except ImportError:
    # Fallback if entity extraction not available
    Entities = None  # type: ignore
    PriceRange = None  # type: ignore

logger = logging.getLogger("app.llm_intent.response_parser")


@dataclass
class LLMIntentResponse:
    """Structured representation of the LLM output."""

    intent: str
    intent_category: str
    action_code: str
    confidence: float
    processing_time_ms: Optional[float] = None
    reasoning: Optional[str] = None
    entities: Optional[Any] = None  # Type: Entities or None


EXPECTED_SCHEMA: Dict[str, str] = {
    "action_code": "One of the ActionCode values consumed downstream",
    "confidence": "Float between 0.0 and 1.0",
    "reasoning": "Optional short justification string for audit trails",
    "secondary_intents": "Optional list of secondary intent codes",
    "entities_extracted": "Optional list of extracted entities",
}
"""Documentation of the properties expected from the LLM classifier."""


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text, handling common formatting issues.
    
    Handles:
    - JSON wrapped in code blocks (```json ... ```)
    - JSON wrapped in backticks
    - Plain JSON strings
    - Trailing text after JSON
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Remove code block markers
    if text.startswith("```"):
        # Extract content between ```json and ```
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # Try without language tag
            text = text.strip("```").strip()
    
    # Remove leading/trailing backticks
    text = text.strip("`")
    
    # Try to find JSON object in text
    # Look for first { and last }
    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _infer_intent_from_action_code(action_code: str) -> str:
    """Infer intent name from action code."""
    # Convert action code to intent format
    # E.g., "ADD_TO_CART" -> "add_to_cart"
    return action_code.lower().replace("_", "_")


def _infer_category_from_action_code(action_code: str) -> str:
    """
    Infer category from action code using common prefixes.
    
    This is a fallback - ideally the LLM should return intent_category,
    but if not, we infer from action code patterns.
    """
    code_upper = action_code.upper()
    
    # Map common prefixes to categories
    if code_upper.startswith("SEARCH") or code_upper.startswith("VIEW") or code_upper.startswith("FILTER"):
        return "SEARCH_DISCOVERY"
    elif code_upper.startswith("ADD_TO_CART") or code_upper.startswith("REMOVE") or "CART" in code_upper or "WISHLIST" in code_upper:
        return "CART_WISHLIST"
    elif code_upper.startswith("CHECKOUT") or code_upper.startswith("PAYMENT"):
        return "CHECKOUT_PAYMENT"
    elif code_upper.startswith("ORDER"):
        return "ORDERS_FULFILLMENT"
    elif code_upper.startswith("RETURN") or code_upper.startswith("REFUND"):
        return "RETURNS_REFUNDS"
    elif code_upper.startswith("REVIEW") or code_upper.startswith("RATING"):
        return "REVIEWS_RATINGS"
    elif code_upper.startswith("PASSWORD") or code_upper.startswith("LOGIN") or code_upper.startswith("ACCOUNT"):
        return "ACCOUNT_PROFILE"
    elif code_upper.startswith("HELP") or code_upper.startswith("CONTACT") or code_upper.startswith("SUPPORT"):
        return "SUPPORT_HELP"
    elif code_upper.startswith("FRAUD") or code_upper.startswith("SECURITY"):
        return "SECURITY_FRAUD"
    elif code_upper.startswith("PERSONALIZED") or code_upper.startswith("RECOMMEND"):
        return "PERSONALIZATION"
    elif code_upper.startswith("COUPON") or code_upper.startswith("PROMOTION"):
        return "PROMOTIONS_LOYALTY"
    else:
        return "SUPPORT_HELP"  # Default fallback


def parse_llm_response(raw: Union[str, Dict[str, Any]]) -> LLMIntentResponse:
    """
    Parse LLM response from string or dict format.
    
    Handles:
    - JSON string responses (from GPT-4)
    - Dict responses (from parsed JSON)
    - Text responses with embedded JSON
    
    Args:
        raw: Raw response from LLM (string or dict)
        
    Returns:
        LLMIntentResponse with parsed fields
        
    Raises:
        ValueError: If response cannot be parsed or required fields are missing
    """
    parsed_json: Optional[Dict[str, Any]] = None
    
    # Handle different input formats
    if isinstance(raw, dict):
        # Already a dict - check if it has the expected structure
        if "action_code" in raw:
            parsed_json = raw
        elif "raw_response" in raw:
            # Nested response format
            parsed_json = _extract_json_from_text(raw["raw_response"])
        else:
            # Try to extract JSON from string representation
            parsed_json = _extract_json_from_text(str(raw))
    elif isinstance(raw, str):
        # String response - try to parse JSON
        parsed_json = _extract_json_from_text(raw)
    else:
        # Unknown format - try to convert to string and parse
        parsed_json = _extract_json_from_text(str(raw))
    
    if parsed_json is None:
        raw_str = str(raw)[:200] if raw else "None"
        logger.warning(f"Could not parse LLM response as JSON: {raw_str}")
        raise ValueError(f"LLM response is not valid JSON: {str(raw)[:100]}...")
    
    # Extract required fields
    try:
        action_code = parsed_json.get("action_code") or parsed_json.get("actionCode")
        if not action_code:
            raise ValueError("Missing required field: action_code")
        
        confidence = float(parsed_json.get("confidence", 0.5))
        confidence = max(0.0, min(confidence, 1.0))  # Clamp to [0, 1]
        
        # Extract optional fields
        reasoning = parsed_json.get("reasoning") or parsed_json.get("reason")
        intent_category = (
            parsed_json.get("intent_category") or 
            parsed_json.get("intentCategory") or
            _infer_category_from_action_code(action_code)
        )
        intent = (
            parsed_json.get("intent") or
            _infer_intent_from_action_code(action_code)
        )
        
        # Extract entities if available
        entities = None
        if Entities and "entities" in parsed_json:
            try:
                entities_data = parsed_json["entities"]
                if entities_data and isinstance(entities_data, dict):
                    # Parse price_range if present
                    price_range_data = entities_data.get("price_range")
                    if price_range_data and isinstance(price_range_data, dict) and PriceRange:
                        price_range = PriceRange(**price_range_data)
                    else:
                        price_range = None
                    
                    # Create Entities object
                    entities = Entities(
                        product_type=entities_data.get("product_type"),
                        category=entities_data.get("category"),
                        brand=entities_data.get("brand"),
                        color=entities_data.get("color"),
                        size=entities_data.get("size"),
                        price_range=price_range
                    )
                    logger.debug(f"Extracted entities: {entities}")
            except Exception as e:
                logger.warning(f"Failed to parse entities from LLM response: {e}")
                entities = None
        
        logger.debug(
            f"Parsed LLM response: action_code={action_code}, "
            f"confidence={confidence:.2f}, category={intent_category}, "
            f"entities={'present' if entities else 'none'}"
        )
        
        return LLMIntentResponse(
            intent=intent,
            intent_category=intent_category,
            action_code=action_code,
            confidence=confidence,
            processing_time_ms=parsed_json.get("processing_time_ms"),
            reasoning=reasoning,
            entities=entities,
        )
        
    except (KeyError, TypeError, ValueError) as exc:
        logger.error(f"Failed to parse LLM response fields: {exc}, raw: {parsed_json}")
        raise ValueError(f"Malformed LLM intent response: {exc}") from exc


__all__ = ["LLMIntentResponse", "EXPECTED_SCHEMA", "parse_llm_response"]
