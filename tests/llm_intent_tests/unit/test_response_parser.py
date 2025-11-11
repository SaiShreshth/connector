"""
Unit tests for LLM response parser.

Tests JSON parsing, confidence clamping, and field inference.
"""

import pytest
import json

from app.ai.llm_intent.response_parser import parse_llm_response, LLMIntentResponse


def test_response_parser_json_string():
    """Test that response parser handles JSON string responses."""
    json_response = '''
    {
      "action_code": "ADD_TO_CART",
      "confidence": 0.95,
      "reasoning": "User wants to add item",
      "secondary_intents": [],
      "entities_extracted": []
    }
    '''
    
    result = parse_llm_response(json_response)
    
    assert result.action_code == "ADD_TO_CART"
    assert result.confidence == 0.95
    assert result.reasoning == "User wants to add item"
    assert result.intent_category is not None
    assert result.intent is not None


def test_response_parser_json_code_block():
    """Test that response parser handles JSON in code blocks."""
    code_block_response = '''```json
    {
      "action_code": "SEARCH_PRODUCT",
      "confidence": 0.88,
      "reasoning": "Searching for products"
    }
    ```'''
    
    result = parse_llm_response(code_block_response)
    
    assert result.action_code == "SEARCH_PRODUCT"
    assert result.confidence == 0.88


def test_response_parser_dict_format():
    """Test that response parser handles dict format."""
    dict_response = {
        "action_code": "ORDER_STATUS",
        "confidence": 0.92,
        "reasoning": "Check order status"
    }
    
    result = parse_llm_response(dict_response)
    
    assert result.action_code == "ORDER_STATUS"
    assert result.confidence == 0.92


def test_response_parser_confidence_clamping():
    """Test that confidence is clamped to [0, 1]."""
    # Test over 1.0
    result1 = parse_llm_response({
        "action_code": "TEST",
        "confidence": 1.5
    })
    assert result1.confidence == 1.0
    
    # Test under 0.0
    result2 = parse_llm_response({
        "action_code": "TEST",
        "confidence": -0.5
    })
    assert result2.confidence == 0.0


def test_response_parser_missing_action_code():
    """Test error handling when action_code is missing."""
    # Should raise ValueError when action_code is missing
    # The error message may vary depending on parsing path
    with pytest.raises(ValueError):
        parse_llm_response({
            "confidence": 0.5
        })


def test_response_parser_category_inference():
    """Test that intent_category is inferred from action_code if missing."""
    result = parse_llm_response({
        "action_code": "ADD_TO_CART",
        "confidence": 0.9
    })
    
    # Should infer category from action code
    assert result.intent_category == "CART_WISHLIST"


def test_response_parser_intent_inference():
    """Test that intent is inferred from action_code if missing."""
    result = parse_llm_response({
        "action_code": "SEARCH_PRODUCT",
        "confidence": 0.85
    })
    
    # Should infer intent from action code
    assert result.intent is not None
    assert "search" in result.intent.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

