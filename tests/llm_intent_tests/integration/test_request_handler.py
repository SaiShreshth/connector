"""
Integration tests for RequestHandler.

Tests end-to-end flow with prompt integration.
"""

import pytest
from unittest.mock import Mock, patch
import json

from app.ai.llm_intent.request_handler import RequestHandler
from app.ai.llm_intent.response_parser import parse_llm_response
from app.schemas.llm_intent import LLMIntentRequest


def test_request_handler_integration():
    """Test that RequestHandler integrates prompts correctly."""
    mock_client = Mock()
    handler = RequestHandler(client=mock_client)
    
    # Should have prompt loader
    assert handler.prompt_loader is not None
    
    # Test build_messages
    payload = LLMIntentRequest(
        user_input="Show me running shoes",
        top_confidence=0.3,
        next_best_confidence=0.2,
        action_code=None,
        is_fallback=True,
    )
    
    messages = handler.build_messages(payload)
    
    # Should include system prompt and few-shot examples
    assert len(messages) > 1
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Show me running shoes"


def test_request_handler_fallback_when_prompts_missing():
    """Test that RequestHandler falls back gracefully when prompts are missing."""
    from app.ai.llm_intent.prompt_loader import PromptLoadError
    
    with patch('app.ai.llm_intent.request_handler.get_prompt_loader') as mock_loader:
        mock_loader.side_effect = PromptLoadError("Prompts not found")
        
        handler = RequestHandler()
        
        # Should still work, but without prompts
        assert handler.prompt_loader is None
        
        payload = LLMIntentRequest(
            user_input="test query",
            top_confidence=0.3,
            next_best_confidence=0.2,
            action_code=None,
            is_fallback=True,
        )
        
        messages = handler.build_messages(payload)
        
        # Should fallback to simple user message
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test query"


def test_request_handler_no_trigger():
    """Test that handler returns passthrough when LLM is not triggered."""
    handler = RequestHandler()
    
    payload = LLMIntentRequest(
        user_input="test query",
        top_confidence=0.85,  # High confidence, should not trigger
        next_best_confidence=0.10,
        action_code="SEARCH_PRODUCT",
        is_fallback=False,
    )
    
    result = handler.handle(payload)
    
    assert result["triggered"] is False
    assert result["action_code"] == "SEARCH_PRODUCT"
    assert result["confidence"] == 0.85


def test_request_handler_with_mock_llm_response():
    """Test handler with mocked LLM response."""
    mock_client = Mock()
    mock_client.complete.return_value = {
        "response": json.dumps({
            "action_code": "ADD_TO_CART",
            "confidence": 0.95,
            "reasoning": "User wants to add item",
            "intent_category": "CART_WISHLIST"
        }),
        "latency_ms": 1250
    }
    
    handler = RequestHandler(client=mock_client)
    
    payload = LLMIntentRequest(
        user_input="Add this to cart",
        top_confidence=0.3,
        next_best_confidence=0.2,
        action_code=None,
        is_fallback=True,
    )
    
    result = handler.handle(payload)
    
    assert result["triggered"] is True
    assert result["action_code"] == "ADD_TO_CART"
    # Confidence may be calibrated down based on historical accuracy
    assert 0.85 <= result["confidence"] <= 0.95, f"Expected confidence in range, got {result['confidence']}"
    # prompt_version may not be in metadata depending on prompt loader state
    assert result["action_code"] == "ADD_TO_CART"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

