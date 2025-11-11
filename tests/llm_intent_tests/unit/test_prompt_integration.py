"""
Tests for prompt integration in LLM intent classification.

Tests that system prompts and few-shot examples are properly loaded
and integrated into the message building process.
"""

import pytest
from unittest.mock import Mock, patch
import json

from app.ai.llm_intent.prompt_loader import PromptLoader, PromptLoadError, get_prompt_loader
from app.ai.llm_intent.request_handler import RequestHandler
from app.ai.llm_intent.response_parser import parse_llm_response
from app.schemas.llm_intent import LLMIntentRequest


def test_prompt_loader_loads_system_prompt():
    """Test that system prompt is loaded correctly."""
    loader = PromptLoader()
    loader.load()
    
    assert loader.system_prompt is not None
    assert len(loader.system_prompt) > 0
    assert "action_code" in loader.system_prompt.lower() or "action" in loader.system_prompt.lower()


def test_prompt_loader_loads_few_shot_examples():
    """Test that few-shot examples are loaded correctly."""
    loader = PromptLoader()
    loader.load()
    
    examples = loader.few_shot_examples
    assert examples is not None
    assert len(examples) > 0
    
    # Check schema
    for example in examples:
        assert "user" in example
        assert "assistant" in example
        assert "action_code" in example["assistant"]
        assert "confidence" in example["assistant"]


def test_prompt_loader_build_messages():
    """Test that build_messages creates correct message array."""
    loader = PromptLoader()
    loader.load()
    
    user_query = "Add this to my cart"
    messages = loader.build_messages(user_query)
    
    # Should have: system + (user + assistant) pairs + final user query
    assert len(messages) > 0
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == user_query
    
    # Check that few-shot examples are included (should be user-assistant pairs)
    example_count = len(loader.few_shot_examples)
    # System (1) + few-shot pairs (example_count * 2) + user query (1)
    expected_min = 1 + (example_count * 2) + 1
    assert len(messages) >= expected_min


def test_prompt_loader_caching():
    """Test that prompt loader uses caching."""
    loader1 = get_prompt_loader()
    loader2 = get_prompt_loader()
    
    # Should be the same instance (singleton)
    assert loader1 is loader2


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

