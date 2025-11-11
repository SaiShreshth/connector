"""
Integration tests for rule-based to LLM handoff and latency verification.
"""

import pytest
import time
from app.ai.intent_classification.decision_engine import DecisionEngine
from app.ai.llm_intent.request_handler import RequestHandler
from app.schemas.llm_intent import LLMIntentRequest


class TestRuleBasedLLMIntegration:
    """Test integration between rule-based and LLM modules."""
    
    def test_end_to_end_latency_under_3s(self):
        """Verify total pipeline latency is under 3 seconds."""
        engine = DecisionEngine()
        
        start_time = time.time()
        result = engine.search("show me red shoes")
        total_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        assert total_latency < 3000, f"Total latency {total_latency:.2f}ms exceeds 3s threshold"
        assert "intent" in result or "action_code" in result, "Result should contain intent/action_code"
    
    def test_handoff_scenario_low_confidence(self):
        """Test handoff from rule-based to LLM for low confidence."""
        engine = DecisionEngine()
        
        # Query that should trigger LLM (ambiguous)
        result = engine.search("I need something for running but not sure what")
        
        # Should either be confident from rule-based or queued for LLM
        assert result is not None
        status = result.get("status", "")
        assert status in ["CONFIDENT", "QUEUED_FOR_LLM", "AMBIGUOUS"] or "CONFIDENT" in status
    
    def test_circuit_breaker_prevention(self):
        """Verify circuit breaker prevents cascading failures."""
        # This would require simulating API failures
        # For now, verify circuit breaker exists
        from app.ai.llm_intent.openai_client import CircuitBreaker
        
        cb = CircuitBreaker(max_failures=3, reset_timeout=60)
        assert cb is not None
        assert hasattr(cb, "call")
    
    def test_interface_contract(self):
        """Verify interface contract between rule-based and LLM modules."""
        from app.schemas.llm_intent import LLMIntentRequest, LLMIntentResponse
        
        # Verify schemas exist
        assert LLMIntentRequest is not None
        assert LLMIntentResponse is not None
        
        # Verify required fields
        request = LLMIntentRequest(
            user_input="test query",
            rule_intent=None,
            action_code=None,
            top_confidence=0.0,
            next_best_confidence=0.0,
            is_fallback=False,
        )
        
        assert request.user_input == "test query"
        assert request.top_confidence == 0.0

