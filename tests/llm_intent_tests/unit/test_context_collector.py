"""
Unit tests for Context Collector (TASK-17)
"""

import unittest
from typing import Dict, Any
from app.ai.llm_intent.context_collector import ContextCollector, get_context_collector


class TestContextCollector(unittest.TestCase):
    """Test ContextCollector functionality."""
    
    def setUp(self):
        self.collector = ContextCollector()
    
    def test_collect_conversation_history(self):
        """Test conversation history collection with limit."""
        history = [
            {"role": "user", "content": "message 1"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "message 2"},
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "content": "message 3"},
            {"role": "user", "content": "message 4"},
            {"role": "user", "content": "message 5"},
            {"role": "user", "content": "message 6"},
        ]
        
        # Test with limit
        result = self.collector.collect_conversation_history(history, limit=5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[-1]["content"], "message 6")  # Most recent
        
        # Test without limit (all messages)
        result_all = self.collector.collect_conversation_history(history, limit=100)
        self.assertEqual(len(result_all), len(history))
    
    def test_collect_conversation_history_empty(self):
        """Test with empty conversation history."""
        result = self.collector.collect_conversation_history([])
        self.assertEqual(result, [])
    
    def test_collect_session_context(self):
        """Test session context collection."""
        # This will return empty if session doesn't exist (expected)
        context = self.collector.collect_session_context(
            user_id="test_user",
            session_id="test_session"
        )
        
        self.assertIn("browsing_history", context)
        self.assertIn("cart_items", context)
        self.assertIn("viewed_products", context)
        self.assertIsInstance(context["browsing_history"], list)
        self.assertIsInstance(context["cart_items"], list)
        self.assertIsInstance(context["viewed_products"], list)
    
    def test_collect_all_context(self):
        """Test collecting all context."""
        conversation_history = [
            {"role": "user", "content": "test query"}
        ]
        
        result = self.collector.collect_all_context(
            conversation_history=conversation_history,
            user_id="test_user",
            session_id="test_session",
            history_limit=5
        )
        
        self.assertIn("conversation_history", result)
        self.assertIn("session_context", result)
        self.assertEqual(len(result["conversation_history"]), 1)
        self.assertIn("browsing_history", result["session_context"])


if __name__ == "__main__":
    unittest.main()

