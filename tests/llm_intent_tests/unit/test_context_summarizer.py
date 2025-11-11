"""
Unit tests for Context Summarizer (TASK-17)
"""

import unittest
from typing import Dict, Any
from app.ai.llm_intent.context_summarizer import ContextSummarizer, get_context_summarizer


class TestContextSummarizer(unittest.TestCase):
    """Test ContextSummarizer functionality."""
    
    def setUp(self):
        self.summarizer = ContextSummarizer(max_tokens=100)
    
    def test_calculate_tokens(self):
        """Test token calculation."""
        text = "This is a test message"
        tokens = self.summarizer.calculate_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)
    
    def test_summarize_conversation_small(self):
        """Test summarizing small conversation (fits within limit)."""
        history = [
            {"role": "user", "content": "message 1"},
            {"role": "assistant", "content": "response 1"},
        ]
        
        result = self.summarizer.summarize_conversation(history, max_tokens=1000)
        self.assertEqual(len(result), 2)  # Should keep all
    
    def test_summarize_conversation_large(self):
        """Test summarizing large conversation (exceeds limit)."""
        history = [
            {"role": "user", "content": "message " + "x" * 100} for _ in range(10)
        ]
        
        result = self.summarizer.summarize_conversation(history, max_tokens=100)
        # Should truncate/summarize
        self.assertLessEqual(len(result), len(history))
        
        # Check token count
        total_tokens = sum(
            self.summarizer.calculate_tokens(str(msg.get("content", "")))
            for msg in result
        )
        self.assertLessEqual(total_tokens, 150)  # Allow some margin
    
    def test_truncate_context(self):
        """Test context truncation."""
        context = {
            "conversation_history": [
                {"role": "user", "content": "message " + "x" * 50} for _ in range(5)
            ],
            "session_context": {
                "browsing_history": [{"product": f"item_{i}"} for i in range(10)],
                "cart_items": [{"item": "cart_item"}],
                "viewed_products": []
            }
        }
        
        result = self.summarizer.truncate_context(context, max_tokens=200)
        
        self.assertIn("conversation_history", result)
        self.assertIn("session_context", result)
        
        # Session context should be limited
        self.assertLessEqual(len(result["session_context"]["browsing_history"]), 5)
        self.assertLessEqual(len(result["session_context"]["cart_items"]), 10)


if __name__ == "__main__":
    unittest.main()

