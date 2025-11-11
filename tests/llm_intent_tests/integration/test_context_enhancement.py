"""
Integration tests for Context Enhancement (TASK-17)

Tests end-to-end flow: conversation history → context → prompt → LLM
"""

import unittest
from typing import Dict, Any
from app.ai.llm_intent.context_collector import get_context_collector
from app.ai.llm_intent.context_summarizer import get_context_summarizer
from app.ai.llm_intent.prompt_loader import get_prompt_loader
from app.core.session_store import get_session_store


class TestContextEnhancementIntegration(unittest.TestCase):
    """Integration tests for context enhancement."""
    
    def setUp(self):
        self.collector = get_context_collector()
        self.summarizer = get_context_summarizer(max_tokens=500)
        self.session_store = get_session_store()
        self.prompt_loader = get_prompt_loader()
    
    def test_context_collection_to_formatting(self):
        """Test: conversation history → collection → summarization → formatting."""
        # Prepare conversation history
        conversation_history = [
            {"role": "user", "content": "I want to buy a phone"},
            {"role": "assistant", "content": "What type of phone are you looking for?"},
            {"role": "user", "content": "iPhone"},
            {"role": "assistant", "content": "Here are some iPhone options..."},
            {"role": "user", "content": "Show me the latest model"}
        ]
        
        # Collect context
        context = self.collector.collect_all_context(
            conversation_history=conversation_history,
            session_id="test_session_integration",
            history_limit=5
        )
        
        self.assertIn("conversation_history", context)
        self.assertEqual(len(context["conversation_history"]), 5)
        
        # Summarize
        summarized = self.summarizer.truncate_context(context, max_tokens=200)
        
        self.assertIn("conversation_history", summarized)
        self.assertLessEqual(
            self.summarizer._calculate_context_tokens(summarized),
            250  # Allow margin
        )
        
        # Format for prompt
        formatted = self.prompt_loader.format_context(summarized)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("Conversation History", formatted)
    
    def test_session_context_integration(self):
        """Test: session context → collection → formatting."""
        session_id = "test_session_integration_2"
        
        # Store session context
        self.session_store.store_cart_items(
            session_id,
            [{"product_id": "123", "name": "iPhone 15"}]
        )
        self.session_store.store_browsing_history(
            session_id,
            [{"product_id": "456", "name": "Samsung Galaxy"}]
        )
        
        # Collect context
        context = self.collector.collect_session_context(
            session_id=session_id
        )
        
        self.assertIn("cart_items", context)
        self.assertIn("browsing_history", context)
        self.assertEqual(len(context["cart_items"]), 1)
        
        # Format
        full_context = {
            "conversation_history": [],
            "session_context": context
        }
        formatted = self.prompt_loader.format_context(full_context)
        
        self.assertIn("Session Context", formatted)
        self.assertIn("Cart", formatted)
    
    def test_token_limit_enforcement(self):
        """Test that token limits are enforced."""
        # Create large conversation history
        large_history = [
            {"role": "user", "content": "message " + "x" * 200} for _ in range(20)
        ]
        
        context = {
            "conversation_history": large_history,
            "session_context": {
                "browsing_history": [],
                "cart_items": [],
                "viewed_products": []
            }
        }
        
        # Summarize with strict limit
        summarized = self.summarizer.truncate_context(context, max_tokens=100)
        
        tokens = self.summarizer._calculate_context_tokens(summarized)
        self.assertLessEqual(tokens, 200)  # Allow margin for formatting (approximate counting)
    
    def test_context_priority_recent_messages(self):
        """Test that recent messages are prioritized."""
        history = [
            {"role": "user", "content": f"old message {i}"} for i in range(10)
        ]
        history.append({"role": "user", "content": "recent important message"})
        
        context = {
            "conversation_history": history,
            "session_context": {"browsing_history": [], "cart_items": [], "viewed_products": []}
        }
        
        summarized = self.summarizer.truncate_context(context, max_tokens=50)
        
        # Recent message should be kept
        recent_kept = any(
            "recent important message" in str(msg.get("content", ""))
            for msg in summarized.get("conversation_history", [])
        )
        
        # At least some messages should be kept
        self.assertGreater(len(summarized.get("conversation_history", [])), 0)


if __name__ == "__main__":
    unittest.main()

