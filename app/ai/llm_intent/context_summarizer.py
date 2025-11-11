"""
Context Summarizer

Summarizes and truncates context to stay within token limits.
"""

import os
from typing import Dict, List, Optional, Any
from loguru import logger
import json

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using approximate token estimation")


class ContextSummarizer:
    """
    Summarizes context to stay within token limits.
    
    Strategy:
    - Keep last 2-3 messages full
    - Summarize older messages (extract key intent/product mentions)
    - Limit product lists to top N items
    - Prioritize recent browsing history
    """
    
    def __init__(self, model_name: str = "gpt-4", max_tokens: int = 2000):
        """
        Initialize context summarizer.
        
        Args:
            model_name: Model name for token counting (default: gpt-4)
            max_tokens: Maximum token budget for context (default: 2000)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.encoding = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base encoding (used by GPT-4 and GPT-3.5-turbo)
                self.encoding = tiktoken.get_encoding("cl100k_base")
                logger.info(f"✅ ContextSummarizer using tiktoken for token counting")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load tiktoken encoding: {e}")
                self.encoding = None
    
    def calculate_tokens(self, text: str) -> int:
        """
        Calculate number of tokens in text.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"⚠️ Token counting error: {e}, using approximation")
        
        # Approximate: ~4 characters per token
        return len(text) // 4
    
    def summarize_conversation(
        self,
        conversation_history: List[Dict[str, Any]],
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Summarize conversation history to stay within token limits.
        
        Strategy:
        - Keep last 2-3 messages full
        - Summarize older messages (extract key info)
        
        Args:
            conversation_history: List of conversation messages
            max_tokens: Token limit (uses self.max_tokens if not provided)
        
        Returns:
            Summarized conversation history
        """
        if not conversation_history:
            return []
        
        max_tokens = max_tokens or self.max_tokens
        
        # Keep last 3 messages full, summarize older ones
        if len(conversation_history) <= 3:
            # All messages fit, return as-is
            return conversation_history
        
        # Split into recent (keep full) and older (summarize)
        recent_messages = conversation_history[-3:]
        older_messages = conversation_history[:-3]
        
        # Calculate tokens for recent messages
        recent_tokens = sum(
            self.calculate_tokens(str(msg.get("content", "")))
            for msg in recent_messages
        )
        
        # Calculate available tokens for older messages
        available_tokens = max(0, max_tokens - recent_tokens)
        
        # Summarize older messages
        summarized_older = self._summarize_messages(older_messages, available_tokens)
        
        # Combine: summarized older + recent full
        result = summarized_older + recent_messages
        
        total_tokens = sum(
            self.calculate_tokens(str(msg.get("content", "")))
            for msg in result
        )
        
        logger.debug(
            f"Summarized conversation: {len(conversation_history)} → {len(result)} messages, "
            f"{total_tokens}/{max_tokens} tokens"
        )
        
        return result
    
    def _summarize_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Summarize a list of older messages.
        
        Extract key information:
        - User queries (keep full)
        - Assistant responses (summarize to key points)
        - Product mentions
        - Intent mentions
        """
        if not messages or max_tokens <= 0:
            return []
        
        summarized = []
        current_tokens = 0
        
        # Process messages in reverse (oldest first)
        for msg in messages:
            content = str(msg.get("content", ""))
            msg_tokens = self.calculate_tokens(content)
            
            role = msg.get("role", "user")
            
            # For user messages, try to keep them (they're queries)
            # For assistant messages, summarize
            if role == "user":
                if current_tokens + msg_tokens <= max_tokens:
                    summarized.append(msg)
                    current_tokens += msg_tokens
                else:
                    # Truncate if necessary
                    truncated = self._truncate_text(content, max_tokens - current_tokens)
                    if truncated:
                        summarized.append({**msg, "content": truncated})
                    break
            else:
                # Summarize assistant responses
                summary = self._extract_key_points(content)
                summary_tokens = self.calculate_tokens(summary)
                
                if current_tokens + summary_tokens <= max_tokens:
                    summarized.append({**msg, "content": f"[Summary] {summary}"})
                    current_tokens += summary_tokens
                else:
                    # Can't fit even summary, skip
                    break
        
        return summarized
    
    def _extract_key_points(self, text: str) -> str:
        """
        Extract key points from assistant response.
        
        Looks for:
        - Product mentions
        - Intent/action codes
        - Key recommendations
        """
        # Simple extraction: look for capitalized words (likely products/actions)
        # and important keywords
        lines = text.split('\n')
        key_lines = []
        
        for line in lines:
            line_lower = line.lower()
            # Keep lines with important keywords
            if any(keyword in line_lower for keyword in [
                "product", "item", "recommend", "search", "add to cart",
                "purchase", "intent", "action", "category"
            ]):
                # Truncate long lines
                if len(line) > 100:
                    line = line[:100] + "..."
                key_lines.append(line)
        
        return " | ".join(key_lines[:5])  # Keep max 5 key points
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if not text:
            return ""
        
        # Approximate: start with character limit
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate and add ellipsis
        truncated = text[:max_chars - 3] + "..."
        
        # Refine to actual token count
        while self.calculate_tokens(truncated) > max_tokens and len(truncated) > 10:
            truncated = truncated[:-10] + "..."
        
        return truncated
    
    def truncate_context(
        self,
        context: Dict[str, Any],
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Truncate entire context dictionary to stay within token limits.
        
        Prioritizes:
        1. Recent conversation history (most recent messages)
        2. Recent browsing history
        3. Cart items
        4. Older conversation history
        5. Older browsing history
        
        Args:
            context: Context dict with conversation_history and session_context
            max_tokens: Token limit (uses self.max_tokens if not provided)
        
        Returns:
            Truncated context dict
        """
        max_tokens = max_tokens or self.max_tokens
        
        result = {
            "conversation_history": [],
            "session_context": {
                "browsing_history": [],
                "cart_items": [],
                "viewed_products": []
            }
        }
        
        # Summarize conversation history
        conv_history = context.get("conversation_history", [])
        result["conversation_history"] = self.summarize_conversation(conv_history, max_tokens // 2)
        
        # Limit session context
        session_context = context.get("session_context", {})
        
        # Cart items: keep all (usually small)
        cart_items = session_context.get("cart_items", [])
        result["session_context"]["cart_items"] = cart_items[:10]  # Max 10 items
        
        # Browsing history: keep recent
        browsing_history = session_context.get("browsing_history", [])
        result["session_context"]["browsing_history"] = browsing_history[-5:]  # Last 5 items
        
        # Viewed products: keep recent
        viewed_products = session_context.get("viewed_products", [])
        result["session_context"]["viewed_products"] = viewed_products[-5:]  # Last 5 items
        
        # Calculate total tokens
        total_tokens = self._calculate_context_tokens(result)
        
        # If still over limit, further truncate
        if total_tokens > max_tokens:
            # Further reduce conversation history
            remaining_tokens = max_tokens - self._calculate_session_tokens(result["session_context"])
            result["conversation_history"] = self.summarize_conversation(
                conv_history,
                remaining_tokens
            )
        
        logger.debug(f"Truncated context: {total_tokens}/{max_tokens} tokens")
        
        return result
    
    def _calculate_context_tokens(self, context: Dict[str, Any]) -> int:
        """Calculate total tokens in context dict."""
        tokens = 0
        
        # Conversation history
        for msg in context.get("conversation_history", []):
            tokens += self.calculate_tokens(str(msg.get("content", "")))
        
        # Session context
        tokens += self._calculate_session_tokens(context.get("session_context", {}))
        
        return tokens
    
    def _calculate_session_tokens(self, session_context: Dict[str, Any]) -> int:
        """Calculate tokens in session context."""
        tokens = 0
        
        # Convert to JSON string for counting
        import json
        try:
            json_str = json.dumps(session_context)
            tokens += self.calculate_tokens(json_str)
        except Exception:
            # Fallback: approximate
            tokens += len(str(session_context)) // 4
        
        return tokens


# Global instance
_context_summarizer: Optional[ContextSummarizer] = None


def get_context_summarizer(max_tokens: int = 2000) -> ContextSummarizer:
    """Get global context summarizer instance."""
    global _context_summarizer
    if _context_summarizer is None:
        _context_summarizer = ContextSummarizer(max_tokens=max_tokens)
    elif _context_summarizer.max_tokens != max_tokens:
        # Recreate if max_tokens changed
        _context_summarizer = ContextSummarizer(max_tokens=max_tokens)
    return _context_summarizer

