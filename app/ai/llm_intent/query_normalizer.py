"""
Query Normalizer for LLM Response Cache

Normalizes user queries before caching to improve cache hit rates by treating
semantically similar variations as identical queries.
"""

import re
import inspect
from typing import Optional


class QueryNormalizer:
    """
    Normalizes user queries to improve cache hit rates.
    
    Normalization includes:
    - Lowercasing
    - Whitespace normalization
    - Special character handling
    - Common phrase variations
    """
    
    # Common variations to normalize
    VARIATIONS = {
        "could you": "can you",
        "would you": "can you",
        "please": "",
        "kindly": "",
        "i want to": "i want",
        "i would like to": "i want",
        "i'd like to": "i want",
        "i need to": "i need",
        "show me": "show",
        "tell me": "",  # Remove "tell me" entirely
        "give me": "give",
    }
    
    def __init__(self):
        """Initialize the query normalizer."""
        # Pre-compile regex patterns for efficiency
        self._multi_space_pattern = re.compile(r'\s+')
        # Note: apostrophes (\') ARE included to preserve contractions like "what's", "I'm"
        self._special_char_pattern = re.compile(r'[^\w\s\-\'?.,!]')
        self._punctuation_pattern = re.compile(r'([?.!,])\1+')  # Repeated punctuation
    
    def normalize(self, query: str) -> str:
        """
        Normalize a query for caching.
        
        Args:
            query: Raw user query
            
        Returns:
            Normalized query string
            
        Example:
            >>> normalizer = QueryNormalizer()
            >>> normalizer.normalize("Show me RED shoes please!!!")
            'show red shoes'
        """
        if not query or not isinstance(query, str):
            return ""
        
        # Step 1: Convert to lowercase
        normalized = query.lower()
        
        # Step 2: Normalize whitespace first (so variations can match correctly)
        # This ensures "show   me" matches the pattern "show me"
        normalized = self._multi_space_pattern.sub(' ', normalized)
        
        # Step 3: Remove extra/repeated punctuation (keep single instances)
        normalized = self._punctuation_pattern.sub(r'\1', normalized)
        
        # Step 4: Replace common variations (now whitespace is normalized, so patterns match)
        for original, replacement in self.VARIATIONS.items():
            normalized = normalized.replace(original, replacement)
        
        # Step 5: Remove special characters (keep alphanumeric, spaces, hyphens, apostrophes, basic punctuation)
        # Note: apostrophes are preserved to keep contractions like "what's", "I'm"
        normalized = self._special_char_pattern.sub('', normalized)
        
        # Step 6: Normalize whitespace again (in case replacements created extra spaces)
        normalized = self._multi_space_pattern.sub(' ', normalized)
        
        # Step 7: Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        # Step 8: Remove trailing punctuation intelligently
        # - Single words: remove ALL trailing punctuation (?, !, ., ,)
        # - Multi-word queries: 
        #   * Simple questions (starts with question words): preserve question marks
        #   * Complex queries: remove trailing question marks for better cache matching
        if ' ' not in normalized:
            # Single word: remove all trailing punctuation
            normalized = normalized.rstrip('?.!,')
        else:
            # Multi-word: Check if it's a simple question pattern
            question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose'}
            words = normalized.split()
            first_word = words[0].lower() if words else ""
            
            # Check if first word starts with or contains a question word (handles contractions like "what's")
            is_question = any(first_word.startswith(qw) or qw in first_word for qw in question_words)
            
            # If it starts with a question word and is short (<= 5 words), preserve question mark
            # Otherwise remove trailing punctuation for better cache matching
            if is_question and len(words) <= 5:
                # Simple question: preserve question marks, remove other trailing punctuation
                normalized = normalized.rstrip('.,!')
            else:
                # Complex query: remove all trailing punctuation including question marks
                normalized = normalized.rstrip('?.!,')
        
        # Step 9: Final strip to remove any trailing whitespace left after punctuation removal
        normalized = normalized.strip()
        
        return normalized
    
    def is_cacheable(self, query: str, min_length: Optional[int] = None) -> bool:
        """
        Check if a query is worth caching.
        
        Args:
            query: User query
            min_length: Minimum query length for caching (default: 3 chars)
                       If None, uses default of 3 with 6-char threshold for single words.
                       If explicitly provided, uses that value as threshold for single words too.
            
        Returns:
            True if query should be cached, False otherwise
        """
        normalized = self.normalize(query)
        
        # Use default if not provided
        if min_length is None:
            min_length = 3
            use_default_single_word_threshold = True
        else:
            use_default_single_word_threshold = False
        
        # Don't cache empty or very short queries
        if len(normalized) < min_length:
            return False
        
        # Don't cache single-word queries (usually too ambiguous)
        # Default: require 6 chars for single words
        # When min_length is explicitly provided: use min_length as threshold
        if ' ' not in normalized:
            if use_default_single_word_threshold:
                # Using default: require 6 chars for single words
                single_word_threshold = 6
            else:
                # Custom min_length provided: use it as threshold
                single_word_threshold = min_length
            
            if len(normalized) < single_word_threshold:
                return False
        
        return True


# Singleton instance
_normalizer_instance: Optional[QueryNormalizer] = None


def get_query_normalizer() -> QueryNormalizer:
    """
    Get or create singleton QueryNormalizer instance.
    
    Returns:
        Singleton QueryNormalizer instance
    """
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = QueryNormalizer()
    return _normalizer_instance


__all__ = ["QueryNormalizer", "get_query_normalizer"]

