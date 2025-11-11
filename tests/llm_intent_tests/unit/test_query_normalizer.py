"""
Unit tests for QueryNormalizer
"""

import pytest
from app.ai.llm_intent.query_normalizer import QueryNormalizer, get_query_normalizer


class TestQueryNormalizer:
    """Test suite for QueryNormalizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = QueryNormalizer()
    
    def test_basic_normalization(self):
        """Test basic query normalization."""
        query = "Show me RED shoes please!!!"
        expected = "show red shoes"
        assert self.normalizer.normalize(query) == expected
    
    def test_lowercase_conversion(self):
        """Test lowercase conversion."""
        query = "FIND PRODUCTS"
        expected = "find products"
        assert self.normalizer.normalize(query) == expected
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        query = "show   me    products"
        expected = "show products"
        assert self.normalizer.normalize(query) == expected
    
    def test_variation_normalization(self):
        """Test common phrase variation normalization."""
        test_cases = [
            ("could you show me products", "can you show products"),
            ("would you help me", "can you help me"),
            ("I would like to buy", "i want buy"),
            ("I'd like to see", "i want see"),
        ]
        
        for query, expected in test_cases:
            assert self.normalizer.normalize(query) == expected
    
    def test_punctuation_removal(self):
        """Test punctuation handling."""
        test_cases = [
            ("Hello!!!", "hello"),
            ("What???", "what"),
            ("Great...", "great"),
            ("Yes, please!!!", "yes,"),  # Comma preserved
        ]
        
        for query, expected in test_cases:
            assert self.normalizer.normalize(query) == expected
    
    def test_special_character_removal(self):
        """Test special character removal."""
        query = "show me @products #sale $50"
        expected = "show products sale 50"
        assert self.normalizer.normalize(query) == expected
    
    def test_empty_query(self):
        """Test empty query handling."""
        assert self.normalizer.normalize("") == ""
        assert self.normalizer.normalize("   ") == ""
    
    def test_none_query(self):
        """Test None query handling."""
        assert self.normalizer.normalize(None) == ""
    
    def test_preserve_hyphens(self):
        """Test that hyphens are preserved."""
        query = "all-in-one solution"
        expected = "all-in-one solution"
        assert self.normalizer.normalize(query) == expected
    
    def test_preserve_apostrophes(self):
        """Test that apostrophes are preserved in contractions."""
        query = "I'm looking for shoes"
        expected = "i'm looking for shoes"  # Apostrophes preserved for better semantic matching
        assert self.normalizer.normalize(query) == expected
    
    def test_is_cacheable_valid(self):
        """Test is_cacheable for valid queries."""
        assert self.normalizer.is_cacheable("show me products")
        assert self.normalizer.is_cacheable("find red shoes")
        assert self.normalizer.is_cacheable("help me")
    
    def test_is_cacheable_too_short(self):
        """Test is_cacheable for too short queries."""
        assert not self.normalizer.is_cacheable("hi")
        assert not self.normalizer.is_cacheable("ok")
        assert not self.normalizer.is_cacheable("")
    
    def test_is_cacheable_single_word(self):
        """Test is_cacheable for single-word queries."""
        assert not self.normalizer.is_cacheable("hello")
        assert not self.normalizer.is_cacheable("help")
        # But longer single words should be cacheable
        assert self.normalizer.is_cacheable("products")
    
    def test_is_cacheable_custom_min_length(self):
        """Test is_cacheable with custom min_length."""
        assert not self.normalizer.is_cacheable("hi", min_length=3)
        assert self.normalizer.is_cacheable("hello", min_length=3)
    
    def test_get_singleton(self):
        """Test singleton pattern."""
        normalizer1 = get_query_normalizer()
        normalizer2 = get_query_normalizer()
        assert normalizer1 is normalizer2
    
    def test_complex_query(self):
        """Test normalization of complex real-world query."""
        query = "Could you PLEASE show me some red Nike shoes under $100???"
        expected = "can you show some red nike shoes under 100"
        assert self.normalizer.normalize(query) == expected
    
    def test_multiple_variations(self):
        """Test multiple variations in same query."""
        query = "Would you please kindly tell me about products"
        expected = "can you about products"
        assert self.normalizer.normalize(query) == expected
    
    def test_repeated_punctuation(self):
        """Test repeated punctuation normalization."""
        query = "What's this?????"
        expected = "what's this?"
        assert self.normalizer.normalize(query) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

