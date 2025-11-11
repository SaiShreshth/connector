"""
Unit tests for EntityValidator
"""

import pytest
from app.ai.entity_extraction.validator import EntityValidator, get_entity_validator
from app.ai.entity_extraction.schema import Entities, PriceRange


class TestEntityValidator:
    """Test suite for EntityValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = EntityValidator()
    
    def test_normalize_brand_known(self):
        """Test brand normalization for known brands."""
        brand, warning = self.validator._normalize_brand("nike")
        assert brand == "Nike"
        assert warning == ""
        
        brand, warning = self.validator._normalize_brand("APPLE")
        assert brand == "Apple"
        assert warning == ""
    
    def test_normalize_brand_unknown(self):
        """Test brand normalization for unknown brands."""
        brand, warning = self.validator._normalize_brand("unknownbrand")
        assert brand == "Unknownbrand"
        assert "Unknown brand" in warning
    
    def test_normalize_color_known(self):
        """Test color normalization for known colors."""
        color, warning = self.validator._normalize_color("RED")
        assert color == "red"
        assert warning == ""
        
        color, warning = self.validator._normalize_color("grey")
        assert color == "gray"  # Normalized to gray
        assert warning == ""
    
    def test_normalize_color_unknown(self):
        """Test color normalization for unknown colors."""
        color, warning = self.validator._normalize_color("unknowncolor")
        assert color == "unknowncolor"
        assert "Unknown color" in warning
    
    def test_validate_price_range_valid(self):
        """Test validation of valid price range."""
        price_range = PriceRange(min=50, max=100, currency="USD")
        warning = self.validator._validate_price_range(price_range)
        assert warning == ""
    
    def test_validate_price_range_invalid_min_greater_than_max(self):
        """Test validation of invalid price range (min > max)."""
        price_range = PriceRange(min=150, max=100, currency="USD")
        warning = self.validator._validate_price_range(price_range)
        assert "min" in warning.lower() and "max" in warning.lower()
    
    def test_validate_price_range_negative(self):
        """Test validation of negative prices."""
        price_range = PriceRange(min=-50, max=100, currency="USD")
        warning = self.validator._validate_price_range(price_range)
        assert "negative" in warning.lower()
    
    def test_validate_price_range_unrealistic(self):
        """Test validation of unrealistic prices."""
        price_range = PriceRange(min=None, max=2000000, currency="USD")
        warning = self.validator._validate_price_range(price_range)
        assert "unrealistic" in warning.lower() or "exceeds" in warning.lower()
    
    def test_validate_and_normalize_complete_entities(self):
        """Test validation and normalization of complete entities."""
        entities = Entities(
            product_type="SHOES",
            category="RUNNING",
            brand="nike",
            color="RED",
            size="10",
            price_range=PriceRange(min=50, max=100, currency="USD")
        )
        
        normalized, warnings = self.validator.validate_and_normalize(entities)
        
        assert normalized.product_type == "shoes"  # Normalized to lowercase
        assert normalized.category == "running"  # Normalized to lowercase
        assert normalized.brand == "Nike"  # Normalized to title case
        assert normalized.color == "red"  # Normalized to lowercase
        assert normalized.size == "10"  # Normalized to uppercase
        assert len(warnings) == 0  # No warnings for valid entities
    
    def test_validate_and_normalize_with_warnings(self):
        """Test validation with warnings for unknown values."""
        entities = Entities(
            product_type="gadget",
            category="unknowncat",
            brand="unknownbrand",
            color="unknowncolor",
            size="M",
            price_range=PriceRange(min=200, max=100, currency="USD")
        )
        
        normalized, warnings = self.validator.validate_and_normalize(entities)
        
        # Should have warnings for unknown brand, color, category, and invalid price range
        assert len(warnings) >= 3
        assert any("brand" in w.lower() for w in warnings)
        assert any("color" in w.lower() for w in warnings)
        assert any("category" in w.lower() for w in warnings)
        assert any("price" in w.lower() for w in warnings)
    
    def test_is_valid_entity_set_with_entities(self):
        """Test is_valid_entity_set with populated entities."""
        entities = Entities(
            product_type="shoes",
            category=None,
            brand=None,
            color=None,
            size=None,
            price_range=None
        )
        
        assert self.validator.is_valid_entity_set(entities) is True
    
    def test_is_valid_entity_set_empty(self):
        """Test is_valid_entity_set with empty entities."""
        entities = Entities(
            product_type=None,
            category=None,
            brand=None,
            color=None,
            size=None,
            price_range=None
        )
        
        assert self.validator.is_valid_entity_set(entities) is False
    
    def test_is_valid_entity_set_with_price_range_only(self):
        """Test is_valid_entity_set with only price range."""
        entities = Entities(
            product_type=None,
            category=None,
            brand=None,
            color=None,
            size=None,
            price_range=PriceRange(min=None, max=100, currency="USD")
        )
        
        assert self.validator.is_valid_entity_set(entities) is True
    
    def test_get_singleton(self):
        """Test singleton pattern."""
        validator1 = get_entity_validator()
        validator2 = get_entity_validator()
        assert validator1 is validator2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

