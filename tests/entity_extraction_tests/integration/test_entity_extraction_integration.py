"""
Integration tests for entity extraction in classification pipeline
"""

import pytest
from app.ai.entity_extraction.extractor import EntityExtractor
from app.ai.entity_extraction.validator import EntityValidator
from app.ai.entity_extraction.schema import Entities


class TestEntityExtractionIntegration:
    """Integration tests for entity extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EntityExtractor()
        self.validator = EntityValidator()
    
    def test_extract_and_validate_complete_query(self):
        """Test extraction and validation of query with multiple entities."""
        query = "Show me red Nike running shoes under $100"
        
        # Extract entities
        entities_dict = self.extractor.extract_entities(query)
        
        # Verify extraction
        assert entities_dict is not None
        assert entities_dict.get("color") == "red"
        assert entities_dict.get("brand") == "nike"
        assert entities_dict.get("product_name") == "shoes"
        
        # Verify price extraction
        price_range = entities_dict.get("price_range", {})
        assert price_range.get("max") == 100
        assert price_range.get("currency") in ["USD", "$"]
    
    def test_extract_price_range_variants(self):
        """Test price range extraction with different formats."""
        test_cases = [
            ("shoes under $100", {"max": 100}),
            ("between 50 and 100 dollars", {"min": 50, "max": 100}),
            ("from 1000 to 2000", {"min": 1000, "max": 2000}),
        ]
        
        for query, expected_range in test_cases:
            entities = self.extractor.extract_entities(query)
            price_range = entities.get("price_range", {})
            
            if "min" in expected_range:
                assert price_range.get("min") == expected_range["min"], f"Failed for: {query}"
            if "max" in expected_range:
                assert price_range.get("max") == expected_range["max"], f"Failed for: {query}"
    
    def test_extract_brand_variations(self):
        """Test brand extraction with different cases."""
        test_cases = [
            ("Nike shoes", "nike"),
            ("ADIDAS running gear", "adidas"),
            ("Samsung phone", "samsung"),
        ]
        
        for query, expected_brand in test_cases:
            entities = self.extractor.extract_entities(query)
            assert entities.get("brand") == expected_brand, f"Failed for: {query}"
    
    def test_extract_color_variations(self):
        """Test color extraction."""
        test_cases = [
            ("red shoes", "red"),
            ("BLACK shirt", "black"),
            ("blue phone case", "blue"),
        ]
        
        for query, expected_color in test_cases:
            entities = self.extractor.extract_entities(query)
            assert entities.get("color") == expected_color, f"Failed for: {query}"
    
    def test_extract_size(self):
        """Test size extraction."""
        query = "size 10 running shoes"
        entities = self.extractor.extract_entities(query)
        assert entities.get("size") == "10"
    
    def test_no_entities_in_query(self):
        """Test extraction when no entities are present."""
        query = "help me with my order"
        entities = self.extractor.extract_entities(query)
        
        # Should return dict with None values
        assert entities is not None
        assert entities.get("product_name") is None
        assert entities.get("brand") is None
        assert entities.get("color") is None


class TestEntityExtractionEndToEnd:
    """End-to-end tests for entity extraction in classification."""
    
    def test_rule_based_extraction_flow(self):
        """Test complete flow: extract -> validate -> use in classification."""
        query = "Show me black Samsung phone under 30000"
        
        # Step 1: Extract
        extractor = EntityExtractor()
        entities_dict = extractor.extract_entities(query)
        
        # Step 2: Convert to schema (simulating what LLM parser does)
        from app.ai.entity_extraction.schema import PriceRange
        
        price_range = entities_dict.get("price_range", {})
        entities = Entities(
            product_type=entities_dict.get("product_name"),
            category=entities_dict.get("category"),
            brand=entities_dict.get("brand"),
            color=entities_dict.get("color"),
            size=entities_dict.get("size"),
            price_range=PriceRange(**price_range) if price_range.get("min") or price_range.get("max") else None
        )
        
        # Step 3: Validate
        validator = EntityValidator()
        normalized, warnings = validator.validate_and_normalize(entities)
        
        # Verify results
        assert normalized.brand == "Samsung"  # Normalized
        assert normalized.color == "black"
        assert validator.is_valid_entity_set(normalized)
        
        # Check that entities are ready for downstream use
        entities_dict = normalized.dict()
        assert entities_dict["brand"] == "Samsung"
        assert entities_dict["color"] == "black"
        assert entities_dict["price_range"]["max"] == 30000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

