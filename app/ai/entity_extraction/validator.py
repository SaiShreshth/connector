"""
Entity Validator for Entity Extraction

Validates and normalizes extracted entities to ensure data quality and consistency.
"""

import logging
from typing import List, Tuple
from app.ai.entity_extraction.schema import Entities, PriceRange

logger = logging.getLogger(__name__)


class EntityValidator:
    """
    Validates and normalizes extracted entities.
    
    Features:
    - Validates entity values against known lists
    - Normalizes formats (case, spelling)
    - Handles ambiguous cases with warnings
    - Validates price ranges for consistency
    """
    
    # Known brands (expandable)
    KNOWN_BRANDS = {
        "nike", "adidas", "puma", "reebok", "under armour",
        "apple", "samsung", "google", "oneplus", "xiaomi",
        "sony", "lg", "dell", "hp", "lenovo",
        "zara", "h&m", "uniqlo", "gap", "levis"
    }
    
    # Known colors
    KNOWN_COLORS = {
        "red", "blue", "black", "white", "green",
        "yellow", "orange", "purple", "pink", "brown",
        "gray", "grey", "beige", "navy", "maroon"
    }
    
    # Known categories
    KNOWN_CATEGORIES = {
        "running", "casual", "sports", "formal", "outdoor",
        "electronics", "clothing", "footwear", "accessories",
        "smartphones", "laptops", "tablets", "watches"
    }
    
    # Brand name normalization (lowercase -> proper case)
    BRAND_NORMALIZATION = {
        "nike": "Nike",
        "adidas": "Adidas",
        "puma": "Puma",
        "reebok": "Reebok",
        "under armour": "Under Armour",
        "apple": "Apple",
        "samsung": "Samsung",
        "google": "Google",
        "oneplus": "OnePlus",
        "xiaomi": "Xiaomi",
        "sony": "Sony",
        "lg": "LG",
        "dell": "Dell",
        "hp": "HP",
        "lenovo": "Lenovo",
        "zara": "Zara",
        "h&m": "H&M",
        "uniqlo": "Uniqlo",
        "gap": "Gap",
        "levis": "Levi's"
    }
    
    def __init__(self):
        """Initialize entity validator."""
        logger.info("EntityValidator initialized")
    
    def validate_and_normalize(self, entities: Entities) -> Tuple[Entities, List[str]]:
        """
        Validate and normalize entities.
        
        Args:
            entities: Entities object to validate
            
        Returns:
            Tuple of (normalized_entities, warnings)
            where warnings is a list of validation warning messages
        """
        warnings = []
        
        # Normalize brand
        if entities.brand:
            entities.brand, brand_warning = self._normalize_brand(entities.brand)
            if brand_warning:
                warnings.append(brand_warning)
        
        # Normalize color
        if entities.color:
            entities.color, color_warning = self._normalize_color(entities.color)
            if color_warning:
                warnings.append(color_warning)
        
        # Normalize category
        if entities.category:
            entities.category = entities.category.lower().strip()
            if entities.category not in self.KNOWN_CATEGORIES:
                warnings.append(f"Unknown category: '{entities.category}'")
        
        # Normalize product_type
        if entities.product_type:
            entities.product_type = entities.product_type.lower().strip()
        
        # Normalize size
        if entities.size:
            entities.size = entities.size.upper().strip()
        
        # Validate price range
        if entities.price_range:
            price_warning = self._validate_price_range(entities.price_range)
            if price_warning:
                warnings.append(price_warning)
        
        logger.debug(f"Validated entities with {len(warnings)} warnings")
        return entities, warnings
    
    def _normalize_brand(self, brand: str) -> Tuple[str, str]:
        """
        Normalize brand name.
        
        Args:
            brand: Brand name to normalize
            
        Returns:
            Tuple of (normalized_brand, warning_message)
        """
        brand_lower = brand.lower().strip()
        
        # Check if known brand
        if brand_lower in self.KNOWN_BRANDS:
            # Normalize to proper case
            normalized = self.BRAND_NORMALIZATION.get(brand_lower, brand.title())
            return normalized, ""
        else:
            # Unknown brand - keep as title case but warn
            return brand.title(), f"Unknown brand: '{brand}'"
    
    def _normalize_color(self, color: str) -> Tuple[str, str]:
        """
        Normalize color name.
        
        Args:
            color: Color name to normalize
            
        Returns:
            Tuple of (normalized_color, warning_message)
        """
        color_lower = color.lower().strip()
        
        # Normalize grey/gray
        if color_lower == "grey":
            color_lower = "gray"
        
        # Check if known color
        if color_lower in self.KNOWN_COLORS:
            return color_lower, ""
        else:
            # Unknown color - keep as lowercase but warn
            return color_lower, f"Unknown color: '{color}'"
    
    def _validate_price_range(self, price_range: PriceRange) -> str:
        """
        Validate price range for logical consistency.
        
        Args:
            price_range: PriceRange object to validate
            
        Returns:
            Warning message if invalid, empty string otherwise
        """
        # Check if min > max
        if price_range.min is not None and price_range.max is not None:
            if price_range.min > price_range.max:
                return f"Invalid price range: min ({price_range.min}) > max ({price_range.max})"
        
        # Check for negative prices
        if price_range.min is not None and price_range.min < 0:
            return f"Invalid price range: negative min price ({price_range.min})"
        
        if price_range.max is not None and price_range.max < 0:
            return f"Invalid price range: negative max price ({price_range.max})"
        
        # Check for unrealistic prices (too high)
        max_price = 1000000  # 1 million
        if price_range.max and price_range.max > max_price:
            return f"Unrealistic price range: max ({price_range.max}) exceeds {max_price}"
        
        return ""
    
    def is_valid_entity_set(self, entities: Entities) -> bool:
        """
        Check if entities contain at least one meaningful value.
        
        Args:
            entities: Entities object to check
            
        Returns:
            True if at least one entity field is populated
        """
        return any([
            entities.product_type,
            entities.category,
            entities.brand,
            entities.color,
            entities.size,
            (entities.price_range and (
                entities.price_range.min or entities.price_range.max
            ))
        ])


# Singleton instance
_validator_instance = None


def get_entity_validator() -> EntityValidator:
    """
    Get or create singleton EntityValidator instance.
    
    Returns:
        Singleton EntityValidator instance
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = EntityValidator()
    return _validator_instance


__all__ = ["EntityValidator", "get_entity_validator"]

