"""
Defines all supported user intents and example utterances
for both keyword-based and embedding-based classifiers.
"""

INTENT_EXAMPLES = {
    "ADD_TO_CART": [
        "add this to my cart",
        "put in my shopping cart",
        "buy this item",
        "I want to purchase this",
    ],
    "REMOVE_FROM_CART": [
        "remove from cart",
        "delete this item from cart",
        "cancel this product",
        "take this out of my cart",
    ],
    "CHECK_ORDER_STATUS": [
        "track my order",
        "check my delivery status",
        "where is my order",
        "order status please",
    ],
    "CREATE_ACCOUNT": [
        "sign up",
        "register new account",
        "create a profile",
        "make a new account",
    ],
    "REORDER": [
        "buy again",
        "reorder my last purchase",
        "order this item again",
    ],
    "SEARCH_PRODUCT": [
        "show me red shoes",
        "find blue t-shirts under 500",
        "search for iphone",
        "look for samsung phones",
        "show me blue jeans under 2000",
    ],
    "ORDER_FOOD": [
        "order pizza",
        "I want to order food",
        "buy a burger",
        "get me some fries",
        "order something to eat",
        "I want to eat something",
    ],
}
>>>>>>> 6bdce236d75757b4a514e17eac9ad00d1d7ea989
intents.py
Central intent management module.

This module provides a clean, simple interface for intent operations.
It wraps the intents_modular system and provides easy-to-use helper functions.

All the core logic is in intents_modular/, but this module provides:
- Simple helper functions for common operations
- Easy access to intents and action codes
- Convenient methods for intent lookup and description
"""

from typing import Optional, Dict, List, Any
from .intents_modular.taxonomy import (
    get_intent_taxonomy,
    get_action_code_for_phrase,
    get_all_intent_categories,
    get_all_action_codes,
)
from .intents_modular.enums import ActionCode, IntentCategory
from .intents_modular.models import IntentDefinition


class IntentManager:
    """
    Manages all available intents and their action codes.
    
    This is a convenience wrapper around IntentTaxonomy that provides
    a simpler interface for common operations.
    """

    def __init__(self):
        """Initialize the IntentManager with the global taxonomy."""
        self.taxonomy = get_intent_taxonomy()
        self.action_codes = get_all_action_codes()
        self.categories = get_all_intent_categories()

    def get_all_action_codes(self) -> List[str]:
        """Return all available action codes as strings."""
        return self.action_codes

    def get_all_categories(self) -> List[str]:
        """Return all available intent categories as strings."""
        return self.categories

    def get_action_code(self, action_code: str) -> Optional[str]:
        """
        Validate and return the action code if it exists.
        
        Args:
            action_code: The action code to validate
            
        Returns:
            The action code if valid, None otherwise
        """
        if action_code in self.action_codes:
            return action_code
        return None

    def get_intent_by_action_code(self, action_code: str) -> Optional[IntentDefinition]:
        """
        Get the intent definition for a given action code.
        
        Args:
            action_code: The action code to look up
            
        Returns:
            IntentDefinition if found, None otherwise
        """
        return self.taxonomy.get_intent_by_action_code(action_code)

    def get_examples(self, action_code: str) -> List[str]:
        """
        Get example phrases for a given action code.
        
        Args:
            action_code: The action code to get examples for
            
        Returns:
            List of example phrases
        """
        return self.taxonomy.get_example_phrases_for_action(action_code)

    def find_intent_by_phrase(self, phrase: str) -> Optional[Dict[str, str]]:
        """
        Find the action code corresponding to an example phrase.
        This is a simple lookup (exact match only, not ML-based).
        
        Args:
            phrase: The phrase to search for
            
        Returns:
            Dict with 'action_code' and 'phrase' if found, None otherwise
        """
        action_code = get_action_code_for_phrase(phrase)
        if action_code:
            return {
                "action_code": action_code,
                "phrase": phrase,
                "source": "exact_match"
            }
        return None

    def get_intents_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all intents in a specific category.
        
        Args:
            category: The category name (e.g., "SEARCH_DISCOVERY")
            
        Returns:
            List of intent definitions as dictionaries
        """
        try:
            category_enum = IntentCategory(category)
            intent_defs = self.taxonomy.get_intents_by_category(category_enum)
            return [
                {
                    "action_code": intent.action_code.value,
                    "category": intent.category.value,
                    "description": intent.description,
                    "example_phrases": intent.example_phrases,
                }
                for intent in intent_defs
            ]
        except ValueError:
            return []

    def describe(self) -> List[Dict[str, Any]]:
        """
        Return a list of all intents with their action codes, categories, and examples.
        Useful for debugging or visualization.
        
        Returns:
            List of dictionaries containing intent information
        """
        results = []
        for action_code in self.action_codes:
            intent_def = self.get_intent_by_action_code(action_code)
            if intent_def:
                results.append({
                    "action_code": action_code,
                    "category": intent_def.category.value,
                    "description": intent_def.description,
                    "example_phrases": intent_def.example_phrases[:3],  # First 3 examples
                    "confidence_threshold": intent_def.confidence_threshold,
                    "priority": intent_def.priority.name,
                })
        return results

    def search_intents(self, query: str) -> List[Dict[str, Any]]:
        """
        Search intents by description or example phrases.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching intent definitions as dictionaries
        """
        intent_defs = self.taxonomy.search_intents(query)
        return [
            {
                "action_code": intent.action_code.value,
                "category": intent.category.value,
                "description": intent.description,
                "example_phrases": intent.example_phrases,
            }
            for intent in intent_defs
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the intent taxonomy.
        
        Returns:
            Dictionary with statistics about intents, categories, etc.
        """
        return self.taxonomy.get_intent_statistics()


# Create singleton instance for easy access
intent_manager = IntentManager()


# Convenience functions that can be imported directly
def get_all_action_codes_list() -> List[str]:
    """Get all available action codes."""
    return intent_manager.get_all_action_codes()


def get_all_categories_list() -> List[str]:
    """Get all available intent categories."""
    return intent_manager.get_all_categories()


def get_intent_definition(action_code: str) -> Optional[IntentDefinition]:
    """Get intent definition for an action code."""
    return intent_manager.get_intent_by_action_code(action_code)


def get_example_phrases(action_code: str) -> List[str]:
    """Get example phrases for an action code."""
    return intent_manager.get_examples(action_code)


def find_action_code_by_phrase(phrase: str) -> Optional[str]:
    """Find action code for a given phrase (exact match)."""
    result = intent_manager.find_intent_by_phrase(phrase)
    return result["action_code"] if result else None


if __name__ == "__main__":
    # Simple test to verify everything works
    print("✅ Intent Manager initialized")
    print(f"📊 Total action codes: {len(intent_manager.get_all_action_codes())}")
    print(f"📁 Total categories: {len(intent_manager.get_all_categories())}")
    
    # Test lookup
    test_phrase = "add to cart"
    result = intent_manager.find_intent_by_phrase(test_phrase)
    if result:
        print(f"🔍 Found action code for '{test_phrase}': {result['action_code']}")
    else:
        print(f"❌ No exact match found for '{test_phrase}'")
    
    # Test statistics
    stats = intent_manager.get_statistics()
    print(f"📈 Intent Statistics: {stats}")
    
    # Sample description
    sample = intent_manager.describe()[:2]
    print(f"📘 Sample intent descriptions: {len(sample)} shown")
