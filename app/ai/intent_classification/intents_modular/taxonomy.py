"""
Intent Taxonomy

This module provides the IntentTaxonomy class that manages all intent definitions,
their metadata, and provides various methods for classification-related operations.
"""

from typing import Dict, List, Optional, Any
from loguru import logger
from .models import IntentDefinition, IntentResult, ClassificationRequest, ClassificationResponse
from .enums import IntentCategory, ActionCode, IntentPriority
from .definitions import ALL_INTENT_DEFINITIONS


class IntentTaxonomy:
    """Comprehensive intent taxonomy and manager for all intent definitions."""

    def __init__(self):
        """Initialize the taxonomy with all available intent definitions."""
        self.intents: Dict[str, IntentDefinition] = ALL_INTENT_DEFINITIONS
        self.intent_mappings: Dict[str, str] = self._create_intent_mappings()
        self._validate_intents()

    # --------------------------------------------------------------------------
    # Internal setup and validation
    # --------------------------------------------------------------------------

    def _create_intent_mappings(self) -> Dict[str, str]:
        """Map example phrases to their corresponding action codes."""
        mappings: Dict[str, str] = {}
        for intent_name, intent_def in self.intents.items():
            # Handle both string action codes and enum-like objects with .value
            action_code_str = (
                intent_def.action_code.value 
                if hasattr(intent_def.action_code, 'value') 
                else str(intent_def.action_code)
            )
            for phrase in intent_def.example_phrases:
                mappings[phrase.lower()] = action_code_str
        return mappings

    def _validate_intents(self) -> None:
        """Validate all registered intent definitions."""
        for intent_name, intent_def in self.intents.items():
            if not isinstance(intent_def, IntentDefinition):
                raise ValueError(f"Invalid intent definition for '{intent_name}'")

        # Check for duplicate action codes (warn but don't fail - multiple intents can share action codes)
        # Handle both string action codes and enum-like objects with .value
        action_codes = []
        for def_ in self.intents.values():
            action_code_str = (
                def_.action_code.value 
                if hasattr(def_.action_code, 'value') 
                else str(def_.action_code)
            )
            action_codes.append(action_code_str)
        
        if len(action_codes) != len(set(action_codes)):
            # Find duplicates for warning message
            from collections import Counter
            duplicates = [code for code, count in Counter(action_codes).items() if count > 1]
            logger.warning(f"Some action codes are used by multiple intents: {duplicates[:5]}")
            logger.debug("This is allowed - multiple intents can map to the same action code.")

    # --------------------------------------------------------------------------
    # Retrieval Methods
    # --------------------------------------------------------------------------

    def get_intent_by_action_code(self, action_code: str) -> Optional[IntentDefinition]:
        """Retrieve intent definition by its action code."""
        for intent_def in self.intents.values():
            # Handle both string action codes and enum-like objects with .value
            intent_action_code = (
                intent_def.action_code.value 
                if hasattr(intent_def.action_code, 'value') 
                else str(intent_def.action_code)
            )
            if intent_action_code == action_code:
                return intent_def
        return None

    def get_intents_by_category(self, category: IntentCategory) -> List[IntentDefinition]:
        """Return all intents belonging to a given category."""
        return [intent for intent in self.intents.values() if intent.category == category]

    def get_intents_by_priority(self, priority: IntentPriority) -> List[IntentDefinition]:
        """Return all intents with a given priority level."""
        return [intent for intent in self.intents.values() if intent.priority == priority]

    def get_all_categories(self) -> List[IntentCategory]:
        """Return all available intent categories."""
        return list(IntentCategory)

    def get_all_action_codes(self) -> List[ActionCode]:
        """Return all available action codes."""
        return list(ActionCode)

    def get_example_phrases_for_action(self, action_code: str) -> List[str]:
        """Return all example phrases for a specific action code."""
        intent_def = self.get_intent_by_action_code(action_code)
        return intent_def.example_phrases if intent_def else []

    # --------------------------------------------------------------------------
    # Validation and Lookup Utilities
    # --------------------------------------------------------------------------

    def validate_intent(self, intent_name: str) -> bool:
        """Return True if the given intent name exists."""
        return intent_name in self.intents

    def get_intent_definition(self, intent_name: str) -> Optional[IntentDefinition]:
        """Retrieve a specific intent definition by name."""
        return self.intents.get(intent_name)

    # --------------------------------------------------------------------------
    # Statistics and Insights
    # --------------------------------------------------------------------------

    def get_intent_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about the current intent taxonomy."""
        total_intents = len(self.intents)
        categories: Dict[str, int] = {}
        priorities: Dict[str, int] = {}

        for intent_def in self.intents.values():
            # Count by category
            cat = intent_def.category.value
            categories[cat] = categories.get(cat, 0) + 1

            # Count by priority
            pr = intent_def.priority.name
            priorities[pr] = priorities.get(pr, 0) + 1

        return {
            "total_intents": total_intents,
            "total_categories": len(categories),
            "total_action_codes": len(set(def_.action_code for def_ in self.intents.values())),
            "intents_by_category": categories,
            "intents_by_priority": priorities,
            "average_confidence_threshold": (
                sum(def_.confidence_threshold for def_ in self.intents.values()) / total_intents
                if total_intents > 0 else 0.0
            ),
        }

    # --------------------------------------------------------------------------
    # Search and Discovery
    # --------------------------------------------------------------------------

    def search_intents(self, query: str) -> List[IntentDefinition]:
        """Search intents by description or example phrases."""
        query_lower = query.lower()
        results: List[IntentDefinition] = []

        for intent_def in self.intents.values():
            if query_lower in intent_def.description.lower():
                results.append(intent_def)
                continue

            for phrase in intent_def.example_phrases:
                if query_lower in phrase.lower():
                    results.append(intent_def)
                    break

        return results

    def get_related_intents(self, intent_name: str) -> List[IntentDefinition]:
        """Return intents related to a given intent (by category or entities)."""
        if intent_name not in self.intents:
            return []

        target = self.intents[intent_name]
        related: List[IntentDefinition] = []

        for name, intent_def in self.intents.items():
            if name == intent_name:
                continue

            # Match same category
            if intent_def.category == target.category:
                related.append(intent_def)
                continue

            # Match by overlapping entities
            if set(intent_def.required_entities) & set(target.required_entities):
                related.append(intent_def)

        return related

    # --------------------------------------------------------------------------
    # Export Utilities
    # --------------------------------------------------------------------------

    def export_intent_definitions(self, format: str = "dict") -> Any:
        """Export all intent definitions in dict or JSON format."""
        if format == "dict":
            result = {}
            for name, def_ in self.intents.items():
                # Handle both string action codes and enum-like objects with .value
                action_code_str = (
                    def_.action_code.value 
                    if hasattr(def_.action_code, 'value') 
                    else str(def_.action_code)
                )
                result[name] = {
                    "category": def_.category.value if hasattr(def_.category, 'value') else str(def_.category),
                    "action_code": action_code_str,
                    "description": def_.description,
                    "example_phrases": def_.example_phrases,
                    "required_entities": [e.value if hasattr(e, 'value') else str(e) for e in def_.required_entities],
                    "optional_entities": [e.value if hasattr(e, 'value') else str(e) for e in def_.optional_entities],
                    "confidence_threshold": def_.confidence_threshold,
                    "priority": def_.priority.name if hasattr(def_.priority, 'name') else str(def_.priority),
                }
            return result

        elif format == "json":
            import json
            return json.dumps(self.export_intent_definitions("dict"), indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format}")


# ------------------------------------------------------------------------------
# Global Access Helpers
# ------------------------------------------------------------------------------

# Singleton instance
intent_taxonomy = IntentTaxonomy()


def get_intent_taxonomy() -> IntentTaxonomy:
    """Return the global IntentTaxonomy instance."""
    return intent_taxonomy


def get_action_code_for_phrase(phrase: str) -> Optional[str]:
    """Retrieve the action code corresponding to a given phrase."""
    return intent_taxonomy.intent_mappings.get(phrase.lower())


def get_all_intent_categories() -> List[str]:
    """Return all intent category names as strings."""
    return [category.value for category in IntentCategory]


def get_all_action_codes() -> List[str]:
    """Return all action code names as strings."""
    # ActionCode is a class with string attributes, not an Enum
    # Extract all attributes that are strings (not methods or special attributes)
    action_codes = []
    for attr_name in dir(ActionCode):
        if not attr_name.startswith('_'):
            attr_value = getattr(ActionCode, attr_name)
            if isinstance(attr_value, str):
                action_codes.append(attr_value)
    return sorted(list(set(action_codes)))  # Remove duplicates and sort


# Exported symbols
__all__ = [
    "IntentTaxonomy",
    "intent_taxonomy",
    "get_intent_taxonomy",
    "get_action_code_for_phrase",
    "get_all_intent_categories",
    "get_all_action_codes",
]
