"""
Intent definitions for personalization and recommendations functionality.
"""

from app.ai.intent_classification.intents_modular.enums import (
    IntentCategory, ActionCode, IntentPriority, EntityType
)
from app.ai.intent_classification.intents_modular.models import IntentDefinition

personalization_intent_definitions = {
    "personalized_search": IntentDefinition(
        category=IntentCategory.PERSONALIZATION,
        action_code=ActionCode.PERSONALIZED_SEARCH,
        description="User wants personalized search results based on their preferences",
        example_phrases=[
            "Show me personalized results",
            "Search based on my preferences",
            "Personalized search",
            "Search for me",
            "Find what I like",
            "Personalized recommendations",
            "Search my style",
            "Find my preferences",
            "Results tailored to me",
            "Search using my taste"
        ],
        required_entities=[EntityType.SEARCH_TERM],
        optional_entities=[EntityType.USER_ID, EntityType.PREFERENCES],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "personalized_offers": IntentDefinition(
        category=IntentCategory.PERSONALIZATION,
        action_code=ActionCode.PERSONALIZED_OFFERS,
        description="User wants to see personalized offers and deals",
        example_phrases=[
            "Show me personalized offers",
            "What deals do you have for me?",
            "Personalized discounts",
            "Offers for me",
            "My personalized deals",
            "Show me my offers",
            "Personalized promotions",
            "Deals for me",
            "Offers tailored to me",
            "Any special deals for my account?"
        ],
        required_entities=[],
        optional_entities=[EntityType.USER_ID, EntityType.PREFERENCES],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "save_preferences": IntentDefinition(
        category=IntentCategory.PERSONALIZATION,
        action_code=ActionCode.SAVE_PREFERENCES,
        description="User wants to save their preferences",
        example_phrases=[
            "Save my preferences",
            "Remember my choices",
            "Save my settings",
            "Store my preferences",
            "Keep my preferences",
            "Save my options",
            "Remember my preferences",
            "Store my choices",
            "Save these selections",
            "Persist my preferences"
        ],
        required_entities=[EntityType.PREFERENCES],
        optional_entities=[EntityType.USER_ID],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "update_preferences": IntentDefinition(
        category=IntentCategory.PERSONALIZATION,
        action_code=ActionCode.UPDATE_PREFERENCES,
        description="User wants to update their preferences",
        example_phrases=[
            "Update my preferences",
            "Change my preferences",
            "Modify my settings",
            "Update my choices",
            "Change my options",
            "Modify my preferences",
            "Update my settings",
            "Change my choices",
            "Revise my preferences",
            "Adjust my settings"
        ],
        required_entities=[EntityType.PREFERENCES],
        optional_entities=[EntityType.USER_ID],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "view_preferences": IntentDefinition(
        category=IntentCategory.PERSONALIZATION,
        action_code=ActionCode.VIEW_PREFERENCES,
        description="User wants to view their saved preferences",
        example_phrases=[
            "Show my preferences",
            "View my preferences",
            "Display my settings",
            "Show my choices",
            "View my options",
            "Display my preferences",
            "Show my settings",
            "View my choices",
            "What preferences have I saved?",
            "List my saved settings"
        ],
        required_entities=[],
        optional_entities=[EntityType.USER_ID],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "clear_preferences": IntentDefinition(
        category=IntentCategory.PERSONALIZATION,
        action_code=ActionCode.CLEAR_PREFERENCES,
        description="User wants to clear their saved preferences",
        example_phrases=[
            "Clear my preferences",
            "Reset my preferences",
            "Delete my preferences",
            "Clear my settings",
            "Reset my choices",
            "Delete my options",
            "Clear my data",
            "Reset my settings",
            "Remove all saved preferences",
            "Wipe my personalization"
        ],
        required_entities=[],
        optional_entities=[EntityType.USER_ID],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),
}
