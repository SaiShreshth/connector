"""
Intent definitions for analytics and tracking functionality.
"""

from app.ai.intent_classification.intents_modular.enums import (
    IntentCategory, ActionCode, IntentPriority, EntityType
)
from app.ai.intent_classification.intents_modular.models import IntentDefinition

analytics_tracking_intent_definitions = {
    "track_user_action": IntentDefinition(
        category=IntentCategory.ANALYTICS_TRACKING,
        action_code=ActionCode.TRACK_USER_ACTION,
        description="System tracks user actions for analytics",
        example_phrases=[
            "User clicked on product",
            "User viewed page",
            "User scrolled down",
            "User hovered over item",
            "User clicked button",
            "User navigated to section",
            "User interacted with element",
            "User performed action"
        ],
        required_entities=[EntityType.SESSION_ID],
        optional_entities=[EntityType.USER_ID, EntityType.DEVICE_TYPE],
        confidence_threshold=0.90,
        priority=IntentPriority.LOW
    ),

    "track_search_query": IntentDefinition(
        category=IntentCategory.ANALYTICS_TRACKING,
        action_code=ActionCode.TRACK_SEARCH_QUERY,
        description="System tracks search queries for analytics",
        example_phrases=[
            "User searched for products",
            "Search query executed",
            "User performed search",
            "Search term entered",
            "User searched for items",
            "Search performed",
            "Query submitted",
            "Search initiated"
        ],
        required_entities=[EntityType.SEARCH_TERM],
        optional_entities=[EntityType.SESSION_ID, EntityType.USER_ID],
        confidence_threshold=0.90,
        priority=IntentPriority.LOW
    ),

    "track_cart_abandonment": IntentDefinition(
        category=IntentCategory.ANALYTICS_TRACKING,
        action_code=ActionCode.TRACK_CART_ABANDONMENT,
        description="System tracks cart abandonment for analytics",
        example_phrases=[
            "User abandoned cart",
            "Cart left without purchase",
            "User left checkout",
            "Cart abandoned",
            "User didn't complete purchase",
            "Checkout abandoned",
            "Cart left incomplete",
            "Purchase not completed"
        ],
        required_entities=[EntityType.SESSION_ID],
        optional_entities=[EntityType.USER_ID, EntityType.TOTAL_AMOUNT],
        confidence_threshold=0.90,
        priority=IntentPriority.LOW
    ),

    "view_analytics": IntentDefinition(
        category=IntentCategory.ANALYTICS_TRACKING,
        action_code=ActionCode.VIEW_ANALYTICS,
        description="User wants to view analytics data",
        example_phrases=[
            "Show analytics",
            "View analytics dashboard",
            "Display analytics data",
            "Show me analytics",
            "View performance metrics",
            "Display statistics",
            "Show analytics report",
            "View data insights"
        ],
        required_entities=[],
        optional_entities=[EntityType.DATE_RANGE, EntityType.USER_ID],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "export_analytics": IntentDefinition(
        category=IntentCategory.ANALYTICS_TRACKING,
        action_code=ActionCode.EXPORT_ANALYTICS,
        description="User wants to export analytics data",
        example_phrases=[
            "Export analytics data",
            "Download analytics report",
            "Export data to CSV",
            "Download analytics",
            "Export report",
            "Download data",
            "Export analytics",
            "Get analytics export"
        ],
        required_entities=[],
        optional_entities=[EntityType.DATE_RANGE, EntityType.USER_ID],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "listen_behavior": IntentDefinition(
        category=IntentCategory.ANALYTICS_TRACKING,
        action_code=ActionCode.LISTEN_BEHAVIOR,
        description="System listens to user behavior for analytics",
        example_phrases=[
            "User behavior tracked",
            "Behavioral data collected",
            "User patterns analyzed",
            "Behavior monitoring",
            "User activity tracked",
            "Behavioral insights",
            "User patterns recorded",
            "Behavior analysis"
        ],
        required_entities=[EntityType.SESSION_ID],
        optional_entities=[EntityType.USER_ID, EntityType.DEVICE_TYPE],
        confidence_threshold=0.90,
        priority=IntentPriority.LOW
    ),

    "track_conversion": IntentDefinition(
        category=IntentCategory.ANALYTICS_TRACKING,
        action_code=ActionCode.TRACK_CONVERSION,
        description="System tracks conversion events for analytics",
        example_phrases=[
            "Purchase completed",
            "Conversion tracked",
            "Sale completed",
            "Purchase made",
            "Conversion event",
            "Sale recorded",
            "Purchase tracked",
            "Conversion recorded"
        ],
        required_entities=[EntityType.SESSION_ID],
        optional_entities=[EntityType.USER_ID, EntityType.TOTAL_AMOUNT],
        confidence_threshold=0.90,
        priority=IntentPriority.LOW
    ),

    "track_performance": IntentDefinition(
        category=IntentCategory.ANALYTICS_TRACKING,
        action_code=ActionCode.TRACK_PERFORMANCE,
        description="System tracks performance metrics for analytics",
        example_phrases=[
            "Performance metrics tracked",
            "Page load time recorded",
            "Performance data collected",
            "Speed metrics tracked",
            "Performance monitoring",
            "Metrics recorded",
            "Performance analysis",
            "Speed tracking"
        ],
        required_entities=[EntityType.SESSION_ID],
        optional_entities=[EntityType.DEVICE_TYPE, EntityType.BROWSER_TYPE],
        confidence_threshold=0.90,
        priority=IntentPriority.LOW
    ),
}
