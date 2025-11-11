"""
Intent definitions for notifications and subscriptions functionality.
"""

from app.ai.intent_classification.intents_modular.enums import (
    IntentCategory, ActionCode, IntentPriority, EntityType
)
from app.ai.intent_classification.intents_modular.models import IntentDefinition

notifications_subscriptions_intent_definitions = {
    "subscribe_notifications": IntentDefinition(
        category=IntentCategory.NOTIFICATIONS_SUBSCRIPTIONS,
        action_code=ActionCode.SUBSCRIBE_NOTIFICATIONS,
        description="User wants to subscribe to notifications",
        example_phrases=[
            "Subscribe to notifications",
            "Turn on notifications",
            "Enable notifications",
            "I want to receive notifications",
            "Subscribe me to updates",
            "Notify me about new products",
            "Send me notifications",
            "Enable alerts"
        ],
        required_entities=[],
        optional_entities=[EntityType.NOTIFICATION_PREFERENCES],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "unsubscribe_notifications": IntentDefinition(
        category=IntentCategory.NOTIFICATIONS_SUBSCRIPTIONS,
        action_code=ActionCode.UNSUBSCRIBE_NOTIFICATIONS,
        description="User wants to unsubscribe from notifications",
        example_phrases=[
            "Unsubscribe from notifications",
            "Turn off notifications",
            "Disable notifications",
            "Stop sending notifications",
            "Unsubscribe me",
            "No more notifications",
            "Disable alerts",
            "Stop notifications"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "notification_preferences": IntentDefinition(
        category=IntentCategory.NOTIFICATIONS_SUBSCRIPTIONS,
        action_code=ActionCode.NOTIFICATION_PREFERENCES,
        description="User wants to manage notification preferences",
        example_phrases=[
            "Notification settings",
            "Manage notifications",
            "Notification preferences",
            "Update notification settings",
            "Change notification options",
            "Notification configuration",
            "Set notification preferences",
            "Customize notifications"
        ],
        required_entities=[],
        optional_entities=[EntityType.NOTIFICATION_PREFERENCES],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "view_notifications": IntentDefinition(
        category=IntentCategory.NOTIFICATIONS_SUBSCRIPTIONS,
        action_code=ActionCode.VIEW_NOTIFICATIONS,
        description="User wants to view their notifications",
        example_phrases=[
            "Show my notifications",
            "View notifications",
            "Check notifications",
            "See my alerts",
            "Display notifications",
            "Show alerts",
            "View my messages",
            "Check my updates"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "mark_notification_read": IntentDefinition(
        category=IntentCategory.NOTIFICATIONS_SUBSCRIPTIONS,
        action_code=ActionCode.MARK_NOTIFICATION_READ,
        description="User wants to mark a notification as read",
        example_phrases=[
            "Mark as read",
            "Mark notification as read",
            "Read notification",
            "Mark as viewed",
            "Dismiss notification",
            "Mark read",
            "Clear notification",
            "Mark as seen"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "delete_notification": IntentDefinition(
        category=IntentCategory.NOTIFICATIONS_SUBSCRIPTIONS,
        action_code=ActionCode.DELETE_NOTIFICATION,
        description="User wants to delete a notification",
        example_phrases=[
            "Delete notification",
            "Remove notification",
            "Delete this alert",
            "Remove this message",
            "Delete notification",
            "Clear this notification",
            "Remove alert",
            "Delete message"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "subscribe_product": IntentDefinition(
        category=IntentCategory.NOTIFICATIONS_SUBSCRIPTIONS,
        action_code=ActionCode.SUBSCRIBE_PRODUCT,
        description="User wants to subscribe to updates for a specific product",
        example_phrases=[
            "Notify me when this product is back in stock",
            "Subscribe to product updates",
            "Alert me about this product",
            "Notify me of price changes",
            "Subscribe to product notifications",
            "Alert me when available",
            "Notify me about this item",
            "Subscribe to updates",
            "Notify me when back in stock",
            "Stock alert for this product"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.NOTIFICATION_PREFERENCES],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "unsubscribe_product": IntentDefinition(
        category=IntentCategory.NOTIFICATIONS_SUBSCRIPTIONS,
        action_code=ActionCode.UNSUBSCRIBE_PRODUCT,
        description="User wants to unsubscribe from updates for a specific product",
        example_phrases=[
            "Stop notifying me about this product",
            "Unsubscribe from product updates",
            "Cancel product notifications",
            "Stop product alerts",
            "Unsubscribe from this product",
            "No more updates for this product",
            "Cancel product subscription",
            "Stop product notifications"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),
}
