"""
Intent definitions for security and fraud prevention functionality.
"""

from app.ai.intent_classification.intents_modular.enums import (
    IntentCategory, ActionCode, IntentPriority, EntityType
)
from app.ai.intent_classification.intents_modular.models import IntentDefinition

security_fraud_intent_definitions = {
    "report_fraud": IntentDefinition(
        category=IntentCategory.SECURITY_FRAUD,
        action_code=ActionCode.REPORT_FRAUD,
        description="User wants to report fraudulent activity",
        example_phrases=[
            "Report fraud",
            "I want to report fraud",
            "Report suspicious activity",
            "Report fraudulent transaction",
            "Report scam",
            "Report fraud attempt",
            "Report suspicious behavior",
            "Report fraudulent activity"
        ],
        required_entities=[EntityType.ISSUE_DESCRIPTION],
        optional_entities=[EntityType.USER_ID, EntityType.TRANSACTION_ID],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "fraud_detection_help": IntentDefinition(
        category=IntentCategory.SECURITY_FRAUD,
        action_code=ActionCode.FRAUD_DETECTION_HELP,
        description="User needs help with fraud detection or prevention",
        example_phrases=[
            "Help with fraud detection",
            "How to detect fraud",
            "Fraud prevention help",
            "Security help",
            "Fraud protection",
            "How to prevent fraud",
            "Fraud detection assistance",
            "Security assistance"
        ],
        required_entities=[],
        optional_entities=[EntityType.USER_ID],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "lock_account": IntentDefinition(
        category=IntentCategory.SECURITY_FRAUD,
        action_code=ActionCode.LOCK_ACCOUNT,
        description="User wants to lock their account for security",
        example_phrases=[
            "Lock my account",
            "Secure my account",
            "Lock account",
            "Freeze my account",
            "Secure account",
            "Lock my profile",
            "Freeze account",
            "Secure my profile"
        ],
        required_entities=[],
        optional_entities=[EntityType.USER_ID, EntityType.REASON],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "unlock_account": IntentDefinition(
        category=IntentCategory.SECURITY_FRAUD,
        action_code=ActionCode.UNLOCK_ACCOUNT,
        description="User wants to unlock their account",
        example_phrases=[
            "Unlock my account",
            "Unlock account",
            "Unfreeze my account",
            "Unlock my profile",
            "Unfreeze account",
            "Unlock my profile",
            "Restore my account",
            "Unlock access"
        ],
        required_entities=[],
        optional_entities=[EntityType.USER_ID, EntityType.VERIFICATION_CODE],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "security_alert": IntentDefinition(
        category=IntentCategory.SECURITY_FRAUD,
        action_code=ActionCode.SECURITY_ALERT,
        description="System sends security alert to user",
        example_phrases=[
            "Security alert sent",
            "Security notification",
            "Security warning",
            "Security alert",
            "Security notice",
            "Security message",
            "Security update",
            "Security information"
        ],
        required_entities=[EntityType.USER_ID],
        optional_entities=[EntityType.ISSUE_TYPE, EntityType.URGENCY],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),
}
