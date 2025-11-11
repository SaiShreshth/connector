from ..models import IntentDefinition
from ..enums import IntentCategory, ActionCode, IntentPriority, EntityType

returns_refunds_intent_definitions = {
    # 9.0 Returns, Refunds & Warranty
    "view_return_policy": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.VIEW_RETURN_POLICY,
        description="User wants to see the return policy",
        example_phrases=[
            "What's the return policy?",
            "Show me return policy",
            "Return policy",
            "How do returns work?",
            "Return information",
            "Return terms",
            "Return conditions",
            "Return rules",
            "Explain your return policy",
            "How can I return items?"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "view_warranty_policy": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.VIEW_WARRANTY_POLICY,
        description="User wants to see the warranty policy",
        example_phrases=[
            "What's the warranty policy?",
            "Show me warranty policy",
            "Warranty information",
            "How does warranty work?",
            "Warranty terms",
            "Warranty coverage",
            "Warranty details",
            "Warranty policy",
            "Explain the warranty",
            "What's covered under warranty?"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "product_return_eligibility": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.PRODUCT_RETURN_ELIGIBILITY,
        description="User wants to check if a product is eligible for return",
        example_phrases=[
            "Can I return this?",
            "Is this returnable?",
            "Return eligibility",
            "Can this be returned?",
            "Is return allowed?",
            "Returnable product?",
            "Eligible for return?",
            "Can I send this back?",
            "Is this item eligible for returns?",
            "Can I return this product?"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "initiate_return": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.INITIATE_RETURN,
        description="User wants to start a return process",
        example_phrases=[
            "I want to return this",
            "Start return process",
            "Initiate return",
            "Return this item",
            "I need to return",
            "Begin return",
            "Start return",
            "Return this product",
            "Create a return",
            "Start a return request"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[EntityType.REASON, EntityType.RETURN_REASON],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "cancel_return": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.CANCEL_RETURN,
        description="User wants to cancel their return",
        example_phrases=[
            "Cancel my return",
            "I want to cancel return",
            "Stop return process",
            "Cancel return request",
            "Don't return anymore",
            "Cancel return",
            "Stop return",
            "Cancel return process",
            "Withdraw return request",
            "Abort the return"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "track_return": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.TRACK_RETURN,
        description="User wants to track their return",
        example_phrases=[
            "Track my return",
            "Where is my return?",
            "Return tracking",
            "Track return package",
            "Return status",
            "Where's my return?",
            "Track return shipment",
            "Return package tracking",
            "Track my return order",
            "Follow my return"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "return_pickup_schedule": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.RETURN_PICKUP_SCHEDULE,
        description="User wants to schedule return pickup",
        example_phrases=[
            "Schedule return pickup",
            "I want pickup for return",
            "Schedule pickup",
            "Arrange return pickup",
            "Pickup for return",
            "Schedule return collection",
            "Return pickup",
            "Arrange pickup",
            "Book a pickup for my return",
            "Pickup request for return"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[EntityType.DATE_RANGE, EntityType.ADDRESS],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "return_status": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.RETURN_STATUS,
        description="User wants to check the status of their return",
        example_phrases=[
            "What's my return status?",
            "Return status",
            "How is my return?",
            "Return progress",
            "Return update",
            "Status of return",
            "Return information",
            "Check return status",
            "Update on my return",
            "What's the status of my return?"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "warranty_claim": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.WARRANTY_CLAIM,
        description="User wants to make a warranty claim",
        example_phrases=[
            "I want to make a warranty claim",
            "Warranty claim",
            "Claim warranty",
            "Use warranty",
            "Warranty service",
            "Claim under warranty",
            "Warranty repair",
            "Warranty replacement",
            "Open a warranty ticket",
            "Start warranty process"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.ISSUE_TYPE, EntityType.REASON],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "warranty_status": IntentDefinition(
        category=IntentCategory.RETURNS_REFUNDS,
        action_code=ActionCode.WARRANTY_STATUS,
        description="User wants to check warranty status",
        example_phrases=[
            "What's my warranty status?",
            "Warranty status",
            "Check warranty",
            "Warranty information",
            "Is this under warranty?",
            "Warranty coverage status",
            "Warranty details",
            "Check warranty coverage",
            "Is my product still under warranty?",
            "Warranty validity status"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),
}