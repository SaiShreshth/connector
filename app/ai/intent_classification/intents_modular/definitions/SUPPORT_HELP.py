from ..models import IntentDefinition
from ..enums import IntentCategory, ActionCode, IntentPriority, EntityType

support_help_intent_definitions = {
    # 7.0 Support, Help & FAQ
    "contact_support": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.CONTACT_SUPPORT,
        description="User wants to contact customer support",
        example_phrases=[
            "I need help",
            "Contact support",
            "I want to talk to someone",
            "Get help",
            "Support please",
            "I need assistance",
            "Contact customer service",
            "Help me"
        ],
        required_entities=[],
        optional_entities=[EntityType.ISSUE_TYPE, EntityType.URGENCY],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "live_chat": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.LIVE_CHAT,
        description="User wants to start a live chat with support",
        example_phrases=[
            "I want to chat",
            "Start live chat",
            "Chat with support",
            "Live chat please",
            "I want to chat with someone",
            "Start chat",
            "Chat support",
            "Live chat help"
        ],
        required_entities=[],
        optional_entities=[EntityType.ISSUE_TYPE],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "call_support": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.CALL_SUPPORT,
        description="User wants to call customer support",
        example_phrases=[
            "I want to call",
            "Call support",
            "Phone support",
            "I want to speak to someone",
            "Call customer service",
            "Phone number",
            "I want to call support",
            "Call help"
        ],
        required_entities=[],
        optional_entities=[EntityType.PHONE_NUMBER],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "request_callback": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.REQUEST_CALLBACK,
        description="User wants to request a callback from support",
        example_phrases=[
            "Call me back",
            "Request callback",
            "I want a callback",
            "Please call me",
            "Callback request",
            "Call me later",
            "Schedule callback",
            "I need a callback"
        ],
        required_entities=[],
        optional_entities=[EntityType.PHONE_NUMBER, EntityType.DATE_RANGE],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "feedback_submit": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.FEEDBACK_SUBMIT,
        description="User wants to submit feedback",
        example_phrases=[
            "I want to give feedback",
            "Submit feedback",
            "Leave feedback",
            "Give feedback",
            "I have feedback",
            "Submit my feedback",
            "Feedback form",
            "I want to comment"
        ],
        required_entities=[],
        optional_entities=[EntityType.QUESTION, EntityType.ISSUE_TYPE],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "report_issue": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.REPORT_ISSUE,
        description="User wants to report an issue",
        example_phrases=[
            "I want to report an issue",
            "Report problem",
            "I have an issue",
            "Report bug",
            "Something is wrong",
            "I found a problem",
            "Report issue",
            "I need to report something"
        ],
        required_entities=[],
        optional_entities=[EntityType.ISSUE_TYPE, EntityType.URGENCY],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "report_product": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.REPORT_PRODUCT,
        description="User wants to report an issue with a product",
        example_phrases=[
            "Report this product",
            "Product issue",
            "Report product problem",
            "This product has issues",
            "Report defective product",
            "Product complaint",
            "Report product issue",
            "I want to report this product"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.ISSUE_TYPE, EntityType.REASON],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "help_general": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.HELP_GENERAL,
        description="User wants general help or assistance",
        example_phrases=[
            "I need help",
            "Help me",
            "I'm confused",
            "How does this work?",
            "I don't understand",
            "Can you help?",
            "I need assistance",
            "Help please"
        ],
        required_entities=[],
        optional_entities=[EntityType.QUESTION, EntityType.TOPIC],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "faq_returns": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.FAQ_RETURNS,
        description="User has questions about returns",
        example_phrases=[
            "How do I return something?",
            "Return policy",
            "Can I return this?",
            "Return process",
            "How to return",
            "Return information",
            "Returning items",
            "Return help"
        ],
        required_entities=[],
        optional_entities=[EntityType.PRODUCT_ID],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "faq_shipping": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.FAQ_SHIPPING,
        description="User has questions about shipping",
        example_phrases=[
            "How long does shipping take?",
            "Shipping information",
            "When will it arrive?",
            "Shipping times",
            "Delivery information",
            "How fast is shipping?",
            "Shipping options",
            "Delivery times"
        ],
        required_entities=[],
        optional_entities=[EntityType.ORDER_ID],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "faq_payment": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.FAQ_PAYMENT,
        description="User has questions about payment",
        example_phrases=[
            "How do I pay?",
            "Payment methods",
            "What payment options?",
            "How to pay",
            "Payment information",
            "Payment help",
            "Payment questions",
            "How can I pay?"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "faq_account": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.FAQ_ACCOUNT,
        description="User has questions about their account",
        example_phrases=[
            "Account questions",
            "How do I manage my account?",
            "Account help",
            "Account information",
            "Account settings",
            "Account management",
            "Account issues",
            "Account problems"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "faq_coupon": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.FAQ_COUPON,
        description="User has questions about coupons",
        example_phrases=[
            "How do I use coupons?",
            "Coupon help",
            "Coupon questions",
            "How to apply coupon",
            "Coupon information",
            "Coupon problems",
            "Coupon issues",
            "Coupon help"
        ],
        required_entities=[],
        optional_entities=[EntityType.COUPON_CODE],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "faq_tech_support": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.FAQ_TECH_SUPPORT,
        description="User has technical support questions",
        example_phrases=[
            "Technical help",
            "Tech support",
            "Technical issues",
            "Website problems",
            "App issues",
            "Technical problems",
            "Tech questions",
            "Technical support"
        ],
        required_entities=[],
        optional_entities=[EntityType.ISSUE_TYPE],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "faq_delivery": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.FAQ_DELIVERY,
        description="User has questions about delivery",
        example_phrases=[
            "Delivery questions",
            "Delivery help",
            "Delivery information",
            "Delivery problems",
            "Delivery issues",
            "Delivery support",
            "Delivery questions",
            "Delivery help"
        ],
        required_entities=[],
        optional_entities=[EntityType.ORDER_ID],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "faq_refund": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.FAQ_REFUND,
        description="User has questions about refunds",
        example_phrases=[
            "Refund questions",
            "Refund help",
            "How do I get a refund?",
            "Refund information",
            "Refund process",
            "Refund policy",
            "Refund issues",
            "Refund problems"
        ],
        required_entities=[],
        optional_entities=[EntityType.ORDER_ID],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "payment_help": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.PAYMENT_HELP,
        description="User needs help with payment issues",
        example_phrases=[
            "Payment help",
            "I have payment problems",
            "Payment issues",
            "Payment support",
            "Payment problems",
            "Help with payment",
            "Payment assistance",
            "Payment trouble"
        ],
        required_entities=[],
        optional_entities=[EntityType.ISSUE_TYPE, EntityType.ORDER_ID],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "security_help": IntentDefinition(
        category=IntentCategory.SUPPORT_HELP,
        action_code=ActionCode.SECURITY_HELP,
        description="User needs help with security issues",
        example_phrases=[
            "Security help",
            "I have security concerns",
            "Security issues",
            "Account security",
            "Security problems",
            "Security support",
            "Security assistance",
            "Security trouble"
        ],
        required_entities=[],
        optional_entities=[EntityType.ISSUE_TYPE],
        confidence_threshold=0.85,
        priority=IntentPriority.CRITICAL
    ),
}