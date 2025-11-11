from ..models import IntentDefinition
from ..enums import IntentCategory, ActionCode, IntentPriority, EntityType

checkout_payment_intent_definitions = {
    # 4.0 Checkout & Payment
    "checkout_initiate": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.CHECKOUT_INITIATE,
        description="User wants to start the checkout process",
        example_phrases=[
            "I want to checkout",
            "Proceed to checkout",
            "Start checkout",
            "Begin checkout",
            "Checkout now",
            "I'm ready to buy",
            "Complete purchase",
            "Go to checkout"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "checkout_address": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.CHECKOUT_ADDRESS,
        description="User wants to manage shipping address during checkout",
        example_phrases=[
            "Enter shipping address",
            "Add my address",
            "Set delivery address",
            "Where should this be shipped?",
            "Enter delivery address",
            "Add shipping info",
            "Set address",
            "Enter address"
        ],
        required_entities=[],
        optional_entities=[EntityType.ADDRESS],
        confidence_threshold=0.85,
        priority=IntentPriority.CRITICAL
    ),

    "add_new_address": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.ADD_NEW_ADDRESS,
        description="User wants to add a new shipping address",
        example_phrases=[
            "Add new address",
            "I want to add an address",
            "Create new address",
            "Add another address",
            "New shipping address",
            "Add different address",
            "Create address",
            "Add address"
        ],
        required_entities=[],
        optional_entities=[EntityType.ADDRESS],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "update_address": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.UPDATE_ADDRESS,
        description="User wants to update an existing address",
        example_phrases=[
            "Update my address",
            "Change address",
            "Edit address",
            "Modify address",
            "Update shipping address",
            "Change delivery address",
            "Edit my address",
            "Update address info"
        ],
        required_entities=[EntityType.ADDRESS],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "delete_address": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.DELETE_ADDRESS,
        description="User wants to delete an address",
        example_phrases=[
            "Delete address",
            "Remove address",
            "Delete this address",
            "Remove my address",
            "Delete shipping address",
            "Remove delivery address",
            "Delete address info",
            "Remove this address"
        ],
        required_entities=[EntityType.ADDRESS],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "change_address": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.CHANGE_ADDRESS,
        description="User wants to change the shipping address for their order",
        example_phrases=[
            "Change shipping address",
            "I want to change address",
            "Switch address",
            "Change delivery address",
            "Use different address",
            "Change to another address",
            "Switch to new address",
            "Change address"
        ],
        required_entities=[EntityType.ADDRESS],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "checkout_payment": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.CHECKOUT_PAYMENT,
        description="User wants to proceed with payment during checkout",
        example_phrases=[
            "I want to pay",
            "Proceed to payment",
            "Pay now",
            "Payment options",
            "How do I pay?",
            "Payment method",
            "Complete payment",
            "Pay for order"
        ],
        required_entities=[],
        optional_entities=[EntityType.PAYMENT_METHOD],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "payment_options": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.PAYMENT_OPTIONS,
        description="User wants to see available payment methods",
        example_phrases=[
            "What payment methods do you accept?",
            "Show payment options",
            "How can I pay?",
            "Payment methods",
            "What are my payment options?",
            "Show me payment choices",
            "Available payments",
            "Payment options"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "change_payment_method": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.CHANGE_PAYMENT_METHOD,
        description="User wants to change their payment method",
        example_phrases=[
            "Change payment method",
            "I want to use a different card",
            "Switch payment method",
            "Use different payment",
            "Change how I pay",
            "Switch to credit card",
            "Change payment",
            "Use another payment method"
        ],
        required_entities=[],
        optional_entities=[EntityType.PAYMENT_METHOD],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "update_payment_method": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.UPDATE_PAYMENT_METHOD,
        description="User wants to update their payment method details",
        example_phrases=[
            "Update payment method",
            "Change card details",
            "Update payment info",
            "Modify payment method",
            "Update card information",
            "Change payment details",
            "Edit payment method",
            "Update payment"
        ],
        required_entities=[EntityType.PAYMENT_METHOD],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "checkout_confirm": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.CHECKOUT_CONFIRM,
        description="User wants to confirm and complete their order",
        example_phrases=[
            "Confirm order",
            "Place order",
            "Complete purchase",
            "Confirm and pay",
            "Finalize order",
            "Complete checkout",
            "Place my order",
            "Confirm purchase"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "payment_failed_help": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.PAYMENT_FAILED_HELP,
        description="User needs help with a failed payment",
        example_phrases=[
            "My payment failed",
            "Payment didn't work",
            "Help with payment",
            "Payment error",
            "Why did payment fail?",
            "Payment problem",
            "Failed to process payment",
            "Payment issue"
        ],
        required_entities=[],
        optional_entities=[EntityType.ISSUE_TYPE],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "payment_success": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.PAYMENT_SUCCESS,
        description="User wants to confirm their payment was successful",
        example_phrases=[
            "Did payment go through?",
            "Was payment successful?",
            "Payment confirmation",
            "Did it work?",
            "Payment status",
            "Is payment complete?",
            "Payment successful?",
            "Confirm payment"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "payment_pending": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.PAYMENT_PENDING,
        description="User wants to check on a pending payment",
        example_phrases=[
            "Is payment pending?",
            "Payment still processing?",
            "Why is payment pending?",
            "Payment status",
            "Is it still processing?",
            "Pending payment",
            "Payment taking long",
            "Why pending?"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "payment_receipt": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.PAYMENT_RECEIPT,
        description="User wants to see their payment receipt",
        example_phrases=[
            "Show me receipt",
            "I want my receipt",
            "Payment receipt",
            "Show receipt",
            "Where's my receipt?",
            "Print receipt",
            "Email receipt",
            "Get receipt"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "gift_card_balance": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.GIFT_CARD_BALANCE,
        description="User wants to check their gift card balance",
        example_phrases=[
            "Check gift card balance",
            "What's my gift card balance?",
            "Gift card balance",
            "Check balance",
            "How much is on my gift card?",
            "Gift card amount",
            "Check gift card",
            "Balance inquiry"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "gift_card_redeem": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.GIFT_CARD_REDEEM,
        description="User wants to redeem a gift card",
        example_phrases=[
            "Redeem gift card",
            "Use gift card",
            "Apply gift card",
            "Redeem my gift card",
            "Use my gift card",
            "Apply gift card code",
            "Redeem code",
            "Use gift card balance"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "apply_gift_card": IntentDefinition(
        category=IntentCategory.CHECKOUT_PAYMENT,
        action_code=ActionCode.APPLY_GIFT_CARD,
        description="User wants to apply a gift card to their order",
        example_phrases=[
            "Apply gift card",
            "Use gift card for payment",
            "Apply gift card to order",
            "Use gift card",
            "Apply gift card code",
            "Use gift card balance",
            "Apply gift card payment",
            "Use gift card for this order"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),
}