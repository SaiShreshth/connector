from ..models import IntentDefinition
from ..enums import IntentCategory, ActionCode, IntentPriority, EntityType

orders_fulfillment_intent_definitions = {
    # 5.0 Orders & Fulfillment
    "order_status": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.ORDER_STATUS,
        description="User wants to check the status of their order",
        example_phrases=[
            "What's my order status?",
            "Check order status",
            "Order status",
            "Where is my order?",
            "Where is my order ORD123?",
            "Track my order",
            "Status of my order",
            "How is my order?",
            "Order progress",
            "Check my order",
            "Update on my order",
            "What's happening with my order?"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "order_history": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.ORDER_HISTORY,
        description="User wants to see their order history",
        example_phrases=[
            "Show my order history",
            "Order history",
            "My past orders",
            "Previous orders",
            "Show order history",
            "All my orders",
            "Order list",
            "Past purchases",
            "View my previous orders",
            "See my purchase history"
        ],
        required_entities=[],
        optional_entities=[EntityType.DATE_RANGE],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "order_cancel": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.ORDER_CANCEL,
        description="User wants to cancel their order",
        example_phrases=[
            "Cancel my order",
            "I want to cancel",
            "Cancel order",
            "Cancel this order",
            "I need to cancel",
            "Cancel purchase",
            "Cancel my purchase",
            "Stop this order",
            "Abort this order",
            "Please cancel this order"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[EntityType.REASON],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "order_return": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.ORDER_RETURN,
        description="User wants to return an order",
        example_phrases=[
            "I want to return this",
            "Return my order",
            "Return this item",
            "I need to return",
            "Return order",
            "Return purchase",
            "Return this product",
            "I want to return",
            "Create a return",
            "Start a return request"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[EntityType.REASON],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "track_shipment": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.TRACK_SHIPMENT,
        description="User wants to track their shipment",
        example_phrases=[
            "Track my package",
            "Where is my package?",
            "Track shipment",
            "Package tracking",
            "Track delivery",
            "Where's my delivery?",
            "Track my order",
            "Package status",
            "Follow my shipment",
            "Shipment tracking status"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "track_return": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
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
            "Follow my return",
            "Return shipment status"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "cancel_return": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.CANCEL_RETURN,
        description="User wants to cancel their return",
        example_phrases=[
            "Cancel my return",
            "I want to cancel return",
            "Cancel return",
            "Stop return",
            "Cancel return request",
            "Don't return anymore",
            "Cancel return process",
            "Stop return",
            "Withdraw return request",
            "Abort the return"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "refund_request": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.REFUND_REQUEST,
        description="User wants to request a refund",
        example_phrases=[
            "I want a refund",
            "Request refund",
            "Refund my money",
            "I need a refund",
            "Refund request",
            "Give me refund",
            "Process refund",
            "Refund this order",
            "Start a refund",
            "Initiate a refund request"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[EntityType.REASON],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "refund_complete": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.REFUND_COMPLETE,
        description="User wants to check if their refund is complete",
        example_phrases=[
            "Is my refund complete?",
            "Refund status",
            "Refund processed?",
            "Is refund done?",
            "Refund complete?",
            "Refund finished?",
            "Refund processed",
            "Check refund",
            "Has my refund been issued?",
            "Refund completion status"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "order_invoice": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.ORDER_INVOICE,
        description="User wants to see their order invoice",
        example_phrases=[
            "Show me invoice",
            "I want my invoice",
            "Order invoice",
            "Show invoice",
            "Print invoice",
            "Email invoice",
            "Get invoice",
            "Invoice for order",
            "Download my invoice",
            "Send me the invoice"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "reorder": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.REORDER,
        description="User wants to reorder a previous purchase",
        example_phrases=[
            "Reorder this",
            "I want to reorder",
            "Order again",
            "Buy again",
            "Reorder item",
            "Purchase again",
            "Order same thing",
            "Buy this again",
            "Repeat my last order",
            "Order this once more"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[EntityType.QUANTITY],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "delivery_status": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.DELIVERY_STATUS,
        description="User wants to check delivery status",
        example_phrases=[
            "Delivery status",
            "Is it delivered?",
            "Delivery update",
            "Has it arrived?",
            "Delivery progress",
            "Is it delivered yet?",
            "Delivery information",
            "Check delivery",
            "What's the delivery status?",
            "Has my package been delivered?"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "delivery_estimate": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.DELIVERY_ESTIMATE,
        description="User wants to know delivery estimate",
        example_phrases=[
            "When will it arrive?",
            "Delivery estimate",
            "Expected delivery",
            "When is delivery?",
            "Delivery date",
            "Arrival time",
            "When delivered?",
            "Delivery timeline",
            "Estimated arrival time",
            "Expected delivery date"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "schedule_delivery": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.SCHEDULE_DELIVERY,
        description="User wants to schedule delivery",
        example_phrases=[
            "Schedule delivery",
            "I want to schedule delivery",
            "Set delivery time",
            "Schedule my delivery",
            "Choose delivery time",
            "Pick delivery date",
            "Schedule delivery time",
            "Set delivery appointment",
            "Book a delivery slot",
            "Arrange delivery time"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[EntityType.DATE_RANGE],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "change_delivery_date": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.CHANGE_DELIVERY_DATE,
        description="User wants to change delivery date",
        example_phrases=[
            "Change delivery date",
            "I want to change delivery",
            "Reschedule delivery",
            "Change delivery time",
            "Modify delivery date",
            "Update delivery",
            "Change delivery schedule",
            "Reschedule my delivery",
            "Move my delivery date",
            "Adjust delivery time"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[EntityType.DATE_RANGE],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "shipment_details": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.SHIPMENT_DETAILS,
        description="User wants detailed shipment information",
        example_phrases=[
            "Shipment details",
            "Show me shipment info",
            "Detailed shipment",
            "Shipment information",
            "Package details",
            "Shipping details",
            "Shipment specifics",
            "Delivery details",
            "Tracking details for shipment",
            "Full shipping information"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "shipping_cost": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.SHIPPING_COST,
        description="User wants to know shipping cost",
        example_phrases=[
            "What's shipping cost?",
            "How much is shipping?",
            "Shipping cost",
            "Delivery cost",
            "Shipping fee",
            "How much for shipping?",
            "Shipping price",
            "Delivery fee",
            "Cost of shipping this order",
            "How much will delivery cost?"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "shipping_options": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.SHIPPING_OPTIONS,
        description="User wants to see shipping options",
        example_phrases=[
            "What shipping options?",
            "Show shipping options",
            "Shipping choices",
            "Delivery options",
            "Available shipping",
            "Shipping methods",
            "What delivery options?",
            "Show delivery choices",
            "Available carriers and speeds",
            "List shipping methods"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "order_modify": IntentDefinition(
        category=IntentCategory.ORDERS_FULFILLMENT,
        action_code=ActionCode.ORDER_MODIFY,
        description="User wants to modify their order",
        example_phrases=[
            "I want to modify my order",
            "Change my order",
            "Modify order",
            "Update order",
            "Change order details",
            "Modify purchase",
            "Update my order",
            "Change order",
            "Edit my order",
            "Adjust order details"
        ],
        required_entities=[EntityType.ORDER_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),
}