from ..models import IntentDefinition
from ..enums import IntentCategory, ActionCode, IntentPriority, EntityType

promotions_loyalty_intent_definitions = {
    # 8.0 Promotions, Offers & Loyalty
    "view_promotions": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.VIEW_PROMOTIONS,
        description="User wants to see current promotions and offers",
        example_phrases=[
            "Show me promotions",
            "What promotions are available?",
            "Current promotions",
            "Show promotions",
            "Available offers",
            "What deals are there?",
            "Promotional offers",
            "Show me deals",
            "Active promotions",
            "What discounts can I get?"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "view_coupons": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.VIEW_COUPONS,
        description="User wants to see available coupons",
        example_phrases=[
            "Show me coupons",
            "Available coupons",
            "What coupons do I have?",
            "Show coupons",
            "My coupons",
            "Coupon codes",
            "Available discount codes",
            "Show discount codes",
            "List my coupon codes",
            "Any coupons for me?"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "apply_promo": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.APPLY_PROMO,
        description="User wants to apply a promotional offer",
        example_phrases=[
            "Apply promotion",
            "Use this promotion",
            "Apply promo code",
            "Use promotional offer",
            "Apply deal",
            "Use this deal",
            "Apply offer",
            "Use promotion",
            "Apply discount code",
            "Use my coupon"
        ],
        required_entities=[],
        optional_entities=[EntityType.COUPON_CODE],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "remove_promo": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.REMOVE_PROMO,
        description="User wants to remove a promotional offer",
        example_phrases=[
            "Remove promotion",
            "Take off promotion",
            "Remove promo",
            "Cancel promotion",
            "Remove deal",
            "Take off deal",
            "Remove offer",
            "Cancel offer",
            "Clear applied promo",
            "Remove the discount"
        ],
        required_entities=[],
        optional_entities=[EntityType.COUPON_CODE],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "view_bestsellers": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.VIEW_BESTSELLERS,
        description="User wants to see best-selling products",
        example_phrases=[
            "Show me bestsellers",
            "What are the bestsellers?",
            "Best selling products",
            "Show bestsellers",
            "Top sellers",
            "Most popular items",
            "Best selling items",
            "Show me top sellers",
            "Top selling right now",
            "Best sellers list"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "view_new_arrivals": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.VIEW_NEW_ARRIVALS,
        description="User wants to see new arrival products",
        example_phrases=[
            "Show me new arrivals",
            "What's new?",
            "New products",
            "Latest arrivals",
            "Show new items",
            "What just came in?",
            "New arrivals",
            "Show me latest",
            "Just in products",
            "Recently added items"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "view_sale_collection": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.VIEW_SALE_COLLECTION,
        description="User wants to see products on sale",
        example_phrases=[
            "Show me sale items",
            "What's on sale?",
            "Sale collection",
            "Items on sale",
            "Show sales",
            "Discounted items",
            "Sale products",
            "Show me deals",
            "Products on markdown",
            "Show current discounts"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY, EntityType.PRICE_RANGE],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "loyalty_points": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.LOYALTY_POINTS,
        description="User wants to check their loyalty points",
        example_phrases=[
            "How many points do I have?",
            "Check my points",
            "Loyalty points",
            "My points balance",
            "Points status",
            "How many loyalty points?",
            "Check points",
            "Points balance",
            "View my points",
            "Show loyalty balance"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "redeem_points": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.REDEEM_POINTS,
        description="User wants to redeem their loyalty points",
        example_phrases=[
            "I want to redeem points",
            "Redeem my points",
            "Use my points",
            "Redeem loyalty points",
            "Spend points",
            "Use points for purchase",
            "Redeem points for discount",
            "Convert points",
            "Apply points to order",
            "Use points at checkout"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "check_rewards": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.CHECK_REWARDS,
        description="User wants to check their rewards",
        example_phrases=[
            "Check my rewards",
            "What rewards do I have?",
            "My rewards",
            "Available rewards",
            "Rewards status",
            "Check rewards balance",
            "Show me rewards",
            "Rewards information",
            "View my rewards",
            "Any rewards available?"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "claim_reward": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.CLAIM_REWARD,
        description="User wants to claim a reward",
        example_phrases=[
            "I want to claim a reward",
            "Claim my reward",
            "Claim reward",
            "Get my reward",
            "Claim available reward",
            "Redeem reward",
            "Claim my prize",
            "Get reward",
            "Collect my reward",
            "Use my reward now"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "view_gift_options": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.VIEW_GIFT_OPTIONS,
        description="User wants to see gift options",
        example_phrases=[
            "Show me gift options",
            "Gift options",
            "What gift options are there?",
            "Gift choices",
            "Gift ideas",
            "Show gift options",
            "Available gifts",
            "Gift suggestions",
            "Recommend gift ideas",
            "Best gifts to buy"
        ],
        required_entities=[],
        optional_entities=[EntityType.BUDGET],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "price_drop_alerts": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.PRICE_DROP_ALERTS,
        description="User wants to set up price drop alerts",
        example_phrases=[
            "Set up price alerts",
            "I want price drop alerts",
            "Alert me when price drops",
            "Price notification",
            "Price drop notification",
            "Notify me of price changes",
            "Price alert setup",
            "Price drop alerts",
            "Track price changes",
            "Send me price alerts"
        ],
        required_entities=[],
        optional_entities=[EntityType.PRODUCT_ID, EntityType.EMAIL],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "flash_sale_info": IntentDefinition(
        category=IntentCategory.PROMOTIONS_LOYALTY,
        action_code=ActionCode.FLASH_SALE_INFO,
        description="User wants information about flash sales",
        example_phrases=[
            "Tell me about flash sales",
            "Flash sale information",
            "What are flash sales?",
            "Flash sale details",
            "Explain flash sales",
            "Flash sale info",
            "About flash sales",
            "Flash sale explanation",
            "When is the next flash sale?",
            "How do flash sales work?"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),
}