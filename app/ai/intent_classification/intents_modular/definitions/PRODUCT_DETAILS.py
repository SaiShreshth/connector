from ..models import IntentDefinition
from ..enums import IntentCategory, ActionCode, IntentPriority, EntityType

product_details_intent_definitions = {
    # 2.0 Product Details & Information
    "product_info": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_INFO,
        description="User wants general information about a product",
        example_phrases=[
            "Tell me about this product",
            "What is this?",
            "Product information",
            "Show me details",
            "What does this do?",
            "Tell me more",
            "Product specs",
            "What's this product about?",
            "Give me product info",
            "Explain this product",
            "Show product information"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "product_details": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_DETAILS,
        description="User wants detailed information about a specific product",
        example_phrases=[
            "Show me product details",
            "I want to see details",
            "Product specifications",
            "Show full details",
            "Detailed information",
            "Complete product info",
            "Show me everything",
            "Full product details",
            "Display product details",
            "Show item details",
            "Detailed specs"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "product_availability": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_AVAILABILITY,
        description="User wants to check if a product is available",
        example_phrases=[
            "Is this available?",
            "Check availability",
            "Is it in stock?",
            "Can I buy this?",
            "Is this product available?",
            "Stock status",
            "Is it available now?",
            "Check if in stock",
            "Availability status",
            "Available to order?"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "product_price": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_PRICE,
        description="User wants to know the price of a product",
        example_phrases=[
            "How much is this?",
            "What's the price?",
            "Show me the price",
            "How much does it cost?",
            "Price information",
            "What's the cost?",
            "Show price",
            "How much?",
            "Product price",
            "What's the price of this?",
            "Tell me the cost"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.90,
        priority=IntentPriority.HIGH
    ),

    "product_images": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_IMAGES,
        description="User wants to see product images",
        example_phrases=[
            "Show me images",
            "I want to see pictures",
            "Product photos",
            "Show me photos",
            "Display images",
            "I want to see images",
            "Show pictures",
            "Product gallery",
            "Show product images",
            "Show item photos"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "product_video": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_VIDEO,
        description="User wants to see product videos",
        example_phrases=[
            "Show me videos",
            "I want to see video",
            "Product video",
            "Show video",
            "Display videos",
            "I want to watch video",
            "Show me the video",
            "Product demo video",
            "Play product video",
            "Show demo video"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "product_warranty": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_WARRANTY,
        description="User wants to know about product warranty",
        example_phrases=[
            "What's the warranty?",
            "Warranty information",
            "How long is the warranty?",
            "Show me warranty details",
            "What warranty does it have?",
            "Warranty coverage",
            "Tell me about warranty",
            "Warranty terms"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "product_dimensions": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_DIMENSIONS,
        description="User wants to know product dimensions",
        example_phrases=[
            "What are the dimensions?",
            "Show me size",
            "How big is it?",
            "Dimensions please",
            "What size is this?",
            "Show dimensions",
            "How large is it?",
            "Size information",
            "Product size details",
            "Dimensions of this item"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "product_weight": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_WEIGHT,
        description="User wants to know product weight",
        example_phrases=[
            "How much does it weigh?",
            "What's the weight?",
            "Show me weight",
            "Weight information",
            "How heavy is it?",
            "Product weight",
            "Tell me the weight",
            "Weight details",
            "Item weight",
            "Weight of this product"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "product_color_options": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_COLOR_OPTIONS,
        description="User wants to see available color options for a product",
        example_phrases=[
            "What colors are available?",
            "Show me color options",
            "Available colors",
            "What colors does it come in?",
            "Color choices",
            "Show color variants",
            "What color options?",
            "Available colorways",
            "List color options",
            "Show available colours"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "product_size_options": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_SIZE_OPTIONS,
        description="User wants to see available size options for a product",
        example_phrases=[
            "What sizes are available?",
            "Show me size options",
            "Available sizes",
            "What sizes does it come in?",
            "Size choices",
            "Show size variants",
            "What size options?",
            "Available sizes",
            "List size options",
            "Show available sizes"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "product_variants": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_VARIANTS,
        description="User wants to see all available variants of a product",
        example_phrases=[
            "Show me all variants",
            "What variants are available?",
            "Show product variants",
            "Available options",
            "Show me options",
            "What options are there?",
            "Product variations",
            "Show all options",
            "List product variants",
            "Available product versions"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "product_sku_info": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_SKU_INFO,
        description="User wants to know the SKU or product code",
        example_phrases=[
            "What's the SKU?",
            "Show me the product code",
            "What's the item number?",
            "SKU information",
            "Product code",
            "Item number",
            "Show SKU",
            "What's the model number?",
            "Show product SKU",
            "Display SKU"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.75,
        priority=IntentPriority.LOW
    ),

    "product_faq": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_FAQ,
        description="User wants to see frequently asked questions about a product",
        example_phrases=[
            "Show me FAQ",
            "Frequently asked questions",
            "Product FAQ",
            "Common questions",
            "Show me questions",
            "FAQ about this product",
            "What questions are asked?",
            "Show product FAQ",
            "FAQ for this item",
            "Product help questions"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "product_reviews": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_REVIEWS,
        description="User wants to see product reviews",
        example_phrases=[
            "Show me reviews",
            "I want to see reviews",
            "Product reviews",
            "Show reviews",
            "Customer reviews",
            "What do people say?",
            "Show me feedback",
            "User reviews",
            "Read product reviews",
            "Show customer feedback",
            "What are customers saying?"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "product_average_review": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_AVERAGE_REVIEW,
        description="User wants to know the average review rating",
        example_phrases=[
            "What's the average rating?",
            "Show me average rating",
            "Overall rating",
            "Average score",
            "What's the rating?",
            "Show rating",
            "Overall review score",
            "Average review",
            "Mean star rating",
            "Typical rating for this"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "product_rating": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_RATING,
        description="User wants to know the product rating",
        example_phrases=[
            "What's the rating?",
            "Show me the rating",
            "How is it rated?",
            "What stars does it have?",
            "Show rating",
            "Product rating",
            "How many stars?",
            "What's the score?",
            "Star rating for this",
            "Rating out of five"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "product_comparison": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_COMPARISON,
        description="User wants to compare products",
        example_phrases=[
            "Compare these products",
            "I want to compare",
            "Show me comparison",
            "Compare products",
            "Product comparison",
            "Compare these items",
            "Show comparison",
            "Compare side by side",
            "Compare iPhone and Galaxy",
            "Compare iPhone 14 and Galaxy S23",
            "Which is better between these two?",
            "Put these two products head to head"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "compare_feature": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_COMPARISON,
        description="User wants to compare specific features of products",
        example_phrases=[
            "Compare features",
            "Show feature comparison",
            "Compare specifications",
            "Feature comparison",
            "Compare specs",
            "Show me feature differences",
            "Compare capabilities",
            "Feature differences",
            "Side-by-side spec comparison",
            "Which features differ?"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    "product_alternatives": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_ALTERNATIVES,
        description="User wants to see alternative products",
        example_phrases=[
            "Show me alternatives",
            "What are the alternatives?",
            "Alternative products",
            "Show alternatives",
            "Other options",
            "Similar products",
            "Alternative choices",
            "Show me other options",
            "What else is similar?",
            "Recommend alternatives"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "related_accessories": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.RELATED_ACCESSORIES,
        description="User wants to see related accessories for a product",
        example_phrases=[
            "Show me accessories",
            "What accessories are available?",
            "Related accessories",
            "Show accessories",
            "What goes with this?",
            "Accessories for this",
            "Show me add-ons",
            "Related items",
            "Compatible accessories",
            "Add-on recommendations"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "product_share": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.PRODUCT_SHARE,
        description="User wants to share a product with others",
        example_phrases=[
            "Share this product",
            "I want to share this",
            "Share with friends",
            "Send this to someone",
            "Share product",
            "Show me how to share",
            "Share link",
            "Send product link",
            "Copy share link",
            "Share via email"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "request_quote": IntentDefinition(
        category=IntentCategory.PRODUCT_DETAILS,
        action_code=ActionCode.REQUEST_QUOTE,
        description="User wants to request a quote for a product",
        example_phrases=[
            "I want a quote",
            "Request quote",
            "Get me a quote",
            "Quote for this product",
            "I need a quote",
            "Can I get a quote?",
            "Request pricing",
            "Show me quote options",
            "Provide a price quote",
            "Send me a quotation"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.QUANTITY, EntityType.EMAIL],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),
}