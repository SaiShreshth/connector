from ..models import IntentDefinition
from ..enums import IntentCategory, ActionCode, IntentPriority, EntityType

search_discovery_intent_definitions = {
    # 1.1 Product Search Actions
    "search_product": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_PRODUCT,
        description="User wants to search for specific products",
        example_phrases=[
            "I'm looking for a laptop",
            "Show me smartphones",
            "Find me running shoes",
            "Search for headphones",
            "I need a coffee maker",
            "Looking for winter jackets",
            "Can you find me a camera?",
            "Show me gaming laptops",
            "Find products matching my query",
            "Search for this item",
            "Look up this product"
        ],
        required_entities=[EntityType.PRODUCT_QUERY],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND, EntityType.PRICE_RANGE],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "search_category": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_CATEGORY,
        description="User wants to browse products in a specific category",
        example_phrases=[
            "Show me electronics",
            "I want to see clothing",
            "Browse home and garden",
            "Show me sports equipment",
            "I'm looking at books",
            "Display kitchen items",
            "Show me beauty products",
            "I want to see toys",
            "Browse this category",
            "Show items in this category"
        ],
        required_entities=[EntityType.CATEGORY],
        optional_entities=[EntityType.PRICE_RANGE, EntityType.BRAND],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "search_brand": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_BRAND,
        description="User wants to find products from a specific brand",
        example_phrases=[
            "Show me Apple products",
            "I want Nike shoes",
            "Find Samsung phones",
            "Show me Adidas clothing",
            "I'm looking for Sony headphones",
            "Display Canon cameras",
            "Show me Toyota cars",
            "I want Levi's jeans",
            "Browse by this brand",
            "Show products from this brand"
        ],
        required_entities=[EntityType.BRAND],
        optional_entities=[EntityType.CATEGORY, EntityType.PRICE_RANGE],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "search_price_range": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_PRICE_RANGE,
        description="User wants to find products within a specific price range",
        example_phrases=[
            "Show me laptops under $1000",
            "I want phones between $200 and $500",
            "Find shoes under $50",
            "Show me cameras over $500",
            "I'm looking for watches under $200",
            "Display headphones under $100",
            "Show me tablets between $300 and $600",
            "I want books under $20",
            "Show items within this price range",
            "Filter by price"
        ],
        required_entities=[EntityType.PRICE_RANGE],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "search_color": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_COLOR,
        description="User wants to find products in a specific color",
        example_phrases=[
            "Show me red dresses",
            "I want black shoes",
            "Find blue jeans",
            "Show me white phones",
            "I'm looking for green bags",
            "Display silver watches",
            "Show me purple headphones",
            "I want yellow shirts",
            "Items in this colour",
            "Show colour options in products"
        ],
        required_entities=[EntityType.COLOR],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    "search_size": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_SIZE,
        description="User wants to find products in a specific size",
        example_phrases=[
            "Show me size 10 shoes",
            "I want large shirts",
            "Find medium pants",
            "Show me XL jackets",
            "I'm looking for size 32 jeans",
            "Display small bags",
            "Show me extra large hoodies",
            "I want size 8 dresses",
            "Filter by size",
            "Show available sizes in products"
        ],
        required_entities=[EntityType.SIZE],
        optional_entities=[EntityType.CATEGORY, EntityType.COLOR],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "search_rating": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_RATING,
        description="User wants to find products with a specific rating",
        example_phrases=[
            "Show me 5-star products",
            "I want highly rated laptops",
            "Find products with 4+ stars",
            "Show me top rated phones",
            "I'm looking for well-reviewed books",
            "Display highly rated headphones",
            "Show me best rated cameras",
            "I want products with good reviews",
            "Filter by rating",
            "Show items above this star rating"
        ],
        required_entities=[EntityType.REVIEW_RATING],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    "search_discount": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_DISCOUNT,
        description="User wants to find products that are on sale or discounted",
        example_phrases=[
            "Show me products on sale",
            "I want discounted items",
            "Find products with deals",
            "Show me clearance items",
            "I'm looking for sale prices",
            "Display discounted electronics",
            "Show me items with offers",
            "I want products on promotion",
            "Show current deals",
            "Filter to on-sale items"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND, EntityType.PRICE_RANGE],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "search_trending": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_TRENDING,
        description="User wants to see trending or popular products",
        example_phrases=[
            "Show me trending products",
            "What's popular right now?",
            "Find trending items",
            "Show me what's hot",
            "I want to see popular products",
            "Display trending electronics",
            "Show me viral products",
            "What's everyone buying?",
            "Popular this week",
            "Hot right now"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    "search_recent": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_RECENT,
        description="User wants to see recently added or updated products",
        example_phrases=[
            "Show me new arrivals",
            "What's new in electronics?",
            "Find recently added items",
            "Show me latest products",
            "I want to see new releases",
            "Display newest arrivals",
            "Show me just added items",
            "What's the latest?",
            "Recently added products",
            "Newly released items"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    "search_similar": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_SIMILAR,
        description="User wants to find products similar to a specific item",
        example_phrases=[
            "Show me similar products",
            "Find items like this",
            "What else is like this?",
            "Show me comparable items",
            "I want similar options",
            "Display related products",
            "Show me alternatives",
            "Find products like this one",
            "Products similar to this",
            "More like this"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "search_featured": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_FEATURED,
        description="User wants to see featured or highlighted products",
        example_phrases=[
            "Show me featured products",
            "What's featured today?",
            "Find highlighted items",
            "Show me editor's picks",
            "I want to see featured deals",
            "Display spotlight products",
            "Show me recommended items",
            "What's being featured?",
            "Featured this week",
            "Spotlighted products"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    "search_new_arrivals": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_NEW_ARRIVALS,
        description="User wants to see newly arrived products",
        example_phrases=[
            "Show me new arrivals",
            "What just came in?",
            "Find newest products",
            "Show me latest arrivals",
            "I want to see new items",
            "Display just arrived products",
            "Show me fresh arrivals",
            "What's newly available?",
            "New products today",
            "Just in items"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "search_bestsellers": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_BESTSELLERS,
        description="User wants to see best-selling products",
        example_phrases=[
            "Show me bestsellers",
            "What's selling well?",
            "Find top sellers",
            "Show me popular items",
            "I want best-selling products",
            "Display top performers",
            "Show me chart toppers",
            "What's the best seller?",
            "Top selling products",
            "Best sellers right now"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "search_on_sale": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_ON_SALE,
        description="User wants to see products currently on sale",
        example_phrases=[
            "Show me what's on sale",
            "What's currently discounted?",
            "Find sale items",
            "Show me sale products",
            "I want to see sales",
            "Display items on sale",
            "Show me discounted products",
            "What's on special offer?",
            "Show active promotions",
            "Products marked down"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND, EntityType.PRICE_RANGE],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    # 1.2 Discovery & Recommendations
    "product_recommendation": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.PRODUCT_RECOMMENDATION,
        description="User wants product recommendations",
        example_phrases=[
            "What do you recommend?",
            "Give me suggestions",
            "What should I buy?",
            "Show me recommendations",
            "I need recommendations",
            "What would you suggest?",
            "Give me some ideas",
            "What's recommended for me?",
            "Suggest products for me",
            "Any product suggestions?"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY, EntityType.BUDGET, EntityType.PREFERENCES],
        confidence_threshold=0.75,
        priority=IntentPriority.HIGH
    ),

    "personalized_recommendations": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.PERSONALIZED_RECOMMENDATIONS,
        description="User wants personalized product recommendations based on their history",
        example_phrases=[
            "Show me personalized recommendations",
            "What's recommended for me?",
            "Give me my recommendations",
            "Show me suggestions for me",
            "What do you think I'd like?",
            "Give me personal picks",
            "Show me tailored suggestions",
            "What matches my taste?",
            "Recommendations based on my history",
            "Suggest items for my profile"
        ],
        required_entities=[],
        optional_entities=[EntityType.USER_ID, EntityType.PREFERENCES],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "recommend_similar": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.RECOMMEND_SIMILAR,
        description="User wants recommendations for products similar to a specific item",
        example_phrases=[
            "Recommend something similar",
            "What's like this?",
            "Show me similar recommendations",
            "Find me something comparable",
            "What else would I like?",
            "Recommend alternatives",
            "Show me related suggestions",
            "What's in the same style?",
            "Similar items to this",
            "More like this product"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "recommend_complements": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.RECOMMEND_COMPLEMENTS,
        description="User wants recommendations for complementary products",
        example_phrases=[
            "What goes with this?",
            "Show me accessories",
            "Recommend complementary items",
            "What pairs well with this?",
            "Show me matching products",
            "What accessories do I need?",
            "Recommend add-ons",
            "What completes this set?",
            "Complements for this item",
            "Products that go together"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    "view_recommended": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.VIEW_RECOMMENDED,
        description="User wants to view their recommended products",
        example_phrases=[
            "Show me my recommendations",
            "View recommended items",
            "See my suggestions",
            "Show recommended products",
            "Display my picks",
            "View my recommendations",
            "Show me suggested items",
            "See what's recommended",
            "My suggested products",
            "Recommendations for me"
        ],
        required_entities=[],
        optional_entities=[EntityType.USER_ID],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "view_featured": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.VIEW_FEATURED,
        description="User wants to view featured products",
        example_phrases=[
            "Show me featured items",
            "View featured products",
            "See what's featured",
            "Display featured items",
            "Show featured products",
            "View highlighted items",
            "See editor's picks",
            "Show me spotlight items",
            "Today's featured picks",
            "Featured selections"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    "view_promotions": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.VIEW_PROMOTIONS,
        description="User wants to view promotional offers and deals",
        example_phrases=[
            "Show me promotions",
            "View current deals",
            "See special offers",
            "Display promotions",
            "Show me deals",
            "View offers",
            "See what's on promotion",
            "Show me specials",
            "Active promo offers",
            "All running deals"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    # 1.3 Search Filters & Facets
    "filter_price": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.FILTER_PRICE,
        description="User wants to filter products by price range",
        example_phrases=[
            "Filter by price",
            "Show me under $100",
            "Filter price range",
            "Between $50 and $200",
            "Under $500",
            "Over $1000",
            "Price filter",
            "Show cheaper options",
            "Limit results to this budget",
            "Only items within this price"
        ],
        required_entities=[EntityType.PRICE_RANGE],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "filter_rating": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.FILTER_RATING,
        description="User wants to filter products by rating",
        example_phrases=[
            "Filter by rating",
            "Show me 4+ stars",
            "Only highly rated",
            "Filter by reviews",
            "4 star and above",
            "Top rated only",
            "Highly rated products",
            "Filter by stars",
            "Only show items above this rating",
            "Minimum 4 stars"
        ],
        required_entities=[EntityType.REVIEW_RATING],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "filter_brand": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.FILTER_BRAND,
        description="User wants to filter products by brand",
        example_phrases=[
            "Filter by brand",
            "Show only Apple",
            "Nike products only",
            "Filter by manufacturer",
            "Samsung only",
            "Brand filter",
            "Show me this brand",
            "Filter to this brand",
            "Only this brand",
            "Restrict results to brand"
        ],
        required_entities=[EntityType.BRAND],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "filter_availability": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.FILTER_AVAILABILITY,
        description="User wants to filter products by availability",
        example_phrases=[
            "Show only in stock",
            "Filter by availability",
            "In stock only",
            "Available products",
            "Show me what's available",
            "Filter stock status",
            "Only available items",
            "Show in stock items",
            "Exclude out of stock",
            "Available for purchase"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "filter_color": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.FILTER_COLOR,
        description="User wants to filter products by color",
        example_phrases=[
            "Filter by color",
            "Show only red",
            "Black items only",
            "Color filter",
            "Blue products only",
            "Filter to this color",
            "Show me this color",
            "Only white items",
            "Restrict to this colour",
            "Items in selected color"
        ],
        required_entities=[EntityType.COLOR],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "filter_size": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.FILTER_SIZE,
        description="User wants to filter products by size",
        example_phrases=[
            "Filter by size",
            "Show only large",
            "Size filter",
            "Medium only",
            "Filter to this size",
            "Show me this size",
            "Only size 10",
            "Large items only",
            "Restrict to this size",
            "Items in selected size"
        ],
        required_entities=[EntityType.SIZE],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "filter_discount": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.FILTER_DISCOUNT,
        description="User wants to filter products by discount status",
        example_phrases=[
            "Show only discounted",
            "Filter by discount",
            "On sale only",
            "Discounted items",
            "Filter to sales",
            "Show me deals only",
            "Only sale items",
            "Filter by offers",
            "Items with markdowns",
            "Show discounted products only"
        ],
        required_entities=[],
        optional_entities=[EntityType.CATEGORY],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "apply_filters": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.APPLY_FILTERS,
        description="User wants to apply current filter settings",
        example_phrases=[
            "Apply filters",
            "Use these filters",
            "Apply my selections",
            "Filter now",
            "Apply current filters",
            "Use filters",
            "Apply all filters",
            "Filter with these settings",
            "Run with selected filters",
            "Apply selected options"
        ],
        required_entities=[],
        optional_entities=[EntityType.FILTER_CRITERIA],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    "remove_filters": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.REMOVE_FILTERS,
        description="User wants to remove current filter settings",
        example_phrases=[
            "Remove filters",
            "Clear filters",
            "Remove all filters",
            "Reset filters",
            "Clear my filters",
            "Remove current filters",
            "Clear all filters",
            "Reset my selections",
            "Clear every filter",
            "Reset applied filters"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "clear_filter": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.CLEAR_FILTER,
        description="User wants to clear a specific filter",
        example_phrases=[
            "Clear this filter",
            "Remove this filter",
            "Clear price filter",
            "Remove brand filter",
            "Clear color filter",
            "Remove size filter",
            "Clear this selection",
            "Remove this option",
            "Clear selected filter",
            "Delete this filter"
        ],
        required_entities=[],
        optional_entities=[EntityType.FILTER_CRITERIA],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),

    # 1.4 Sorting & View Options
    "sort_price": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SORT_PRICE,
        description="User wants to sort products by price",
        example_phrases=[
            "Sort by price",
            "Order by price",
            "Price low to high",
            "Sort cheapest first",
            "Price high to low",
            "Sort by cost",
            "Order by price range",
            "Sort by price ascending",
            "Sort by price descending",
            "Cheapest first / Most expensive first"
        ],
        required_entities=[],
        optional_entities=[EntityType.SORT_CRITERIA],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "sort_popularity": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SORT_POPULARITY,
        description="User wants to sort products by popularity",
        example_phrases=[
            "Sort by popularity",
            "Most popular first",
            "Order by popularity",
            "Sort by bestsellers",
            "Most bought first",
            "Sort by sales",
            "Popular items first",
            "Order by demand",
            "Trending first",
            "Most viewed first"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "sort_newest": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SORT_NEWEST,
        description="User wants to sort products by newest first",
        example_phrases=[
            "Sort by newest",
            "Newest first",
            "Order by newest",
            "Sort by latest",
            "Most recent first",
            "Sort by date",
            "Newest items first",
            "Order by release date",
            "Latest arrivals first",
            "Newly added first"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "sort_rating": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SORT_RATING,
        description="User wants to sort products by rating",
        example_phrases=[
            "Sort by rating",
            "Highest rated first",
            "Order by rating",
            "Sort by stars",
            "Best rated first",
            "Sort by reviews",
            "Top rated first",
            "Order by score",
            "Most stars first",
            "Highest score first"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "view_grid": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.VIEW_GRID,
        description="User wants to view products in grid layout",
        example_phrases=[
            "Show in grid view",
            "Grid layout",
            "View as grid",
            "Grid format",
            "Show grid view",
            "Switch to grid",
            "Grid display",
            "View grid style",
            "Use grid layout",
            "Display in grid"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.75,
        priority=IntentPriority.LOW
    ),

    "view_list": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.VIEW_LIST,
        description="User wants to view products in list layout",
        example_phrases=[
            "Show in list view",
            "List layout",
            "View as list",
            "List format",
            "Show list view",
            "Switch to list",
            "List display",
            "View list style",
            "Use list layout",
            "Display in list"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.75,
        priority=IntentPriority.LOW
    ),

    "search_material": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SEARCH_MATERIAL,
        description="User wants to find products made of specific materials",
        example_phrases=[
            "Show me cotton shirts",
            "Find leather shoes",
            "Search for wooden furniture",
            "Products made of steel",
            "Silk dresses",
            "Plastic containers",
            "Metal tools",
            "Glass products",
            "Ceramic items",
            "Fabric materials",
            "Products by material type",
            "Items made from this material"
        ],
        required_entities=[EntityType.MATERIAL],
        optional_entities=[EntityType.CATEGORY, EntityType.BRAND],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "sort_relevance": IntentDefinition(
        category=IntentCategory.SEARCH_DISCOVERY,
        action_code=ActionCode.SORT_RELEVANCE,
        description="User wants to sort search results by relevance",
        example_phrases=[
            "Sort by relevance",
            "Most relevant first",
            "Show most relevant results",
            "Order by relevance",
            "Best matches first",
            "Most relevant items",
            "Sort by best match",
            "Relevance order",
            "Highest relevance first",
            "Order by best match"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.75,
        priority=IntentPriority.MEDIUM
    ),
}