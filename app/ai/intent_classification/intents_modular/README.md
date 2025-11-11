# Intent Classification System

## Overview

The Intent Classification System is a modular, scalable framework for understanding and categorizing user intents in an ecommerce chatbot. It provides a structured approach to mapping user queries to specific actions, enabling the chatbot to respond appropriately to customer needs.

## Architecture

The system is built around 4 core modules:

```
intents_modular/
├── enums.py          # Core enums and constants
├── models.py         # Data structures and models
├── taxonomy.py       # Intent management and operations
└── definitions/      # Intent definitions by category
    ├── __init__.py
    ├── SEARCH_DISCOVERY.py
    ├── PRODUCT_DETAILS.py
    ├── CART_WISHLIST.py
    ├── CHECKOUT_PAYMENT.py
    ├── ORDERS_FULFILLMENT.py
    ├── ACCOUNT_PROFILE.py
    ├── SUPPORT_HELP.py
    ├── PROMOTIONS_LOYALTY.py
    ├── RETURNS_REFUNDS.py
    └── REVIEWS_RATINGS.py
```

## 10 Intent Categories

The system organizes 50 essential action codes across 10 core ecommerce categories:

| Category | Action Codes | Description |
|----------|-------------|-------------|
| 🔍 **SEARCH_DISCOVERY** | 8 codes | Product search, filtering, and discovery |
| 📦 **PRODUCT_DETAILS** | 6 codes | Product information and variants |
| 🛒 **CART_WISHLIST** | 8 codes | Shopping cart and wishlist management |
| 💳 **CHECKOUT_PAYMENT** | 7 codes | Checkout flow and payment processing |
| 📋 **ORDERS_FULFILLMENT** | 7 codes | Order tracking and fulfillment |
| 👤 **ACCOUNT_PROFILE** | 5 codes | User account and authentication |
| 🆘 **SUPPORT_HELP** | 5 codes | Customer support and help |
| 🎁 **PROMOTIONS_LOYALTY** | 4 codes | Promotions and loyalty programs |
| 🔄 **RETURNS_REFUNDS** | 5 codes | Returns and refund processing |
| ⭐ **REVIEWS_RATINGS** | 5 codes | Product reviews and ratings |

## Key Components

### IntentCategory Enum
Defines the 10 core ecommerce categories:
```python
class IntentCategory(Enum):
    SEARCH_DISCOVERY = "SEARCH_DISCOVERY"
    PRODUCT_DETAILS = "PRODUCT_DETAILS"
    CART_WISHLIST = "CART_WISHLIST"
    CHECKOUT_PAYMENT = "CHECKOUT_PAYMENT"
    ORDERS_FULFILLMENT = "ORDERS_FULFILLMENT"
    ACCOUNT_PROFILE = "ACCOUNT_PROFILE"
    SUPPORT_HELP = "SUPPORT_HELP"
    PROMOTIONS_LOYALTY = "PROMOTIONS_LOYALTY"
    RETURNS_REFUNDS = "RETURNS_REFUNDS"
    REVIEWS_RATINGS = "REVIEWS_RATINGS"
```

### ActionCode Class
Contains 50 essential action codes for core ecommerce functionality:
```python
# Example action codes
SEARCH_PRODUCT = "SEARCH_PRODUCT"
ADD_TO_CART = "ADD_TO_CART"
CHECKOUT_INITIATE = "CHECKOUT_INITIATE"
ORDER_STATUS = "ORDER_STATUS"
# ... and 46 more
```

### IntentPriority Levels
```python
class IntentPriority(Enum):
    CRITICAL = 1    # Checkout, payment, critical actions
    HIGH = 2        # Search, cart operations
    MEDIUM = 3      # Browsing, recommendations
    LOW = 4         # General interaction
    FALLBACK = 5    # Unknown intents
```

### EntityType Enum
Defines entities that can be extracted from user input:
```python
PRODUCT_QUERY = "product_query"
PRODUCT_ID = "product_id"
CATEGORY = "category"
BRAND = "brand"
PRICE_RANGE = "price_range"
# ... and more
```

## Data Models

### IntentDefinition
Core structure for defining intents:
```python
@dataclass
class IntentDefinition:
    category: IntentCategory
    action_code: ActionCode
    description: str
    example_phrases: List[str]
    required_entities: List[EntityType]
    optional_entities: List[EntityType]
    confidence_threshold: float = 0.7
    priority: IntentPriority = IntentPriority.MEDIUM
```

### IntentResult
Represents classification results:
```python
@dataclass
class IntentResult:
    intent_name: str
    action_code: ActionCode
    confidence_score: float
    extracted_entities: Dict[str, Any]
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None
```

### ClassificationRequest/Response
Input/output structures for the classification process:
```python
@dataclass
class ClassificationRequest:
    user_input: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class ClassificationResponse:
    primary_intent: IntentResult
    alternative_intents: List[IntentResult]
    entities: List[EntityExtraction]
    requires_clarification: bool
    clarification_message: Optional[str] = None
```

## IntentTaxonomy

The `IntentTaxonomy` class provides comprehensive management of all intent definitions:

```python
from .taxonomy import get_intent_taxonomy

# Get the global taxonomy instance
taxonomy = get_intent_taxonomy()

# Get intents by category
search_intents = taxonomy.get_intents_by_category(IntentCategory.SEARCH_DISCOVERY)

# Get intents by priority
critical_intents = taxonomy.get_intents_by_priority(IntentPriority.CRITICAL)

# Search for intents
results = taxonomy.search_intents("product search")

# Get intent statistics
stats = taxonomy.get_intent_statistics()
print(f"Total intents: {stats['total_intents']}")
print(f"Categories: {stats['intents_by_category']}")
```

## Usage Examples

### Basic Usage
```python
from app.ai.intent_classification.intents_modular.taxonomy import get_intent_taxonomy
from app.ai.intent_classification.intents_modular.enums import IntentCategory

# Initialize taxonomy
taxonomy = get_intent_taxonomy()

# Get all search and discovery intents
search_intents = taxonomy.get_intents_by_category(IntentCategory.SEARCH_DISCOVERY)

# Get specific intent definition
intent_def = taxonomy.get_intent_by_action_code("SEARCH_PRODUCT")

# Get example phrases for an action
phrases = taxonomy.get_example_phrases_for_action("ADD_TO_CART")
```

### Working with Intent Definitions
```python
# Get intent definition
intent_def = taxonomy.get_intent_definition("search_product")

# Access intent properties
print(f"Category: {intent_def.category}")
print(f"Action Code: {intent_def.action_code}")
print(f"Description: {intent_def.description}")
print(f"Example phrases: {intent_def.example_phrases}")
print(f"Required entities: {intent_def.required_entities}")
print(f"Confidence threshold: {intent_def.confidence_threshold}")
```

### Getting Statistics
```python
# Get comprehensive statistics
stats = taxonomy.get_intent_statistics()

print(f"Total intents: {stats['total_intents']}")
print(f"Total categories: {stats['total_categories']}")
print(f"Intents by category: {stats['intents_by_category']}")
print(f"Intents by priority: {stats['intents_by_priority']}")
print(f"Average confidence threshold: {stats['average_confidence_threshold']:.2f}")
```

### Exporting Intent Definitions
```python
# Export as dictionary
intent_dict = taxonomy.export_intent_definitions("dict")

# Export as JSON
intent_json = taxonomy.export_intent_definitions("json")
```

## How to Extend

### Adding New Action Codes

1. **Add to enums.py**:
```python
class ActionCode:
    # ... existing codes ...
    NEW_ACTION = "NEW_ACTION"
```

2. **Create Intent Definition**:
```python
# In appropriate category file (e.g., PRODUCT_DETAILS.py)
"new_intent": IntentDefinition(
    category=IntentCategory.PRODUCT_DETAILS,
    action_code=ActionCode.NEW_ACTION,
    description="User wants to perform new action",
    example_phrases=[
        "I want to do new action",
        "Can you help with new action"
    ],
    required_entities=[EntityType.PRODUCT_QUERY],
    optional_entities=[EntityType.CATEGORY],
    confidence_threshold=0.8,
    priority=IntentPriority.MEDIUM
)
```

3. **Update definitions/__init__.py**:
```python
# The new intent will be automatically included in ALL_INTENT_DEFINITIONS
```

### Adding New Categories

1. **Add to IntentCategory enum**:
```python
class IntentCategory(Enum):
    # ... existing categories ...
    NEW_CATEGORY = "NEW_CATEGORY"
```

2. **Create new definition file**:
```python
# Create NEW_CATEGORY.py in definitions/
new_category_intent_definitions = {
    # Intent definitions here
}
```

3. **Update definitions/__init__.py**:
```python
from .NEW_CATEGORY import new_category_intent_definitions

ALL_INTENT_DEFINITIONS = {
    # ... existing definitions ...
    **new_category_intent_definitions,
}
```

## Best Practices

### For Intent Definitions
- **Example Phrases**: Include 5-8 diverse, natural examples
- **Entities**: Be specific about required vs optional entities
- **Confidence Thresholds**: 
  - Critical actions: 0.9+
  - High priority: 0.8-0.9
  - Medium priority: 0.7-0.8
  - Low priority: 0.6-0.7
- **Priority**: Match business importance

### For System Maintenance
- Keep action codes descriptive and consistent
- Use clear, business-focused category names
- Document any changes in intent definitions
- Test new intents with real user queries
- Monitor classification accuracy and adjust thresholds

## Related Documentation

- [Intent Definitions Guide](./definitions/README.md) - Detailed guide for creating and managing intent definitions
- [Entity Types Reference](./enums.py#EntityType) - Complete list of available entity types
- [Action Codes Reference](./enums.py#ActionCode) - All 50 essential action codes

## Support

For questions about the intent classification system:
1. Check the [Intent Definitions Guide](./definitions/README.md)
2. Review the code examples in this documentation
3. Examine existing intent definitions in the `definitions/` folder
4. Consult the `models.py` file for data structure details
