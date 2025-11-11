# Intent Definitions Guide

## Overview

Intent definitions are the core building blocks of the intent classification system. Each definition maps user queries to specific actions, providing the chatbot with the intelligence to understand and respond to customer needs.

## File Structure

The definitions are organized into 10 category files, each corresponding to a core ecommerce domain:

```
definitions/
├── __init__.py              # Merges all definitions into ALL_INTENT_DEFINITIONS
├── SEARCH_DISCOVERY.py      # Product search and discovery (8 intents)
├── PRODUCT_DETAILS.py       # Product information and details (6 intents)
├── CART_WISHLIST.py         # Shopping cart and wishlist (8 intents)
├── CHECKOUT_PAYMENT.py      # Checkout and payment processing (7 intents)
├── ORDERS_FULFILLMENT.py    # Order tracking and fulfillment (7 intents)
├── ACCOUNT_PROFILE.py       # User account management (5 intents)
├── SUPPORT_HELP.py          # Customer support and help (5 intents)
├── PROMOTIONS_LOYALTY.py    # Promotions and loyalty programs (4 intents)
├── RETURNS_REFUNDS.py       # Returns and refunds (5 intents)
└── REVIEWS_RATINGS.py       # Product reviews and ratings (5 intents)
```

## IntentDefinition Schema

Each intent definition follows this structure:

```python
@dataclass
class IntentDefinition:
    category: IntentCategory           # Which domain this intent belongs to
    action_code: ActionCode           # Unique identifier for the action
    description: str                  # Human-readable description
    example_phrases: List[str]        # Training examples for the intent
    required_entities: List[EntityType]    # Must-have entities
    optional_entities: List[EntityType]   # Nice-to-have entities
    confidence_threshold: float = 0.7     # Minimum confidence for classification
    priority: IntentPriority = IntentPriority.MEDIUM  # Business priority
```

### Field Breakdown

#### category
- **Purpose**: Groups related intents together
- **Values**: One of the 10 `IntentCategory` enum values
- **Example**: `IntentCategory.SEARCH_DISCOVERY`

#### action_code
- **Purpose**: Unique identifier for the specific action
- **Values**: One of the 50 `ActionCode` enum values
- **Example**: `ActionCode.SEARCH_PRODUCT`

#### description
- **Purpose**: Clear, concise description of what the intent does
- **Format**: "User wants to [action description]"
- **Example**: "User wants to search for specific products"

#### example_phrases
- **Purpose**: Training data for intent classification
- **Count**: 5-8 diverse examples recommended
- **Format**: Natural language phrases users might say
- **Example**: `["I'm looking for a laptop", "Show me smartphones", "Find me running shoes"]`

#### required_entities
- **Purpose**: Entities that must be present for the intent to be valid
- **Values**: List of `EntityType` enum values
- **Example**: `[EntityType.PRODUCT_QUERY]`

#### optional_entities
- **Purpose**: Entities that enhance the intent but aren't required
- **Values**: List of `EntityType` enum values
- **Example**: `[EntityType.CATEGORY, EntityType.BRAND, EntityType.PRICE_RANGE]`

#### confidence_threshold
- **Purpose**: Minimum confidence score for successful classification
- **Range**: 0.0 to 1.0
- **Guidelines**:
  - Critical actions: 0.9+
  - High priority: 0.8-0.9
  - Medium priority: 0.7-0.8
  - Low priority: 0.6-0.7

#### priority
- **Purpose**: Business importance for intent processing
- **Values**: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `FALLBACK`
- **Usage**: Affects processing order and resource allocation

## Creating New Intents

### Step 1: Choose the Right Category
Determine which of the 10 categories your intent belongs to:
- **SEARCH_DISCOVERY**: Product search, filtering, browsing
- **PRODUCT_DETAILS**: Product information, variants, specifications
- **CART_WISHLIST**: Shopping cart, wishlist, saved items
- **CHECKOUT_PAYMENT**: Checkout flow, payment processing
- **ORDERS_FULFILLMENT**: Order tracking, delivery, fulfillment
- **ACCOUNT_PROFILE**: User accounts, authentication, profiles
- **SUPPORT_HELP**: Customer support, help, FAQ
- **PROMOTIONS_LOYALTY**: Promotions, coupons, loyalty programs
- **RETURNS_REFUNDS**: Returns, refunds, exchanges
- **REVIEWS_RATINGS**: Product reviews, ratings, feedback

### Step 2: Define the Intent
```python
"intent_name": IntentDefinition(
    category=IntentCategory.CHOOSE_CATEGORY,
    action_code=ActionCode.CHOOSE_ACTION_CODE,
    description="User wants to [clear action description]",
    example_phrases=[
        "Natural phrase 1",
        "Natural phrase 2",
        "Natural phrase 3",
        "Natural phrase 4",
        "Natural phrase 5"
    ],
    required_entities=[EntityType.REQUIRED_ENTITY],
    optional_entities=[EntityType.OPTIONAL_ENTITY],
    confidence_threshold=0.8,
    priority=IntentPriority.HIGH
)
```

### Step 3: Add to Category File
Place your intent definition in the appropriate category file:
```python
# In SEARCH_DISCOVERY.py
search_discovery_intent_definitions = {
    # ... existing intents ...
    "your_new_intent": IntentDefinition(
        # ... your definition ...
    ),
}
```

### Step 4: Update Action Code (if needed)
If you need a new action code, add it to `enums.py`:
```python
class ActionCode:
    # ... existing codes ...
    YOUR_NEW_ACTION = "YOUR_NEW_ACTION"
```

## Best Practices

### Writing Good Example Phrases

✅ **Good Examples**:
```python
example_phrases=[
    "I'm looking for a laptop",           # Direct and clear
    "Show me smartphones",                 # Action-oriented
    "Find me running shoes",               # Specific product type
    "Can you help me find a camera?",      # Polite request
    "I need a coffee maker"                # Need-based
]
```

❌ **Avoid**:
```python
example_phrases=[
    "laptop",                             # Too short
    "I want to buy something",            # Too vague
    "Show me the thing that makes coffee" # Unclear
]
```

### Choosing Appropriate Entities

**Required Entities**: Only include entities that are absolutely necessary
```python
# For product search
required_entities=[EntityType.PRODUCT_QUERY]

# For brand-specific search
required_entities=[EntityType.BRAND]

# For price-based search
required_entities=[EntityType.PRICE_RANGE]
```

**Optional Entities**: Include entities that enhance the experience
```python
# For general product search
optional_entities=[EntityType.CATEGORY, EntityType.BRAND, EntityType.PRICE_RANGE]
```

### Setting Confidence Thresholds

| Priority | Threshold | Use Case |
|----------|-----------|----------|
| CRITICAL | 0.9+ | Checkout, payment, account security |
| HIGH | 0.8-0.9 | Search, cart operations, order tracking |
| MEDIUM | 0.7-0.8 | Product browsing, recommendations |
| LOW | 0.6-0.7 | General interaction, help |

### Determining Priority Levels

```python
# Critical: Essential business operations
priority=IntentPriority.CRITICAL  # CHECKOUT_INITIATE, PAYMENT_OPTIONS

# High: Core user actions
priority=IntentPriority.HIGH      # SEARCH_PRODUCT, ADD_TO_CART

# Medium: Supporting actions
priority=IntentPriority.MEDIUM    # PRODUCT_DETAILS, VIEW_REVIEWS

# Low: Nice-to-have features
priority=IntentPriority.LOW       # PRODUCT_SHARE, VIEW_PROMOTIONS
```

## Category Files Overview

### SEARCH_DISCOVERY.py (8 intents)
- Product search, category browsing, brand search
- Price range filtering, basic sorting
- Core discovery functionality

### PRODUCT_DETAILS.py (6 intents)
- Product information retrieval
- Availability checking, price display
- Product variants and reviews

### CART_WISHLIST.py (8 intents)
- Add/remove from cart and wishlist
- Cart management, coupon application
- Essential shopping functionality

### CHECKOUT_PAYMENT.py (7 intents)
- Checkout initiation and confirmation
- Address management, payment options
- Core checkout flow

### ORDERS_FULFILLMENT.py (7 intents)
- Order status and history
- Shipment tracking, delivery status
- Order cancellation and reordering

### ACCOUNT_PROFILE.py (5 intents)
- Account creation and login
- Password management
- Basic authentication

### SUPPORT_HELP.py (5 intents)
- Contact support, general help
- Issue reporting, FAQ access
- Essential support functions

### PROMOTIONS_LOYALTY.py (4 intents)
- View promotions and coupons
- Apply promotional codes
- Sale collection browsing

### RETURNS_REFUNDS.py (5 intents)
- Return policy viewing
- Return initiation and tracking
- Refund requests

### REVIEWS_RATINGS.py (5 intents)
- Add, view, edit, delete reviews
- Product rating display
- Simple review system

## Validation

Intent definitions are automatically validated when the taxonomy is initialized:

### Automatic Validation
- **Example phrases**: Must not be empty
- **Confidence threshold**: Must be between 0.0 and 1.0
- **Action codes**: Must be unique across all definitions
- **Entity types**: Must be valid EntityType enum values

### Manual Validation Checklist
- [ ] Description is clear and concise
- [ ] Example phrases are diverse and natural
- [ ] Required entities are truly necessary
- [ ] Optional entities add value
- [ ] Confidence threshold is appropriate for priority
- [ ] Action code follows naming conventions

## Integration

### How Definitions are Loaded

1. **Individual Files**: Each category file exports its definitions
2. **__init__.py**: Merges all definitions into `ALL_INTENT_DEFINITIONS`
3. **Taxonomy**: Loads all definitions when initialized
4. **Validation**: Automatic validation occurs during loading

### Accessing Definitions

```python
from app.ai.intent_classification.intents_modular.definitions import ALL_INTENT_DEFINITIONS

# Get all definitions
all_intents = ALL_INTENT_DEFINITIONS

# Get specific definition
search_intent = ALL_INTENT_DEFINITIONS["search_product"]

# Get definitions by category
from app.ai.intent_classification.intents_modular.definitions import search_discovery_intent_definitions
search_intents = search_discovery_intent_definitions
```

### Using with Taxonomy

```python
from app.ai.intent_classification.intents_modular.taxonomy import get_intent_taxonomy

taxonomy = get_intent_taxonomy()

# Get intent by action code
intent = taxonomy.get_intent_by_action_code("SEARCH_PRODUCT")

# Get intents by category
search_intents = taxonomy.get_intents_by_category(IntentCategory.SEARCH_DISCOVERY)

# Search intents
results = taxonomy.search_intents("product search")
```

## Testing New Definitions

### 1. Unit Testing
```python
def test_new_intent_definition():
    intent_def = ALL_INTENT_DEFINITIONS["your_new_intent"]
    
    assert intent_def.category == IntentCategory.EXPECTED_CATEGORY
    assert intent_def.action_code == ActionCode.EXPECTED_ACTION
    assert len(intent_def.example_phrases) >= 5
    assert 0.0 <= intent_def.confidence_threshold <= 1.0
```

### 2. Integration Testing
```python
def test_intent_classification():
    taxonomy = get_intent_taxonomy()
    
    # Test that intent can be found
    intent = taxonomy.get_intent_by_action_code("YOUR_NEW_ACTION")
    assert intent is not None
    
    # Test example phrases
    phrases = taxonomy.get_example_phrases_for_action("YOUR_NEW_ACTION")
    assert len(phrases) > 0
```

### 3. Real-world Testing
- Test with actual user queries
- Monitor classification accuracy
- Adjust confidence thresholds as needed
- Gather feedback on intent coverage

## Troubleshooting

### Common Issues

**Intent not found**:
- Check that the intent is added to the correct category file
- Verify that `__init__.py` includes the new definition
- Ensure the action code exists in `enums.py`

**Low classification accuracy**:
- Review and improve example phrases
- Adjust confidence threshold
- Check entity requirements
- Add more diverse training examples

**Validation errors**:
- Check that all required fields are provided
- Verify entity types are valid
- Ensure confidence threshold is in valid range
- Check for duplicate action codes

### Debugging Tips

```python
# Check if intent exists
taxonomy = get_intent_taxonomy()
intent = taxonomy.get_intent_definition("your_intent_name")
if intent is None:
    print("Intent not found - check definition")

# Validate intent definition
try:
    intent_def = IntentDefinition(...)
    print("Intent definition is valid")
except ValueError as e:
    print(f"Validation error: {e}")

# Check taxonomy statistics
stats = taxonomy.get_intent_statistics()
print(f"Total intents: {stats['total_intents']}")
```

## Related Documentation

- [Main Intent Classification README](../README.md) - Overall system documentation
- [Entity Types Reference](../enums.py#EntityType) - Available entity types
- [Action Codes Reference](../enums.py#ActionCode) - All action codes
- [Models Documentation](../models.py) - Data structure details
