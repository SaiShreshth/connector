# Init file for definitions package

from .SEARCH_DISCOVERY import search_discovery_intent_definitions
from .PRODUCT_DETAILS import product_details_intent_definitions
from .CART_WISHLIST import cart_wishlist_intent_definitions
from .CHECKOUT_PAYMENT import checkout_payment_intent_definitions
from .ORDERS_FULFILLMENT import orders_fulfillment_intent_definitions
from .ACCOUNT_PROFILE import account_profile_intent_definitions
from .SUPPORT_HELP import support_help_intent_definitions
from .PROMOTIONS_LOYALTY import promotions_loyalty_intent_definitions
from .RETURNS_REFUNDS import returns_refunds_intent_definitions
from .REVIEWS_RATINGS import reviews_ratings_intent_definitions
from .NOTIFICATIONS_SUBSCRIPTIONS import notifications_subscriptions_intent_definitions
from .ANALYTICS_TRACKING import analytics_tracking_intent_definitions
from .PERSONALIZATION import personalization_intent_definitions
from .SECURITY_FRAUD import security_fraud_intent_definitions

# Merge all intent definitions into a single dictionary
ALL_INTENT_DEFINITIONS = {
    **search_discovery_intent_definitions,
    **product_details_intent_definitions,
    **cart_wishlist_intent_definitions,
    **checkout_payment_intent_definitions,
    **orders_fulfillment_intent_definitions,
    **account_profile_intent_definitions,
    **support_help_intent_definitions,
    **promotions_loyalty_intent_definitions,
    **returns_refunds_intent_definitions,
    **reviews_ratings_intent_definitions,
    **notifications_subscriptions_intent_definitions,
    **analytics_tracking_intent_definitions,
    **personalization_intent_definitions,
    **security_fraud_intent_definitions,
}

__all__ = [
    "ALL_INTENT_DEFINITIONS",
    "search_discovery_intent_definitions",
    "product_details_intent_definitions", 
    "cart_wishlist_intent_definitions",
    "checkout_payment_intent_definitions",
    "orders_fulfillment_intent_definitions",
    "account_profile_intent_definitions",
    "support_help_intent_definitions",
    "promotions_loyalty_intent_definitions",
    "returns_refunds_intent_definitions",
    "reviews_ratings_intent_definitions",
    "notifications_subscriptions_intent_definitions",
    "analytics_tracking_intent_definitions",
    "personalization_intent_definitions",
    "security_fraud_intent_definitions",
]