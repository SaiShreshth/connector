"""Reference data enumerating intent categories and canonical action codes."""

from __future__ import annotations

from typing import Dict, List


INTENT_CATEGORIES: Dict[str, List[str]] = {
    "SEARCH_DISCOVERY": [
        "SEARCH_PRODUCT",
        "SEARCH_CATEGORY",
        "FILTER_PRICE",
        "SORT_POPULARITY",
    ],
    "PRODUCT_DETAILS": [
        "PRODUCT_INFO",
        "PRODUCT_AVAILABILITY",
        "PRODUCT_VARIANTS",
        "PRODUCT_REVIEWS",
    ],
    "CART_WISHLIST": [
        "ADD_TO_CART",
        "REMOVE_FROM_CART",
        "VIEW_CART",
        "ADD_TO_WISHLIST",
    ],
    "CHECKOUT_PAYMENT": [
        "CHECKOUT_INITIATE",
        "PAYMENT_OPTIONS",
        "APPLY_COUPON",
        "PAYMENT_FAILED_HELP",
    ],
    "ORDERS_FULFILLMENT": [
        "ORDER_STATUS",
        "TRACK_SHIPMENT",
        "ORDER_CANCEL",
        "INITIATE_RETURN",
    ],
    "ACCOUNT_PROFILE": [
        "LOGIN",
        "RESET_PASSWORD",
        "UPDATE_PROFILE",
        "DELETE_ACCOUNT",
    ],
    "SUPPORT_HELP": [
        "CONTACT_SUPPORT",
        "HELP_GENERAL",
        "REPORT_ISSUE",
        "FAQ_SHIPPING",
    ],
    "PROMOTIONS_LOYALTY": [
        "VIEW_PROMOTIONS",
        "APPLY_PROMO",
        "LOYALTY_POINTS",
        "REDEEM_POINTS",
    ],
    "RETURNS_REFUNDS": [
        "REFUND_REQUEST",
        "RETURN_STATUS",
        "VIEW_RETURN_POLICY",
        "WARRANTY_CLAIM",
    ],
    "REVIEWS_RATINGS": [
        "ADD_REVIEW",
        "VIEW_REVIEWS",
        "EDIT_REVIEW",
        "FLAG_REVIEW",
    ],
    "NOTIFICATIONS_SUBSCRIPTIONS": [
        "SUBSCRIBE_NOTIFICATIONS",
        "UNSUBSCRIBE_NOTIFICATIONS",
        "VIEW_NOTIFICATIONS",
        "MARK_NOTIFICATION_READ",
    ],
    "ANALYTICS_TRACKING": [
        "VIEW_ANALYTICS",
        "TRACK_USER_ACTION",
        "EXPORT_ANALYTICS",
        "TRACK_CONVERSION",
    ],
    "PERSONALIZATION": [
        "PERSONALIZED_RECOMMENDATIONS",
        "RECOMMEND_SIMILAR",
        "SAVE_PREFERENCES",
        "PERSONALIZED_OFFERS",
    ],
    "SECURITY_FRAUD": [
        "ACCOUNT_SECURITY",
        "TWO_FACTOR_SETUP",
        "REPORT_FRAUD",
        "LOCK_ACCOUNT",
    ],
}
"""All categories the LLM classifier must be able to recognize."""


def get_supported_categories() -> List[str]:
    """Return the ordered list of supported categories."""

    return list(INTENT_CATEGORIES.keys())


__all__ = ["INTENT_CATEGORIES", "get_supported_categories"]
