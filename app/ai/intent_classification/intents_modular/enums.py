from enum import Enum

class IntentCategory(Enum):
    # =============================================================================
    # PRODUCTION-GRADE ECOMMERCE CATEGORIES (15 Core Categories)
    # =============================================================================
    
    # 1. Core Product Discovery & Search
    SEARCH_DISCOVERY = "SEARCH_DISCOVERY"
    
    # 2. Product Information & Details
    PRODUCT_DETAILS = "PRODUCT_DETAILS"
    
    # 3. Shopping Cart & Wishlist Management
    CART_WISHLIST = "CART_WISHLIST"
    
    # 4. Checkout & Payment Processing
    CHECKOUT_PAYMENT = "CHECKOUT_PAYMENT"
    
    # 5. Order Management & Fulfillment
    ORDERS_FULFILLMENT = "ORDERS_FULFILLMENT"
    
    # 6. User Account & Profile Management
    ACCOUNT_PROFILE = "ACCOUNT_PROFILE"
    
    # 7. Customer Support & Help
    SUPPORT_HELP = "SUPPORT_HELP"
    
    # 8. Promotions, Offers & Loyalty Programs
    PROMOTIONS_LOYALTY = "PROMOTIONS_LOYALTY"
    
    # 9. Returns & Refunds
    RETURNS_REFUNDS = "RETURNS_REFUNDS"
    
    # 10. Product Reviews & Ratings
    REVIEWS_RATINGS = "REVIEWS_RATINGS"
    
    # 11. Notifications & Subscriptions
    NOTIFICATIONS_SUBSCRIPTIONS = "NOTIFICATIONS_SUBSCRIPTIONS"
    
    # 12. Analytics & Tracking
    ANALYTICS_TRACKING = "ANALYTICS_TRACKING"
    
    # 13. Personalization & Recommendations
    PERSONALIZATION = "PERSONALIZATION"
    
    # 14. Security & Fraud Prevention
    SECURITY_FRAUD = "SECURITY_FRAUD"


class ActionCode:
    # =============================================================================
    # PRODUCTION-GRADE ECOMMERCE ACTION CODES (150+ Essential Codes)
    # =============================================================================
    # Complete coverage for enterprise-level e-commerce platform
    
    # 1️⃣ SEARCH & DISCOVERY (25 codes)
    SEARCH_PRODUCT = "SEARCH_PRODUCT"
    SEARCH_CATEGORY = "SEARCH_CATEGORY"
    SEARCH_BRAND = "SEARCH_BRAND"
    SEARCH_PRICE_RANGE = "SEARCH_PRICE_RANGE"
    SEARCH_COLOR = "SEARCH_COLOR"
    SEARCH_SIZE = "SEARCH_SIZE"
    SEARCH_MATERIAL = "SEARCH_MATERIAL"
    SEARCH_BESTSELLERS = "SEARCH_BESTSELLERS"
    SEARCH_NEW_ARRIVALS = "SEARCH_NEW_ARRIVALS"
    SEARCH_FEATURED = "SEARCH_FEATURED"
    SEARCH_TRENDING = "SEARCH_TRENDING"
    SEARCH_ON_SALE = "SEARCH_ON_SALE"
    SEARCH_RECENT = "SEARCH_RECENT"
    SEARCH_SIMILAR = "SEARCH_SIMILAR"
    SEARCH_DISCOUNT = "SEARCH_DISCOUNT"
    SEARCH_RATING = "SEARCH_RATING"
    
    # Filters
    FILTER_PRICE = "FILTER_PRICE"
    FILTER_RATING = "FILTER_RATING"
    FILTER_AVAILABILITY = "FILTER_AVAILABILITY"
    FILTER_BRAND = "FILTER_BRAND"
    FILTER_COLOR = "FILTER_COLOR"
    FILTER_SIZE = "FILTER_SIZE"
    FILTER_DISCOUNT = "FILTER_DISCOUNT"
    APPLY_FILTERS = "APPLY_FILTERS"
    REMOVE_FILTERS = "REMOVE_FILTERS"
    CLEAR_FILTER = "CLEAR_FILTER"
    
    # Sorting
    SORT_PRICE = "SORT_PRICE"
    SORT_POPULARITY = "SORT_POPULARITY"
    SORT_NEWEST = "SORT_NEWEST"
    SORT_RATING = "SORT_RATING"
    SORT_RELEVANCE = "SORT_RELEVANCE"
    
    # View Options
    VIEW_GRID = "VIEW_GRID"
    VIEW_LIST = "VIEW_LIST"

    # 2️⃣ PRODUCT DETAILS (20 codes)
    PRODUCT_INFO = "PRODUCT_INFO"
    PRODUCT_DETAILS = "PRODUCT_DETAILS"
    PRODUCT_AVAILABILITY = "PRODUCT_AVAILABILITY"
    PRODUCT_PRICE = "PRODUCT_PRICE"
    PRODUCT_VARIANTS = "PRODUCT_VARIANTS"
    PRODUCT_REVIEWS = "PRODUCT_REVIEWS"
    PRODUCT_IMAGES = "PRODUCT_IMAGES"
    PRODUCT_VIDEO = "PRODUCT_VIDEO"
    PRODUCT_DIMENSIONS = "PRODUCT_DIMENSIONS"
    PRODUCT_WEIGHT = "PRODUCT_WEIGHT"
    PRODUCT_SKU_INFO = "PRODUCT_SKU_INFO"
    PRODUCT_COLOR_OPTIONS = "PRODUCT_COLOR_OPTIONS"
    PRODUCT_SIZE_OPTIONS = "PRODUCT_SIZE_OPTIONS"
    PRODUCT_WARRANTY = "PRODUCT_WARRANTY"
    PRODUCT_FAQ = "PRODUCT_FAQ"
    PRODUCT_COMPARISON = "PRODUCT_COMPARISON"
    PRODUCT_ALTERNATIVES = "PRODUCT_ALTERNATIVES"
    PRODUCT_RECOMMENDATION = "PRODUCT_RECOMMENDATION"
    RELATED_ACCESSORIES = "RELATED_ACCESSORIES"
    PRODUCT_SHARE = "PRODUCT_SHARE"
    REQUEST_QUOTE = "REQUEST_QUOTE"

    # 3️⃣ CART & WISHLIST (15 codes)
    ADD_TO_CART = "ADD_TO_CART"
    REMOVE_FROM_CART = "REMOVE_FROM_CART"
    UPDATE_CART_QUANTITY = "UPDATE_CART_QUANTITY"
    VIEW_CART = "VIEW_CART"
    EMPTY_CART = "EMPTY_CART"
    CART_TOTAL = "CART_TOTAL"
    CART_ITEM_DETAILS = "CART_ITEM_DETAILS"
    SAVE_FOR_LATER = "SAVE_FOR_LATER"
    MOVE_TO_CART = "MOVE_TO_CART"
    MOVE_TO_WISHLIST = "MOVE_TO_WISHLIST"
    GIFT_WRAP = "GIFT_WRAP"
    ADD_NOTE = "ADD_NOTE"
    VIEW_RECENTLY_VIEWED = "VIEW_RECENTLY_VIEWED"
    CLEAR_RECENTLY_VIEWED = "CLEAR_RECENTLY_VIEWED"
    
    # Wishlist
    ADD_TO_WISHLIST = "ADD_TO_WISHLIST"
    REMOVE_FROM_WISHLIST = "REMOVE_FROM_WISHLIST"
    VIEW_WISHLIST = "VIEW_WISHLIST"

    # 4️⃣ CHECKOUT & PAYMENT (20 codes)
    CHECKOUT_INITIATE = "CHECKOUT_INITIATE"
    CHECKOUT_CONFIRM = "CHECKOUT_CONFIRM"
    CHECKOUT_ADDRESS = "CHECKOUT_ADDRESS"
    CHECKOUT_PAYMENT = "CHECKOUT_PAYMENT"
    
    # Address Management
    ADD_NEW_ADDRESS = "ADD_NEW_ADDRESS"
    UPDATE_ADDRESS = "UPDATE_ADDRESS"
    CHANGE_ADDRESS = "CHANGE_ADDRESS"
    DELETE_ADDRESS = "DELETE_ADDRESS"
    
    # Payment
    PAYMENT_OPTIONS = "PAYMENT_OPTIONS"
    CHANGE_PAYMENT_METHOD = "CHANGE_PAYMENT_METHOD"
    UPDATE_PAYMENT_METHOD = "UPDATE_PAYMENT_METHOD"
    PAYMENT_SUCCESS = "PAYMENT_SUCCESS"
    PAYMENT_FAILED_HELP = "PAYMENT_FAILED_HELP"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAYMENT_RECEIPT = "PAYMENT_RECEIPT"
    PAYMENT_HELP = "PAYMENT_HELP"
    
    # Gift Cards
    GIFT_CARD_BALANCE = "GIFT_CARD_BALANCE"
    GIFT_CARD_REDEEM = "GIFT_CARD_REDEEM"
    APPLY_GIFT_CARD = "APPLY_GIFT_CARD"
    
    # Coupons
    APPLY_COUPON = "APPLY_COUPON"
    REMOVE_COUPON = "REMOVE_COUPON"

    # 5️⃣ ORDERS & FULFILLMENT (20 codes)
    ORDER_STATUS = "ORDER_STATUS"
    ORDER_HISTORY = "ORDER_HISTORY"
    ORDER_CANCEL = "ORDER_CANCEL"
    ORDER_MODIFY = "ORDER_MODIFY"
    ORDER_RETURN = "ORDER_RETURN"
    ORDER_INVOICE = "ORDER_INVOICE"
    REORDER = "REORDER"
    
    # Shipping & Delivery
    TRACK_SHIPMENT = "TRACK_SHIPMENT"
    DELIVERY_STATUS = "DELIVERY_STATUS"
    DELIVERY_ESTIMATE = "DELIVERY_ESTIMATE"
    SCHEDULE_DELIVERY = "SCHEDULE_DELIVERY"
    CHANGE_DELIVERY_DATE = "CHANGE_DELIVERY_DATE"
    SHIPMENT_DETAILS = "SHIPMENT_DETAILS"
    SHIPPING_COST = "SHIPPING_COST"
    SHIPPING_OPTIONS = "SHIPPING_OPTIONS"
    
    # Returns
    INITIATE_RETURN = "INITIATE_RETURN"
    CANCEL_RETURN = "CANCEL_RETURN"
    RETURN_STATUS = "RETURN_STATUS"
    TRACK_RETURN = "TRACK_RETURN"
    RETURN_PICKUP_SCHEDULE = "RETURN_PICKUP_SCHEDULE"

    # 6️⃣ ACCOUNT & PROFILE (15 codes)
    CREATE_ACCOUNT = "CREATE_ACCOUNT"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    LOGIN_STATUS = "LOGIN_STATUS"
    LOGIN_WITH_SOCIAL = "LOGIN_WITH_SOCIAL"
    FORGOT_PASSWORD = "FORGOT_PASSWORD"
    RESET_PASSWORD = "RESET_PASSWORD"
    CHANGE_PASSWORD = "CHANGE_PASSWORD"
    UPDATE_PROFILE = "UPDATE_PROFILE"
    VIEW_PROFILE = "VIEW_PROFILE"
    DELETE_ACCOUNT = "DELETE_ACCOUNT"
    VERIFY_EMAIL = "VERIFY_EMAIL"
    RESEND_VERIFICATION = "RESEND_VERIFICATION"
    PROFILE_PICTURE_UPDATE = "PROFILE_PICTURE_UPDATE"
    PREFERENCES_UPDATE = "PREFERENCES_UPDATE"

    # 7️⃣ SUPPORT & HELP (20 codes)
    CONTACT_SUPPORT = "CONTACT_SUPPORT"
    HELP_GENERAL = "HELP_GENERAL"
    LIVE_CHAT = "LIVE_CHAT"
    CALL_SUPPORT = "CALL_SUPPORT"
    REQUEST_CALLBACK = "REQUEST_CALLBACK"
    FEEDBACK_SUBMIT = "FEEDBACK_SUBMIT"
    REPORT_ISSUE = "REPORT_ISSUE"
    REPORT_PRODUCT = "REPORT_PRODUCT"
    SECURITY_HELP = "SECURITY_HELP"
    PAYMENT_HELP = "PAYMENT_HELP"
    
    # FAQ Categories
    FAQ_SHIPPING = "FAQ_SHIPPING"
    FAQ_PAYMENT = "FAQ_PAYMENT"
    FAQ_RETURNS = "FAQ_RETURNS"
    FAQ_REFUND = "FAQ_REFUND"
    FAQ_ACCOUNT = "FAQ_ACCOUNT"
    FAQ_COUPON = "FAQ_COUPON"
    FAQ_DELIVERY = "FAQ_DELIVERY"
    FAQ_TECH_SUPPORT = "FAQ_TECH_SUPPORT"

    # 8️⃣ PROMOTIONS & LOYALTY (15 codes)
    VIEW_PROMOTIONS = "VIEW_PROMOTIONS"
    VIEW_COUPONS = "VIEW_COUPONS"
    APPLY_PROMO = "APPLY_PROMO"
    REMOVE_PROMO = "REMOVE_PROMO"
    VIEW_SALE_COLLECTION = "VIEW_SALE_COLLECTION"
    FLASH_SALE_INFO = "FLASH_SALE_INFO"
    PRICE_DROP_ALERTS = "PRICE_DROP_ALERTS"
    VIEW_BESTSELLERS = "VIEW_BESTSELLERS"
    VIEW_NEW_ARRIVALS = "VIEW_NEW_ARRIVALS"
    VIEW_FEATURED = "VIEW_FEATURED"
    VIEW_RECOMMENDED = "VIEW_RECOMMENDED"
    VIEW_GIFT_OPTIONS = "VIEW_GIFT_OPTIONS"
    
    # Loyalty Programs
    LOYALTY_POINTS = "LOYALTY_POINTS"
    REDEEM_POINTS = "REDEEM_POINTS"
    CHECK_REWARDS = "CHECK_REWARDS"
    CLAIM_REWARD = "CLAIM_REWARD"

    # 9️⃣ RETURNS & REFUNDS (10 codes)
    VIEW_RETURN_POLICY = "VIEW_RETURN_POLICY"
    PRODUCT_RETURN_ELIGIBILITY = "PRODUCT_RETURN_ELIGIBILITY"
    REFUND_REQUEST = "REFUND_REQUEST"
    REFUND_COMPLETE = "REFUND_COMPLETE"
    VIEW_WARRANTY_POLICY = "VIEW_WARRANTY_POLICY"
    WARRANTY_CLAIM = "WARRANTY_CLAIM"
    WARRANTY_STATUS = "WARRANTY_STATUS"

    # 🔟 REVIEWS & RATINGS (10 codes)
    ADD_REVIEW = "ADD_REVIEW"
    VIEW_REVIEWS = "VIEW_REVIEWS"
    EDIT_REVIEW = "EDIT_REVIEW"
    DELETE_REVIEW = "DELETE_REVIEW"
    PRODUCT_AVERAGE_REVIEW = "PRODUCT_AVERAGE_REVIEW"
    PRODUCT_RATING = "PRODUCT_RATING"
    FLAG_REVIEW = "FLAG_REVIEW"
    REVIEW_RECOMMENDATION = "REVIEW_RECOMMENDATION"

    # 1️⃣1️⃣ NOTIFICATIONS & SUBSCRIPTIONS (8 codes)
    SUBSCRIBE_NOTIFICATIONS = "SUBSCRIBE_NOTIFICATIONS"
    UNSUBSCRIBE_NOTIFICATIONS = "UNSUBSCRIBE_NOTIFICATIONS"
    NOTIFICATION_PREFERENCES = "NOTIFICATION_PREFERENCES"
    VIEW_NOTIFICATIONS = "VIEW_NOTIFICATIONS"
    MARK_NOTIFICATION_READ = "MARK_NOTIFICATION_READ"
    DELETE_NOTIFICATION = "DELETE_NOTIFICATION"
    SUBSCRIBE_PRODUCT = "SUBSCRIBE_PRODUCT"
    UNSUBSCRIBE_PRODUCT = "UNSUBSCRIBE_PRODUCT"

    # 1️⃣2️⃣ ANALYTICS & TRACKING (8 codes)
    TRACK_USER_ACTION = "TRACK_USER_ACTION"
    TRACK_SEARCH_QUERY = "TRACK_SEARCH_QUERY"
    TRACK_CART_ABANDONMENT = "TRACK_CART_ABANDONMENT"
    VIEW_ANALYTICS = "VIEW_ANALYTICS"
    EXPORT_ANALYTICS = "EXPORT_ANALYTICS"
    LISTEN_BEHAVIOR = "LISTEN_BEHAVIOR"
    TRACK_CONVERSION = "TRACK_CONVERSION"
    TRACK_PERFORMANCE = "TRACK_PERFORMANCE"

    # 1️⃣4️⃣ PERSONALIZATION & RECOMMENDATIONS (10 codes)
    PERSONALIZED_RECOMMENDATIONS = "PERSONALIZED_RECOMMENDATIONS"
    RECOMMEND_SIMILAR = "RECOMMEND_SIMILAR"
    RECOMMEND_COMPLEMENTS = "RECOMMEND_COMPLEMENTS"
    VIEW_RECOMMENDED = "VIEW_RECOMMENDED"
    SAVE_PREFERENCES = "SAVE_PREFERENCES"
    UPDATE_PREFERENCES = "UPDATE_PREFERENCES"
    VIEW_PREFERENCES = "VIEW_PREFERENCES"
    CLEAR_PREFERENCES = "CLEAR_PREFERENCES"
    PERSONALIZED_SEARCH = "PERSONALIZED_SEARCH"
    PERSONALIZED_OFFERS = "PERSONALIZED_OFFERS"

    # 1️⃣4️⃣ SECURITY & FRAUD (8 codes)
    ACCOUNT_SECURITY = "ACCOUNT_SECURITY"
    TWO_FACTOR_SETUP = "TWO_FACTOR_SETUP"
    TWO_FACTOR_VERIFY = "TWO_FACTOR_VERIFY"
    REPORT_FRAUD = "REPORT_FRAUD"
    FRAUD_DETECTION_HELP = "FRAUD_DETECTION_HELP"
    LOCK_ACCOUNT = "LOCK_ACCOUNT"
    UNLOCK_ACCOUNT = "UNLOCK_ACCOUNT"
    SECURITY_ALERT = "SECURITY_ALERT"


class IntentPriority(Enum):
    """Priority levels for intent processing."""
    CRITICAL = 1    # Checkout, payment, critical actions
    HIGH = 2        # Search, cart operations
    MEDIUM = 3      # Browsing, recommendations
    LOW = 4         # General interaction
    FALLBACK = 5    # Unknown intents


class EntityType(Enum):
    """Types of entities that can be extracted from user input."""
    # Core Product Entities
    PRODUCT_QUERY = "product_query"
    PRODUCT_ID = "product_id"
    PRODUCT_NAME = "product_name"
    PRODUCT_SKU = "product_sku"
    CATEGORY = "category"
    SUBCATEGORY = "subcategory"
    BRAND = "brand"
    PRICE_RANGE = "price_range"
    PRICE = "price"
    QUANTITY = "quantity"
    
    # Product Attributes
    SIZE = "size"
    COLOR = "color"
    MATERIAL = "material"
    WEIGHT = "weight"
    DIMENSIONS = "dimensions"
    WARRANTY = "warranty"
    
    # Order & Transaction Entities
    ORDER_ID = "order_id"
    ORDER_NUMBER = "order_number"
    TRANSACTION_ID = "transaction_id"
    COUPON_CODE = "coupon_code"
    PROMO_CODE = "promo_code"
    PAYMENT_METHOD = "payment_method"
    PAYMENT_STATUS = "payment_status"
    
    # User & Account Entities
    USER_ID = "user_id"
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    ADDRESS = "address"
    ZIP_CODE = "zip_code"
    COUNTRY = "country"
    STATE = "state"
    CITY = "city"
    
    # Search & Filter Entities
    FILTER_CRITERIA = "filter_criteria"
    SORT_CRITERIA = "sort_criteria"
    SEARCH_TERM = "search_term"
    KEYWORD = "keyword"
    
    # Time & Date Entities
    DATE_RANGE = "date_range"
    START_DATE = "start_date"
    END_DATE = "end_date"
    DELIVERY_DATE = "delivery_date"
    ORDER_DATE = "order_date"
    
    # Support & Communication Entities
    REASON = "reason"
    URGENCY = "urgency"
    PRIORITY = "priority"
    QUESTION = "question"
    TOPIC = "topic"
    ISSUE_TYPE = "issue_type"
    ISSUE_DESCRIPTION = "issue_description"
    TICKET_ID = "ticket_id"
    
    # Review & Rating Entities
    REVIEW_RATING = "review_rating"
    REVIEW_TEXT = "review_text"
    REVIEW_TITLE = "review_title"
    REVIEW_ID = "review_id"
    
    # Return & Refund Entities
    RETURN_REASON = "return_reason"
    RETURN_ID = "return_id"
    REFUND_AMOUNT = "refund_amount"
    REFUND_METHOD = "refund_method"
    
    # Shipping & Delivery Entities
    SHIPPING_METHOD = "shipping_method"
    TRACKING_NUMBER = "tracking_number"
    DELIVERY_ADDRESS = "delivery_address"
    PICKUP_LOCATION = "pickup_location"
    
    # Preferences & Settings Entities
    PREFERENCES = "preferences"
    SETTINGS = "settings"
    NOTIFICATION_PREFERENCES = "notification_preferences"
    LANGUAGE = "language"
    CURRENCY = "currency"
    TIMEZONE = "timezone"
    
    # Budget & Financial Entities
    BUDGET = "budget"
    MAX_PRICE = "max_price"
    MIN_PRICE = "min_price"
    DISCOUNT_AMOUNT = "discount_amount"
    TAX_AMOUNT = "tax_amount"
    SHIPPING_COST = "shipping_cost"
    TOTAL_AMOUNT = "total_amount"
    
    # Loyalty & Rewards Entities
    LOYALTY_POINTS = "loyalty_points"
    REWARD_TYPE = "reward_type"
    MEMBERSHIP_LEVEL = "membership_level"
    POINTS_BALANCE = "points_balance"
    
    # Security & Authentication Entities
    PASSWORD = "password"
    SECURITY_QUESTION = "security_question"
    SECURITY_ANSWER = "security_answer"
    VERIFICATION_CODE = "verification_code"
    TWO_FACTOR_CODE = "two_factor_code"
    
    # Analytics & Tracking Entities
    SESSION_ID = "session_id"
    USER_AGENT = "user_agent"
    IP_ADDRESS = "ip_address"
    DEVICE_TYPE = "device_type"
    BROWSER_TYPE = "browser_type"
    
    # Miscellaneous Entities
    MESSAGE = "message"
    COMMENT = "comment"
    FEEDBACK = "feedback"
    SUGGESTION = "suggestion"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    REQUEST = "request"
    INQUIRY = "inquiry"
    CONCERN = "concern"
    APPRECIATION = "appreciation"


class IntentStatus(Enum):
    """Status of intent processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ConfidenceLevel(Enum):
    """Confidence levels for intent classification."""
    VERY_HIGH = 0.95
    HIGH = 0.85
    MEDIUM = 0.70
    LOW = 0.50
    VERY_LOW = 0.30


class ProcessingMode(Enum):
    """Processing modes for intent classification."""
    REALTIME = "realtime"
    BATCH = "batch"
    STREAMING = "streaming"
    SCHEDULED = "scheduled"


class ErrorType(Enum):
    """Types of errors in intent processing."""
    CLASSIFICATION_ERROR = "classification_error"
    ENTITY_EXTRACTION_ERROR = "entity_extraction_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"
