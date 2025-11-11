from ..models import IntentDefinition
from ..enums import IntentCategory, ActionCode, IntentPriority, EntityType

reviews_ratings_intent_definitions = {
    # 10.0 Reviews & Ratings
    "add_review": IntentDefinition(
        category=IntentCategory.REVIEWS_RATINGS,
        action_code=ActionCode.ADD_REVIEW,
        description="User wants to add a review for a product",
        example_phrases=[
            "I want to review this",
            "Add a review",
            "Write a review",
            "Review this product",
            "I want to review",
            "Add my review",
            "Write review",
            "Post a review",
            "Leave a review",
            "Submit my review",
            "Write feedback"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.REVIEW_RATING],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "edit_review": IntentDefinition(
        category=IntentCategory.REVIEWS_RATINGS,
        action_code=ActionCode.EDIT_REVIEW,
        description="User wants to edit their existing review",
        example_phrases=[
            "I want to edit my review",
            "Edit review",
            "Modify my review",
            "Update my review",
            "Change my review",
            "Edit my rating",
            "Update review",
            "Modify review",
            "Revise my review",
            "Correct my review"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.REVIEW_RATING],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "delete_review": IntentDefinition(
        category=IntentCategory.REVIEWS_RATINGS,
        action_code=ActionCode.DELETE_REVIEW,
        description="User wants to delete their review",
        example_phrases=[
            "I want to delete my review",
            "Delete review",
            "Remove my review",
            "Delete my rating",
            "Remove review",
            "Delete my review",
            "Remove my rating",
            "Cancel my review",
            "Erase my review",
            "Take down my review"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "flag_review": IntentDefinition(
        category=IntentCategory.REVIEWS_RATINGS,
        action_code=ActionCode.FLAG_REVIEW,
        description="User wants to flag an inappropriate review",
        example_phrases=[
            "Flag this review",
            "Report this review",
            "Flag inappropriate review",
            "Report review",
            "Flag review",
            "Report this rating",
            "Flag this comment",
            "Report inappropriate",
            "Mark review as abusive",
            "Report spam review"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[EntityType.REASON],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "view_reviews": IntentDefinition(
        category=IntentCategory.REVIEWS_RATINGS,
        action_code=ActionCode.VIEW_REVIEWS,
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
            "Show reviews for headphones",
            "See product reviews",
            "Show me reviews for the wireless headphones",
            "Read customer reviews",
            "Display product reviews"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "review_recommendation": IntentDefinition(
        category=IntentCategory.REVIEWS_RATINGS,
        action_code=ActionCode.REVIEW_RECOMMENDATION,
        description="User wants to see review-based recommendations",
        example_phrases=[
            "Recommend based on reviews",
            "What do reviews suggest?",
            "Review recommendations",
            "Based on reviews",
            "What reviews recommend?",
            "Review-based suggestions",
            "Recommend from reviews",
            "Review insights",
            "Suggestions from customer reviews",
            "Top picks by reviews"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "product_average_review": IntentDefinition(
        category=IntentCategory.REVIEWS_RATINGS,
        action_code=ActionCode.PRODUCT_AVERAGE_REVIEW,
        description="User wants to see the average review rating",
        example_phrases=[
            "What's the average rating?",
            "Show me average rating",
            "Overall rating",
            "Average score",
            "What's the rating?",
            "Show rating",
            "Overall review score",
            "Average review",
            "Mean review rating",
            "Average stars"
        ],
        required_entities=[EntityType.PRODUCT_ID],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),
}