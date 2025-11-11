from ..models import IntentDefinition
from ..enums import IntentCategory, ActionCode, IntentPriority, EntityType

account_profile_intent_definitions = {
    # 6.0 Account, Auth & Profile
    "create_account": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.CREATE_ACCOUNT,
        description="User wants to create a new account",
        example_phrases=[
            "I want to create an account",
            "Sign up",
            "Create account",
            "Register",
            "New account",
            "I want to register",
            "Create new account",
            "Sign me up"
        ],
        required_entities=[],
        optional_entities=[EntityType.EMAIL, EntityType.PHONE_NUMBER],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "login": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.LOGIN,
        description="User wants to log into their account",
        example_phrases=[
            "I want to login",
            "Sign in",
            "Login",
            "Log me in",
            "Access my account",
            "Sign into account",
            "Login to account",
            "I want to sign in"
        ],
        required_entities=[],
        optional_entities=[EntityType.EMAIL, EntityType.USER_ID],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "logout": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.LOGOUT,
        description="User wants to log out of their account",
        example_phrases=[
            "I want to logout",
            "Sign out",
            "Logout",
            "Log me out",
            "Sign out of account",
            "Logout of account",
            "End session",
            "I want to sign out"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.90,
        priority=IntentPriority.MEDIUM
    ),

    "login_status": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.LOGIN_STATUS,
        description="User wants to check if they are logged in",
        example_phrases=[
            "Am I logged in?",
            "Check login status",
            "Am I signed in?",
            "Login status",
            "Am I logged in?",
            "Check if logged in",
            "Sign in status",
            "Am I signed in?"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "login_with_social": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.LOGIN_WITH_SOCIAL,
        description="User wants to login using social media",
        example_phrases=[
            "Login with Google",
            "Sign in with Facebook",
            "Login with social media",
            "Sign in with Apple",
            "Login with social account",
            "Sign in with Google",
            "Login with Facebook",
            "Social login"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "forgot_password": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.FORGOT_PASSWORD,
        description="User wants to reset their forgotten password",
        example_phrases=[
            "I forgot my password",
            "Reset password",
            "Forgot password",
            "Password reset",
            "I can't remember password",
            "Reset my password",
            "Forgot my password",
            "Password help"
        ],
        required_entities=[],
        optional_entities=[EntityType.EMAIL],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "reset_password": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.RESET_PASSWORD,
        description="User wants to reset their password",
        example_phrases=[
            "Reset my password",
            "Change password",
            "New password",
            "Update password",
            "Reset password",
            "Change my password",
            "Update my password",
            "Set new password"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.CRITICAL
    ),

    "change_password": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.CHANGE_PASSWORD,
        description="User wants to change their current password",
        example_phrases=[
            "I want to change my password",
            "Change password",
            "Update password",
            "Modify password",
            "Change my password",
            "Update my password",
            "New password",
            "Change current password"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "update_profile": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.UPDATE_PROFILE,
        description="User wants to update their profile information",
        example_phrases=[
            "Update my profile",
            "Change profile",
            "Edit profile",
            "Modify profile",
            "Update profile info",
            "Change my profile",
            "Edit my profile",
            "Update account info"
        ],
        required_entities=[],
        optional_entities=[EntityType.EMAIL, EntityType.PHONE_NUMBER, EntityType.ADDRESS],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "view_profile": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.VIEW_PROFILE,
        description="User wants to view their profile information",
        example_phrases=[
            "Show my profile",
            "View profile",
            "See my profile",
            "Display profile",
            "Show profile info",
            "View my account",
            "See account info",
            "Show my details"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.MEDIUM
    ),

    "delete_account": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.DELETE_ACCOUNT,
        description="User wants to delete their account",
        example_phrases=[
            "Delete my account",
            "Close account",
            "Remove account",
            "Delete account",
            "Close my account",
            "Remove my account",
            "Cancel account",
            "Terminate account"
        ],
        required_entities=[],
        optional_entities=[EntityType.REASON],
        confidence_threshold=0.90,
        priority=IntentPriority.CRITICAL
    ),

    "account_security": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.ACCOUNT_SECURITY,
        description="User wants to manage account security settings",
        example_phrases=[
            "Account security",
            "Security settings",
            "Account safety",
            "Security options",
            "Account protection",
            "Security features",
            "Account privacy",
            "Security management"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.HIGH
    ),

    "two_factor_setup": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.TWO_FACTOR_SETUP,
        description="User wants to set up two-factor authentication",
        example_phrases=[
            "Set up two-factor",
            "Enable two-factor authentication",
            "2FA setup",
            "Two-factor setup",
            "Enable 2FA",
            "Set up 2FA",
            "Two-factor auth",
            "Security verification"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "two_factor_verify": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.TWO_FACTOR_VERIFY,
        description="User wants to verify two-factor authentication",
        example_phrases=[
            "Verify two-factor",
            "2FA verification",
            "Two-factor code",
            "Verify 2FA",
            "Two-factor verification",
            "Security code",
            "Verification code",
            "2FA code"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "verify_email": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.VERIFY_EMAIL,
        description="User wants to verify their email address",
        example_phrases=[
            "Verify my email",
            "Email verification",
            "Verify email address",
            "Confirm email",
            "Email confirmation",
            "Verify email",
            "Confirm my email",
            "Email verify"
        ],
        required_entities=[],
        optional_entities=[EntityType.EMAIL],
        confidence_threshold=0.85,
        priority=IntentPriority.HIGH
    ),

    "resend_verification": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.RESEND_VERIFICATION,
        description="User wants to resend verification email",
        example_phrases=[
            "Resend verification",
            "Send verification again",
            "Resend email",
            "Send verification email",
            "Resend verification email",
            "Send again",
            "Resend confirmation",
            "Send verification"
        ],
        required_entities=[],
        optional_entities=[EntityType.EMAIL],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),

    "profile_picture_update": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.PROFILE_PICTURE_UPDATE,
        description="User wants to update their profile picture",
        example_phrases=[
            "Update profile picture",
            "Change profile photo",
            "Update photo",
            "Change picture",
            "New profile picture",
            "Update avatar",
            "Change avatar",
            "Profile image"
        ],
        required_entities=[],
        optional_entities=[],
        confidence_threshold=0.80,
        priority=IntentPriority.LOW
    ),

    "preferences_update": IntentDefinition(
        category=IntentCategory.ACCOUNT_PROFILE,
        action_code=ActionCode.PREFERENCES_UPDATE,
        description="User wants to update their account preferences",
        example_phrases=[
            "Update preferences",
            "Change preferences",
            "Modify preferences",
            "Update settings",
            "Change settings",
            "Account preferences",
            "User preferences",
            "Update my preferences"
        ],
        required_entities=[],
        optional_entities=[EntityType.PREFERENCES],
        confidence_threshold=0.80,
        priority=IntentPriority.MEDIUM
    ),
}