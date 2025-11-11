"""
User-friendly error formatting, escalation metadata, and UNCLEAR responses.

Provides specific error handlers for different failure modes:
- Timeout errors
- Rate limit errors
- Service unavailable (503)
- Authentication errors
- Context length exceeded
- Generic errors
"""

from __future__ import annotations
import logging
from typing import Dict, List
import time

logger = logging.getLogger(__name__)


def handle_timeout_error(user_query: str, context: Dict) -> Dict:
    """
    Handle LLM API timeout errors.
    
    Args:
        user_query: User's input query
        context: Additional context (error details, timestamps, etc.)
        
    Returns:
        User-friendly error response with retry recommendation
    """
    return {
        "user_message": "The request is taking longer than expected. Please try again.",
        "action_code": "TIMEOUT_ERROR",
        "retry_recommended": True,
        "retry_after_seconds": 5,
        "original_input": user_query,
        "suggestions": [
            "Try again in a few seconds",
            "Simplify your request if it's very detailed"
        ],
        "metadata": context
    }


def handle_rate_limit_error(user_query: str, context: Dict) -> Dict:
    """
    Handle LLM API rate limit exceeded errors.
    
    Args:
        user_query: User's input query
        context: Additional context (error details, timestamps, etc.)
        
    Returns:
        User-friendly error response with retry delay
    """
    return {
        "user_message": "We're experiencing high traffic. Please try again in a moment.",
        "action_code": "RATE_LIMIT_ERROR",
        "retry_recommended": True,
        "retry_after_seconds": context.get("retry_after", 60),
        "original_input": user_query,
        "suggestions": [
            "Wait a minute before trying again",
            "Browse our catalog while waiting"
        ],
        "metadata": context
    }


def handle_service_unavailable_error(user_query: str, context: Dict) -> Dict:
    """
    Handle 503 service unavailable and other server errors.
    
    Args:
        user_query: User's input query
        context: Additional context (error details, timestamps, etc.)
        
    Returns:
        User-friendly error response with escalation notice
    """
    return {
        "user_message": "The service is temporarily unavailable. Our team has been notified.",
        "action_code": "SERVICE_UNAVAILABLE",
        "retry_recommended": True,
        "retry_after_seconds": 120,
        "escalated": True,
        "original_input": user_query,
        "suggestions": [
            "Try again in a few minutes",
            "Contact support if the issue persists"
        ],
        "metadata": context
    }


def handle_auth_error(user_query: str, context: Dict) -> Dict:
    """
    Handle authentication errors (API key issues).
    
    Args:
        user_query: User's input query
        context: Additional context (error details, timestamps, etc.)
        
    Returns:
        User-friendly error response with critical escalation
    """
    return {
        "user_message": "We're experiencing technical difficulties. Our team has been notified.",
        "action_code": "AUTH_ERROR",
        "retry_recommended": False,
        "escalated": True,
        "critical": True,
        "original_input": user_query,
        "suggestions": [
            "Please try again later",
            "Contact support for immediate assistance"
        ],
        "metadata": context
    }


def handle_context_length_error(user_query: str, context: Dict) -> Dict:
    """
    Handle context length exceeded errors.
    
    Args:
        user_query: User's input query
        context: Additional context (error details, timestamps, etc.)
        
    Returns:
        User-friendly error response with simplification suggestion
    """
    return {
        "user_message": "Your request is too complex. Please try a simpler query.",
        "action_code": "CONTEXT_LENGTH_EXCEEDED",
        "retry_recommended": True,
        "original_input": user_query,
        "suggestions": [
            "Break your query into smaller parts",
            "Use fewer words to describe what you need",
            "Ask one question at a time"
        ],
        "metadata": context
    }


def log_error_with_full_context(error: Exception, context: Dict) -> None:
    """
    Log error with complete context for debugging and monitoring.
    
    Args:
        error: The exception that occurred
        context: Dict with request context (user_input, user_id, session_id, etc.)
    """
    logger.error(
        f"LLM Error: {type(error).__name__}: {str(error)}",
        extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_code": context.get("error_code", "unknown"),
            "user_query": context.get("user_input", "")[:200],  # Truncate long queries
            "user_id": context.get("user_id"),
            "session_id": context.get("session_id"),
            "request_id": context.get("request_id"),
            "timestamp": context.get("timestamp", time.time()),
            "error_details": context.get("error_details", {}),
        },
        exc_info=True
    )


def build_user_error_response(user_message: str, internal_error: str) -> Dict:
    """
    Return a generic friendly message payload for end-users while keeping internal context for logs.
    
    Args:
        user_message: User's input
        internal_error: Internal error message (not shown to user)
        
    Returns:
        Generic error response dict
    """
    return {
        "user_message": "Sorry — something went wrong while processing your request. Please try again in a moment.",
        "internal_error": internal_error,
        "suggestions": [
            "Try again in a few seconds",
            "Provide more details (product name, order number, or clear action)"
        ],
        "original_input": user_message
    }


def build_unclear_intent_response(suggested_questions: List[str]) -> Dict:
    """
    Return the canonical UNCLEAR response which the system should use when
    the model returns ambiguous/low-confidence answers.
    """
    return {
        "action_code": "UNCLEAR",
        "confidence": 0.0,
        "reasoning": "Ambiguous input — require user clarification",
        "clarifying_questions": suggested_questions
    }
