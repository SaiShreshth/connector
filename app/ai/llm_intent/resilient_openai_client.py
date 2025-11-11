"""
Robust OpenAI client wrapper implementing:
- retries with exponential backoff
- classification of common errors (rate limit, timeout, 5xx, auth, context length)
- structured logging with full request context
- returns structured responses and error metadata
"""

from __future__ import annotations
import time
import logging
from typing import Any, Dict, Optional
import random

# If you have the official OpenAI library installed, import it here.
try:
    import openai
except Exception:
    openai = None

logger = logging.getLogger(__name__)


def _classify_error(exc: Exception) -> str:
    """
    Classify OpenAI API errors into categories.
    
    Returns error code string for structured handling.
    """
    err_str = str(exc).lower()
    
    # Check for timeout
    if "timeout" in err_str or "timed out" in err_str:
        return "timeout"
    
    # Check for context length exceeded
    if "context_length_exceeded" in err_str or "maximum context length" in err_str:
        return "context_length_exceeded"
    
    # Check for authentication errors
    if "authentication" in err_str or "invalid api key" in err_str or "unauthorized" in err_str:
        return "auth_error"
    
    # Check HTTP status codes
    if hasattr(exc, "http_status"):
        status = getattr(exc, "http_status")
        if status == 401:
            return "auth_error"
        elif status == 429:
            return "rate_limit"
        elif 500 <= status < 600:
            return "server_error"
    
    # Check for status_code attribute (alternate naming)
    if hasattr(exc, "status_code"):
        status = getattr(exc, "status_code")
        if status == 401:
            return "auth_error"
        elif status == 429:
            return "rate_limit"
        elif 500 <= status < 600:
            return "server_error"
    
    # Check exception type names
    exc_type = type(exc).__name__
    if "Timeout" in exc_type or "timeout" in exc_type:
        return "timeout"
    if "RateLimit" in exc_type or "rate_limit" in exc_type:
        return "rate_limit"
    if "Auth" in exc_type or "authentication" in exc_type:
        return "auth_error"
    
    return "unknown_error"


class LLMRequestError(Exception):
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ResilientOpenAIClient:
    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo", max_retries: int = 3, base_backoff: float = 0.5):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_backoff = base_backoff

        if openai:
            openai.api_key = api_key

    def complete(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform the LLM call with retry and backoff. Returns parsed dict on success,
        or raises LLMRequestError on final failure.
        
        Args:
            request_body: Dict containing 'messages', 'temperature', 'max_tokens', etc.
            
        Returns:
            Dict with 'raw_content' and 'usage' on success
            
        Raises:
            LLMRequestError: After all retries exhausted, with error code and details
        """
        last_exc = None
        last_error_code = "unknown_error"
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Use OpenAI chat completion if available
                if openai is None:
                    # Running in mock/offline env — simulate deterministic response
                    logger.info("Running in simulation mode (no OpenAI package)")
                    response = {
                        "intent": "simulated_intent",
                        "intent_category": "SIMULATION",
                        "action_code": request_body.get("metadata", {}).get("action_code", "HELP_GENERAL"),
                        "confidence": 0.9,
                        "reasoning": "Simulated response (no openai package installed)",
                        "processing_time_ms": 50,
                    }
                    return response

                # Log request attempt with context
                messages_count = len(request_body.get("messages", []))
                logger.info(
                    f"LLM request attempt {attempt}/{self.max_retries}",
                    extra={
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "model": self.model_name,
                        "messages_count": messages_count,
                        "temperature": request_body.get("temperature", 0.0),
                        "max_tokens": request_body.get("max_tokens", 200),
                    }
                )

                # Call OpenAI API
                completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=request_body.get("messages"),
                    max_tokens=request_body.get("max_tokens", 200),
                    temperature=request_body.get("temperature", 0.0),
                    timeout=request_body.get("timeout_s", 10),
                )
                
                # Extract assistant content — expecting JSON text
                content = completion["choices"][0]["message"]["content"]
                
                logger.info(f"LLM request succeeded on attempt {attempt}")
                
                # Return raw content container
                return {"raw_content": content, "usage": completion.get("usage")}
                
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                
                # Classify error using enhanced classification
                last_error_code = _classify_error(exc)
                
                # Structured logging with full context
                logger.error(
                    f"LLM request failed on attempt {attempt}/{self.max_retries}",
                    extra={
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "error_code": last_error_code,
                        "error_type": type(exc).__name__,
                        "error_message": err_str[:200],  # Truncate long messages
                        "model": self.model_name,
                        "request_summary": {
                            "messages_count": len(request_body.get("messages", [])),
                            "temperature": request_body.get("temperature"),
                            "max_tokens": request_body.get("max_tokens"),
                        }
                    },
                    exc_info=(attempt == self.max_retries)  # Full traceback only on last attempt
                )
                
                # If last attempt, raise immediately
                if attempt >= self.max_retries:
                    break
                
                # Calculate exponential backoff with jitter
                jitter = random.uniform(0, 0.1 * attempt)
                backoff = self.base_backoff * (2 ** (attempt - 1)) + jitter
                
                logger.info(f"Retrying in {backoff:.2f}s...")
                time.sleep(backoff)
        
        # All retries exhausted - raise LLMRequestError with details
        raise LLMRequestError(
            f"LLM request failed after {self.max_retries} attempts: {last_error_code}",
            code=last_error_code,
            details={
                "exception": str(last_exc),
                "exception_type": type(last_exc).__name__,
                "attempts": self.max_retries,
                "model": self.model_name,
            }
        )
