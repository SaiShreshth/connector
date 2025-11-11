import os
import time
import logging
from typing import Any, Dict, Optional
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError
import threading
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

logger = logging.getLogger("openai_client")
logger.setLevel(logging.INFO)

# ========== Circuit Breaker State ==========
class CircuitBreaker:
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.failures = 0
        self.lock = threading.Lock()
        self.tripped = False
        self.trip_time = 0
        self.reset_timeout = reset_timeout

    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.tripped and (time.time() - self.trip_time) < self.reset_timeout:
                raise RuntimeError("Circuit breaker open: refusing to call OpenAI API")
            elif self.tripped:
                # Cooldown expired: reset
                self.tripped = False
                self.failures = 0

        try:
            out = func(*args, **kwargs)
        except Exception as e:
            with self.lock:
                self.failures += 1
                if self.failures >= self.max_failures:
                    self.tripped = True
                    self.trip_time = time.time()
                    logger.error(f"OpenAI circuit breaker tripped: {e}")
            raise
        else:
            with self.lock:
                self.failures = 0
            return out

# ========== Main Client ==========
class OpenAIClient:
    """
    Advanced OpenAI API wrapper with authentication, retry, circuit breaker, logging, and config management.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_attempts: int = 3,
        circuit_max_failures: int = 3,
        circuit_cooldown: int = 60,  # seconds
        log_calls: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.temperature = temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("OPENAI_MAX_TOKENS", "400"))
        self.timeout = timeout or float(os.getenv("OPENAI_TIMEOUT_SECS", "30"))
        self.retry_attempts = retry_attempts
        self.circuit_breaker = CircuitBreaker(max_failures=circuit_max_failures, reset_timeout=circuit_cooldown)
        self.log_calls = log_calls

        # Initialize OpenAI client (v1.x API)
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        else:
            self.client = None

    # === RETRY/Timeout decorator ===
    def _retry_decorator(self, method):
        return retry(
            wait=wait_exponential(multiplier=1, min=2, max=15),
            stop=stop_after_attempt(self.retry_attempts),
            retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError, APIConnectionError)),
            reraise=True,
        )(method)

    # === Call the API ===
    def _call_openai(self, messages: list, temperature=None, max_tokens=None, stream=False, extra_kwargs=None) -> Dict[str, Any]:
        """Single wrapped call. Handles exceptions for outer retry and CB."""
        # Compose parameters:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "timeout": self.timeout,
            "stream": stream,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. API key is required.")
        
        t0 = time.time()
        try:
            # Use OpenAI v1.x API
            response = self.client.chat.completions.create(**kwargs)
            latency = time.time() - t0
            
            # Parse and validate response (v1.x response structure)
            result = self._parse_response(response)
            usage = response.usage.model_dump() if response.usage else {}
            
            # Log
            if self.log_calls:
                logger.info(f"OpenAI call | model={self.model_name} | temp={kwargs['temperature']} | max_tokens={kwargs['max_tokens']} | latency={latency:.2f}s | prompt_tokens={usage.get('prompt_tokens','?')} | completion_tokens={usage.get('completion_tokens','?')}")
            return {"latency_ms": int(latency * 1000), "response": result, "usage": usage}
        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit: {e}")
            raise
        except APITimeoutError as e:
            logger.error(f"OpenAI timeout: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI unexpected error: {e}")
            raise

    def complete(self, payload: Dict[str, Any], **opts) -> Dict[str, Any]:
        """
        Send a ChatCompletion request to OpenAI.
        Accepts model/temperature/max_tokens/stream in payload/opts, with fallback to defaults.
        Handles retries, logging, and error/timeout gracefully.
        """
        messages = payload["messages"]
        temperature = payload.get("temperature") or opts.get("temperature")
        max_tokens = payload.get("max_tokens") or opts.get("max_tokens")
        stream = opts.get("stream", False)
        extra_kwargs = opts.get("extra_kwargs")
        retry_call = self._retry_decorator(lambda *a, **k: self._call_openai(*a, **k))

        try:
            result = self.circuit_breaker.call(
                retry_call,
                messages,
                temperature,
                max_tokens,
                stream,
                extra_kwargs,
            )
            return result
        except RetryError as r:
            logger.error(f"OpenAIClient: Gave up after retries: {r.last_attempt.exception()}")
            return {"error": str(r.last_attempt.exception()), "type": "retry_exhausted"}
        except RuntimeError as cb:
            return {"error": str(cb), "type": "circuit_open"}
        except Exception as ex:
            logger.error(f"OpenAIClient: Fatal error: {ex}")
            return {"error": str(ex)}

    def _parse_response(self, response: Any) -> str:
        """Parses and validates the LLM's response object from OpenAI (v1.x format)."""
        try:
            # OpenAI v1.x response is an object, not a dict
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    if not content:
                        raise ValueError("No content in OpenAI response message")
                    return content
            
            # Fallback for dict format (shouldn't happen with v1.x but handle it)
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if not choices:
                    raise ValueError("No choices in OpenAI response")
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if not content:
                    raise ValueError("No content in OpenAI response message")
                return content
            
            raise ValueError(f"Unexpected OpenAI response format: {type(response)}")
        except Exception as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            raise

    # Optional: Provide info/probe endpoints
    def status(self):
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }