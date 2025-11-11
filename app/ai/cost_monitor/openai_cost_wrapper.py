"""
Wrapper layer for OpenAIClient that adds:
- Token usage tracking
- Cost estimation
- Latency measurement
- Budget guardrail
- Application-level rate limiting
"""

import time
from typing import Any, Dict
from loguru import logger
from app.ai.llm_intent.openai_client import OpenAIClient
from app.ai.cost_monitor.usage_tracker import UsageTracker
from app.ai.cost_monitor.rate_limiter import RateLimiter

# Configurable per-request limits
TARGET_P95_LATENCY_MS = 2000
MAX_REQUEST_COST_USD = 0.01

# Example per-1k token cost rates (adjust as per your model)
MODEL_PRICING = {
    "gpt-4o-mini": 0.00015,   # per 1K tokens (example)
    "gpt-4o": 0.0003,
    "gpt-3.5-turbo": 0.0005,
}


class CostAwareOpenAIClient:
    """
    Thin cost-monitoring wrapper around the existing OpenAIClient.
    Does NOT modify or override the original openai_client.py.
    """

    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAIClient(api_key, model_name)
        self.model_name = model_name
        self.usage_tracker = UsageTracker()

        # ✅ Added rate limiter (60 calls/minute by default)
        self.rate_limiter = RateLimiter(max_calls=60, window_seconds=60)

    def complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call OpenAIClient.complete() but track cost + latency metrics."""
        # ✅ Step 1: Enforce rate limit
        if not self.rate_limiter.allow():
            return {
                "error": "Rate limit exceeded — please retry after a few seconds."
            }

        start_time = time.time()

        try:
            response = self.client.complete(payload)
        except NotImplementedError:
            # Mock for testing if real API not implemented yet
            response = {
                "choices": [{"message": {"content": "Mocked LLM response"}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 45},
            }

        latency_ms = (time.time() - start_time) * 1000

        if not response or "error" in response:
            logger.error(f"LLM request failed or returned error: {response.get('error') if response else 'unknown'}")
            return response

        # Extract usage info
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Calculate cost
        cost_per_1k = MODEL_PRICING.get(self.model_name, 0.0003)
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000) * cost_per_1k

        # Record metrics
        self.usage_tracker.record(
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

        if cost > MAX_REQUEST_COST_USD:
            logger.warning(f"High cost alert: ${cost:.4f} > ${MAX_REQUEST_COST_USD:.4f} per request limit")

        return response
