"""
alert_notifier.py

Handles escalation alerts with intelligent frequency-based filtering.

Features:
- Escalation thresholds (count within 1 hour window)
- Severity levels (info, warning, error, critical)
- Time-windowed error counting
- Webhook integration with fallback to structured logging
"""
import os
import json
import logging
import time
from collections import defaultdict, deque
from typing import Dict

logger = logging.getLogger("alert_notifier")
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
# Escalation thresholds (count within 1 hour window)
# -------------------------------------------------------------------
ESCALATION_THRESHOLDS = {
    "rate_limit": 10,           # Escalate after 10 rate limits in 1 hour
    "timeout": 20,              # Escalate after 20 timeouts in 1 hour
    "server_error": 5,          # Escalate after 5 server errors in 1 hour
    "auth_error": 1,            # Escalate immediately on auth error
    "context_length_exceeded": 5,  # Escalate after 5 context length errors in 1 hour
    "unknown_error": 15,        # Escalate after 15 unknown errors in 1 hour
}

# -------------------------------------------------------------------
# Track error counts (in-memory, time-windowed)
# For production with multiple workers, consider using Redis for distributed tracking
# -------------------------------------------------------------------
error_counts: Dict[str, deque] = defaultdict(lambda: deque())


def should_escalate(error_type: str) -> bool:
    """
    Check if error count exceeds threshold for escalation.
    Uses a 1-hour sliding window for counting.
    
    Args:
        error_type: Type of error (e.g., 'rate_limit', 'timeout')
        
    Returns:
        True if threshold exceeded and should escalate, False otherwise
    """
    threshold = ESCALATION_THRESHOLDS.get(error_type, 10)
    now = time.time()
    one_hour_ago = now - 3600  # 1 hour window
    
    # Get error queue for this error type
    error_queue = error_counts[error_type]
    
    # Remove old entries (outside 1 hour window)
    while error_queue and error_queue[0] < one_hour_ago:
        error_queue.popleft()
    
    # Add current error timestamp
    error_queue.append(now)
    
    # Check if threshold exceeded
    count = len(error_queue)
    if count >= threshold:
        logger.warning(f"Escalation threshold reached for {error_type}: {count}/{threshold}")
        return True
    else:
        logger.debug(f"Error count for {error_type}: {count}/{threshold} (below threshold)")
        return False


def send_alert(
    event_type: str,
    context: dict,
    severity: str = "warning"
):
    """
    Sends an escalation alert to the configured webhook or logs it locally.
    
    Uses frequency-based filtering to avoid alert spam. Critical and error
    severity always escalate, while warning and info are filtered by frequency.
    
    Severity levels:
    - info: FYI, no immediate action needed
    - warning: Needs attention soon
    - error: Needs immediate attention
    - critical: System failure, page on-call
    
    Args:
        event_type: Type of alert (e.g., 'RATE_LIMIT_EXCEEDED')
        context: Additional context (query text, error details, etc.)
        severity: Severity level (default: 'warning')
    """
    # Check if should escalate based on frequency
    # Critical and error always escalate, warning and info are throttled
    if severity not in ["critical", "error"]:
        # Extract error_code from context if available
        error_code = context.get("error_code", event_type.lower())
        if not should_escalate(error_code):
            logger.debug(f"Skipping escalation for {event_type} (below threshold)")
            return
    
    webhook_url = os.getenv("ESCALATION_WEBHOOK_URL")
    
    # Enhanced payload with severity and environment
    payload = {
        "event_type": event_type,
        "severity": severity,
        "context": context,
        "timestamp": time.time(),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "service": "chatNShop-intent-classification",
    }
    
    # Try to send to webhook if configured
    if webhook_url:
        try:
            response = requests.post(webhook_url, json=payload, timeout=3)
            if response.status_code == 200:
                logger.info(f"Escalation sent to webhook: {event_type} ({severity})")
            else:
                logger.warning(
                    f"Webhook escalation failed ({response.status_code}): {response.text}"
                )
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}", exc_info=True)
    else:
        # Fallback to structured log
        logger.warning(f"[ESCALATION-{severity.upper()}] {json.dumps(payload, indent=2)}")
