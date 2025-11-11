# app/ai/cost_monitor/cost_alerts.py
import os
import requests
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from app.ai.cost_monitor.usage_tracker import UsageTracker
except Exception:
    UsageTracker = None

try:
    from app.ai.feedback.accuracy_tracker import AccuracyTracker
    from app.ai.feedback.feedback_store import get_feedback_store
    FEEDBACK_AVAILABLE = True
except Exception:
    FEEDBACK_AVAILABLE = False

SLACK_WEBHOOK = os.getenv("COST_ALERT_SLACK_WEBHOOK", "").strip()
DAILY_THRESHOLD = float(os.getenv("COST_ALERT_DAILY_THRESHOLD_USD", "2.0"))
ACCURACY_DROP_THRESHOLD = float(os.getenv("ACCURACY_DROP_THRESHOLD", "0.05"))  # 5% drop
LATENCY_SPIKE_THRESHOLD_MS = float(os.getenv("LATENCY_SPIKE_THRESHOLD_MS", "3000"))  # 3 seconds
ERROR_RATE_SPIKE_THRESHOLD = float(os.getenv("ERROR_RATE_SPIKE_THRESHOLD", "0.05"))  # 5% error rate

def check_and_alert(day: str = None) -> Dict[str, Any]:
    """
    Check today's cost against threshold. If exceeded and webhook set, send Slack alert.
    Returns a dict describing totals and whether an alert was sent.
    """
    day = day or datetime.utcnow().strftime("%Y-%m-%d")
    result: Dict[str, Any] = {"date": day, "daily_total": 0.0, "threshold": DAILY_THRESHOLD, "alert_sent": False}

    if UsageTracker is None:
        result["error"] = "UsageTracker not available"
        return result

    try:
        tracker = UsageTracker()
        daily = tracker.get_daily(day)
        total_cost = float(daily.get("cost", 0.0))
        result["daily_total"] = round(total_cost, 8)

        if total_cost > DAILY_THRESHOLD:
            result["exceeded"] = True
            if SLACK_WEBHOOK:
                payload = {"text": f"ALERT: LLM cost for {day} = ${total_cost:.4f} (threshold ${DAILY_THRESHOLD})"}
                try:
                    r = requests.post(SLACK_WEBHOOK, json=payload, timeout=5)
                    result["alert_sent"] = True
                    result["slack_status"] = r.status_code
                except Exception as e:
                    result["alert_error"] = str(e)
            else:
                # Slack not configured — still indicate we would have alerted
                result["alert_sent"] = False
                result["note"] = "SLACK_WEBHOOK not configured"
        else:
            result["exceeded"] = False

    except Exception as e:
        result["error"] = str(e)

    return result


def check_accuracy_drop() -> Optional[Dict[str, Any]]:
    """
    Check for accuracy drop and send alert if detected.
    
    Returns:
        Dict with alert info if drop detected, None otherwise
    """
    if not FEEDBACK_AVAILABLE:
        return None
    
    try:
        accuracy_tracker = AccuracyTracker()
        summary = accuracy_tracker.get_accuracy_summary()
        trends = accuracy_tracker.get_accuracy_trends(days=7)
        
        if len(trends) < 2:
            return None  # Not enough data
        
        current_accuracy = trends[0].get("accuracy", 0.0) if trends else 0.0
        previous_accuracy = trends[-1].get("accuracy", 0.0) if len(trends) > 1 else 0.0
        
        if previous_accuracy > 0 and (previous_accuracy - current_accuracy) > ACCURACY_DROP_THRESHOLD:
            message = f"ALERT: Accuracy dropped from {previous_accuracy:.2%} to {current_accuracy:.2%} (drop: {(previous_accuracy - current_accuracy):.2%})"
            
            if SLACK_WEBHOOK:
                try:
                    payload = {"text": message}
                    requests.post(SLACK_WEBHOOK, json=payload, timeout=5)
                except Exception:
                    pass
            
            return {
                "alert_type": "accuracy_drop",
                "message": message,
                "current_accuracy": current_accuracy,
                "previous_accuracy": previous_accuracy,
                "drop": previous_accuracy - current_accuracy,
            }
    except Exception:
        pass
    
    return None


def check_latency_spike() -> Optional[Dict[str, Any]]:
    """
    Check for latency spike and send alert if detected.
    Note: This is a simplified check. In production, you'd track latency percentiles.
    
    Returns:
        Dict with alert info if spike detected, None otherwise
    """
    # For now, this is a placeholder. In production, you'd query latency metrics
    # from your monitoring system and check p95 latency
    return None


def check_error_rate_spike() -> Optional[Dict[str, Any]]:
    """
    Check for error rate spike and send alert if detected.
    
    Returns:
        Dict with alert info if spike detected, None otherwise
    """
    # This would typically check error logs or metrics
    # For now, placeholder implementation
    return None


def check_all_alerts() -> Dict[str, Any]:
    """Check all alert conditions and return results."""
    results = {
        "cost_alert": check_and_alert(),
        "accuracy_alert": check_accuracy_drop(),
        "latency_alert": check_latency_spike(),
        "error_rate_alert": check_error_rate_spike(),
    }
    
    return results
