import os
import json
from datetime import datetime
from threading import Lock
from typing import Dict, Any, Optional
from loguru import logger

_lock = Lock()
LOG_PATH = os.path.join("data", "usage_log.json")

# Simple per-model pricing (USD per 1K tokens)
MODEL_PRICING = {
    "gpt-4o-mini": 0.000005,
    "gpt-4-turbo": 0.00001,
    "gpt-3.5-turbo": 0.0000015,
}


def _ensure_storage():
    """Ensure the persistent log file and directories exist."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            json.dump({"daily": {}, "monthly": {}}, f)


class UsageTracker:
    """Tracks token usage, calculates cost, and persists daily/monthly summaries."""

    def __init__(self, log_path: str = LOG_PATH):
        self.log_path = log_path
        _ensure_storage()

        # Initialize detectors and alert systems
        try:
            from app.ai.cost_monitor.cost_spike_detector import CostSpikeDetector
            from app.ai.cost_monitor.alert_manager import AlertManager
            self.spike_detector = CostSpikeDetector()
            self.alert_manager = AlertManager()
        except Exception as e:
            logger.warning(f"UsageTracker: optional components not loaded → {e}")

    # ------------------------------
    # File operations
    # ------------------------------
    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.log_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"daily": {}, "monthly": {}}

    def _write(self, data: Dict[str, Any]) -> None:
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------
    # Core record logic
    # ------------------------------
    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: Optional[float] = None,
    ):
        """
        Record a new LLM API usage event.
        Aggregates token counts and cost at daily/monthly granularity.
        """
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        price_per_token = MODEL_PRICING.get(model, MODEL_PRICING["gpt-3.5-turbo"])
        cost = total_tokens * price_per_token

        now = datetime.utcnow()
        day_key = now.strftime("%Y-%m-%d")
        month_key = now.strftime("%Y-%m")

        with _lock:
            data = self._read()

            # ---- Daily ----
            daily = data["daily"].get(day_key, {"tokens": 0, "cost": 0.0, "requests": 0})
            daily["tokens"] += total_tokens
            daily["cost"] = round(daily["cost"] + cost, 8)
            daily["requests"] += 1
            data["daily"][day_key] = daily

            # ---- Monthly ----
            monthly = data["monthly"].get(month_key, {"tokens": 0, "cost": 0.0, "requests": 0})
            monthly["tokens"] += total_tokens
            monthly["cost"] = round(monthly["cost"] + cost, 8)
            monthly["requests"] += 1
            data["monthly"][month_key] = monthly

            # persist
            self._write(data)

        # ---- Optional alert for high usage ----
        try:
            from app.ai.cost_monitor.cost_alerts import check_and_alert
            check_and_alert()
        except Exception as e:
            logger.warning(f"UsageTracker: cost alert check failed → {e}")

        # ---- Spike detection ----
        try:
            history = [
                {"date": day, "tokens": val["tokens"], "cost": val["cost"]}
                for day, val in sorted(data.get("daily", {}).items())
            ]
            if hasattr(self, "spike_detector"):
                result = self.spike_detector.detect(history)
                if result.get("spike_detected"):
                    msg = result.get("reason")
                    if hasattr(self, "alert_manager"):
                        self.alert_manager.send_alert(msg)
                    else:
                        logger.warning(f"Spike Alert: {msg}")
        except Exception as e:
            logger.warning(f"UsageTracker: spike detection failed → {e}")

        return {
            "day": day_key,
            "month": month_key,
            "cost": cost,
            "total_tokens": total_tokens,
        }

    # ------------------------------
    # Data access
    # ------------------------------
    def get_daily(self, day: Optional[str] = None) -> Dict[str, Any]:
        data = self._read()
        day = day or datetime.utcnow().strftime("%Y-%m-%d")
        return data.get("daily", {}).get(day, {"tokens": 0, "cost": 0.0, "requests": 0})

    def get_monthly(self, month: Optional[str] = None) -> Dict[str, Any]:
        data = self._read()
        month = month or datetime.utcnow().strftime("%Y-%m")
        return data.get("monthly", {}).get(month, {"tokens": 0, "cost": 0.0, "requests": 0})

    # ------------------------------
    # Summary helper
    # ------------------------------
    def summary(self):
        """Logs a detailed cost summary for the current day and month."""
        data = self._read()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        month = datetime.utcnow().strftime("%Y-%m")

        daily = data.get("daily", {}).get(today, {"tokens": 0, "cost": 0.0, "requests": 0})
        monthly = data.get("monthly", {}).get(month, {"tokens": 0, "cost": 0.0, "requests": 0})

        logger.info("📊 Cost Summary")
        logger.info(f"Date: {today}")
        logger.info(f"Month: {month}")
        logger.info(f"Total Requests Today: {daily['requests']}")
        logger.info(f"Total Tokens Today: {daily['tokens']}")
        logger.info(f"Total Cost Today: ${daily['cost']:.6f}")
        logger.info(f"Total Requests This Month: {monthly['requests']}")
        logger.info(f"Total Tokens This Month: {monthly['tokens']}")
        logger.info(f"Total Cost This Month: ${monthly['cost']:.6f}")

    # ------------------------------
    # NEW: get_daily_summary (for scheduler)
    # ------------------------------
    def get_daily_summary(self) -> Dict[str, Any]:
        """
        Returns a compact summary of today's usage for the scheduler or dashboard.
        Compatible with cost_spike_detector and alert_manager.
        """
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            data = self._read()
            today_data = data.get("daily", {}).get(today, {"tokens": 0, "cost": 0.0, "requests": 0})

            return {
                "date": today,
                "total_tokens": today_data["tokens"],
                "total_cost": today_data["cost"],
                "total_requests": today_data["requests"],
            }

        except Exception as e:
            logger.error(f"UsageTracker: get_daily_summary failed → {e}")
            return {
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_requests": 0,
            }
# ------------------------------------------------------------
# ✅ Manual Test (temporary, safe to keep)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n🔍 Running manual UsageTracker test...\n")
    tracker = UsageTracker()

    # Simulate a few fake API calls
    tracker.record(model="gpt-4o-mini", prompt_tokens=200, completion_tokens=150)
    tracker.record(model="gpt-4-turbo", prompt_tokens=500, completion_tokens=300)
    tracker.record(model="gpt-3.5-turbo", prompt_tokens=100, completion_tokens=80)

    tracker.summary()  # full breakdown
    print("📦 Daily Summary:", tracker.get_daily_summary())
    print("\n✅ Manual test completed successfully.\n")
