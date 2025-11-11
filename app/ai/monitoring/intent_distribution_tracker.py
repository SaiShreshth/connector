"""
Intent Distribution Tracker

Tracks intent distribution over time for monitoring and analysis.
"""

import os
import json
from datetime import datetime, timezone
from collections import defaultdict
from threading import Lock
from typing import Dict, Any, Optional

_lock = Lock()
INTENT_DISTRIBUTION_FILE = os.path.join("data", "intent_distribution.json")


def _ensure_storage():
    """Ensure the data directory exists."""
    os.makedirs(os.path.dirname(INTENT_DISTRIBUTION_FILE), exist_ok=True)


class IntentDistributionTracker:
    """Tracks intent distribution over time."""

    def __init__(self, distribution_file: str = INTENT_DISTRIBUTION_FILE):
        self.distribution_file = distribution_file
        _ensure_storage()

    def _read_distribution(self) -> Dict[str, Any]:
        """Read intent distribution data from file."""
        if not os.path.exists(self.distribution_file):
            return {"daily": {}}
        
        try:
            with open(self.distribution_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"daily": {}}

    def _write_distribution(self, data: Dict[str, Any]) -> None:
        """Write intent distribution data to file."""
        with _lock:
            with open(self.distribution_file, "w") as f:
                json.dump(data, f, indent=2)

    def record_intent(self, action_code: str, date: Optional[str] = None) -> None:
        """
        Record an intent classification.
        
        Args:
            action_code: The action code that was classified
            date: Date in YYYY-MM-DD format (defaults to today)
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        data = self._read_distribution()
        
        if "daily" not in data:
            data["daily"] = {}
        
        if date not in data["daily"]:
            data["daily"][date] = {}
        
        data["daily"][date][action_code] = data["daily"][date].get(action_code, 0) + 1
        
        self._write_distribution(data)

    def get_daily_distribution(self, date: Optional[str] = None) -> Dict[str, int]:
        """Get intent distribution for a specific day."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        data = self._read_distribution()
        return data.get("daily", {}).get(date, {})

    def get_weekly_distribution(self) -> Dict[str, int]:
        """Get aggregated intent distribution for the last 7 days."""
        from datetime import timedelta
        
        data = self._read_distribution()
        weekly = defaultdict(int)
        
        for i in range(7):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_dist = data.get("daily", {}).get(date, {})
            for action_code, count in daily_dist.items():
                weekly[action_code] += count
        
        return dict(weekly)

    def get_monthly_distribution(self) -> Dict[str, int]:
        """Get aggregated intent distribution for the last 30 days."""
        from datetime import timedelta
        
        data = self._read_distribution()
        monthly = defaultdict(int)
        
        for i in range(30):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_dist = data.get("daily", {}).get(date, {})
            for action_code, count in daily_dist.items():
                monthly[action_code] += count
        
        return dict(monthly)

    def get_top_intents(self, period: str = "daily", limit: int = 10) -> list:
        """
        Get top N intents by count.
        
        Args:
            period: "daily", "weekly", or "monthly"
            limit: Number of top intents to return
        
        Returns:
            List of tuples (action_code, count) sorted by count descending
        """
        if period == "daily":
            distribution = self.get_daily_distribution()
        elif period == "weekly":
            distribution = self.get_weekly_distribution()
        else:
            distribution = self.get_monthly_distribution()
        
        sorted_intents = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        return sorted_intents[:limit]


_intent_distribution_tracker_instance: Optional[IntentDistributionTracker] = None


def get_intent_distribution_tracker() -> IntentDistributionTracker:
    """Get or create singleton IntentDistributionTracker instance."""
    global _intent_distribution_tracker_instance
    if _intent_distribution_tracker_instance is None:
        _intent_distribution_tracker_instance = IntentDistributionTracker()
    return _intent_distribution_tracker_instance
