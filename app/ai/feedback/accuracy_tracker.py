"""
Accuracy Tracker

Tracks accuracy improvement over time for continuous improvement monitoring.
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from collections import defaultdict

from app.ai.feedback.feedback_store import get_feedback_store

ACCURACY_HISTORY_FILE = os.path.join("data", "accuracy_history.json")


class AccuracyTracker:
    """Tracks accuracy improvement over time."""

    def __init__(self, history_file: str = ACCURACY_HISTORY_FILE):
        self.history_file = history_file
        self.feedback_store = get_feedback_store()

    def _read_history(self) -> Dict[str, Any]:
        """Read accuracy history from file."""
        if not os.path.exists(self.history_file):
            return {"daily": {}, "weekly": {}, "monthly": {}}
        
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"daily": {}, "weekly": {}, "monthly": {}}

    def _write_history(self, data: Dict[str, Any]) -> None:
        """Write accuracy history to file."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    def calculate_accuracy(self, period: str = "daily") -> Dict[str, Any]:
        """
        Calculate accuracy for a given period.
        
        Args:
            period: "daily", "weekly", or "monthly"
        
        Returns:
            Dict with accuracy metrics
        """
        stats = self.feedback_store.get_stats()
        
        total_feedback = stats.get("total_feedback", 0)
        total_misclassifications = stats.get("total_misclassifications", 0)
        
        if total_feedback == 0:
            return {
                "period": period,
                "date": datetime.now(timezone.utc).isoformat(),
                "accuracy": 0.0,
                "total_feedback": 0,
                "misclassifications": 0,
            }
        
        accuracy = (total_feedback - total_misclassifications) / total_feedback
        
        return {
            "period": period,
            "date": datetime.now(timezone.utc).isoformat(),
            "accuracy": accuracy,
            "total_feedback": total_feedback,
            "misclassifications": total_misclassifications,
        }

    def record_daily_accuracy(self) -> None:
        """Record today's accuracy in history."""
        history = self._read_history()
        accuracy_data = self.calculate_accuracy("daily")
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        history["daily"][date_key] = accuracy_data
        self._write_history(history)

    def get_accuracy_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get accuracy trends over the last N days."""
        history = self._read_history()
        daily_data = history.get("daily", {})
        
        trends = []
        for i in range(days):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            if date in daily_data:
                trends.append(daily_data[date])
        
        return sorted(trends, key=lambda x: x.get("date", ""))

    def get_accuracy_summary(self) -> Dict[str, Any]:
        """Get overall accuracy summary."""
        history = self._read_history()
        daily_data = history.get("daily", {})
        
        if not daily_data:
            return {
                "current_accuracy": 0.0,
                "trend": "no_data",
                "days_tracked": 0,
            }
        
        # Get most recent accuracy
        sorted_dates = sorted(daily_data.keys(), reverse=True)
        if sorted_dates:
            current = daily_data[sorted_dates[0]]
            current_accuracy = current.get("accuracy", 0.0)
            
            # Calculate trend
            if len(sorted_dates) >= 7:
                week_ago = daily_data.get(sorted_dates[6] if len(sorted_dates) > 6 else sorted_dates[-1], {})
                week_ago_accuracy = week_ago.get("accuracy", 0.0)
                trend = "improving" if current_accuracy > week_ago_accuracy else "declining" if current_accuracy < week_ago_accuracy else "stable"
            else:
                trend = "insufficient_data"
            
            return {
                "current_accuracy": current_accuracy,
                "trend": trend,
                "days_tracked": len(sorted_dates),
                "latest_date": sorted_dates[0],
            }
        
        return {
            "current_accuracy": 0.0,
            "trend": "no_data",
            "days_tracked": 0,
        }
