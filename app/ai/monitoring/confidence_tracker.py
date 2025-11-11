"""
Confidence Score Distribution Tracker

Tracks confidence score distribution for monitoring and analysis.
"""

import os
import json
from datetime import datetime, timezone
from collections import defaultdict
from threading import Lock
from typing import Dict, Any, Optional

_lock = Lock()
CONFIDENCE_DISTRIBUTION_FILE = os.path.join("data", "confidence_distribution.json")


def _ensure_storage():
    """Ensure the data directory exists."""
    os.makedirs(os.path.dirname(CONFIDENCE_DISTRIBUTION_FILE), exist_ok=True)


class ConfidenceTracker:
    """Tracks confidence score distribution."""

    def __init__(self, distribution_file: str = CONFIDENCE_DISTRIBUTION_FILE):
        self.distribution_file = distribution_file
        _ensure_storage()
        self.bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def _read_distribution(self) -> Dict[str, Any]:
        """Read confidence distribution data from file."""
        if not os.path.exists(self.distribution_file):
            return {"daily": {}}
        
        try:
            with open(self.distribution_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"daily": {}}

    def _write_distribution(self, data: Dict[str, Any]) -> None:
        """Write confidence distribution data to file."""
        with _lock:
            with open(self.distribution_file, "w") as f:
                json.dump(data, f, indent=2)

    def _get_bin(self, confidence: float) -> str:
        """Get the bin label for a confidence score."""
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= confidence < self.bins[i + 1]:
                return f"{self.bins[i]:.1f}-{self.bins[i+1]:.1f}"
        return "1.0-1.0"  # Handle edge case

    def record_confidence(self, confidence: float, date: Optional[str] = None) -> None:
        """
        Record a confidence score.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            date: Date in YYYY-MM-DD format (defaults to today)
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        data = self._read_distribution()
        
        if "daily" not in data:
            data["daily"] = {}
        
        if date not in data["daily"]:
            data["daily"][date] = {}
        
        bin_label = self._get_bin(confidence)
        data["daily"][date][bin_label] = data["daily"][date].get(bin_label, 0) + 1
        
        self._write_distribution(data)

    def get_daily_distribution(self, date: Optional[str] = None) -> Dict[str, int]:
        """Get confidence distribution for a specific day."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        data = self._read_distribution()
        return data.get("daily", {}).get(date, {})

    def get_weekly_distribution(self) -> Dict[str, int]:
        """Get aggregated confidence distribution for the last 7 days."""
        from datetime import timedelta
        
        data = self._read_distribution()
        weekly = defaultdict(int)
        
        for i in range(7):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_dist = data.get("daily", {}).get(date, {})
            for bin_label, count in daily_dist.items():
                weekly[bin_label] += count
        
        return dict(weekly)

    def get_distribution_summary(self, period: str = "daily") -> Dict[str, Any]:
        """
        Get summary statistics for confidence distribution.
        
        Args:
            period: "daily", "weekly", or "monthly"
        
        Returns:
            Dict with distribution summary
        """
        if period == "daily":
            distribution = self.get_daily_distribution()
        elif period == "weekly":
            distribution = self.get_weekly_distribution()
        else:
            # Monthly
            from datetime import timedelta
            data = self._read_distribution()
            monthly = defaultdict(int)
            for i in range(30):
                date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
                daily_dist = data.get("daily", {}).get(date, {})
                for bin_label, count in daily_dist.items():
                    monthly[bin_label] += count
            distribution = dict(monthly)
        
        total = sum(distribution.values())
        
        if total == 0:
            return {
                "total": 0,
                "distribution": {},
                "percentages": {},
            }
        
        percentages = {bin_label: (count / total * 100) for bin_label, count in distribution.items()}
        
        return {
            "total": total,
            "distribution": distribution,
            "percentages": percentages,
        }


_confidence_tracker_instance: Optional[ConfidenceTracker] = None


def get_confidence_tracker() -> ConfidenceTracker:
    """Get or create singleton ConfidenceTracker instance."""
    global _confidence_tracker_instance
    if _confidence_tracker_instance is None:
        _confidence_tracker_instance = ConfidenceTracker()
    return _confidence_tracker_instance
