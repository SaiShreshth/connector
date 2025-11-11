"""
Detects unusual cost or token usage spikes based on recent trends.
"""

import datetime
import statistics
from typing import List, Dict, Tuple


class CostSpikeDetector:
    def __init__(self, threshold_factor: float = 2.0):
        """
        Args:
            threshold_factor: Multiplier above which a cost is considered a spike (e.g., 2× average)
        """
        self.threshold_factor = threshold_factor

    def detect(self, history: List[Dict]) -> Dict:
        """
        Analyzes cost history and detects spikes.
        Args:
            history: list of daily summaries → [{"date": "2025-10-27", "tokens": 500, "cost": 0.002}, ...]
        Returns:
            {"spike_detected": bool, "reason": str or None, "today": float, "avg": float}
        """
        if len(history) < 2:
            return {"spike_detected": False, "reason": "Not enough history"}

        today = history[-1]
        past = history[:-1]

        avg_cost = statistics.mean(day["cost"] for day in past)
        avg_tokens = statistics.mean(day["tokens"] for day in past)

        spike_cost = today["cost"] > avg_cost * self.threshold_factor
        spike_tokens = today["tokens"] > avg_tokens * self.threshold_factor

        if spike_cost or spike_tokens:
            reason = (
                f"🚨 Spike detected! "
                f"Today's cost (${today['cost']:.4f}) or tokens ({today['tokens']}) "
                f"exceeds {self.threshold_factor}× average."
            )
            return {
                "spike_detected": True,
                "reason": reason,
                "today": today,
                "avg": {"cost": avg_cost, "tokens": avg_tokens},
            }

        return {
            "spike_detected": False,
            "reason": None,
            "today": today,
            "avg": {"cost": avg_cost, "tokens": avg_tokens},
        }

    # ------------------------------------------------------------------
    # ✅ NEW METHOD → Used by scheduler for automatic spike monitoring
    # ------------------------------------------------------------------
    def detect_spike(self, summary: Dict) -> Tuple[bool, str]:
        """
        Wrapper for scheduler-based check.
        Expects summary like:
        {
            "daily_history": [
                {"date": "2025-10-27", "tokens": 500, "cost": 0.002},
                {"date": "2025-10-28", "tokens": 900, "cost": 0.003},
                {"date": "2025-10-29", "tokens": 1100, "cost": 0.004},
            ]
        }
        Returns:
            (spike_detected: bool, report: str)
        """
        try:
            history = summary.get("daily_history", [])
            if not history:
                return False, "[INFO] No daily history available for spike detection."

            result = self.detect(history)
            if result["spike_detected"]:
                return True, result["reason"]
            else:
                return False, "[INFO] No spike detected in current trend."
        except Exception as e:
            return False, f"[ERROR] Spike detection failed: {e}"
