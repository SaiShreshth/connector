"""
Pattern Analysis for Misclassifications
"""

from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional

from app.ai.feedback.feedback_store import get_feedback_store


class PatternAnalyzer:
    """Analyzes misclassification patterns to identify common issues."""

    def __init__(self, feedback_store=None):
        self.feedback_store = feedback_store or get_feedback_store()

    def analyze_patterns(
        self,
        action_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze misclassification patterns."""
        misclassifications = self.feedback_store.get_misclassifications(
            action_code=action_code,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )
        
        if not misclassifications:
            return {
                "total_misclassifications": 0,
                "patterns": {},
                "insights": [],
            }
        
        by_predicted = defaultdict(list)
        confidence_scores = []
        
        for record in misclassifications:
            predicted = record.get("predicted_action_code", "UNKNOWN")
            confidence = record.get("confidence", 0.0)
            by_predicted[predicted].append(record)
            confidence_scores.append(confidence)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        low_confidence_count = sum(1 for c in confidence_scores if c < 0.5)
        
        patterns = {}
        for predicted, records in by_predicted.items():
            actual_counts = Counter(r.get("actual_action_code", "UNKNOWN") for r in records)
            patterns[predicted] = {
                "count": len(records),
                "most_common_actual": dict(actual_counts.most_common(5)),
                "avg_confidence": sum(r.get("confidence", 0.0) for r in records) / len(records),
                "sample_queries": [r.get("query", "") for r in records[:5]],
            }
        
        insights = []
        if low_confidence_count > len(misclassifications) * 0.3:
            insights.append({
                "type": "low_confidence",
                "severity": "high",
                "message": f"{low_confidence_count} misclassifications ({low_confidence_count/len(misclassifications)*100:.1f}%) had confidence < 0.5",
                "recommendation": "Review low-confidence classifications and consider improving matching",
            })
        
        return {
            "total_misclassifications": len(misclassifications),
            "avg_confidence": avg_confidence,
            "low_confidence_percentage": (low_confidence_count / len(misclassifications) * 100) if misclassifications else 0.0,
            "patterns_by_predicted": patterns,
            "top_misclassified_action_codes": dict(Counter(r.get("predicted_action_code", "UNKNOWN") for r in misclassifications).most_common(10)),
            "insights": insights,
        }
