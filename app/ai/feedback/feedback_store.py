"""
Feedback Storage System

Stores misclassifications and feedback data for continuous improvement.
Uses JSONL format for simplicity and easy querying.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Any, Optional, List
from loguru import logger

_lock = Lock()
FEEDBACK_FILE = os.path.join("data", "feedback.jsonl")
MISCLASSIFICATIONS_FILE = os.path.join("data", "misclassifications.jsonl")


def _ensure_storage():
    """Ensure the data directory exists."""
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)


class FeedbackStore:
    """Thread-safe storage for feedback and misclassifications."""

    def __init__(self, feedback_file: str = FEEDBACK_FILE, misclassifications_file: str = MISCLASSIFICATIONS_FILE):
        self.feedback_file = feedback_file
        self.misclassifications_file = misclassifications_file
        _ensure_storage()

    def _append_to_file(self, filepath: str, data: Dict[str, Any]) -> None:
        """Append a JSON record to a JSONL file (thread-safe)."""
        with _lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _read_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Read all records from a JSONL file."""
        if not os.path.exists(filepath):
            return []
        
        records = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
        return records

    def record_feedback(
        self,
        request_id: str,
        query: str,
        predicted_action_code: str,
        actual_action_code: Optional[str] = None,
        confidence: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
        correct: bool = True,
        comment: Optional[str] = None,
        expected_action_code: Optional[str] = None,
    ) -> str:
        """Record user feedback about a classification."""
        feedback_id = str(uuid.uuid4())
        actual = actual_action_code or expected_action_code
        
        feedback_record = {
            "feedback_id": feedback_id,
            "request_id": request_id,
            "query": query,
            "predicted_action_code": predicted_action_code,
            "actual_action_code": actual,
            "correct": correct,
            "confidence": confidence,
            "context": context or {},
            "comment": comment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reviewed": False,
            "exported": False,
        }
        
        self._append_to_file(self.feedback_file, feedback_record)
        
        if not correct:
            misclassification_record = feedback_record.copy()
            misclassification_record["misclassification_id"] = feedback_id
            self._append_to_file(self.misclassifications_file, misclassification_record)
        
        return feedback_id

    def get_misclassifications(
        self,
        action_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get misclassified examples with optional filters."""
        records = self._read_file(self.misclassifications_file)
        
        if action_code:
            records = [r for r in records if r.get("predicted_action_code") == action_code]
        
        if start_date:
            records = [r for r in records if r.get("timestamp", "") >= start_date]
        
        if end_date:
            records = [r for r in records if r.get("timestamp", "") <= end_date]
        
        records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return records[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        feedback_records = self._read_file(self.feedback_file)
        misclassification_records = self._read_file(self.misclassifications_file)
        
        total_feedback = len(feedback_records)
        total_misclassifications = len(misclassification_records)
        
        action_code_counts = {}
        for record in misclassification_records:
            code = record.get("predicted_action_code", "UNKNOWN")
            action_code_counts[code] = action_code_counts.get(code, 0) + 1
        
        return {
            "total_feedback": total_feedback,
            "total_misclassifications": total_misclassifications,
            "misclassifications_by_action_code": action_code_counts,
            "accuracy_rate": (
                (total_feedback - total_misclassifications) / total_feedback
                if total_feedback > 0 else 0.0
            ),
        }


_feedback_store_instance: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Get or create singleton FeedbackStore instance."""
    global _feedback_store_instance
    if _feedback_store_instance is None:
        _feedback_store_instance = FeedbackStore()
    return _feedback_store_instance
