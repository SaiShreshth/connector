"""
Classification Logger

Logs all LLM classifications with query and context for feedback and analysis.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Any, Optional

_lock = Lock()
CLASSIFICATION_LOG_FILE = os.path.join("data", "classification_logs.jsonl")


def _ensure_log_file():
    """Ensure the log file directory exists."""
    os.makedirs(os.path.dirname(CLASSIFICATION_LOG_FILE), exist_ok=True)


class ClassificationLogger:
    """Logs all classification results for feedback and continuous improvement."""

    def __init__(self, log_file: str = CLASSIFICATION_LOG_FILE):
        self.log_file = log_file
        _ensure_log_file()

    def log_classification(
        self,
        query: str,
        action_code: str,
        confidence: float,
        source: str,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a classification result."""
        log_entry = {
            "log_id": str(uuid.uuid4()),
            "request_id": request_id or str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "action_code": action_code,
            "confidence": confidence,
            "source": source,
            "context": context or {},
            "metadata": metadata or {},
        }
        
        with _lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_llm_classification(
        self,
        query: str,
        action_code: str,
        confidence: float,
        request_id: Optional[str] = None,
        prompt_version: Optional[str] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None,
        context_snippets: Optional[list] = None,
        entities: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log an LLM classification with LLM-specific metadata."""
        metadata = {
            "prompt_version": prompt_version,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "entities": entities,
            **kwargs
        }
        
        context = {}
        if context_snippets:
            context["context_snippets"] = context_snippets
        
        self.log_classification(
            query=query,
            action_code=action_code,
            confidence=confidence,
            source="llm",
            request_id=request_id,
            context=context,
            metadata=metadata,
        )

    def log_rule_based_classification(
        self,
        query: str,
        action_code: str,
        confidence: float,
        request_id: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        match_type: Optional[str] = None,
        matched_text: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log a rule-based (keyword/embedding) classification."""
        metadata = {
            "match_type": match_type,
            "matched_text": matched_text,
            "entities": entities,
            **kwargs
        }
        
        self.log_classification(
            query=query,
            action_code=action_code,
            confidence=confidence,
            source="keyword",
            request_id=request_id,
            context={},
            metadata=metadata,
        )


_classification_logger_instance: Optional[ClassificationLogger] = None


def get_classification_logger() -> ClassificationLogger:
    """Get or create singleton ClassificationLogger instance."""
    global _classification_logger_instance
    if _classification_logger_instance is None:
        _classification_logger_instance = ClassificationLogger()
    return _classification_logger_instance
