"""
Monitoring API Endpoints

Provides endpoints for monitoring metrics including intent distribution,
confidence scores, latency, and accuracy.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import logging

from app.ai.monitoring.intent_distribution_tracker import get_intent_distribution_tracker
from app.ai.monitoring.confidence_tracker import get_confidence_tracker
from app.ai.feedback.accuracy_tracker import AccuracyTracker
from app.ai.feedback.feedback_store import get_feedback_store

logger = logging.getLogger("monitoring_api")

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

intent_tracker = get_intent_distribution_tracker()
confidence_tracker = get_confidence_tracker()
accuracy_tracker = AccuracyTracker()
feedback_store = get_feedback_store()


@router.get("/intent-distribution")
async def get_intent_distribution(
    period: str = Query("daily", regex="^(daily|weekly|monthly)$", description="Time period"),
) -> Dict[str, Any]:
    """Get intent distribution over time."""
    try:
        if period == "daily":
            distribution = intent_tracker.get_daily_distribution()
        elif period == "weekly":
            distribution = intent_tracker.get_weekly_distribution()
        else:
            distribution = intent_tracker.get_monthly_distribution()
        
        top_intents = intent_tracker.get_top_intents(period=period, limit=10)
        
        return {
            "period": period,
            "total_intents": sum(distribution.values()),
            "distribution": distribution,
            "top_intents": [{"action_code": code, "count": count} for code, count in top_intents],
        }
    except Exception as e:
        logger.error(f"Error getting intent distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/confidence-distribution")
async def get_confidence_distribution(
    period: str = Query("daily", regex="^(daily|weekly|monthly)$", description="Time period"),
) -> Dict[str, Any]:
    """Get confidence score distribution."""
    try:
        summary = confidence_tracker.get_distribution_summary(period=period)
        return {
            "period": period,
            **summary,
        }
    except Exception as e:
        logger.error(f"Error getting confidence distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accuracy-metrics")
async def get_accuracy_metrics() -> Dict[str, Any]:
    """Get accuracy metrics and trends."""
    try:
        summary = accuracy_tracker.get_accuracy_summary()
        trends = accuracy_tracker.get_accuracy_trends(days=30)
        
        return {
            **summary,
            "trends": trends,
        }
    except Exception as e:
        logger.error(f"Error getting accuracy metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample-queries")
async def get_sample_queries(
    limit: int = Query(100, ge=1, le=1000, description="Number of sample queries"),
) -> Dict[str, Any]:
    """Get sample queries for quality review."""
    try:
        import json
        import os
        
        log_file = os.path.join("data", "classification_logs.jsonl")
        queries = []
        
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    try:
                        entry = json.loads(line.strip())
                        queries.append({
                            "query": entry.get("query", ""),
                            "action_code": entry.get("action_code", ""),
                            "confidence": entry.get("confidence", 0.0),
                            "source": entry.get("source", ""),
                            "timestamp": entry.get("timestamp", ""),
                        })
                    except Exception:
                        continue
        
        return {
            "count": len(queries),
            "queries": queries,
        }
    except Exception as e:
        logger.error(f"Error getting sample queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))
