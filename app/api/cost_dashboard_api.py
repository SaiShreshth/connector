"""
Simple API endpoint to expose LLM cost and token usage metrics.
"""

from fastapi import APIRouter
from app.ai.cost_monitor.usage_tracker import UsageTracker

router = APIRouter(prefix="/api", tags=["Cost Monitoring"])
usage_tracker = UsageTracker()

@router.get("/cost_metrics")
def get_cost_metrics():
    """
    Returns:
        {
          "daily": {"tokens": ..., "cost": ..., "requests": ...},
          "monthly": {"tokens": ..., "cost": ..., "requests": ...}
        }
    """
    daily = usage_tracker.get_daily()
    monthly = usage_tracker.get_monthly()
    return {"daily": daily, "monthly": monthly}
