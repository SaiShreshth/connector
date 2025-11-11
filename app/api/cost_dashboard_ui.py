"""
Visual Cost Dashboard (HTML + Chart.js)
Enhanced with intent distribution and confidence score monitoring
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.ai.cost_monitor.usage_tracker import UsageTracker
from app.ai.monitoring.intent_distribution_tracker import get_intent_distribution_tracker
from app.ai.monitoring.confidence_tracker import get_confidence_tracker
import os

# ✅ Router with prefix ensures endpoint = /dashboard/cost
router = APIRouter(prefix="/dashboard", tags=["Cost Dashboard"])

# ✅ Template directory — matches your project's folder structure
templates = Jinja2Templates(directory=os.path.join("app", "templates"))

# ✅ Initialize trackers
tracker = UsageTracker()
intent_tracker = get_intent_distribution_tracker()
confidence_tracker = get_confidence_tracker()

# ✅ Dashboard route
@router.get("/cost", response_class=HTMLResponse)
async def show_cost_dashboard(request: Request):
    """Renders the enhanced cost dashboard page with monitoring metrics"""
    daily = tracker.get_daily()
    monthly = tracker.get_monthly()
    
    # Get intent distribution (top 10)
    top_intents = intent_tracker.get_top_intents(period="daily", limit=10)
    intent_distribution = {
        "labels": [code for code, _ in top_intents],
        "counts": [count for _, count in top_intents],
    }
    
    # Get confidence distribution
    confidence_summary = confidence_tracker.get_distribution_summary(period="daily")
    confidence_distribution = {
        "labels": list(confidence_summary.get("distribution", {}).keys()),
        "counts": list(confidence_summary.get("distribution", {}).values()),
        "percentages": list(confidence_summary.get("percentages", {}).values()),
    }

    return templates.TemplateResponse(
        "cost_dashboard.html",
        {
            "request": request,
            "daily": daily,
            "monthly": monthly,
            "intent_distribution": intent_distribution,
            "confidence_distribution": confidence_distribution,
        },
    )
