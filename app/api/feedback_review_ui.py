"""
Feedback Review UI

Simple HTML interface for reviewing and flagging misclassifications.
"""

from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.ai.feedback.feedback_store import get_feedback_store
import os

router = APIRouter(prefix="/dashboard", tags=["Feedback Review"])
templates = Jinja2Templates(directory=os.path.join("app", "templates"))

feedback_store = get_feedback_store()


@router.get("/feedback-review", response_class=HTMLResponse)
async def feedback_review_page(
    request: Request,
    action_code: str = Query(None, description="Filter by action code"),
    limit: int = Query(50, ge=1, le=200, description="Number of records to show"),
):
    """Render feedback review interface."""
    try:
        misclassifications = feedback_store.get_misclassifications(
            action_code=action_code,
            limit=limit,
        )
        stats = feedback_store.get_stats()
        
        return templates.TemplateResponse(
            "feedback_review.html",
            {
                "request": request,
                "misclassifications": misclassifications,
                "stats": stats,
                "action_code_filter": action_code,
                "limit": limit,
            },
        )
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>Error loading feedback review</h1><p>{str(e)}</p></body></html>",
            status_code=500,
        )
