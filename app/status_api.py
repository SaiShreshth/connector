# app/status_api.py
from fastapi import APIRouter, HTTPException, Query
from app.ai.intent_classification.cache.status_store import StatusStore

# ✅ Router instance
router = APIRouter(prefix="/status", tags=["Status API"])

# ✅ Initialize store (persistent + in-memory)
status_store = StatusStore()


# -------------------------------------------------------------------
# 🔹 Fetch status for a given request ID
# -------------------------------------------------------------------
@router.get("/{request_id}")
def get_status(request_id: str):
    """
    Fetch detailed status (including tokens, cost, timestamps)
    Example: /status/12345
    """
    data = status_store.get_status(request_id)
    if data.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=data["message"])
    return data


# -------------------------------------------------------------------
# 🔹 Set or update status manually
# -------------------------------------------------------------------
@router.post("/{request_id}")
def set_status(request_id: str, status: str, message: str = ""):
    """
    Manually set or update a request's status.
    Example: /status/12345?status=completed&message=Done
    """
    status_store.set_status(request_id, status, message)
    return {
        "message": "Status updated",
        "request_id": request_id,
        "status": status,
        "info": message,
    }


# -------------------------------------------------------------------
# 🔹 Log token usage & cost manually (admin or debug purpose)
# -------------------------------------------------------------------
@router.post("/{request_id}/log")
def log_cost_usage(
    request_id: str,
    prompt_tokens: int = Query(..., description="Prompt token count"),
    completion_tokens: int = Query(..., description="Completion token count"),
    cost: float = Query(..., description="Total API cost in USD"),
):
    """
    Log token usage & cost for a given request manually.
    Example:
    /status/req123/log?prompt_tokens=100&completion_tokens=40&cost=0.0025
    """
    status_store.log_usage(request_id, prompt_tokens, completion_tokens, cost)
    return {
        "message": "Usage logged",
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost,
    }


# -------------------------------------------------------------------
# 🔹 Fetch daily + monthly totals (for dashboard)
# -------------------------------------------------------------------
@router.get("/summary/daily")
def get_daily_total():
    """Get total cost spent today"""
    return {"daily_total_cost": status_store.get_daily_total()}


@router.get("/summary/monthly")
def get_monthly_total():
    """Get total cost spent this month"""
    return {"monthly_total_cost": status_store.get_monthly_total()}
