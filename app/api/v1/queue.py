"""
Queue Management API Endpoints (CNS-21)

Provides endpoints for monitoring and managing the queue infrastructure.
Queue Management API Endpoints

Provides endpoints for monitoring and managing the queue infrastructure.
Includes status tracking endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/queue", tags=["Queue"])

# Import queue components
try:
    from app.queue.queue_manager import queue_manager
    from app.queue.monitor import queue_monitor
    QUEUE_ENABLED = True
except ImportError:
    queue_manager = None
    queue_monitor = None
    QUEUE_ENABLED = False

# Import status store
try:
    from app.core.status_store import status_store
    STATUS_STORE_AVAILABLE = True
except ImportError:
    status_store = None
    STATUS_STORE_AVAILABLE = False


@router.get("/health")
async def queue_health() -> Dict[str, Any]:
    """Check queue system health."""
    if not QUEUE_ENABLED or queue_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Queue infrastructure not available"
        )
    
    is_healthy = queue_manager.health_check()
    return {
        "healthy": is_healthy,
        "available": queue_manager.is_available() if queue_manager else False
    }


@router.get("/stats")
async def get_queue_stats() -> Dict[str, Any]:
    """Get queue statistics."""
    if not QUEUE_ENABLED or queue_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Queue infrastructure not available"
        )
    
    stats = queue_manager.get_queue_stats()
    return stats


@router.get("/metrics")
async def get_queue_metrics() -> Dict[str, Any]:
    """Get queue performance metrics."""
    if not QUEUE_ENABLED or queue_monitor is None:
        raise HTTPException(
            status_code=503,
            detail="Queue monitoring not available"
        )
    
    metrics = queue_monitor.get_metrics()
    return metrics


@router.post("/clear/{queue_name}")
async def clear_queue(queue_name: str) -> Dict[str, Any]:
    """
    Clear specific queue (admin only).
    
    Allowed queue names:
    - ambiguous_queries
    - classification_results
    - dead_letter
    """
    if not QUEUE_ENABLED or queue_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Queue infrastructure not available"
        )
    
    # Map queue names to actual Redis keys
    queue_map = {
        "ambiguous_queries": "chatns:queue:ambiguous_queries",
        "classification_results": "chatns:queue:classification_results",
        "dead_letter": "chatns:queue:dead_letter"
    }
    
    if queue_name not in queue_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid queue name. Allowed: {list(queue_map.keys())}"
        )
    
    try:
        queue_manager.clear_queue(queue_map[queue_name])
        return {
            "status": "cleared",
            "queue": queue_name,
            "message": f"Queue '{queue_name}' cleared successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear queue: {str(e)}"
        )


@router.get("/status/{request_id}")
async def get_request_status(request_id: str) -> Dict[str, Any]:
    """
    Get status of a classification request.
    
    Returns request status with timestamps and result (if completed).
    Fast lookup (<10ms p95).
    """
    if not STATUS_STORE_AVAILABLE or status_store is None:
        raise HTTPException(
            status_code=503,
            detail="Status store not available"
        )
    
    try:
        status = status_store.get(request_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Request {request_id} not found. It may have expired or never existed."
            )
        
        # Convert to dict for JSON response
        status_dict = status.model_dump(mode='json')
        
        # Convert datetime to ISO string
        for field in ['queued_at', 'started_at', 'completed_at']:
            if status_dict.get(field):
                if hasattr(status_dict[field], 'isoformat'):
                    status_dict[field] = status_dict[field].isoformat() + 'Z'
        
        return status_dict
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve status: {str(e)}"
        )

