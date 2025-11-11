"""
Feedback API Endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import csv
import io
import json
import logging

from app.ai.feedback.feedback_store import get_feedback_store
from app.ai.feedback.pattern_analyzer import PatternAnalyzer

logger = logging.getLogger("feedback_api")

router = APIRouter(prefix="/feedback", tags=["Feedback"])

feedback_store = get_feedback_store()
pattern_analyzer = PatternAnalyzer()


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    request_id: str = Field(..., description="Original request ID from classification")
    correct: bool = Field(..., description="Whether the classification was correct")
    expected_action_code: Optional[str] = Field(None, description="Expected/correct action code")
    comment: Optional[str] = Field(None, description="Optional comment or notes")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    feedback_id: str
    message: str
    request_id: str


@router.post("/classifications/flag")
async def flag_misclassification(
    request_id: str = Query(..., description="Request ID to flag"),
    query: str = Query(..., description="Original query text"),
    predicted_action_code: str = Query(..., description="Predicted action code"),
    expected_action_code: str = Query(..., description="Expected action code"),
    confidence: float = Query(0.0, description="Confidence score"),
    comment: Optional[str] = Query(None, description="Optional comment"),
) -> Dict[str, Any]:
    """Flag an incorrect classification."""
    try:
        feedback_id = feedback_store.record_feedback(
            request_id=request_id,
            query=query,
            predicted_action_code=predicted_action_code,
            expected_action_code=expected_action_code,
            confidence=confidence,
            correct=False,
            comment=comment,
        )
        
        return {
            "feedback_id": feedback_id,
            "message": "Misclassification flagged successfully",
            "request_id": request_id,
        }
    except Exception as e:
        logger.error(f"Error flagging misclassification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classifications")
async def submit_feedback(
    request_id: str = Query(..., description="Request ID"),
    query: str = Query(..., description="Original query text"),
    predicted_action_code: str = Query(..., description="Predicted action code"),
    correct: bool = Query(True, description="Whether classification was correct"),
    expected_action_code: Optional[str] = Query(None, description="Expected action code (if incorrect)"),
    confidence: float = Query(0.0, description="Confidence score"),
    comment: Optional[str] = Query(None, description="Optional comment"),
) -> Dict[str, Any]:
    """Submit feedback about a classification (correct or incorrect)."""
    try:
        feedback_id = feedback_store.record_feedback(
            request_id=request_id,
            query=query,
            predicted_action_code=predicted_action_code,
            expected_action_code=expected_action_code,
            confidence=confidence,
            correct=correct,
            comment=comment,
        )
        
        return {
            "feedback_id": feedback_id,
            "message": "Feedback recorded successfully",
            "request_id": request_id,
        }
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/misclassifications")
async def get_misclassifications(
    action_code: Optional[str] = Query(None, description="Filter by action code"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
) -> Dict[str, Any]:
    """Get list of misclassified examples."""
    try:
        misclassifications = feedback_store.get_misclassifications(
            action_code=action_code,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
        
        return {
            "count": len(misclassifications),
            "misclassifications": misclassifications,
        }
    except Exception as e:
        logger.error(f"Error getting misclassifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
async def export_misclassifications(
    format: str = Query("csv", regex="^(csv|json)$", description="Export format (csv or json)"),
    action_code: Optional[str] = Query(None, description="Filter by action code"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
) -> StreamingResponse:
    """Export misclassified examples for prompt refinement."""
    try:
        misclassifications = feedback_store.get_misclassifications(
            action_code=action_code,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )
        
        if format == "csv":
            output = io.StringIO()
            if misclassifications:
                writer = csv.DictWriter(
                    output,
                    fieldnames=[
                        "timestamp",
                        "query",
                        "predicted_action_code",
                        "actual_action_code",
                        "confidence",
                        "comment",
                    ],
                )
                writer.writeheader()
                for record in misclassifications:
                    writer.writerow({
                        "timestamp": record.get("timestamp", ""),
                        "query": record.get("query", ""),
                        "predicted_action_code": record.get("predicted_action_code", ""),
                        "actual_action_code": record.get("actual_action_code", ""),
                        "confidence": record.get("confidence", 0.0),
                        "comment": record.get("comment", ""),
                    })
            
            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=misclassifications.csv"},
            )
        
        else:
            return JSONResponse(
                content={
                    "count": len(misclassifications),
                    "misclassifications": misclassifications,
                },
                headers={"Content-Disposition": "attachment; filename=misclassifications.json"},
            )
    
    except Exception as e:
        logger.error(f"Error exporting misclassifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics."""
    try:
        stats = feedback_store.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_patterns(
    action_code: Optional[str] = Query(None, description="Filter by action code"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
) -> Dict[str, Any]:
    """Analyze misclassification patterns."""
    try:
        patterns = pattern_analyzer.analyze_patterns(
            action_code=action_code,
            start_date=start_date,
            end_date=end_date,
        )
        return patterns
    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))
