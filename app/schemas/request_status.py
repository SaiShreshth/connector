"""
RequestStatus schema - defines structure for request tracking data.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ResultSchema(BaseModel):
    action_code: Optional[str] = Field(None, description="Action or intent label")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    entities: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RequestStatus(BaseModel):
    request_id: str = Field(..., description="UUID for the request")
    status: str = Field(..., description="Current status (QUEUED, PROCESSING, COMPLETED, FAILED)")
    queued_at: Optional[datetime] = Field(None, description="When request entered queue")
    started_at: Optional[datetime] = Field(None, description="When processing started")
    completed_at: Optional[datetime] = Field(None, description="When processing finished")
    result: Optional[ResultSchema] = Field(None, description="Result payload for completed requests")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details if failed")
    retry_count: int = Field(default=0, description="Retry attempts count")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "COMPLETED",
                "queued_at": "2025-10-19T10:30:00Z",
                "started_at": "2025-10-19T10:30:01Z",
                "completed_at": "2025-10-19T10:30:03Z",
                "result": {
                    "action_code": "SEARCH",
                    "confidence": 0.87,
                    "entities": {"query": "running shoes"}
                },
                "error": None,
                "retry_count": 0
            }
        }

