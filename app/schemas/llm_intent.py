"""Pydantic models for the LLM intent classification API."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMIntentRequest(BaseModel):
    """Request payload forwarded from the rule-based pipeline to the LLM."""

    user_input: str = Field(..., description="Raw user utterance requiring clarification.")
    rule_intent: Optional[str] = Field(
        None, description="Intent name predicted by the rule-based classifier."
    )
    action_code: Optional[str] = Field(
        None, description="Action code emitted by the rule-based classifier."
    )
    top_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Highest confidence score from the rule-based pass."
    )
    next_best_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Runner-up confidence score from the rule-based pass."
    )
    is_fallback: bool = Field(
        False, description="Whether the rule-based flow already emitted a fallback response."
    )
    context_snippets: List[Any] = Field(
        default_factory=list,
        description="Optional contextual snippets (conversation history, metadata, etc.). Can be strings or dicts with role/content.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic information from upstream components.",
    )


class LLMIntentResponse(BaseModel):
    """Normalized response structure returned by the hybrid classifier."""

    triggered: bool = Field(..., description="True when the LLM classifier was invoked.")
    intent: str = Field(..., description="Resolved intent name (LLM or rule-based).")
    intent_category: str = Field(..., description="Resolved intent category.")
    action_code: str = Field(..., description="Resolved action code consumed downstream.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score associated with the resolved intent."
    )
    requires_clarification: bool = Field(
        False, description="Whether the caller should prompt the user for clarification."
    )
    clarification_message: Optional[str] = Field(
        None, description="Suggested clarification prompt to present to the user."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata (trigger reason, processing stats, etc.).",
    )


class LLMIntentSimpleRequest(BaseModel):
    """Minimal request payload: only raw user input."""

    user_input: str = Field(..., description="Raw user utterance requiring clarification.")