"""Intent classification API endpoints."""

from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException, Depends

from app.ai.llm_intent.request_handler import RequestHandler
from app.ai.llm_intent.openai_client import OpenAIClient
from app.schemas.llm_intent import LLMIntentRequest, LLMIntentResponse, LLMIntentSimpleRequest
from app.core.auth import optional_auth, UserContext
import logging


router = APIRouter(prefix="/api/v1/llm-intent", tags=["LLM Intent"])

# Instantiate with a real OpenAI client if environment variables are set
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
openai_temp = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "400"))

logger = logging.getLogger("intent_api")

if openai_api_key:
    client = OpenAIClient(
        api_key=openai_api_key,
        model_name=openai_model,
        temperature=openai_temp,
        max_tokens=openai_max_tokens
    )
    handler = RequestHandler(client)
    logger.info("✅ LLM Intent handler initialized with OpenAI client")
else:
    logger.warning("⚠️ OPENAI_API_KEY not set, LLM Intent handler will use simulated responses")
    handler = RequestHandler()  # Will return simulated responses


@router.post("/classify", response_model=LLMIntentResponse, summary="Direct LLM intent classification")
async def classify_intent(
    request: LLMIntentRequest,
    current_user: UserContext = Depends(optional_auth)
) -> LLMIntentResponse:
    """
    Direct LLM intent classification endpoint.
    
    This endpoint processes the request synchronously and returns the LLM response immediately.
    It accepts full LLMIntentRequest parameters including:
    - user_input: The user's query
    - rule_intent: Previous intent from rule-based system (optional)
    - action_code: Previous action code (optional)
    - top_confidence: Confidence score from rule-based (optional, default 0.0)
    - next_best_confidence: Runner-up confidence (optional, default 0.0)
    - is_fallback: Whether this is a fallback request (optional, default False)
    - context_snippets: Conversation history or context (optional)
    - metadata: Additional metadata (optional)
    
    Returns LLM classification result immediately.
    """
    
    # Ensure handler has OpenAI client
    if handler.client is None:
        # Try to re-initialize with API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAIClient(
                api_key=api_key,
                model_name=openai_model,
                temperature=openai_temp,
                max_tokens=openai_max_tokens
            )
            handler.client = client
            logger.info("✅ Re-initialized LLM handler with OpenAI client")
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OpenAI API not configured",
                    "message": "OPENAI_API_KEY environment variable is not set. Cannot process LLM requests."
                }
            )
    
    # Process with LLM handler (synchronous, direct processing)
    # Force LLM processing: set is_fallback=True to bypass trigger checks
    # This endpoint always calls LLM directly
    # Create a new request with is_fallback=True to ensure LLM is called
    forced_request = LLMIntentRequest(
        user_input=request.user_input,
        rule_intent=request.rule_intent,
        action_code=request.action_code,
        top_confidence=request.top_confidence,
        next_best_confidence=request.next_best_confidence,
        is_fallback=True,  # Force LLM processing
        context_snippets=request.context_snippets,
        metadata=request.metadata
    )
    
    try:
        result = handler.handle(forced_request)
        
        # Always return LLM response (handler will process with LLM since is_fallback=True)
        return LLMIntentResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ LLM classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "LLM processing failed",
                "message": str(e)
            }
        )
