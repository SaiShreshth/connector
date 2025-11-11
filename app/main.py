"""
Intent Classification API - Main Application Entry Point
"""
import os
import time
import uuid
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Routers
from app.status_api import router as status_router
from app.api.cost_dashboard_api import router as cost_dashboard_router
from app.api.cost_dashboard_ui import router as cost_dashboard_ui_router
from app.api.ab_testing_api import router as ab_testing_router
from app.api.testing_framework_api import router as testing_framework_router
from app.api.feedback_review_ui import router as feedback_review_ui_router
from app.ai.cost_monitor.scheduler import start_scheduler

# Qdrant client setup
from qdrant_client import QdrantClient, models

# Load environment variables
load_dotenv()

# Validate environment variables at startup
from app.core.env_validator import validate_environment
try:
    env_settings = validate_environment()
except SystemExit:
    raise  # Re-raise to prevent startup
except Exception as e:
    logger.error(f"Failed to validate environment: {e}")
    raise SystemExit(1)

# Configure loguru
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Remove default handler and add custom ones
logger.remove()
logger.add(
    os.path.join(LOG_DIR, "app.log"),
    rotation="100 MB",
    retention="30 days",
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    enqueue=True,  # Thread-safe logging
)
logger.add(
    lambda msg: print(msg, end=""),
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


# Structured logging with correlation ID and error context
def log_with_context(level: str, message: str, error: Exception | None = None, context: str | None = None):
    correlation_id = str(uuid.uuid4())
    extra_context = {"correlation_id": correlation_id}
    if context:
        extra_context["context"] = context
    
    if error:
        logger.bind(**extra_context).error(
            message,
            exc_info=error
        )
    else:
        log_method = getattr(logger.bind(**extra_context), level.lower(), logger.info)
        log_method(message)
    return correlation_id


# Note: Real ResilientOpenAIClient is in app.ai.llm_intent.resilient_openai_client
# and is used via RequestHandler for proper LLM calls

# Import decision engine and routers
logger.info("Attempting to import Decision Engine...")
from app.ai.intent_classification.decision_engine import get_intent_classification
from app.api.v1.intent import router as intent_router

try:
    from app.api.v1.queue import router as queue_router
    QUEUE_ROUTER_AVAILABLE = True
except ImportError:
    QUEUE_ROUTER_AVAILABLE = False

try:
    from app.api.v1.cache import router as cache_router
    CACHE_ROUTER_AVAILABLE = True
except ImportError:
    CACHE_ROUTER_AVAILABLE = False

logger.info("Successfully imported Decision Engine.")

# Queue Infrastructure Import
try:
    from app.queue.queue_manager import queue_manager
    from app.queue.monitor import queue_monitor
    QUEUE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Queue infrastructure not available: {e}")
    queue_manager = None
    queue_monitor = None
    QUEUE_AVAILABLE = False

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
PRODUCT_COLLECTION_NAME = "chatnshop_products"
VECTOR_SIZE = 384  # Must match embedding model
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Use QDRANT_URL if set, otherwise construct from host/port
if not QDRANT_URL or QDRANT_URL == "http://localhost:6333":
    QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# Initialize Qdrant Client with Retry Logic
logger.info(f"Attempting to connect to Qdrant at {QDRANT_URL}...")
qdrant_client = None
retries = 5
wait_time = 3

for i in range(retries):
    try:
        if QDRANT_API_KEY:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
        else:
            qdrant_client = QdrantClient(url=QDRANT_URL, timeout=10)
        qdrant_client.get_collections()
        logger.info(f"Connected to Qdrant at {QDRANT_URL}")
        break
    except Exception as e:
        logger.warning(f"Attempt {i + 1} failed: Could not connect to Qdrant. Error: {e}")
        if i < retries - 1:
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            logger.error(f"FAILED to initialize Qdrant client after {retries} attempts.")


# Lifespan hook (app startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown manager."""
    logger.info("Starting Intent Classification API...")

    # Initialize Queue Infrastructure
    if QUEUE_AVAILABLE and queue_manager:
        try:
            if queue_manager.health_check():
                logger.info("Queue infrastructure ready (Redis connected)")
            else:
                logger.warning("Queue infrastructure available but Redis not connected")
        except Exception as e:
            logger.warning(f"Queue health check failed: {e}")
    else:
        logger.warning("Queue infrastructure not available (continuing without async processing)")

    # Validate configuration files
    try:
        from app.core.config_manager import CONFIG_CACHE, load_all_configs
        load_all_configs()
        if "rules" in CONFIG_CACHE:
            logger.info("✅ Configuration files validated and loaded")
        else:
            logger.warning("⚠️  Configuration file 'rules.json' not found - using defaults")
    except Exception as e:
        logger.warning(f"Configuration validation warning: {e} - continuing with defaults")
    
    # Model Warmup - Warm up Decision Engine (load models)
    try:
        get_intent_classification("warm up")
        logger.info("Models loaded and Decision Engine is warm.")
    except Exception as e:
        logger.error(f"ERROR during model warmup: {e}")
        log_with_context("ERROR", "Model warmup failed", e)
        # Don't fail startup if model warmup fails - might be recoverable

    # Verify or create Qdrant collection
    if qdrant_client:
        try:
            # Check if collection exists
            collections = qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if PRODUCT_COLLECTION_NAME not in collection_names:
                qdrant_client.create_collection(
                    collection_name=PRODUCT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
                )
                logger.info(f"✅ Qdrant collection '{PRODUCT_COLLECTION_NAME}' created.")
            else:
                logger.info(f"✅ Qdrant collection '{PRODUCT_COLLECTION_NAME}' exists and is accessible.")
        except Exception as e:
            logger.error(f"❌ Could not create/verify Qdrant collection: {e}")
            log_with_context("ERROR", "Could not create/verify Qdrant collection", e)
            # Don't fail startup - might be recoverable
    else:
        logger.warning("⚠️  Qdrant client not initialized, skipping collection creation.")

    # Initialize Cost Monitoring Scheduler
    try:
        start_scheduler()
        logger.info("Cost monitoring scheduler initialized.")
    except Exception as e:
        logger.warning(f"Scheduler init failed: {e}")
    
    # Initialize Review Scheduler (for weekly/monthly reports)
    try:
        from app.ai.feedback.review_scheduler import start_review_scheduler
        review_scheduler = start_review_scheduler()
        if review_scheduler:
            logger.info("Review scheduler initialized.")
    except Exception as e:
        logger.warning(f"Review scheduler failed: {e}")

    logger.info("Intent Classification API started successfully!")
    
    yield
    
    # Graceful shutdown
    logger.info("Shutting down Intent Classification API...")
    
    # Close async Redis connections
    try:
        from app.core.async_redis_client import close_async_redis
        await close_async_redis()
        logger.info("✅ Async Redis connections closed")
    except Exception as e:
        logger.warning(f"Error closing async Redis: {e}")
    
    # Close other async resources if needed
    # Add cleanup for other async clients here
    
    logger.info("✅ Shutdown complete.")


# FastAPI app setup
app = FastAPI(
    title=os.getenv("APP_NAME", "Intent Classification API"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="Hybrid rule-based + LLM intent classification backend for chatNShop",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import authentication and rate limiting
from app.core.auth import optional_auth, require_auth, UserContext
from app.core.rate_limiter import get_rate_limiter, rate_limit_middleware


# Request ID and Timing Middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add correlation ID to all requests for tracing."""
    # Get or generate request ID
    correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    
    # Add to request state for access in endpoints
    request.state.correlation_id = correlation_id
    
    # Bind logger with correlation ID for this request
    logger_context = logger.bind(correlation_id=correlation_id)
    
    # Log request start
    start_time = time.time()
    logger_context.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        # Add correlation ID to response headers
        response.headers["X-Request-ID"] = correlation_id
        
        # Log request completion
        duration_ms = (time.time() - start_time) * 1000
        logger_context.info(
            f"Response: {request.method} {request.url.path} - "
            f"{response.status_code} ({duration_ms:.2f}ms)"
        )
        
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger_context.error(
            f"Request failed: {request.method} {request.url.path} - "
            f"{type(e).__name__} ({duration_ms:.2f}ms)",
            exc_info=e
        )
        raise


# Root Endpoint
@app.get("/", tags=["Health"])
async def root() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Intent Classification API",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "message": "Hybrid Intent Classification System is running!"
    }


# Health Endpoint (Comprehensive)
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    qdrant_status = "disconnected"
    if qdrant_client:
        try:
            qdrant_client.get_collections()
            qdrant_status = "connected"
        except Exception:
            qdrant_status = "unhealthy"

    redis_status = "unavailable"
    queue_stats = {}
    if QUEUE_AVAILABLE and queue_manager:
        try:
            if queue_manager.health_check():
                redis_status = "connected"
                queue_stats = queue_manager.get_queue_stats()
            else:
                redis_status = "disconnected"
        except Exception as e:
            redis_status = f"error: {str(e)}"

    openai_status = "not_configured"
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        openai_status = "configured" if api_key else "not_configured"
    except Exception:
        openai_status = "unknown"

    overall_status = "healthy"
    if qdrant_status == "unhealthy" or redis_status.startswith("error"):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {
            "qdrant": qdrant_status,
            "redis": redis_status,
            "openai": openai_status
        },
        "queue": queue_stats,
        "version": os.getenv("APP_VERSION", "1.0.0")
    }


# Readiness Probe (Kubernetes-style)
@app.get("/health/readiness", tags=["Health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe endpoint.
    Returns 200 only if all critical dependencies are ready to serve traffic.
    """
    checks = {
        "qdrant": False,
        "redis": False,
    }
    
    # Check Qdrant
    if qdrant_client:
        try:
            qdrant_client.get_collections()
            checks["qdrant"] = True
        except Exception:
            checks["qdrant"] = False
    
    # Check Redis (via queue manager)
    if QUEUE_AVAILABLE and queue_manager:
        try:
            checks["redis"] = queue_manager.health_check()
        except Exception:
            checks["redis"] = False
    
    # Readiness = all critical services ready
    is_ready = all(checks.values())
    
    status_code = 200 if is_ready else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": checks,
        }
    )


# Liveness Probe (Kubernetes-style)
@app.get("/health/liveness", tags=["Health"])
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness probe endpoint.
    Returns 200 if application is running (doesn't check dependencies).
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# Classification Models
class ClassificationInput(BaseModel):
    text: str = Field(..., max_length=10000, description="User input text (max 10,000 characters)")


class ClassificationOutput(BaseModel):
    action_code: str = Field(..., description="Resolved action code")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence (0-1)")
    matched_keywords: List[str] = Field(default_factory=list, description="Matched keyword phrases")
    original_text: str = Field(..., description="Original input text")
    status: str = Field(..., description="Resolution status (e.g., CONFIDENT_KEYWORD, FALLBACK_*)")
    intent: Optional[Dict[str, Any]] = Field(default=None, description="Raw top intent payload for debug")
    entities: Optional[Dict[str, Any]] = Field(default=None, description="Extracted entities (product, brand, color, etc.)")


# Classification Endpoint
@app.post(
    "/classify",
    tags=["Intent Classification"],
    response_model=ClassificationOutput,
    summary="Classify user input into an intent",
    responses={
        200: {
            "description": "Classification result",
            "content": {
                "application/json": {
                    "example": {
                        "action_code": "SEARCH_PRODUCT",
                        "confidence_score": 0.92,
                        "matched_keywords": ["search", "shoes"],
                        "original_text": "Show me red Nike running shoes under $100",
                        "status": "LLM_CLASSIFICATION",
                        "intent": {
                            "id": "SEARCH_PRODUCT",
                            "score": 0.92,
                            "source": "llm",
                        },
                        "entities": {
                            "product_type": "shoes",
                            "category": "running",
                            "brand": "Nike",
                            "color": "red",
                            "size": None,
                            "price_range": {"min": None, "max": 100, "currency": "USD"}
                        }
                    }
                }
            },
        },
        500: {
            "description": "Internal error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Classification Failed",
                        "message": "An internal error occurred while processing the request.",
                        "detail": "<error details>",
                    }
                }
            },
        },
    },
)
async def classify_intent(
    user_input: ClassificationInput,
    request: Request,
    current_user: UserContext = Depends(optional_auth)
) -> ClassificationOutput:
    # Rate limiting (async)
    limiter = get_rate_limiter()
    await limiter.check_rate_limit_async(request, current_user.user_id if current_user.is_authenticated else None)
    
    # Get correlation ID from middleware (already set)
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    user_id = current_user.user_id if current_user.is_authenticated else "anonymous"
    logger.bind(correlation_id=correlation_id, user_id=user_id).info(f"Received text: {user_input.text}")

    try:
        result = get_intent_classification(user_input.text)
        top = result.get("intent", {}) if isinstance(result, dict) else {}
        action_code = top.get("id") or top.get("intent")
        confidence_score = top.get("score")
        matched_kw = top.get("matched_text")

        result.setdefault("action_code", action_code or "UNKNOWN_INTENT")
        result.setdefault("confidence_score", confidence_score or 0.0)
        if matched_kw:
            result.setdefault("matched_keywords", [matched_kw])
        result["original_text"] = user_input.text
        
        # Track intent distribution and confidence (classification logging is done in decision_engine)
        try:
            from app.ai.monitoring.intent_distribution_tracker import get_intent_distribution_tracker
            from app.ai.monitoring.confidence_tracker import get_confidence_tracker
            
            intent_tracker = get_intent_distribution_tracker()
            confidence_tracker = get_confidence_tracker()
            
            intent_tracker.record_intent(result["action_code"])
            confidence_tracker.record_confidence(float(result["confidence_score"]))
        except Exception:
            pass  # Non-critical, continue

        return ClassificationOutput(
            action_code=result["action_code"],
            confidence_score=float(result["confidence_score"]),
            matched_keywords=result.get("matched_keywords", []),
            original_text=user_input.text,
            status=result.get("status", "UNKNOWN"),
            intent=result.get("intent"),
            entities=result.get("entities"),
        )
    except Exception as e:
        error_id = log_with_context("ERROR", "Classification failed", e, context=user_input.text)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Classification Failed",
                "message": "Internal processing error. Please retry.",
                "error_id": error_id,
            },
        )


# Include routers
app.include_router(intent_router)
app.include_router(status_router)
app.include_router(cost_dashboard_router)
app.include_router(cost_dashboard_ui_router)
app.include_router(ab_testing_router)
app.include_router(testing_framework_router)

if QUEUE_ROUTER_AVAILABLE:
    try:
        app.include_router(queue_router, prefix="/api/v1")
    except Exception as e:
        logger.warning(f"Failed to include queue router: {e}")

if CACHE_ROUTER_AVAILABLE:
    try:
        app.include_router(cache_router, prefix="/api/v1")
    except Exception as e:
        logger.warning(f"Failed to include cache router: {e}")

# Include feedback and monitoring routers
try:
    from app.api.v1.feedback import router as feedback_router
    from app.api.v1.monitoring import router as monitoring_router
    app.include_router(feedback_router, prefix="/api/v1")
    app.include_router(monitoring_router, prefix="/api/v1")
    app.include_router(feedback_review_ui_router)
    logger.info("Feedback and monitoring routers registered")
except Exception as e:
    logger.warning(f"Failed to include feedback/monitoring routers: {e}")


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = log_with_context("ERROR", f"Unhandled exception at {request.url}", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "error_id": error_id,
            "path": str(request.url),
        },
    )


# Entrypoint
def run():
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        workers=int(os.getenv("WORKERS", 1)),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    run()
