from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
import os
from urllib.parse import urlparse
# from loguru import logger

import logging

# Configure logging to show INFO level and above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from app.config import Settings, build_pg_dsn
from app.db.postgres import init_pool, close_pool
from app.llm_connectors.shopify_client import ShopifyClient

# -----------------------------------------------------------------------------
# FASTAPI APP CONFIGURATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="E-commerce Orchestrator API",
    description="Routes user input → Intent Classifier → Appropriate Backend",
    version="1.0.0"
)


# -----------------------------------------------------------------------------
# STARTUP / SHUTDOWN
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    settings = Settings()
    app.state.settings = settings
    logger.info("Orchestrator startup: settings loaded")

    has_dsn = bool(build_pg_dsn(settings))
    logger.info("PostgreSQL DSN provided: %s", has_dsn)
    logger.info(
        "Shopify configured: domain=%s, token_present=%s",
        settings.shopify_shop_domain,
        bool(settings.shopify_access_token),
    )

    # Init Postgres (best-effort)
    dsn = build_pg_dsn(settings)
    if dsn:
        await init_pool(dsn)
    else:
        logger.warning("No PostgreSQL DSN configured, Shopify logs will use JSONL fallback")

    # Init Shopify client (if configured)
    if settings.shopify_shop_domain and settings.shopify_access_token:
        app.state.shopify_client = ShopifyClient(
            shop_domain=settings.shopify_shop_domain,
            access_token=settings.shopify_access_token,
            storefront_access_token=settings.shopify_storefront_access_token,
            api_version=settings.shopify_api_version,
            timeout_seconds=settings.shopify_timeout_seconds,
            log_response_bodies=settings.shopify_log_response_bodies,
        )
        logger.info("Shopify client initialized")

        # Validate API tokens on startup
        await validate_shopify_tokens(app.state.shopify_client, settings)
    else:
        app.state.shopify_client = None
        logger.warning("Shopify credentials not configured; Shopify calls are disabled")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    # Close Shopify client
    client: ShopifyClient | None = getattr(app.state, "shopify_client", None)
    if client:
        await client.aclose()
    # Close PG pool
    await close_pool()

# -----------------------------------------------------------------------------
# SERVICE URL CONFIGURATION
# -----------------------------------------------------------------------------
# Intent Classifier API (Your main.py service)
INTENT_CLASSIFIER_URL = os.getenv("INTENT_CLASSIFIER_URL", "http://localhost:8000/classify")

# Example backend APIs
same = "https://happyruh.myshopify.com/api/2024-10/graphql.json"
SEARCH_API_URL = os.getenv("SEARCH_API_URL", same)
CART_API_URL = os.getenv("CART_API_URL", same)
ORDER_API_URL = os.getenv("ORDER_API_URL", same)
CHECKOUT_API_URL = os.getenv("CHECKOUT_API_URL", same)

# -----------------------------------------------------------------------------
# REQUEST MODEL
# -----------------------------------------------------------------------------
class UserQuery(BaseModel):
    text: str

# -----------------------------------------------------------------------------
# ORCHESTRATOR ENDPOINT
# -----------------------------------------------------------------------------
@app.post("/orchestrate")
async def orchestrate(query: UserQuery):
    """
    Main orchestration endpoint.
    1️⃣ Accepts user input.
    2️⃣ Calls intent classifier.
    3️⃣ Routes request to the proper backend based on intent.
    """
    logger.info(f"ORCHESTRATOR: Received /orchestrate request | text='{query.text}'")

    # STEP 1: Call the intent classifier
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            classifier_response = await client.post(INTENT_CLASSIFIER_URL, json={"text": query.text})
            classifier_response.raise_for_status()
            intent_data = classifier_response.json()
            logger.info(f"ORCHESTRATOR: Intent classification result | action_code={intent_data.get('action_code')} confidence={intent_data.get('confidence_score')}")
        except Exception as e:
            logger.error(f"ORCHESTRATOR: Intent classification failed | error={e}")
            raise HTTPException(status_code=500, detail=f"Intent classification failed: {str(e)}")

    # STEP 2: Parse classifier response
    action_code = intent_data.get("action_code", "UNKNOWN_INTENT")
    logger.info(f"ORCHESTRATOR: Parsed action_code={action_code}")
    # Fallback: some classifiers return intent under intent.payload.intent even if action_code is UNKNOWN_INTENT
    if action_code == "UNKNOWN_INTENT":
        nested_intent = (
            (intent_data.get("intent") or {}).get("payload") or {}
        ).get("intent")
        if isinstance(nested_intent, str) and nested_intent.strip():
            action_code = nested_intent.strip()

    confidence = float(intent_data.get("confidence_score", 0))
    # Ensure entities is a dict (classifier may return null/None)
    entities = intent_data.get("entities") or {}
    status = intent_data.get("status", "UNKNOWN")

    # STEP 3: Handle low-confidence cases
    if confidence < 0.6:
        return {
            "status": "LOW_CONFIDENCE",
            "message": "I'm not sure what you mean. Could you rephrase?",
            "intent_result": intent_data
        }

    # STEP 4: Decide routing logic
    backend_url = None
    action_label = None

    if action_code.startswith("SEARCH_"):
        backend_url = SEARCH_API_URL
        action_label = "Product Search"
    elif action_code in ["ADD_TO_CART", "REMOVE_FROM_CART", "VIEW_CART"]:
        backend_url = CART_API_URL
        action_label = "Cart Management"
    elif action_code.startswith("ORDER_") or action_code in ["TRACK_SHIPMENT", "DELIVERY_STATUS"]:
        backend_url = ORDER_API_URL
        action_label = "Order Management"
    elif action_code.startswith("CHECKOUT_"):
        backend_url = CHECKOUT_API_URL
        action_label = "Checkout"
    else:
        return {
            "status": "UNKNOWN_INTENT",
            "message": f"No routing rule found for intent '{action_code}'",
            "intent_result": intent_data
        }

    # STEP 5: Call the selected backend API
    backend_response_data = {}
    try:
        parsed = urlparse(backend_url)
        is_shopify = "myshopify.com" in (parsed.netloc or "")
        logger.info(
            "Routing decision | action=%s label=%s backend_url=%s shopify=%s",
            action_code,
            action_label,
            backend_url,
            is_shopify,
        )
        if is_shopify:
            client: ShopifyClient | None = getattr(app.state, "shopify_client", None)
            if not client:
                backend_response_data = {"error": "Shopify client not configured"}
            else:
                path = parsed.path or "/"
                # Preserve query if present
                if parsed.query:
                    path = f"{path}?{parsed.query}"
                logger.info("ORCHESTRATOR: Calling Shopify via connector | path=%s entities=%s", path, entities)
                # For GraphQL, build proper query from entities
                if path.endswith("/graphql.json") and entities:
                    # Build GraphQL query for product search
                    query = """
                    query($query: String!) {
                        products(first: 10, query: $query) {
                            edges {
                                node {
                                    id
                                    title
                                    handle
                                    description
                                    priceRange {
                                        minVariantPrice {
                                            amount
                                            currencyCode
                                        }
                                    }
                                    images(first: 1) {
                                        edges {
                                            node {
                                                url
                                                altText
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    """
                    variables = {"query": entities.get("query", "raw")}
                    json_body = {"query": query, "variables": variables}
                else:
                    json_body = entities
                # Add proper headers for authentication
                headers = {}
                if path.startswith("/api/") and app.state.settings.shopify_storefront_access_token:
                    headers["X-Shopify-Storefront-Access-Token"] = app.state.settings.shopify_storefront_access_token
                resp = await client.post(path, json_body=json_body, headers=headers)
                logger.info(f"ORCHESTRATOR: Shopify response | status={resp.status_code} is_success={resp.is_success}")
                if resp.is_success:
                    backend_response_data = resp.json()
                else:
                    backend_response_data = {"error": f"Backend responded with {resp.status_code}"}
        else:
            async with httpx.AsyncClient(timeout=15.0) as client:
                logger.info("Calling backend via httpx | url=%s", backend_url)
                backend_response = await client.post(backend_url, json=entities)
                if backend_response.is_success:
                    backend_response_data = backend_response.json()
                else:
                    backend_response_data = {"error": f"Backend responded with {backend_response.status_code}"}
    except Exception as e:
        backend_response_data = {"error": f"Failed to call backend: {str(e)}"}

    # STEP 6: Return unified response
    return {
        "status": "SUCCESS",
        "intent": action_code,
        "confidence": confidence,
        "entities": entities,
        "classified_status": status,
        "orchestrator_message": f"Intent '{action_code}' routed to {action_label} backend.",
        "backend_response": backend_response_data
    }


# -----------------------------------------------------------------------------
# HEALTH CHECK ENDPOINT
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Orchestrator API",
        "message": "The Orchestrator is running and ready to route requests."
    }


# -----------------------------------------------------------------------------
# TOKEN VALIDATION FUNCTION
# -----------------------------------------------------------------------------
async def validate_shopify_tokens(client: ShopifyClient, settings: Settings) -> None:
    """Validate Shopify API tokens on startup."""
    logger.info("Validating Shopify API tokens...")

    # Test admin token
    try:
        path = f"/admin/api/{settings.shopify_api_version}/shop.json"
        # path = "/admin/api/2024-10/products.json"
        resp = await client.get(path)
        if resp.is_success:
            data = resp.json()
            shop_name = data.get("shop", {}).get("name", "Unknown")
            logger.info(f"✅ Admin token valid - Shop: {shop_name}")
        else:
            logger.error(f"❌ Admin token invalid - Status: {resp.status_code}")
    except Exception as e:
        logger.error(f"❌ Admin token validation failed: {str(e)}")

    # Test storefront token
    if settings.shopify_storefront_access_token:
        try:
            path = f"/api/{settings.shopify_api_version}/graphql.json"
            query = """
                query SearchProducts($q:String!){ search(query:$q, types: PRODUCT, first:10){ edges{ node{ ... on Product { id title handle images(first:1){edges{node{url}}} variants(first:10){edges{node{id title price{amount currencyCode}}}} } } } } }
                """
            variables = { "q": "{{search_query}}" }
            json_body = {"query": query, "variables":variables}
            headers = {"X-Shopify-Storefront-Access-Token": settings.shopify_storefront_access_token}
            resp = await client.post(path, json_body=json_body, headers=headers)
            if resp.is_success:
                data = resp.json()
                shop_name = data.get("data", {}).get("shop", {}).get("name", "Unknown")
                logger.info(f"✅ Storefront token valid - Shop: {shop_name}")
            else:
                logger.error(f"❌ Storefront token invalid - Status: {resp.status_code}")
        except Exception as e:
            logger.error(f"❌ Storefront token validation failed: {str(e)}")
    else:
        logger.warning("⚠️  No storefront access token configured")

# -----------------------------------------------------------------------------
# DEBUG: Basic Shopify ping to verify logging
# -----------------------------------------------------------------------------
@app.get("/debug/shopify/ping")
async def shopify_ping(request: Request):
    logger.info("ORCHESTRATOR: Received /debug/shopify/ping request")
    client: ShopifyClient | None = getattr(app.state, "shopify_client", None)
    if not client:
        logger.error("ORCHESTRATOR: Shopify client not configured")
        raise HTTPException(status_code=503, detail="Shopify client not configured")
    # GET shop info as a lightweight call
    path = f"/admin/api/{app.state.settings.shopify_api_version}/shop.json"
    logger.info(f"ORCHESTRATOR: Calling Shopify client.get | path={path}")
    try:
        resp = await client.get(path)
        logger.info(f"ORCHESTRATOR: Shopify response received | status={resp.status_code}")
        data = resp.json()
        return {"status": "ok", "shop": data.get("shop", {}).get("name")}
    except Exception as e:
        logger.error(f"ORCHESTRATOR: Shopify ping failed | error={e}")
        # Still logged with timestamps by the client
        raise HTTPException(status_code=502, detail=f"Shopify ping failed: {str(e)}")
