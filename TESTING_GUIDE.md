# Complete Testing Guide - ChatNShop Intent Classification API

This guide helps you test **every feature** we've implemented. Follow this systematically to verify everything works as expected.

---

## 📋 Pre-Testing Checklist

Before starting, ensure:

- [ ] Application is running: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- [ ] Redis is running: `redis-cli ping` should return `PONG`
- [ ] Qdrant is running: Check http://localhost:6333/dashboard
- [ ] Environment variables are set (check `.env` file)
- [ ] Swagger UI is accessible: http://localhost:8000/docs
- [ ] Application logs are visible in terminal

---

## 🚀 Quick Start

### Access Points:
- **Swagger UI**: http://localhost:8000/docs (Best for testing)
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000/

### Start Testing:
1. Open Swagger UI: http://localhost:8000/docs
2. Use "Try it out" button on each endpoint
3. Check responses match expected format
4. Verify no errors in terminal logs

---

## 1. ✅ HEALTH & STATUS ENDPOINTS

### 1.1 GET `/` - Root Endpoint
**Test Steps:**
1. Click endpoint in Swagger
2. Click "Try it out"
3. Click "Execute"

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "Intent Classification API",
  "version": "1.0.0",
  "message": "Hybrid Intent Classification System is running!"
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Service name is correct
- ✅ Version matches

---

### 1.2 GET `/health` - Comprehensive Health Check
**Test Steps:**
1. Execute endpoint
2. Check all service statuses

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-XX...",
  "services": {
    "qdrant": "connected",
    "redis": "connected",
    "openai": "configured"
  },
  "queue": {
    "status": "available",
    "ambiguous_queue_size": 0,
    "result_queue_size": 0,
    "dead_letter_queue_size": 0,
    "timestamp": "2025-11-XX..."
  },
  "version": "1.0.0"
}
```

**Verify:**
- ✅ Status: "healthy" or "degraded"
- ✅ Qdrant: "connected" (if available)
- ✅ Redis: "connected" (if available)
- ✅ OpenAI: "configured" or "not_configured"
- ✅ Timestamp is recent

**Test Edge Cases:**
- Stop Redis → Should show "disconnected" or "unhealthy"
- Stop Qdrant → Should show "disconnected" or "unhealthy"

---

### 1.3 GET `/health/readiness` - Readiness Probe (Kubernetes)
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "status": "ready",
  "timestamp": "2025-01-XX...",
  "checks": {
    "qdrant": true,
    "redis": true
  }
}
```

**Verify:**
- ✅ Status code: 200 (if ready) or 503 (if not ready)
- ✅ All checks show `true` when services are up
- ✅ Returns 503 if any critical service is down

---

### 1.4 GET `/health/liveness` - Liveness Probe (Kubernetes)
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "status": "alive",
  "timestamp": "2025-01-XX..."
}
```

**Verify:**
- ✅ Status code: Always 200
- ✅ Response is fast (<100ms)
- ✅ Doesn't check dependencies (always alive if app is running)

---

## 2. 🎯 INTENT CLASSIFICATION ENDPOINTS

### 2.1 POST `/classify` - Main Classification Endpoint
**This is the PRIMARY endpoint - Test thoroughly!**

#### Test Case 1: Simple Keyword Match (Fast Path)
**Request:**
```json
{
  "text": "add to cart"
}
```

**Expected Response:**
```json
{
  "action_code": "ADD_TO_CART",
  "confidence_score": 1,
  "matched_keywords": ["add to cart"],
  "original_text": "add to cart",
  "status": "CONFIDENT_KEYWORD",
  "intent": {
    "id": "ADD_TO_CART",
    "intent": "ADD_TO_CART",
    "action": "ADD_TO_CART",
    "score": 1,
    "source": "keyword",
    "match_type": "exact",
    "matched_text": "add to cart"
  },
  "entities": null
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Action code is valid
- ✅ Confidence score: 0.0-1.0
- ✅ Status: "CONFIDENT_KEYWORD" (keyword match)
- ✅ Response time: <100ms (fast keyword path)
- ✅ Check logs: Should see "keyword match" or similar

---

#### Test Case 2: Product Search with Entities
**Request:**
```json
{
  "text": "show me red Nike running shoes under $100"
}
```

**Expected Response:**
```json
{
  "action_code": "SEARCH_PRODUCT",
  "confidence_score": 0.88,
  "matched_keywords": ["search", "shoes"],
  "original_text": "show me red Nike running shoes under $100",
  "status": "LLM_CLASSIFICATION",
  "intent": {
    "id": "SEARCH_PRODUCT",
    "intent": "SEARCH_PRODUCT",
    "action": "SEARCH_PRODUCT",
    "score": 0.88,
    "source": "llm",
    "match_type": "partial",
    "matched_text": "show me"
  },
  "entities": {
    "product_type": "shoes",
    "category": "running",
    "brand": "Nike",
    "color": "red",
    "price_range": {
      "min": null,
      "max": 100,
      "currency": "USD"
    }
  }
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Entities extracted: brand (Nike), color (red), price_range
- ✅ Action code: SEARCH_PRODUCT
- ✅ Response time: <2 seconds
- ✅ Check logs: Should see entity extraction

---

#### Test Case 3: Complex Intent - Returns
**Request:**
```json
{
  "text": "I want to return the blue shirt I ordered last week"
}
```

**Expected Response:**
```json
{
  "action_code": "INITIATE_RETURN",
  "confidence_score": 0.85,
  "original_text": "I want to return the blue shirt I ordered last week",
  "status": "LLM_CLASSIFICATION",
  "entities": {
    "product_type": "shirt",
    "color": "blue"
  }
}
```

**Verify:**
- ✅ Action code: RETURN_ORDER
- ✅ Entities extracted correctly
- ✅ Confidence reasonable (>0.6)

---

#### Test Case 4: Ambiguous Query (Should use LLM)
**Request:**
```json
{
  "text": "I need help with something"
}
```

**Expected Response:**
```json
{
  "action_code": "GET_HELP",
  "confidence_score": 0.65,
  "status": "LLM_CLASSIFICATION",
  ...
}
```

**Verify:**
- ✅ Uses LLM fallback (status may show this)
- ✅ Returns reasonable intent
- ✅ Response time: <2 seconds

---

#### Test Case 5: Edge Cases

**Empty String:**
```json
{
  "text": ""
}
```
**Verify:**
- ✅ Returns error 422 (validation error)

**Very Long String (10,000+ chars):**
```json
{
  "text": "a".repeat(10000)
}
```
**Verify:**
- ✅ Handles gracefully (may timeout or truncate)
- ✅ Error response if rejected

**Special Characters:**
```json
{
  "text": "search for @#$%^&*() products"
}
```
**Verify:**
- ✅ Handles special characters
- ✅ Returns valid response

**Unicode/Emoji:**
```json
{
  "text": "search for 👟 shoes 🎁"
}
```
**Verify:**
- ✅ Handles Unicode/emoji
- ✅ Returns valid response

---

### 2.2 POST `/api/v1/llm-intent/classify` - Direct LLM Classification
**Test Case:**
**Request:**
```json
{
  "user_input": "find me a gift for my girlfriend",
  "rule_intent": null,
  "action_code": null,
  "top_confidence": 0.45,
  "next_best_confidence": 0.30,
  "is_fallback": true,
  "context_snippets": [],
  "metadata": {}
}
```

**Expected Response:**
```json
{
  "action_code": "SEARCH_PRODUCT",
  "confidence": 0.82,
  "entities": {},
  ...
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Uses LLM directly (is_fallback=true)
- ✅ Response includes action_code and confidence
- ✅ Response time: <2 seconds

---

## 3. 💾 CACHE MANAGEMENT ENDPOINTS

### 3.1 GET `/api/v1/cache/metrics` - Cache Statistics
**Test Steps:**
1. Execute endpoint
2. Check metrics after using `/classify` endpoint

**Expected Response:**
```json
{
  "hit_rate": 0.35,
  "miss_rate": 0.65,
  "total_requests": 100,
  "cache_size": 45,
  "avg_latency_ms": 8.5,
  "p95_latency_ms": 12.0
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Hit rate + miss rate = 1.0 (approximately)
- ✅ Cache size >= 0
- ✅ Latency metrics are reasonable

**Test Workflow:**
1. Clear cache first (use `/api/v1/cache/clear`)
2. Make 5 requests to `/classify` with same query
3. Check metrics → Should show cache hits on subsequent requests

---

### 3.2 GET `/api/v1/cache/health` - Cache Health Check
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "enabled": true,
  "redis_available": true,
  "embedding_model_loaded": true,
  "cache_size": 45,
  "max_cache_size": 10000,
  "similarity_threshold": 0.95,
  "ttl_seconds": 86400,
  "status": "healthy"
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Status: "healthy", "degraded", or "unhealthy"
- ✅ Redis available: true (if Redis is running)
- ✅ Embedding model loaded: true

---

### 3.3 GET `/api/v1/cache/top?limit=10` - Top Cached Queries
**Test Steps:**
1. Make several classification requests first
2. Execute this endpoint

**Expected Response:**
```json
{
  "top_queries": [
    {
      "query": "show me red shoes",
      "hit_count": 15
    },
    {
      "query": "add to cart",
      "hit_count": 10
    }
  ],
  "total_tracked": 2
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Returns queries with hit counts
- ✅ Ordered by hit count (descending)
- ✅ Limit parameter works (try limit=5, limit=20)

---

### 3.4 POST `/api/v1/cache/clear` - Clear All Cache
**Test Steps:**
1. Check cache size first (use metrics endpoint)
2. Execute this endpoint
3. Verify cache is cleared (check metrics again)

**Expected Response:**
```json
{
  "success": true,
  "message": "Cache cleared successfully"
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Success: true
- ✅ After clearing, cache metrics show size = 0

**⚠️ WARNING:** This clears ALL cache entries. Use carefully.

---

### 3.5 POST `/api/v1/cache/invalidate` - Invalidate Specific Query
**Test Steps:**
1. Make a classification request: `"text": "show me shoes"`
2. Check cache metrics (should show this query cached)
3. Invalidate it:
```json
{
  "query": "show me shoes"
}
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Cache entry invalidated successfully",
  "query": "show me shoes"
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Success: true
- ✅ After invalidation, cache metrics may decrease

---

## 4. 📊 QUEUE MANAGEMENT ENDPOINTS

### 4.1 GET `/api/v1/queue/health` - Queue Health
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "healthy": true,
  "available": true
}
```

**Verify:**
- ✅ Status code: 200 (if queue is available) or 503 (if not)
- ✅ Healthy: true when Redis queue is available
- ✅ Available: true when queue manager is initialized

---

### 4.2 GET `/api/v1/queue/stats` - Queue Statistics
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "ambiguous_queue_size": 0,
  "result_queue_size": 0,
  "dead_letter_queue_size": 0,
  "total_processed": 150,
  "total_failed": 2
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Queue sizes >= 0
- ✅ Statistics are accurate

---

### 4.3 GET `/api/v1/queue/metrics` - Queue Performance Metrics
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "avg_processing_time_ms": 850,
  "throughput_per_minute": 45,
  "error_rate": 0.02
}
```

**Verify:**
- ✅ Status code: 200 or 503
- ✅ Metrics are reasonable (processing time < 5000ms)
- ✅ Error rate < 0.1 (10%)

---

### 4.4 GET `/api/v1/queue/status/{request_id}` - Check Request Status
**Test Steps:**
1. Create an async request (if available)
2. Get request_id from response
3. Check status with this endpoint

**Expected Response:**
```json
{
  "request_id": "abc-123",
  "status": "COMPLETED",
  "queued_at": "2025-01-XX...",
  "started_at": "2025-01-XX...",
  "completed_at": "2025-01-XX..."
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Status: QUEUED, PROCESSING, COMPLETED, or FAILED
- ✅ Timestamps are valid

---

## 5. 📈 MONITORING ENDPOINTS

### 5.1 GET `/api/v1/monitoring/intent-distribution?period=daily` - Intent Distribution
**Test Steps:**
1. Make several classification requests first
2. Execute endpoint with period=daily

**Expected Response:**
```json
{
  "period": "daily",
  "total_intents": 150,
  "distribution": {
    "SEARCH_PRODUCT": 45,
    "ADD_TO_CART": 30,
    "RETURN_ORDER": 15
  },
  "top_intents": [
    {
      "action_code": "SEARCH_PRODUCT",
      "count": 45
    }
  ]
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Distribution sums to total_intents
- ✅ Test all periods: daily, weekly, monthly

---

### 5.2 GET `/api/v1/monitoring/confidence-distribution?period=daily` - Confidence Scores
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "period": "daily",
  "distribution": {
    "0.8-1.0": 60,
    "0.6-0.8": 30,
    "0.4-0.6": 10
  },
  "average_confidence": 0.82
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Distribution buckets are reasonable
- ✅ Average confidence is between 0.0-1.0

---

### 5.3 GET `/api/v1/monitoring/accuracy-metrics` - Accuracy Metrics
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "overall_accuracy": 0.87,
  "total_classifications": 1000,
  "correct_classifications": 870,
  "accuracy_by_intent": {}
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Accuracy between 0.0-1.0
- ✅ Total >= correct

---

### 5.4 GET `/api/v1/monitoring/sample-queries` - Sample Queries
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "samples": [
    {
      "query": "show me shoes",
      "action_code": "SEARCH_PRODUCT",
      "confidence": 0.92
    }
  ]
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Returns sample data
- ✅ Each sample has query, action_code, confidence

---

## 6. 💬 FEEDBACK ENDPOINTS

### 6.1 POST `/api/v1/feedback/classifications/flag` - Flag Misclassification
**Test Steps:**
1. First, make a classification request
2. Note the request_id or details
3. Flag it:
```json
{
  "request_id": "test-123",
  "is_correct": false,
  "comment": "This was classified incorrectly as SEARCH_PRODUCT"
}
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Feedback recorded"
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Success: true
- ✅ Feedback is stored (check stats endpoint)

---

### 6.2 POST `/api/v1/feedback/classifications` - Submit Feedback
**Test Case:**
```json
{
  "action_code": "SEARCH_PRODUCT",
  "confidence": 0.85,
  "is_correct": true,
  "user_input": "show me shoes",
  "comment": "Correct classification"
}
```

**Expected Response:**
```json
{
  "success": true,
  "feedback_id": "feedback-123"
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Feedback ID is returned
- ✅ Feedback is stored

---

### 6.3 GET `/api/v1/feedback/misclassifications` - Get Misclassifications
**Test Steps:**
1. Submit some feedback with is_correct=false first
2. Execute this endpoint

**Expected Response:**
```json
{
  "total": 5,
  "misclassifications": [
    {
      "query": "show me shoes",
      "expected_action": "SEARCH_PRODUCT",
      "actual_action": "ADD_TO_CART",
      "confidence": 0.65
    }
  ]
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Returns misclassified queries
- ✅ Shows expected vs actual

---

### 6.4 GET `/api/v1/feedback/stats` - Feedback Statistics
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "total_feedback": 100,
  "positive_feedback": 87,
  "negative_feedback": 13,
  "accuracy_rate": 0.87
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Stats are accurate
- ✅ Positive + negative = total

---

## 7. 📝 STATUS TRACKING ENDPOINTS

### 7.1 GET `/status/{request_id}` - Get Request Status
**Test Steps:**
1. Create a request (use classify endpoint)
2. Use request_id from response headers (X-Request-ID) or logs
3. Check status

**Expected Response:**
```json
{
  "request_id": "abc-123",
  "status": "COMPLETED",
  "created_at": "2025-01-XX...",
  "updated_at": "2025-01-XX..."
}
```

**Verify:**
- ✅ Status code: 200 or 404 (if not found)
- ✅ Status is valid
- ✅ Timestamps are valid

---

### 7.2 POST `/status/{request_id}` - Update Status
**Test Case:**
```json
{
  "status": "PROCESSING",
  "message": "Currently processing request"
}
```

**Expected Response:**
```json
{
  "success": true,
  "request_id": "abc-123",
  "status": "PROCESSING"
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Status is updated
- ✅ Verify with GET endpoint

---

### 7.3 GET `/status/summary/daily` - Daily Status Summary
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "date": "2025-01-XX",
  "total_requests": 1000,
  "completed": 950,
  "failed": 50,
  "pending": 0
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Totals are accurate
- ✅ Date is today

---

### 7.4 GET `/status/summary/monthly` - Monthly Status Summary
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "month": "2025-01",
  "total_requests": 30000,
  "completed": 28500,
  "failed": 1500
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Month format is correct (YYYY-MM)
- ✅ Statistics are reasonable

---

## 8. 🧪 A/B TESTING ENDPOINTS

### 8.1 POST `/api/ab-testing/experiments` - Create Experiment
**Test Case:**
```json
{
  "experiment_id": "test-exp-1",
  "name": "Test Prompt Experiment",
  "description": "Testing different prompt versions",
  "variants": [
    {
      "variant_id": "A",
      "variant_type": "control"
    },
    {
      "variant_id": "B",
      "variant_type": "treatment"
    }
  ]
}
```

**Expected Response:**
```json
{
  "experiment_id": "test-exp-1",
  "status": "running",
  "created_at": "2025-01-XX..."
}
```

**Verify:**
- ✅ Status code: 200 or 201
- ✅ Experiment ID matches
- ✅ Status is "running"

---

### 8.2 GET `/api/ab-testing/experiments` - List Experiments
**Test Steps:**
1. Create an experiment first
2. Execute this endpoint

**Expected Response:**
```json
{
  "experiments": [
    {
      "experiment_id": "test-exp-1",
      "name": "Test Prompt Experiment",
      "status": "running"
    }
  ]
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Returns list of experiments
- ✅ Includes created experiments

---

### 8.3 GET `/api/ab-testing/experiments/{experiment_id}` - Get Experiment
**Test Steps:**
1. Use experiment_id from previous step

**Expected Response:**
```json
{
  "experiment_id": "test-exp-1",
  "name": "Test Prompt Experiment",
  "status": "running",
  "variants": [...],
  "results": {...}
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Complete experiment details
- ✅ Includes variants and results

---

### 8.4 GET `/api/ab-testing/experiments/{experiment_id}/compare` - Compare Variants
**Test Steps:**
1. Wait for experiment to collect data
2. Execute comparison

**Expected Response:**
```json
{
  "experiment_id": "test-exp-1",
  "winner": "B",
  "confidence": 0.95,
  "variant_stats": {
    "A": {"conversions": 100, "rate": 0.10},
    "B": {"conversions": 120, "rate": 0.12}
  }
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Statistical comparison is valid
- ✅ Winner is determined

---

### 8.5 POST `/api/ab-testing/experiments/{experiment_id}/stop` - Stop Experiment
**Test Steps:**
1. Use experiment_id
2. Execute stop

**Expected Response:**
```json
{
  "success": true,
  "experiment_id": "test-exp-1",
  "status": "stopped"
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Status changes to "stopped"
- ✅ Experiment no longer routes traffic

---

## 9. 🔬 TESTING FRAMEWORK ENDPOINTS

### 9.1 POST `/api/testing/experiments` - Create Test Experiment
**Test Case:**
```json
{
  "experiment_id": "test-framework-1",
  "name": "Framework Test",
  "description": "Testing framework",
  "variants": [
    {"variant_id": "A", "config": {}},
    {"variant_id": "B", "config": {}}
  ],
  "use_bandit": true
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Experiment is created
- ✅ Bandit algorithm is enabled

---

### 9.2 GET `/api/testing/experiments` - List Test Experiments
**Verify:**
- ✅ Returns list of test experiments
- ✅ Includes framework experiments

---

### 9.3 GET `/api/testing/experiments/{experiment_id}/results` - Get Results
**Verify:**
- ✅ Returns experiment results
- ✅ Includes statistical analysis

---

### 9.4 GET `/api/testing/bandit/state` - Get Bandit State
**Verify:**
- ✅ Returns current bandit algorithm state
- ✅ Shows variant allocations

---

### 9.5 POST `/api/testing/experiments/{experiment_id}/rollback` - Rollback Experiment
**Test Case:**
```json
{
  "target_variant": "A"
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Traffic routes to target variant
- ✅ Rollback is successful

---

## 10. 💰 COST DASHBOARD ENDPOINTS

### 10.1 GET `/api/cost_metrics` - Get Cost Metrics
**Test Steps:**
1. Execute endpoint

**Expected Response:**
```json
{
  "daily": {
    "tokens": 50000,
    "cost": 12.34,
    "requests": 1000
  },
  "monthly": {
    "tokens": 1500000,
    "cost": 370.20,
    "requests": 30000
  }
}
```

**Verify:**
- ✅ Status code: 200
- ✅ Costs are calculated correctly
- ✅ Token counts are reasonable
- ✅ Daily and monthly metrics present

---

### 10.2 GET `/cost` - Cost Dashboard UI
**Test Steps:**
1. Open in browser: http://localhost:8000/cost
2. Check dashboard displays

**Verify:**
- ✅ Page loads without errors
- ✅ Charts/graphs display (if implemented)
- ✅ Cost data is visible

---

### 10.3 GET `/feedback-review` - Feedback Review UI
**Test Steps:**
1. Open in browser: http://localhost:8000/feedback-review
2. Check interface

**Verify:**
- ✅ Page loads
- ✅ Feedback data is displayed
- ✅ Interface is functional

---

## 11. 🔐 AUTHENTICATION TESTING

### 11.1 Test Optional Authentication
**Endpoints with Optional Auth:**
- `/classify`
- `/api/v1/llm-intent/classify`

**Test Steps:**
1. Call endpoint WITHOUT Authorization header
2. Should work (optional auth)

**Verify:**
- ✅ Works without token
- ✅ Response includes anonymous user context

---

### 11.2 Test JWT Token (If You Have Tokens)
**Test Steps:**
1. Generate a JWT token (if you have endpoint) or use existing
2. Add header: `Authorization: Bearer <token>`
3. Call protected endpoint

**Expected:**
- ✅ Request is authenticated
- ✅ User context includes user_id

**Invalid Token Test:**
1. Use invalid token: `Bearer invalid-token-123`
2. Should still work (optional auth allows invalid tokens)

---

## 12. 🚦 RATE LIMITING TESTING

### 12.1 Test Rate Limits
**Test Steps:**
1. Make rapid requests to `/classify` endpoint
2. Send 600+ requests in 1 minute
3. Check for rate limit response

**Expected After Limit:**
```json
{
  "detail": "Rate limit exceeded: 500 requests per minute"
}
```

**Verify:**
- ✅ Status code: 429 (Too Many Requests)
- ✅ Error message is clear
- ✅ Headers include `Retry-After`

**Check Rate Limit Headers:**
- Look for `X-RateLimit-Remaining` in response headers
- Should decrease with each request

---

## 13. 📊 INTEGRATION TESTING SCENARIOS

### Scenario 1: Complete User Flow
1. ✅ Health check → All services healthy
2. ✅ Classify query → Get intent + entities
3. ✅ Check cache metrics → Should show cache hit on repeat
4. ✅ Check monitoring → Should show new classification
5. ✅ Submit feedback → Mark as correct/incorrect
6. ✅ Check feedback stats → Should reflect submission

### Scenario 2: Cache Flow
1. ✅ Clear cache → Cache size = 0
2. ✅ Classify same query 3 times → First miss, next 2 hits
3. ✅ Check cache metrics → Hit rate should increase
4. ✅ Check top queries → Should show your query
5. ✅ Invalidate cache → Query removed
6. ✅ Verify cache metrics → Size decreased

### Scenario 3: Error Handling
1. ✅ Invalid input → 422 error with clear message
2. ✅ Missing service (stop Redis) → Degraded health
3. ✅ Rate limit exceeded → 429 error
4. ✅ Server error → 500 with error_id

### Scenario 4: Monitoring Flow
1. ✅ Make 10 classifications → Generate data
2. ✅ Check intent distribution → Should show your intents
3. ✅ Check confidence distribution → Should show scores
4. ✅ Check accuracy metrics → Should reflect performance

---

## 14. 🐛 TROUBLESHOOTING CHECKLIST

### If Endpoint Returns 500 Error:
- [ ] Check application logs in terminal
- [ ] Verify Redis is running: `redis-cli ping`
- [ ] Verify Qdrant is running: http://localhost:6333/dashboard
- [ ] Check environment variables are set
- [ ] Verify OpenAI API key is configured (if using LLM)

### If Endpoint Returns 503 Service Unavailable:
- [ ] Check health endpoint → Which service is down?
- [ ] Verify dependencies are running
- [ ] Check network connectivity

### If Classification is Slow:
- [ ] Check if it's first request (model loading takes time)
- [ ] Verify cache is working (should be faster on repeat)
- [ ] Check logs for timeouts
- [ ] Verify Redis/Qdrant performance

### If Cache Not Working:
- [ ] Check cache health endpoint
- [ ] Verify Redis is connected
- [ ] Check cache metrics → Is it enabled?
- [ ] Verify embedding model is loaded

---

## 15. ✅ FINAL VERIFICATION CHECKLIST

After testing, verify:

- [ ] All health endpoints return 200
- [ ] `/classify` works with various inputs
- [ ] Cache is working (check metrics after repeated queries)
- [ ] Monitoring endpoints show data
- [ ] Queue endpoints show status
- [ ] Feedback endpoints accept submissions
- [ ] Rate limiting works (test with rapid requests)
- [ ] Error handling is proper (test invalid inputs)
- [ ] Authentication works (if tokens available)
- [ ] All UI endpoints load in browser
- [ ] No critical errors in application logs
- [ ] Response times are reasonable (<2s for most)

---

## 📝 NOTES FOR TESTING

### Expected Response Times:
- Keyword match: <100ms
- Embedding match: <200ms
- LLM classification: <2 seconds
- Cache lookup: <10ms
- Health checks: <50ms

### Logs to Monitor:
- Check terminal for application logs
- Look for error messages
- Verify correlation IDs in logs match request IDs
- Check for rate limit warnings

### What to Document:
- Note any endpoints that fail
- Record response times that seem slow
- Document any unexpected behaviors
- Note missing features you expected

---

## 🎯 QUICK TEST SCRIPT

Run these in order to quickly verify core functionality:

```bash
# 1. Health Check
curl http://localhost:8000/health

# 2. Simple Classification
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "add to cart"}'

# 3. Cache Metrics
curl http://localhost:8000/api/v1/cache/metrics

# 4. Monitoring
curl http://localhost:8000/api/v1/monitoring/intent-distribution?period=daily
```

---

## 📚 Additional Resources

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Application Logs**: Check terminal output

---

**Happy Testing! 🚀**

If you find any issues, note them down with:
- Endpoint name
- Request details
- Expected vs Actual response
- Error messages
- Steps to reproduce

