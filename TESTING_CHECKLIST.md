# Testing Checklist - Quick Reference

Use this checklist to systematically test every endpoint. Check off each item as you test it.

---

## 🚀 Quick Start Commands

```bash
# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access Swagger UI
open http://localhost:8000/docs

# Check Redis
redis-cli ping

# Check Qdrant
curl http://localhost:6333/collections
```

---

## ✅ Health & Status (4 endpoints)

- [ ] `GET /` - Root endpoint
- [ ] `GET /health` - Comprehensive health check
- [ ] `GET /health/readiness` - Readiness probe
- [ ] `GET /health/liveness` - Liveness probe

**Verification:**
- [ ] All return 200 when services are up
- [ ] Health shows Qdrant, Redis, OpenAI status
- [ ] Readiness returns 503 when services are down

---

## 🎯 Intent Classification (2 endpoints)

### Main Classification
- [ ] `POST /classify` - Main endpoint
  - [ ] Test: "add to cart" (keyword match)
  - [ ] Test: "show me red Nike shoes under $100" (with entities)
  - [ ] Test: "I want to return my order" (complex)
  - [ ] Test: Empty string (should error)
  - [ ] Test: Very long string (edge case)

### LLM Intent
- [ ] `POST /api/v1/llm-intent/classify` - Direct LLM
  - [ ] Test with is_fallback=true
  - [ ] Verify LLM response format

**Verification:**
- [ ] Response time < 2 seconds
- [ ] Action codes are valid
- [ ] Confidence scores between 0-1
- [ ] Entities extracted correctly
- [ ] Error handling for invalid input

---

## 💾 Cache Management (6 endpoints)

- [ ] `GET /api/v1/cache/metrics` - Cache statistics
- [ ] `GET /api/v1/cache/health` - Cache health
- [ ] `GET /api/v1/cache/top?limit=10` - Top queries
- [ ] `POST /api/v1/cache/clear` - Clear all cache
- [ ] `POST /api/v1/cache/invalidate` - Invalidate query
- [ ] `POST /api/v1/cache/invalidate/{query_hash}` - Invalidate by hash

**Workflow Test:**
- [ ] Clear cache → Size = 0
- [ ] Make 3 same requests → Cache hits increase
- [ ] Check metrics → Hit rate > 0
- [ ] Check top queries → Your query appears
- [ ] Invalidate query → Cache size decreases

---

## 📊 Queue Management (5 endpoints)

- [ ] `GET /api/v1/queue/health` - Queue health
- [ ] `GET /api/v1/queue/stats` - Queue statistics
- [ ] `GET /api/v1/queue/metrics` - Performance metrics
- [ ] `GET /api/v1/queue/status/{request_id}` - Request status
- [ ] `POST /api/v1/queue/clear/{queue_name}` - Clear queue

**Verification:**
- [ ] Health shows queue availability
- [ ] Stats show queue sizes
- [ ] Metrics show processing rates
- [ ] Status tracking works

---

## 📈 Monitoring (4 endpoints)

- [ ] `GET /api/v1/monitoring/intent-distribution?period=daily`
- [ ] `GET /api/v1/monitoring/intent-distribution?period=weekly`
- [ ] `GET /api/v1/monitoring/intent-distribution?period=monthly`
- [ ] `GET /api/v1/monitoring/confidence-distribution?period=daily`
- [ ] `GET /api/v1/monitoring/accuracy-metrics`
- [ ] `GET /api/v1/monitoring/sample-queries`

**Verification:**
- [ ] Data appears after making classifications
- [ ] All periods work (daily/weekly/monthly)
- [ ] Distribution data is accurate

---

## 💬 Feedback (6 endpoints)

- [ ] `POST /api/v1/feedback/classifications/flag` - Flag misclassification
- [ ] `POST /api/v1/feedback/classifications` - Submit feedback
- [ ] `GET /api/v1/feedback/misclassifications` - Get misclassifications
- [ ] `GET /api/v1/feedback/export` - Export feedback
- [ ] `GET /api/v1/feedback/stats` - Feedback statistics
- [ ] `GET /api/v1/feedback/patterns` - Feedback patterns

**Workflow Test:**
- [ ] Submit feedback → Verify stored
- [ ] Flag misclassification → Verify recorded
- [ ] Check stats → Reflects submissions
- [ ] Check misclassifications → Shows flagged items

---

## 📝 Status Tracking (4 endpoints)

- [ ] `GET /status/{request_id}` - Get status
- [ ] `POST /status/{request_id}` - Update status
- [ ] `POST /status/{request_id}/log` - Log status
- [ ] `GET /status/summary/daily` - Daily summary
- [ ] `GET /status/summary/monthly` - Monthly summary

**Verification:**
- [ ] Status tracking works
- [ ] Summaries are accurate
- [ ] Updates persist

---

## 🧪 A/B Testing (5 endpoints)

- [ ] `POST /api/ab-testing/experiments` - Create experiment
- [ ] `GET /api/ab-testing/experiments` - List experiments
- [ ] `GET /api/ab-testing/experiments/{experiment_id}` - Get experiment
- [ ] `POST /api/ab-testing/experiments/{experiment_id}/stop` - Stop experiment
- [ ] `GET /api/ab-testing/experiments/{experiment_id}/compare` - Compare variants
- [ ] `GET /api/ab-testing/health` - A/B testing health

**Workflow Test:**
- [ ] Create experiment → Verify created
- [ ] List experiments → Shows your experiment
- [ ] Compare variants → Shows statistical comparison
- [ ] Stop experiment → Status changes to stopped

---

## 🔬 Testing Framework (9 endpoints)

- [ ] `POST /api/testing/experiments` - Create test experiment
- [ ] `GET /api/testing/experiments` - List test experiments
- [ ] `GET /api/testing/experiments/{experiment_id}/results` - Get results
- [ ] `GET /api/testing/bandit/state` - Get bandit state
- [ ] `POST /api/testing/bandit/reset` - Reset bandit
- [ ] `POST /api/testing/experiments/{experiment_id}/rollback` - Rollback
- [ ] `GET /api/testing/experiments/{experiment_id}/backups` - List backups
- [ ] `POST /api/testing/experiments/{experiment_id}/backup` - Create backup
- [ ] `GET /api/testing/status` - Testing status
- [ ] `GET /api/testing/health` - Testing health

---

## 💰 Cost Dashboard (3 endpoints)

- [ ] `GET /api/cost_metrics` - Cost metrics API
- [ ] `GET /cost` - Cost dashboard UI (browser)
- [ ] `GET /feedback-review` - Feedback review UI (browser)

**Verification:**
- [ ] Cost metrics show daily/monthly data
- [ ] UI pages load without errors
- [ ] Charts/data display correctly

---

## 🔐 Authentication Testing

- [ ] Test endpoints WITHOUT auth token → Should work (optional auth)
- [ ] Test endpoints WITH valid token → Should include user_id
- [ ] Test endpoints WITH invalid token → Should still work (optional auth)

**Endpoints to test:**
- [ ] `/classify` with/without token
- [ ] `/api/v1/llm-intent/classify` with/without token

---

## 🚦 Rate Limiting Testing

- [ ] Make 10 rapid requests → All succeed
- [ ] Make 500+ requests in 1 minute → Most succeed
- [ ] Make 600+ requests in 1 minute → Get 429 error
- [ ] Check response headers → `X-RateLimit-Remaining` decreases
- [ ] Wait 1 minute → Rate limit resets

**Verification:**
- [ ] 429 status code when limit exceeded
- [ ] Clear error message
- [ ] `Retry-After` header present
- [ ] Rate limit resets after window

---

## 🔄 Integration Scenarios

### Scenario 1: Complete Classification Flow
- [ ] Health check → All healthy
- [ ] Classify query → Get response
- [ ] Check cache → Query cached
- [ ] Check monitoring → Classification recorded
- [ ] Submit feedback → Feedback stored
- [ ] Check stats → Stats updated

### Scenario 2: Cache Flow
- [ ] Clear cache
- [ ] Classify same query 3x
- [ ] Check metrics → Hit rate increases
- [ ] Check top queries → Query appears
- [ ] Invalidate → Cache decreases

### Scenario 3: Error Handling
- [ ] Invalid input → 422 error
- [ ] Stop Redis → Health shows degraded
- [ ] Rate limit → 429 error
- [ ] Server error → 500 with error_id

### Scenario 4: Monitoring Flow
- [ ] Make 10 classifications
- [ ] Check intent distribution → Shows intents
- [ ] Check confidence → Shows scores
- [ ] Check accuracy → Shows metrics

---

## 📊 Performance Checks

- [ ] Keyword match: <100ms
- [ ] Embedding match: <200ms
- [ ] LLM classification: <2 seconds
- [ ] Cache lookup: <10ms
- [ ] Health checks: <50ms
- [ ] No memory leaks (monitor over time)

---

## 🐛 Error Scenarios

- [ ] Invalid JSON → 422 error
- [ ] Missing required fields → 422 error
- [ ] Empty string → Validation error
- [ ] Very long string → Handled gracefully
- [ ] Special characters → Handled correctly
- [ ] Redis down → Degraded but works
- [ ] Qdrant down → Degraded but works
- [ ] OpenAI API error → Fallback works

---

## 📝 Logging & Observability

- [ ] Check application logs → Correlation IDs present
- [ ] Check request IDs in logs → Match response headers
- [ ] Check error logs → Clear error messages
- [ ] Check timing logs → Response times logged
- [ ] Check rate limit logs → Warnings appear

---

## 🎯 Final Verification

### Core Functionality
- [ ] All health endpoints work
- [ ] Classification works for various inputs
- [ ] Cache is operational
- [ ] Monitoring shows data
- [ ] Feedback accepts submissions

### Edge Cases
- [ ] Error handling works
- [ ] Rate limiting works
- [ ] Authentication works
- [ ] Invalid inputs handled

### Integration
- [ ] End-to-end flows work
- [ ] Services interact correctly
- [ ] No critical errors in logs
- [ ] Response times acceptable

---

## 📋 Issues Found

Document any issues you find:

1. **Endpoint**: _______________
   - **Issue**: _______________
   - **Steps to Reproduce**: _______________
   - **Expected**: _______________
   - **Actual**: _______________

2. **Endpoint**: _______________
   - **Issue**: _______________
   - **Steps to Reproduce**: _______________
   - **Expected**: _______________
   - **Actual**: _______________

---

**Total Endpoints to Test: ~50+**
**Estimated Time: 2-3 hours for thorough testing**

Good luck! 🚀

