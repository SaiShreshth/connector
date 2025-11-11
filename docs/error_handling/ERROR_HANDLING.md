# Error Handling and Fallback Mechanisms - Task 21

## Overview

Comprehensive error handling system for LLM-based intent classification with intelligent fallback, retry logic, escalation, and cache integration.

## Architecture

```
User Request
    ↓
[Cache Check] → Cache Hit? → Return cached response
    ↓ (miss)
[Rule-Based] → High confidence? → Return rule result
    ↓ (low confidence)
[LLM Request with Retry]
    ├→ Success → Cache result → Return
    ├→ Timeout → Retry (3x with backoff)
    ├→ Rate Limit → Alert + Retry
    ├→ Server Error → Alert + Retry
    ├→ Auth Error → Critical Alert + Fallback
    └→ All Failed → Escalate + Cache Fallback → UNCLEAR
```

---

## Error Types and Handling

### 1. Timeout Errors

**Detection:**
- String contains "timeout" or "timed out"
- Exception type contains "Timeout"

**Handling:**
- Error code: `timeout`
- Retry: Yes (3 attempts with exponential backoff)
- User message: "The request is taking longer than expected. Please try again."
- Alert: Warning level after 20 timeouts in 1 hour

**Response:**
```json
{
  "user_message": "The request is taking longer than expected. Please try again.",
  "action_code": "TIMEOUT_ERROR",
  "retry_recommended": true,
  "retry_after_seconds": 5,
  "suggestions": [
    "Try again in a few seconds",
    "Simplify your request if it's very detailed"
  ]
}
```

### 2. Rate Limit Errors

**Detection:**
- HTTP status 429
- String contains "rate limit"
- Exception type contains "RateLimit"

**Handling:**
- Error code: `rate_limit`
- Retry: Yes (3 attempts with backoff)
- User message: "We're experiencing high traffic. Please try again in a moment."
- Alert: Warning level after 10 rate limits in 1 hour

**Response:**
```json
{
  "user_message": "We're experiencing high traffic. Please try again in a moment.",
  "action_code": "RATE_LIMIT_ERROR",
  "retry_recommended": true,
  "retry_after_seconds": 60,
  "suggestions": [
    "Wait a minute before trying again",
    "Browse our catalog while waiting"
  ]
}
```

### 3. Server Errors (5xx)

**Detection:**
- HTTP status 500-599
- Error code contains "server_error"

**Handling:**
- Error code: `server_error`
- Retry: Yes (3 attempts)
- User message: "The service is temporarily unavailable. Our team has been notified."
- Alert: Error level after 5 server errors in 1 hour

**Response:**
```json
{
  "user_message": "The service is temporarily unavailable. Our team has been notified.",
  "action_code": "SERVICE_UNAVAILABLE",
  "retry_recommended": true,
  "retry_after_seconds": 120,
  "escalated": true,
  "suggestions": [
    "Try again in a few minutes",
    "Contact support if the issue persists"
  ]
}
```

### 4. Authentication Errors

**Detection:**
- HTTP status 401
- String contains "authentication" or "invalid api key"

**Handling:**
- Error code: `auth_error`
- Retry: No
- User message: "We're experiencing technical difficulties. Our team has been notified."
- Alert: Critical level (immediate escalation)

**Response:**
```json
{
  "user_message": "We're experiencing technical difficulties. Our team has been notified.",
  "action_code": "AUTH_ERROR",
  "retry_recommended": false,
  "escalated": true,
  "critical": true,
  "suggestions": [
    "Please try again later",
    "Contact support for immediate assistance"
  ]
}
```

### 5. Context Length Exceeded

**Detection:**
- String contains "context_length_exceeded" or "maximum context length"

**Handling:**
- Error code: `context_length_exceeded`
- Retry: No (context won't shrink on retry)
- User message: "Your request is too complex. Please try a simpler query."
- Alert: Warning level after 5 occurrences in 1 hour

**Response:**
```json
{
  "user_message": "Your request is too complex. Please try a simpler query.",
  "action_code": "CONTEXT_LENGTH_EXCEEDED",
  "retry_recommended": true,
  "suggestions": [
    "Break your query into smaller parts",
    "Use fewer words to describe what you need",
    "Ask one question at a time"
  ]
}
```

---

## Retry Strategy

### Configuration

```python
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds
```

### Exponential Backoff

```python
backoff = base_backoff * (2 ** (attempt - 1)) + jitter
# Attempt 1: 0.5s + jitter
# Attempt 2: 1.0s + jitter
# Attempt 3: 2.0s + jitter
```

### When to Retry

- ✅ Timeout errors
- ✅ Rate limit errors
- ✅ Server errors (5xx)
- ❌ Auth errors (fatal)
- ❌ Context length errors (won't change)

### Retry Flow

```python
for attempt in range(1, MAX_RETRIES + 1):
    try:
        result = llm_call()
        cache_result(result)  # Cache successful results
        return result
    except LLMRequestError as exc:
        log_error_with_context(exc)
        send_alert_if_threshold_met(exc)
        
        if attempt < MAX_RETRIES:
            backoff = calculate_backoff(attempt)
            sleep(backoff)
        else:
            # All retries failed
            escalate()
            return fallback_response()
```

---

## Fallback Mechanisms

### Fallback Priority

1. **Cache Fallback** (Task 19 Integration)
   - Check cache with lower similarity threshold (0.90 instead of 0.95)
   - Return similar successful response if found
   - Reduces error impact by using historical data

2. **Default Fallback**
   - Return UNCLEAR intent with clarifying questions
   - Graceful degradation - system never crashes

### Cache Fallback Implementation

```python
def try_cached_fallback(user_query: str, similarity_threshold: float = 0.90):
    """Try to find similar cached response as fallback."""
    try:
        cache = get_response_cache()
        cached = cache.get(user_query, similarity_threshold=similarity_threshold)
        if cached:
            logger.info(f"Using cached response as fallback")
            return cached
    except Exception as e:
        logger.warning(f"Cache fallback failed: {e}")
    return None
```

### UNCLEAR Response

```json
{
  "intent": "unclear",
  "action_code": "UNCLEAR",
  "confidence": 0.0,
  "requires_clarification": true,
  "clarification_message": "I'm not sure I understood that. Could you rephrase or provide more details?",
  "clarifying_questions": [
    "Do you want to remove something from your cart?",
    "Are you trying to add a product?",
    "Would you like to view or clear your cart?"
  ]
}
```

---

## Escalation System

### Severity Levels

1. **Info** - FYI, no immediate action needed
2. **Warning** - Needs attention soon (throttled by frequency)
3. **Error** - Needs immediate attention (always escalates)
4. **Critical** - System failure, page on-call (always escalates)

### Escalation Thresholds

```python
ESCALATION_THRESHOLDS = {
    "rate_limit": 10,           # 10 in 1 hour
    "timeout": 20,              # 20 in 1 hour
    "server_error": 5,          # 5 in 1 hour
    "auth_error": 1,            # Immediate
    "context_length_exceeded": 5, # 5 in 1 hour
    "unknown_error": 15,        # 15 in 1 hour
}
```

### Escalation Flow

```python
def should_escalate(error_type: str) -> bool:
    """Check if error count exceeds threshold (1-hour sliding window)."""
    threshold = ESCALATION_THRESHOLDS.get(error_type, 10)
    count = len([e for e in error_queue if e > (now - 3600)])
    return count >= threshold
```

### Alert Payload

```json
{
  "event_type": "RATE_LIMIT_EXCEEDED",
  "severity": "warning",
  "context": {
    "user_input": "show me red shoes",
    "error_code": "rate_limit",
    "attempt": 3,
    "timestamp": 1730000000
  },
  "timestamp": 1730000000,
  "environment": "production",
  "service": "chatNShop-intent-classification"
}
```

### Webhook Integration

```python
# Configure webhook URL
export ESCALATION_WEBHOOK_URL="https://alerts.company.com/webhook"

# Sends POST request to webhook
# Falls back to structured logging if webhook unavailable
```

---

## Logging

### Structured Logging

All errors logged with full context:

```python
logger.error(
    f"LLM Error: {type(error).__name__}: {str(error)}",
    extra={
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_code": "rate_limit",
        "user_query": "show me red shoes",
        "user_id": "user_123",
        "session_id": "sess_456",
        "timestamp": 1730000000,
        "request_id": "req_789"
    },
    exc_info=True  # Include stack trace
)
```

### Log Levels

- **DEBUG**: Cache hits, successful operations
- **INFO**: Request attempts, successful completions
- **WARNING**: Retries, cache failures
- **ERROR**: API failures, error handling
- **CRITICAL**: Complete failure, escalation

---

## Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo

# Retry Configuration
MAX_RETRIES=3
RETRY_BASE_BACKOFF=0.5

# Cache Configuration
ENABLE_LLM_CACHE=true
CACHE_SIMILARITY_THRESHOLD=0.95
FALLBACK_SIMILARITY_THRESHOLD=0.90

# Escalation Configuration
ESCALATION_WEBHOOK_URL=https://alerts.company.com/webhook
ENVIRONMENT=production
```

### Configurable Thresholds

```python
# In app/core/alert_notifier.py
ESCALATION_THRESHOLDS = {
    "rate_limit": 10,
    "timeout": 20,
    # ... customize per environment
}
```

---

## Integration with Other Systems

### Task 19: LLM Caching

- **Pre-LLM**: Check cache before calling LLM
- **Post-LLM**: Cache successful responses
- **Fallback**: Use cache with lower threshold when LLM fails

### Task 20: Entity Extraction

- **Preservation**: Entities extracted even during errors
- **Fallback**: Rule-based entity extraction when LLM fails

### Task 17: Context Enhancement

- **Logging**: Context included in error logs
- **Debugging**: Full session context for troubleshooting

### Task 18: Confidence Calibration

- **Low Confidence**: Triggers UNCLEAR response
- **Prevention**: Reduces errors by avoiding uncertain classifications

---

## Error Flow Example

### Scenario: Rate Limit Error

```
1. User query: "show me red nike shoes"
2. Check cache → miss
3. Rule-based confidence: 0.45 (low)
4. Attempt 1: LLM call
   ├→ Error: 429 Rate Limit
   ├→ Classify error: rate_limit
   ├→ Log with full context
   ├→ Check threshold: 8/10 (below threshold, no alert)
   ├→ Wait 1s backoff
5. Attempt 2: LLM call
   ├→ Error: 429 Rate Limit
   ├→ Check threshold: 9/10 (below threshold)
   ├→ Wait 2s backoff
6. Attempt 3: LLM call
   ├→ Error: 429 Rate Limit
   ├→ Check threshold: 10/10 (THRESHOLD MET!)
   ├→ Send alert: RATE_LIMIT_EXCEEDED (severity: warning)
7. All retries failed → Escalate
8. Try cache fallback (similarity 0.90)
   ├→ Found: "nike running shoes" query (similarity: 0.92)
   ├→ Return cached response with metadata
9. User receives response from cache fallback
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Configure `ESCALATION_WEBHOOK_URL`
- [ ] Set appropriate escalation thresholds
- [ ] Enable LLM caching (`ENABLE_LLM_CACHE=true`)
- [ ] Configure retry parameters
- [ ] Set up alerting infrastructure
- [ ] Test error scenarios
- [ ] Monitor error rates and latency

### Monitoring Metrics

1. **Error Rates**
   - Timeout rate
   - Rate limit rate
   - Server error rate
   - Overall error rate

2. **Retry Metrics**
   - Average retry count
   - Retry success rate
   - Time spent in retries

3. **Fallback Metrics**
   - Cache fallback hit rate
   - UNCLEAR response rate
   - Overall fallback rate

4. **Escalation Metrics**
   - Alert frequency by type
   - Time to resolution
   - Escalation trigger rate

---

## Testing

### Unit Tests

Test specific error handlers:

```python
def test_handle_timeout_error():
    context = {"user_input": "test query", "error_code": "timeout"}
    response = handle_timeout_error("test query", context)
    
    assert response["action_code"] == "TIMEOUT_ERROR"
    assert response["retry_recommended"] == True
    assert "timeout" in response["user_message"].lower()
```

### Integration Tests

Test end-to-end error flow:

```python
def test_llm_error_with_cache_fallback():
    # 1. Cache a successful response
    cache.set("red shoes", successful_response)
    
    # 2. Trigger LLM error
    with mock.patch("openai.ChatCompletion.create") as mock_llm:
        mock_llm.side_effect = RateLimitError()
        
        # 3. Make request
        response = handler.handle(request)
        
        # 4. Verify fallback from cache
        assert response["metadata"]["fallback_source"] == "cache"
        assert response["confidence"] > 0.7
```

---

## Troubleshooting

### High Error Rate

**Symptoms**: Many LLM errors, frequent escalations

**Solutions:**
1. Check OpenAI API status
2. Verify API key validity
3. Review rate limits
4. Increase cache usage to reduce LLM calls

### Cache Fallback Not Working

**Symptoms**: UNCLEAR responses despite cached similar queries

**Solutions:**
1. Verify cache is enabled
2. Check similarity threshold (try lowering to 0.85)
3. Ensure cache has sufficient data
4. Review cache metrics (`/api/v1/cache/metrics`)

### Excessive Alerts

**Symptoms**: Too many webhook/log alerts

**Solutions:**
1. Increase escalation thresholds
2. Filter by severity (only error/critical)
3. Adjust time window (1 hour → 2 hours)
4. Add alert deduplication

---

## Future Enhancements

1. **Circuit Breaker Pattern**
   - Open circuit after N consecutive failures
   - Half-open state for testing
   - Auto-recovery detection

2. **Distributed Error Tracking**
   - Redis-based error counting for multiple workers
   - Cross-instance escalation coordination

3. **ML-Based Error Prediction**
   - Predict errors before they occur
   - Proactive fallback selection

4. **A/B Testing Error Responses**
   - Test different user messages
   - Optimize for user satisfaction

---

## Summary

Task 21 implements a **production-ready error handling system** with:

✅ **Comprehensive Error Classification** (timeout, rate limit, 5xx, auth, context length)  
✅ **Intelligent Retry Logic** (3 attempts, exponential backoff)  
✅ **Multi-Level Fallback** (cache fallback → UNCLEAR)  
✅ **Frequency-Based Escalation** (thresholds, severity levels, webhooks)  
✅ **Structured Logging** (full context, stack traces)  
✅ **Integration with Tasks 17-20** (caching, context, entities, confidence)  

**Result**: **Resilient, self-healing system** that provides graceful degradation and actionable insights for operations teams.

