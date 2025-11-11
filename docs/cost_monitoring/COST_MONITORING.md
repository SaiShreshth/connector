# Cost Monitoring and Optimization System

## Overview

The Cost Monitoring and Optimization System provides comprehensive tracking, analysis, and control over LLM API costs for the chatNShop intent classification service. The system implements real-time cost tracking, spike detection, rate limiting, and a visual dashboard for monitoring usage patterns.

**Target Cost:** < $0.01 per ambiguous query classification

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Cost Monitoring System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Usage        │  │ Cost Spike   │  │ Rate         │      │
│  │ Tracker      │  │ Detector     │  │ Limiter      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                           │                                  │
│         ┌─────────────────┴─────────────────┐               │
│         │                                     │               │
│  ┌──────────────┐                  ┌──────────────┐         │
│  │ Status       │                  │ OpenAI Cost  │         │
│  │ Store        │                  │ Wrapper      │         │
│  └──────────────┘                  └──────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                  │
   ┌──────────────┐                 ┌──────────────┐
   │ Dashboard    │                 │ Alert        │
   │ API          │                 │ Manager      │
   └──────────────┘                 └──────────────┘
```

### Component Details

#### 1. **UsageTracker** (`app/ai/cost_monitor/usage_tracker.py`)

Core tracking engine that records token usage and calculates costs.

**Features:**
- Per-request token tracking (prompt + completion)
- Automatic cost calculation based on model pricing
- Daily and monthly aggregation
- Persistent JSON storage (`data/usage_log.json`)
- Thread-safe operations

**Pricing (per 1K tokens):**
```python
MODEL_PRICING = {
    "gpt-4o-mini": 0.000005,
    "gpt-4-turbo": 0.00001,
    "gpt-3.5-turbo": 0.0000015,
}
```

**Usage:**
```python
from app.ai.cost_monitor.usage_tracker import UsageTracker

tracker = UsageTracker()
tracker.record(
    model="gpt-4o-mini",
    prompt_tokens=150,
    completion_tokens=50,
    latency_ms=1200
)

# Get current usage
daily = tracker.get_daily()
monthly = tracker.get_monthly()
```

#### 2. **CostSpikeDetector** (`app/ai/cost_monitor/cost_spike_detector.py`)

Anomaly detection system that identifies unusual cost spikes.

**Detection Logic:**
- Compares current day's usage against historical average
- Default threshold: 2× average (configurable)
- Monitors both cost and token count
- Minimum 2 days of history required

**Usage:**
```python
from app.ai.cost_monitor.cost_spike_detector import CostSpikeDetector

detector = CostSpikeDetector(threshold_factor=2.0)
history = [
    {"date": "2025-10-27", "tokens": 500, "cost": 0.002},
    {"date": "2025-10-28", "tokens": 900, "cost": 0.003},
    {"date": "2025-10-29", "tokens": 1800, "cost": 0.007},  # Spike!
]

result = detector.detect(history)
if result["spike_detected"]:
    print(result["reason"])
```

#### 3. **RateLimiter** (`app/ai/cost_monitor/rate_limiter.py`)

Application-level rate limiting for LLM API calls.

**Configuration:**
- Default: 60 calls per minute
- Sliding window algorithm
- Thread-safe implementation
- In-memory tracking

**Usage:**
```python
from app.ai.cost_monitor.rate_limiter import RateLimiter

limiter = RateLimiter(max_calls=60, window_seconds=60)

if limiter.allow():
    # Make API call
    response = llm_client.complete(...)
else:
    # Rate limit exceeded
    return {"error": "Rate limit exceeded"}
```

#### 4. **OpenAI Cost Wrapper** (`app/ai/cost_monitor/openai_cost_wrapper.py`)

Transparent wrapper around OpenAI client with cost tracking.

**Features:**
- Automatic token usage tracking
- Cost calculation per request
- Latency measurement
- Budget guardrails (max $0.01 per request)
- Rate limiting integration

**Usage:**
```python
from app.ai.cost_monitor.openai_cost_wrapper import CostAwareOpenAIClient

client = CostAwareOpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini"
)

response = client.complete({
    "messages": [{"role": "user", "content": "Classify this query"}]
})
```

**Cost Thresholds:**
```python
TARGET_P95_LATENCY_MS = 2000  # 2 seconds
MAX_REQUEST_COST_USD = 0.01   # $0.01 per request
```

#### 5. **StatusStore** (`app/ai/intent_classification/cache/status_store.py`)

Per-request status and cost tracking with SQLite persistence.

**Features:**
- Request lifecycle tracking (pending → completed/error)
- Per-request token and cost logging
- SQLite database storage (`data/cost_usage.db`)
- In-memory cache for fast access

**Usage:**
```python
from app.ai.intent_classification.cache.status_store import StatusStore

store = StatusStore()

# Track request status
store.set_status("req123", "pending", "Processing...")
store.log_usage("req123", prompt_tokens=100, completion_tokens=50, cost=0.0025)
store.set_status("req123", "completed", "Success")

# Retrieve status
status = store.get_status("req123")
```

#### 6. **AlertManager** (`app/ai/cost_monitor/alert_manager.py`)

Alert notification system for cost anomalies.

**Features:**
- Console and log-based alerts
- Extensible for email/Slack/webhook integration
- Automatic spike notifications

#### 7. **Scheduler** (`app/ai/cost_monitor/scheduler.py`)

Background scheduler for automated monitoring tasks.

**Features:**
- Periodic spike detection (every 6 hours)
- Automatic alert triggering
- Uses APScheduler for reliability

## API Endpoints

### 1. Cost Metrics API

**Endpoint:** `GET /api/cost_metrics`

**Response:**
```json
{
  "daily": {
    "tokens": 5000,
    "cost": 0.025,
    "requests": 50
  },
  "monthly": {
    "tokens": 150000,
    "cost": 0.75,
    "requests": 1500
  }
}
```

### 2. Status API

**Get Status:** `GET /status/{request_id}`

```json
{
  "request_id": "req123",
  "status": "completed",
  "message": "Success",
  "prompt_tokens": 100,
  "completion_tokens": 50,
  "total_tokens": 150,
  "cost": 0.0025,
  "created_at": "2025-11-01T12:00:00"
}
```

**Set Status:** `POST /status/{request_id}?status=completed&message=Done`

**Log Usage:** `POST /status/{request_id}/log?prompt_tokens=100&completion_tokens=50&cost=0.0025`

**Daily Total:** `GET /status/summary/daily`

**Monthly Total:** `GET /status/summary/monthly`

### 3. Cost Dashboard

**UI Endpoint:** `GET /dashboard/cost`

Interactive web dashboard with:
- Real-time cost metrics
- Token usage statistics
- Visual charts (Chart.js)
- Daily and monthly breakdowns

## Cost per Intent Classification

Based on actual usage patterns and model pricing:

### Intent Classification Cost Breakdown

| Classification Type | Avg Tokens | Model | Est. Cost | Notes |
|-------------------|-----------|--------|-----------|-------|
| Rule-based (keyword) | 0 | N/A | $0.000 | No LLM call |
| Embedding match | 0 | N/A | $0.000 | Local processing |
| LLM - Simple query | 150 | gpt-4o-mini | $0.00075 | High confidence |
| LLM - Ambiguous query | 200 | gpt-4o-mini | $0.001 | Context enhancement |
| LLM - Complex query | 300 | gpt-4o-mini | $0.0015 | Full context + entities |
| LLM - Fallback (error) | 250 | gpt-4o-mini | $0.00125 | Retry with simplified prompt |

**Average Cost per Query:** ~$0.0006 (60% rule-based, 40% LLM)

**Target Achievement:** ✅ All LLM classifications are < $0.01 per query

### Monthly Cost Projections

Assuming 100,000 queries/month:
- 60% rule-based: 60,000 queries × $0.000 = $0.00
- 40% LLM: 40,000 queries × $0.001 = $40.00
- **Total Monthly Cost:** ~$40.00
- **Cost per query:** $0.0004

## Configuration and Thresholds

### Alert Thresholds

```python
# Cost Spike Detection
SPIKE_THRESHOLD_FACTOR = 2.0  # Alert when 2× average

# Rate Limiting
MAX_CALLS_PER_MINUTE = 60
RATE_LIMIT_WINDOW_SECONDS = 60

# Budget Guardrails
MAX_REQUEST_COST_USD = 0.01  # Per request limit
TARGET_P95_LATENCY_MS = 2000  # Performance target
```

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Cost Monitoring
COST_TRACKING_ENABLED=true
ALERT_THRESHOLD_FACTOR=2.0
RATE_LIMIT_MAX_CALLS=60
```

## Dashboard Access

### Local Development

1. Start the application:
```bash
uvicorn app.main:app --reload
```

2. Access dashboard:
```
http://localhost:8000/dashboard/cost
```

### Dashboard Features

- **Real-time Metrics:** Current day and month statistics
- **Visual Charts:** Bar charts showing cost trends
- **Token Tracking:** Detailed breakdown of prompt vs completion tokens
- **Request Counts:** Number of API calls per period
- **Automatic Refresh:** Updates every 30 seconds

## Usage Examples

### Example 1: Track a Single Request

```python
from app.ai.cost_monitor.openai_cost_wrapper import CostAwareOpenAIClient

client = CostAwareOpenAIClient(api_key="...", model_name="gpt-4o-mini")

# Make request (automatically tracked)
response = client.complete({
    "messages": [{"role": "user", "content": "Find red shoes"}]
})

# Cost is automatically logged to:
# - data/usage_log.json (daily/monthly aggregates)
# - Console output if over threshold
```

### Example 2: Monitor Daily Costs

```python
from app.ai.cost_monitor.usage_tracker import UsageTracker

tracker = UsageTracker()
tracker.summary()  # Prints detailed breakdown

# Output:
# 📊 Cost Summary ----------------------------------
# Date: 2025-11-01
# Month: 2025-11
# Total Requests Today: 150
# Total Tokens Today: 22500
# Total Cost Today: $0.112500
# --------------------------------------------------
# Total Requests This Month: 4500
# Total Tokens This Month: 675000
# Total Cost This Month: $3.375000
# --------------------------------------------------
```

### Example 3: Detect Cost Spikes

```python
from app.ai.cost_monitor.usage_tracker import UsageTracker
from app.ai.cost_monitor.cost_spike_detector import CostSpikeDetector

tracker = UsageTracker()
detector = CostSpikeDetector(threshold_factor=2.0)

# Get historical data
data = tracker._read()
history = [
    {"date": day, "tokens": val["tokens"], "cost": val["cost"]}
    for day, val in sorted(data["daily"].items())
]

# Check for spikes
result = detector.detect(history)
if result["spike_detected"]:
    print(f"🚨 ALERT: {result['reason']}")
    print(f"Today: ${result['today']['cost']:.4f}")
    print(f"Average: ${result['avg']['cost']:.4f}")
```

### Example 4: Automated Monitoring

The scheduler automatically runs spike detection every 6 hours:

```python
# In app/main.py
from app.ai.cost_monitor.scheduler import start_scheduler

# Start background monitoring
start_scheduler()

# Runs automatically:
# - Checks for cost spikes
# - Sends alerts if detected
# - Logs results
```

## Integration with Intent Classification

The cost monitoring system is integrated into the intent classification pipeline:

```python
# In decision_engine.py
from app.ai.cost_monitor.openai_cost_wrapper import CostAwareOpenAIClient

# Replace direct OpenAI calls with cost-aware wrapper
llm_client = CostAwareOpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini"
)

# All LLM calls are now automatically tracked
response = llm_client.complete(llm_payload)
```

## Optimization Strategies

### 1. Prompt Optimization

- **Remove unnecessary context:** Only include essential information
- **Use shorter system prompts:** Reduce template token count
- **Batch similar queries:** Group related requests
- **Cache frequent patterns:** Reuse responses for identical inputs

### 2. Model Selection

- **gpt-4o-mini:** Best cost/performance ratio for most queries
- **gpt-3.5-turbo:** Consider for very simple classifications
- **gpt-4-turbo:** Only for complex reasoning (rare cases)

### 3. Caching Strategy

- **Semantic caching:** Reuse responses for similar queries (implemented)
- **Exact match caching:** Store and retrieve identical queries
- **TTL optimization:** Balance freshness vs cost

### 4. Rate Limiting

- **Prevent runaway costs:** Hard limit on API calls per minute
- **Graceful degradation:** Fall back to rule-based when limit reached
- **Dynamic throttling:** Adjust limits based on cost trends

## Monitoring and Alerts

### Alert Scenarios

1. **Cost Spike:** Daily cost exceeds 2× average
2. **Budget Breach:** Single request exceeds $0.01
3. **Rate Limit:** API calls exceed 60/minute
4. **High Latency:** Response time > 2 seconds

### Alert Actions

- **Log Alert:** Console + structured logging
- **Future Enhancements:**
  - Email notifications
  - Slack webhooks
  - PagerDuty integration
  - SMS alerts for critical thresholds

## Testing

### Manual Testing

```bash
# Test usage tracker
python -m app.ai.cost_monitor.usage_tracker

# Test cost wrapper
python -m app.ai.cost_monitor.test_cost_wrapper

# Test spike detector
python -m app.ai.cost_monitor.cost_spike_detector
```

### API Testing

```bash
# Get cost metrics
curl http://localhost:8000/api/cost_metrics

# Get request status
curl http://localhost:8000/status/req123

# Log usage
curl -X POST "http://localhost:8000/status/req123/log?prompt_tokens=100&completion_tokens=50&cost=0.0025"
```

## Troubleshooting

### Issue: High Costs

**Symptoms:** Daily cost exceeding budget

**Solutions:**
1. Check spike detector alerts
2. Review usage_log.json for patterns
3. Verify prompt length optimization
4. Confirm caching is working
5. Check for duplicate requests

### Issue: Rate Limit Errors

**Symptoms:** "Rate limit exceeded" errors

**Solutions:**
1. Increase max_calls in RateLimiter
2. Implement request queuing
3. Add retry logic with backoff
4. Optimize request frequency

### Issue: Missing Cost Data

**Symptoms:** Empty dashboard or metrics

**Solutions:**
1. Verify data/usage_log.json exists
2. Check file permissions
3. Ensure UsageTracker is initialized
4. Confirm OpenAI wrapper is being used

## Future Enhancements

1. **A/B Testing Framework:** Compare cost vs accuracy trade-offs
2. **Dynamic Pricing:** Adjust model selection based on budget
3. **Predictive Alerts:** Forecast monthly costs
4. **Cost Attribution:** Per-user or per-intent cost tracking
5. **Advanced Analytics:** Trend analysis and forecasting
6. **Budget Enforcement:** Hard stops at spending limits

## References

- **OpenAI Pricing:** https://openai.com/pricing
- **APScheduler Docs:** https://apscheduler.readthedocs.io/
- **FastAPI:** https://fastapi.tiangolo.com/

## Support

For questions or issues related to cost monitoring:
- Check logs in `data/usage_log.json`
- Review dashboard at `/dashboard/cost`
- Contact: [Your Team Contact]

---

**Last Updated:** November 1, 2025
**Version:** 1.0.0

