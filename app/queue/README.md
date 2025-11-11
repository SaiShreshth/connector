# Queue Infrastructure Documentation (CNS-21)

## Overview

Message queue infrastructure for asynchronous LLM intent classification. Routes ambiguous queries from CNS-12 to LLM processing with retry logic, dead letter queue, and monitoring.

---

## Architecture

```
CNS-12 Ambiguity Resolver
         ↓
   (AMBIGUOUS/UNCLEAR)
         ↓
  Ambiguous Query Queue (Redis)
         ↓
    LLM Worker (CNS-23+)
         ↓
Classification Result Queue
         ↓
    Return to User
         
Failed Messages → Dead Letter Queue
```

---

## Technology Decision

**Selected:** Redis

**Why Redis:**
- ✅ Already in `requirements.txt`
- ✅ Fast in-memory performance
- ✅ Built-in persistence and durability
- ✅ Sorted sets for priority queues
- ✅ Simple deployment
- ✅ Good Python client library

**Alternatives Considered:**
- RabbitMQ: More features but heavier
- AWS SQS: Cloud-only, vendor lock-in
- Azure Service Bus: Cloud-only, vendor lock-in

---

## Queue Configuration

### Queue Names

1. **`chatns:queue:ambiguous_queries`** (Input)
   - Purpose: Ambiguous/unclear queries from CNS-12
   - Type: Priority queue (sorted set)
   - TTL: 24 hours

2. **`chatns:queue:classification_results`** (Output)
   - Purpose: LLM classification results
   - Type: FIFO queue (list)
   - TTL: 24 hours

3. **`chatns:queue:dead_letter`** (DLQ)
   - Purpose: Failed messages after max retries
   - Type: FIFO queue (list)
   - No TTL (manual cleanup)

---

## Configuration

### Environment Variables

```bash
# Redis Connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password  # Optional

# Queue Settings
MESSAGE_TTL=86400  # 24 hours
MAX_RETRY_ATTEMPTS=3
RETRY_DELAY=5  # seconds

# Connection Pool
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5

# Priority Queue
ENABLE_PRIORITY_QUEUE=true
PRIORITY_HIGH=1        # Premium users
PRIORITY_NORMAL=5      # Regular users
PRIORITY_LOW=10        # Batch processing
```

---

## Usage Examples

### 1. Integrate with CNS-12 Ambiguity Resolver

```python
from app.ai.intent_classification.ambiguity_resolver import detect_intent
from app.queue.integration import send_to_llm_queue

# User query
user_query = "add shoes and track my order"

# Detect ambiguity (CNS-12)
result = detect_intent(user_query)

# If ambiguous, send to LLM queue
if result["action"] in ["AMBIGUOUS", "UNCLEAR"]:
    message_id = send_to_llm_queue(
        query=user_query,
        ambiguity_result=result,
        user_id="user_123",
        is_premium=False  # Regular user
    )
    print(f"Sent to LLM queue: {message_id}")
else:
    # Clear intent, no LLM needed
    print(f"Clear intent: {result['action']}")
```

### 2. Enqueue Message Directly

```python
from app.queue.queue_manager import queue_manager

message_id = queue_manager.enqueue_ambiguous_query(
    query="what is this?",
    context={
        "ambiguity_type": "UNCLEAR",
        "possible_intents": {},
        "user_id": "user_456"
    },
    priority=5  # Normal priority
)
```

### 3. Dequeue Message (LLM Worker)

```python
from app.queue.queue_manager import queue_manager

# Get next message
message = queue_manager.dequeue_ambiguous_query(timeout=10)

if message:
    print(f"Processing: {message['query']}")
    
    # Process with LLM...
    result = llm_classify(message['query'])
    
    # Enqueue result
    queue_manager.enqueue_classification_result(
        message_id=message['message_id'],
        result=result,
        processing_time=2.5
    )
```

### 4. Handle Failures with Retry

```python
from app.queue.queue_manager import queue_manager

message = queue_manager.dequeue_ambiguous_query()

try:
    # Process message
    result = process_message(message)
except Exception as e:
    # Retry or move to DLQ
    if not queue_manager.retry_message(message):
        queue_manager.move_to_dead_letter_queue(
            message=message,
            error=str(e)
        )
```

### 5. Monitor Queue Health

```python
from app.queue.queue_manager import queue_manager
from app.queue.monitor import queue_monitor

# Get queue statistics
stats = queue_manager.get_queue_stats()
print(f"Ambiguous queue size: {stats['ambiguous_queue_size']}")
print(f"DLQ size: {stats['dead_letter_queue_size']}")

# Get performance metrics
metrics = queue_monitor.get_metrics()
print(f"Throughput: {metrics['throughput']} msg/sec")
print(f"Avg processing time: {metrics['avg_processing_time']}s")
```

---

## Retry Policy

**Exponential Backoff:**
- Attempt 1: Immediate
- Attempt 2: 5 seconds delay
- Attempt 3: 10 seconds delay
- Attempt 4: 20 seconds delay
- After 3 retries → Dead Letter Queue

**Why Exponential Backoff:**
- Avoids overwhelming downstream systems
- Gives temporary issues time to resolve
- Reduces resource consumption

---

## Priority Queue

Messages are processed by priority:

| Priority | Score | Use Case |
|----------|-------|----------|
| High | 1 | Premium users, urgent requests |
| Normal | 5 | Regular users (default) |
| Low | 10 | Batch processing, analytics |

Lower score = higher priority (processed first)

---

## Monitoring Metrics

```python
{
    "ambiguous_queue_size": 15,          # Pending messages
    "result_queue_size": 3,               # Unprocessed results
    "dead_letter_queue_size": 1,          # Failed messages
    "messages_enqueued": 234,             # Total enqueued
    "messages_dequeued": 220,             # Total processed
    "messages_failed": 5,                 # Total failures
    "messages_retried": 12,               # Total retries
    "avg_processing_time": 1.8,           # Seconds
    "throughput": 3.5,                    # Messages/second
    "uptime_seconds": 3600                # System uptime
}
```

---

## Health Check

```python
from app.queue.queue_manager import queue_manager

if queue_manager.health_check():
    print("✅ Queue system healthy")
else:
    print("❌ Queue system down")
```

---

## Production Deployment

### 1. Redis Setup

**Local Development:**
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

**Production:**
- Use Redis Cluster for high availability
- Enable AOF persistence
- Set up replication
- Configure memory limits
- Enable authentication

### 2. Environment Configuration

Create `.env` file:
```bash
REDIS_HOST=redis-cluster.example.com
REDIS_PORT=6379
REDIS_PASSWORD=secure_password
MESSAGE_TTL=86400
MAX_RETRY_ATTEMPTS=3
```

### 3. Application Integration

Add to `main.py`:
```python
from app.queue.queue_manager import queue_manager
from app.queue.monitor import queue_monitor

@app.on_event("startup")
async def startup_event():
    # Initialize queue system
    if queue_manager.health_check():
        logger.info("✅ Queue system ready")
    else:
        logger.error("❌ Queue system unavailable")

@app.get("/health/queue")
async def queue_health():
    return {
        "healthy": queue_manager.health_check(),
        "stats": queue_manager.get_queue_stats()
    }
```

---

## Testing

### Unit Tests

```python
# tests/test_queue.py
from app.queue.queue_manager import QueueManager

def test_enqueue_dequeue():
    qm = QueueManager()
    
    # Enqueue message
    msg_id = qm.enqueue_ambiguous_query(
        query="test query",
        context={"user_id": "test"}
    )
    
    # Dequeue message
    message = qm.dequeue_ambiguous_query()
    assert message["message_id"] == msg_id
    assert message["query"] == "test query"
```

### Integration Test

```python
def test_full_flow():
    from app.queue.integration import send_to_llm_queue
    from app.ai.intent_classification.ambiguity_resolver import detect_intent
    
    # Ambiguous query
    result = detect_intent("add and search")
    assert result["action"] == "AMBIGUOUS"
    
    # Send to queue
    msg_id = send_to_llm_queue(
        query="add and search",
        ambiguity_result=result
    )
    
    assert msg_id is not None
```

---

## Troubleshooting

### Issue 1: Connection Refused

**Error:** `redis.ConnectionError: Connection refused`

**Solution:**
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Start Redis
docker start redis
# or
redis-server
```

### Issue 2: Queue Growing Too Large

**Symptoms:** `ambiguous_queue_size` keeps increasing

**Solutions:**
1. Scale up LLM workers
2. Increase worker concurrency
3. Check for worker errors
4. Adjust message TTL

### Issue 3: Messages in DLQ

**Symptoms:** `dead_letter_queue_size` > 0

**Solutions:**
1. Check DLQ messages:
```python
# Get DLQ messages for analysis
dlq_messages = redis_client.lrange("chatns:queue:dead_letter", 0, -1)
for msg in dlq_messages:
    print(json.loads(msg))
```

2. Fix underlying issue
3. Replay messages or discard

---

## API Endpoints (Optional)

Add to FastAPI app:

```python
from fastapi import APIRouter
from app.queue.queue_manager import queue_manager
from app.queue.monitor import queue_monitor

router = APIRouter(prefix="/queue", tags=["Queue"])

@router.get("/stats")
async def get_queue_stats():
    """Get queue statistics"""
    return queue_manager.get_queue_stats()

@router.get("/metrics")
async def get_queue_metrics():
    """Get queue performance metrics"""
    return queue_monitor.get_metrics()

@router.post("/clear/{queue_name}")
async def clear_queue(queue_name: str):
    """Clear specific queue (admin only)"""
    queue_manager.clear_queue(queue_name)
    return {"status": "cleared", "queue": queue_name}
```

---

## Future Enhancements

1. **Multiple Redis Instances:** For high availability
2. **Message Compression:** Reduce memory usage
3. **Custom Serialization:** Faster than JSON
4. **Queue Partitioning:** Scale horizontally
5. **Rate Limiting:** Per-user throttling
6. **Message Deduplication:** Prevent duplicates

---

## Related Tasks

- **CNS-12:** Ambiguity Resolver (source of ambiguous queries)
- **CNS-23:** Request Status Tracking (consumer of this queue)
- **CNS-27:** Fallback and Error Handling (uses DLQ)
- **CNS-29:** Cost Monitoring (tracks queue metrics)

---

**Last Updated:** October 28, 2025  
**Version:** 1.0  
**Status:** ✅ Production Ready
