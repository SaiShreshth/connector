# ChatNShop - Project Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Component Relationships](#component-relationships)
4. [Technology Stack](#technology-stack)
5. [Data Flow](#data-flow)
6. [Deployment Architecture](#deployment-architecture)
7. [Design Decisions & Rationale](#design-decisions--rationale)

---

## System Overview

**ChatNShop** is an intelligent intent classification system designed for e-commerce applications. It combines rule-based pattern matching, semantic search, and LLM-powered classification to accurately understand user queries and route them to appropriate handlers.

### Core Capabilities

- **Hybrid Intent Classification**: Combines keyword matching, embedding-based semantic search, and LLM fallback
- **Real-time Processing**: Sub-100ms response for most queries
- **Asynchronous Processing**: Queue-based system for ambiguous queries
- **Cost Optimization**: Caching, threshold-based routing, and usage tracking
- **Entity Extraction**: Identifies products, brands, prices, colors, sizes from queries
- **A/B Testing**: Built-in experimentation framework
- **Production Ready**: Monitoring, error handling, and graceful degradation

### Key Differentiators

```
┌─────────────────────────────────────────────────────────────┐
│  Why Hybrid Approach?                                       │
├─────────────────────────────────────────────────────────────┤
│  ✓ Fast: Keyword matching for obvious intents (< 50ms)     │
│  ✓ Smart: Embeddings catch semantic variations (< 100ms)   │
│  ✓ Accurate: LLM handles complex/ambiguous cases (< 2s)    │
│  ✓ Cost-effective: LLM only when needed (saves 80% costs)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

The system is organized into 5 distinct layers:

### 1. API Layer (`app/main.py`, `app/api/v1/`)

**Purpose**: HTTP interface for external clients

**Components**:
- FastAPI application server
- Request validation (Pydantic models)
- CORS middleware
- Error handling & logging
- Health check endpoints

**Endpoints**:
- `POST /classify` - Main classification endpoint
- `GET /health` - System health status
- `GET /status/{request_id}` - Async request status
- Cost monitoring & A/B testing APIs

### 2. Orchestration Layer (`app/ai/intent_classification/decision_engine.py`)

**Purpose**: Coordinates classification strategies

**Responsibilities**:
- Route requests through appropriate classifiers
- Apply priority rules and short-circuits
- Manage fallback logic
- Handle configuration hot-reload
- Cache management

**Decision Flow**:
```
Query → Normalize → Keyword Match
                         ↓
                   High Confidence? → YES → Return
                         ↓ NO
                   Embedding Match
                         ↓
                   Hybrid Blend
                         ↓
                   Confident? → YES → Return
                         ↓ NO
                   LLM Available? → YES → LLM Fallback
                         ↓ NO
                   Generic Fallback (SEARCH_PRODUCT)
```

### 3. Classification Layer

**Components**:

#### a) Rule-Based Classifier (`app/ai/intent_classification/`)
- **Keyword Matcher** (`keyword_matcher.py`): Pattern-based matching
- **Embedding Matcher** (`embedding_matcher.py`): Semantic similarity
- **Hybrid Classifier** (`hybrid_classifier.py`): Weighted blending
- **Confidence Evaluator** (`confidence_threshold.py`): Threshold checks

#### b) LLM Classifier (`app/ai/llm_intent/`)
- **Request Handler** (`request_handler.py`): Main LLM orchestrator
- **OpenAI Client** (`openai_client.py`): API integration with retry
- **Prompt Loader** (`prompt_loader.py`): Version-controlled prompts
- **Response Parser** (`response_parser.py`): Extract structured data

#### c) Entity Extractor (`app/ai/entity_extraction/`)
- **Extractor** (`extractor.py`): Identifies products, brands, prices, etc.
- **Validator** (`validator.py`): Validates extracted entities
- **Normalizer** (`normalizer.py`): Standardizes entity formats

### 4. Infrastructure Layer

#### a) Queue System (`app/queue/`)
**Technology**: Redis-based priority queues

**Components**:
- **Queue Manager** (`queue_manager.py`): Enqueue/dequeue operations
- **Queue Producer** (`queue_producer.py`): Publish ambiguous queries
- **Worker** (`worker.py`): Process queued requests
- **Monitor** (`monitor.py`): Queue health metrics

**Why Redis?**
- In-memory speed for low latency
- Native priority queue support
- Already in tech stack (no new dependencies)
- Simple pub/sub for worker coordination

#### b) Status Store (`app/core/status_store.py`)
**Technology**: Redis with in-memory fallback

**Purpose**: Track asynchronous request status
- Request lifecycle: QUEUED → PROCESSING → COMPLETED/FAILED
- Fast lookup by request_id (< 10ms)
- Auto-expiration after 1 hour

#### c) Caching Layer (`app/ai/llm_intent/response_cache.py`)
**Technology**: Redis + Qdrant vector cache

**Strategies**:
- Exact match cache (Redis): Immediate hits
- Semantic cache (Qdrant): Similar query detection
- TTL-based expiration (24 hours)
- Cache hit rate > 30% target

### 5. Data Layer

#### a) Qdrant Vector Database
**Purpose**: Store embeddings for semantic search

**Collections**:
- `intent_embeddings`: Reference intent vectors
- `query_cache`: Cached query embeddings
- `chatnshop_products`: Product catalog (future)

**Why Qdrant?**
- Optimized for vector similarity search
- Supports filtering and hybrid search
- Easy Python integration
- Docker deployment ready

#### b) Redis
**Purpose**: Fast key-value store

**Use Cases**:
- Queue management
- Request status tracking
- Cache storage
- Session data
- Rate limiting

#### c) SQLite (Development)
**Purpose**: Cost tracking and A/B testing data

**Databases**:
- `cost_usage.db`: API usage and cost metrics
- `ab_testing.db`: Experiment results

---

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Request                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (main.py)                 │
│  • CORS, validation, error handling                              │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              Decision Engine (Orchestrator)                      │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐            │
│  │  Keyword   │→ │  Embedding  │→ │    Hybrid    │            │
│  │  Matcher   │  │   Matcher   │  │  Classifier  │            │
│  └────────────┘  └─────────────┘  └──────────────┘            │
│         ↓                                                        │
│  ┌────────────────────────────────────────────┐                │
│  │      Confidence Threshold Check            │                │
│  └────────────┬───────────────────────────────┘                │
│               ↓                                                  │
│        High Confidence? → YES → Return Result                   │
│               ↓ NO                                               │
└───────────────┼──────────────────────────────────────────────────┘
                ↓
┌───────────────┴──────────────────────────────────────────────────┐
│           LLM Fallback Path (Async Optional)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │Queue Producer│→ │ Redis Queue  │→ │    Worker    │          │
│  └──────────────┘  └──────────────┘  └──────┬───────┘          │
│                                              ↓                    │
│                                    ┌──────────────────┐          │
│                                    │  LLM Request     │          │
│                                    │  Handler         │          │
│                                    └────────┬─────────┘          │
│                                             ↓                     │
│                                    ┌──────────────────┐          │
│                                    │  OpenAI API      │          │
│                                    │  (GPT-4)         │          │
│                                    └────────┬─────────┘          │
│                                             ↓                     │
│                                    ┌──────────────────┐          │
│                                    │ Response Parser  │          │
│                                    │ Entity Extractor │          │
│                                    └────────┬─────────┘          │
│                                             ↓                     │
│                                    ┌──────────────────┐          │
│                                    │  Status Store    │          │
│                                    │  (COMPLETED)     │          │
│                                    └──────────────────┘          │
└──────────────────────────────────────────────────────────────────┘

Supporting Services (Always Available):
├─ Configuration Manager (Hot reload)
├─ Cost Monitor (Track usage)
├─ Alert Manager (Notify on issues)
├─ A/B Testing Framework (Experiments)
└─ Logging & Metrics (Observability)
```

---

## Technology Stack

### Core Technologies

| Component | Technology | Version | Why Chosen |
|-----------|-----------|---------|------------|
| **Language** | Python | 3.12+ | Modern features, best performance, numpy compatibility |
| **Web Framework** | FastAPI | 0.115.0 | Async support, auto docs, Pydantic validation |
| **Server** | Uvicorn | 0.32.0 | ASGI server, high performance |
| **LLM** | OpenAI GPT-4 | GPT-4-turbo | Best accuracy for intent classification |
| **Embeddings** | SentenceTransformers | 3.1.1 | Fast, accurate semantic similarity |
| **Embedding Model** | all-MiniLM-L6-v2 | - | 384 dims, fast inference, good quality |
| **Vector DB** | Qdrant | 1.9.1 | Optimized for similarity search |
| **Cache/Queue** | Redis | 5.1.1 | Fast, reliable, simple |
| **HTTP Client** | httpx | 0.27.2 | Async support, connection pooling |

### ML/AI Stack

```python
# Embedding Generation
sentence-transformers==3.1.1  # Semantic similarity
torch==2.4.1                  # Deep learning backend
transformers==4.45.2          # HuggingFace models

# Vector Search
qdrant-client==1.9.1          # Vector database

# LLM Integration
openai==1.52.2                # GPT-4 API
tenacity==9.0.0               # Retry logic
```

### Infrastructure

```python
# Data Layer
redis==5.1.1                  # Cache, queue, status
sqlalchemy==2.0.36            # ORM for cost tracking

# Configuration
python-dotenv==1.0.1          # Environment variables
pyyaml==6.0.2                 # Config files

# Monitoring
loguru==0.7.2                 # Structured logging
APScheduler==3.10.4           # Scheduled jobs
```

### Why These Choices?

**FastAPI over Flask/Django**:
- Native async/await support (critical for concurrent requests)
- Automatic API documentation
- Built-in validation with Pydantic
- Modern, high performance

**Redis over RabbitMQ**:
- Already in stack (no new dependency)
- Simpler to operate
- Native priority queues
- Sub-millisecond latency

**Qdrant over Pinecone/Weaviate**:
- Self-hosted (no vendor lock-in)
- Docker deployment
- Lower latency (local)
- Cost-effective

**SentenceTransformers over OpenAI Embeddings**:
- No API costs
- Faster (local inference)
- Offline capable
- Good enough accuracy for intent classification

---

## Data Flow

### Synchronous Flow (Fast Path - 95% of requests)

```
1. Client sends query → POST /classify
   Example: "add to cart"

2. FastAPI validates request
   → ClassificationInput(text="add to cart")

3. Decision Engine receives query
   → normalize_text("add to cart")

4. Keyword Matcher checks
   → Finds exact match: "add to cart" → ADD_TO_CART
   → Confidence: 0.95 (above threshold 0.85)

5. Short-circuit: Skip embedding (priority rule)

6. Return immediately
   {
     "action_code": "ADD_TO_CART",
     "confidence_score": 0.95,
     "matched_keywords": ["add to cart"],
     "status": "CONFIDENT_KEYWORD"
   }

Time: ~30-50ms
```

### Hybrid Flow (Semantic Matching - 4% of requests)

```
1. Client sends query → POST /classify
   Example: "show me running kicks"

2. Keyword Matcher checks
   → Low confidence: "kicks" is slang, not in keywords
   → Score: 0.45 (below priority threshold)

3. Embedding Matcher activates
   → Encode query to 384-dim vector
   → Cosine similarity vs reference intents
   → Best match: SEARCH_PRODUCT (0.78)

4. Hybrid Classifier blends
   → keyword_score * 0.6 + embedding_score * 0.4
   → Final score: 0.58

5. Confidence check
   → Above threshold → Return

Time: ~80-120ms
```

### Asynchronous Flow (LLM Fallback - 1% of requests)

```
1. Client sends ambiguous query
   Example: "I need something for my trip"

2. Rule-based + Embedding both low confidence
   → keyword: 0.35
   → embedding: 0.40
   → Not confident enough

3. Queue Producer publishes to Redis
   → Generate request_id: "uuid-123"
   → Message: {query, context, rule_based_result}
   → Priority: normal

4. Return immediately to client
   {
     "request_id": "uuid-123",
     "status": "QUEUED",
     "message": "Processing asynchronously"
   }

5. Worker picks up from queue
   → Update status: PROCESSING

6. LLM Request Handler
   → Load prompt template
   → Add conversation context
   → Call OpenAI API
   → Parse response
   → Extract entities

7. Update status store
   → status: COMPLETED
   → result: {action_code, confidence, entities}

8. Client polls: GET /status/uuid-123
   → Returns completed result

Time: 200ms-2s (doesn't block client)
```

---

## Deployment Architecture

### Development Environment

```
┌─────────────────────────────────────────────────────────┐
│  Local Machine                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Python Application                               │  │
│  │  • uvicorn app.main:app                          │  │
│  │  • Port: 8000                                     │  │
│  └───────────┬──────────────────────────────────────┘  │
│              │                                          │
│  ┌───────────┴──────────────────────────────────────┐  │
│  │  Docker Compose Services                          │  │
│  │  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │   Qdrant    │  │   Redis     │               │  │
│  │  │   :6333     │  │   :6379     │               │  │
│  │  └─────────────┘  └─────────────┘               │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Commands**:
```bash
# Start services
docker-compose up -d

# Run application
source venv/bin/activate
uvicorn app.main:app --reload
```

### Production Environment (Docker)

```
┌──────────────────────────────────────────────────────────┐
│  Docker Network: chatnshop                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  intent-api Container                              │  │
│  │  • Python app                                       │  │
│  │  • Port: 8000                                       │  │
│  │  • Health check: /health                           │  │
│  │  • Auto-restart: unless-stopped                    │  │
│  └─────────┬──────────────────────────────────────────┘  │
│            │                                              │
│  ┌─────────┴──────────────┐  ┌────────────────────────┐ │
│  │  qdrant Container       │  │  redis Container       │ │
│  │  • Port: 6333 (HTTP)    │  │  • Port: 6379         │ │
│  │  • Port: 6334 (gRPC)    │  │  • Persistence: AOF   │ │
│  │  • Volume: qdrant_data  │  │  • Volume: redis_data │ │
│  │  • Health: /ready       │  │  • Health: PING       │ │
│  └─────────────────────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────────┘

External Access:
• API: http://localhost:8000
• Docs: http://localhost:8000/docs
• Qdrant: http://localhost:6333/dashboard
```

**Commands**:
```bash
# Build and start all services
docker-compose up --build

# View logs
docker-compose logs -f intent-api

# Scale workers
docker-compose up --scale intent-api=3
```

---

## Design Decisions & Rationale

### 1. Why Hybrid Approach (Not Just LLM)?

**Decision**: Use rule-based → embeddings → LLM cascade

**Rationale**:
- **Cost**: LLM APIs expensive ($0.01-0.03 per request)
- **Latency**: GPT-4 takes 500ms-2s
- **Accuracy**: 85% of queries are simple ("add to cart", "checkout")
- **Reliability**: LLM might be down, need fallback

**Impact**: 80% cost savings, 10x faster for common queries

### 2. Why Redis Queues (Not RabbitMQ)?

**Decision**: Use Redis for queue management

**Rationale**:
- Already using Redis for cache/status
- Simpler to operate (one less service)
- Native priority queue support
- Sub-millisecond enqueue/dequeue
- Good enough for our scale (< 10K req/sec)

**Trade-off**: Less features than RabbitMQ, but we don't need them

### 3. Why Async Queue for LLM (Not Sync)?

**Decision**: Queue ambiguous queries for async LLM processing

**Rationale**:
- Don't block client for 2 seconds
- Batch processing more efficient
- Can retry failed LLM calls
- Client can poll or use webhooks
- Better user experience (show "processing...")

**Trade-off**: More complex (status tracking), but worth it

### 4. Why Confidence Thresholds?

**Decision**: Multiple confidence levels trigger different paths

**Rationale**:
- High confidence (> 0.85): Skip expensive steps
- Medium confidence (0.6-0.85): Try embedding
- Low confidence (< 0.6): Use LLM or fallback
- Reduces unnecessary processing
- Optimizes cost vs accuracy

**Tunable**: Can adjust per intent category

### 5. Why Semantic Caching?

**Decision**: Cache LLM responses with similarity matching

**Rationale**:
- Users rephrase similar queries
- "show me shoes" ≈ "display footwear" ≈ "looking for shoes"
- Exact cache misses these variations
- Semantic cache catches them (> 0.95 similarity)
- 30%+ cache hit rate achieved

**Implementation**: Qdrant vector search on query embeddings

### 6. Why Hot-Reload Config?

**Decision**: Configuration changes without restart

**Rationale**:
- Adjust thresholds based on monitoring
- A/B test different weights
- Fix issues without downtime
- Deploy config changes in seconds
- Essential for production tuning

**Implementation**: File watcher + config manager

### 7. Why Entity Extraction in LLM?

**Decision**: Extract entities (brand, price, color) when using LLM

**Rationale**:
- LLM already understanding query
- Can extract structured data in same call
- No additional API cost
- Richer response for downstream services
- One call instead of two

**Alternative**: Separate entity extraction service (rejected - more latency)

### 8. Why A/B Testing Built-In?

**Decision**: Framework for experimentation

**Rationale**:
- Need to test different prompts
- Compare keyword vs embedding weights
- Measure accuracy improvements
- Data-driven optimization
- Essential for ML systems

**Implementation**: Traffic splitting + metrics tracking

---

## Performance Characteristics

### Latency Targets

| Path | Target | Typical | P95 | P99 |
|------|--------|---------|-----|-----|
| Keyword match | < 50ms | 30ms | 45ms | 60ms |
| Hybrid (keyword+embedding) | < 100ms | 80ms | 120ms | 150ms |
| LLM fallback (async) | < 2s | 500ms | 1.5s | 2.5s |
| Cache hit | < 10ms | 5ms | 8ms | 12ms |

### Throughput

- **Keyword matching**: 1000+ req/sec (single instance)
- **Hybrid matching**: 500+ req/sec
- **With caching**: 2000+ req/sec
- **Horizontal scaling**: Linear with instances

### Resource Usage

**Single Instance**:
- CPU: 1-2 cores (spike to 4 during embedding)
- RAM: 2-4 GB (model loading)
- Disk: < 1 GB (code + models)

**Dependencies**:
- Redis: 100-500 MB RAM
- Qdrant: 500MB-2GB RAM (depends on vector count)

---

## Security Considerations

### API Security
- **Authentication**: Bearer token (add if needed)
- **Rate Limiting**: Redis-based (500 req/min per IP)
- **Input Validation**: Pydantic schemas
- **CORS**: Configurable origins

### Data Privacy
- **No PII Storage**: Queries not persisted (except logs)
- **Log Sanitization**: Mask sensitive data
- **Encryption**: TLS for external APIs (OpenAI)
- **API Keys**: Environment variables, never in code

### Infrastructure Security
- **Docker**: Non-root user (appuser)
- **Network**: Services isolated by Docker network
- **Secrets**: Volume mounts for keys
- **Updates**: Regular dependency updates

---

## Monitoring & Observability

### Metrics Tracked

1. **Performance Metrics**
   - Request latency (p50, p95, p99)
   - Throughput (req/sec)
   - Error rate
   - Cache hit rate

2. **Classification Metrics**
   - Intent distribution
   - Confidence score distribution
   - Keyword vs embedding vs LLM usage
   - Ambiguous query rate

3. **Cost Metrics**
   - OpenAI API calls
   - Token usage
   - Daily/monthly spend
   - Cost per intent

4. **Infrastructure Metrics**
   - Redis queue depth
   - Worker processing time
   - Qdrant query latency
   - Health check status

### Logging

**Structured Logging Format**:
```json
{
  "timestamp": "2025-11-01T10:30:00Z",
  "level": "INFO",
  "correlation_id": "uuid-123",
  "message": "Classification completed",
  "query": "add to cart",
  "intent": "ADD_TO_CART",
  "confidence": 0.95,
  "latency_ms": 45,
  "source": "keyword"
}
```

### Alerts

- Cost spike (> $100/day)
- Error rate spike (> 5%)
- Latency degradation (p95 > 200ms)
- Queue backup (> 1000 messages)
- Service down (health check fail)

---

## Scalability

### Horizontal Scaling

```
Load Balancer
     ↓
┌────┴────┬─────────┬─────────┐
│ API-1   │  API-2  │  API-3  │  (Stateless)
└────┬────┴────┬────┴────┬────┘
     │         │         │
     └────┬────┴────┬────┘
          ↓         ↓
     ┌────────┬────────┐
     │ Redis  │ Qdrant │  (Shared state)
     └────────┴────────┘
```

**Why it works**:
- API instances are stateless
- Shared state in Redis/Qdrant
- Docker makes deployment easy
- Linear scaling up to 100s of instances

### Vertical Scaling

**Bottlenecks**:
1. Embedding generation (CPU-bound)
   - Solution: Use GPU or batch processing
2. Qdrant similarity search (RAM-bound)
   - Solution: More RAM or sharding
3. Redis (RAM-bound)
   - Solution: Redis cluster

### Database Scaling

**Qdrant Scaling**:
- Sharding by collection
- Read replicas for search
- Can handle millions of vectors

**Redis Scaling**:
- Redis Cluster for horizontal scaling
- Separate instances for queue/cache
- Can handle 100K+ ops/sec

---

**Last Updated**: November 2025  
**Version**: 1.0.0   
**Maintained By**: ChatNShop Development Team

