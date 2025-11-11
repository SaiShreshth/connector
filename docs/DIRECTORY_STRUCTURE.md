# ChatNShop - Directory Structure Explained

## Table of Contents
1. [Project Root](#project-root)
2. [Application Directory (`app/`)](#application-directory-app)
3. [Tests Directory (`tests/`)](#tests-directory-tests)
4. [Configuration (`config/`)](#configuration-config)
5. [Documentation (`docs/`)](#documentation-docs)
6. [Supporting Files](#supporting-files)

---

## Project Root

```
chatNShop/
├── app/                    # Main application code
├── tests/                  # Test suites
├── config/                 # Configuration files
├── docs/                   # Documentation
├── data/                   # Runtime data (databases, logs)
├── logs/                   # Application logs
├── Results/                # Test/benchmark results
├── tools/                  # Utility scripts
├── venv/                   # Python virtual environment
├── .env                    # Environment variables (not in git)
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── docker-compose.yml      # Docker services definition
├── Dockerfile              # Application container
├── requirements.txt        # Python dependencies
├── pytest.ini              # Pytest configuration
├── README.md               # Project overview
├── SETUP.md                # Setup instructions
└── SYSTEM_FLOW.md          # System flow documentation
```

### Key Root Files

**`.env`** - Environment configuration
```bash
# Database connections
REDIS_HOST=localhost
QDRANT_URL=http://localhost:6333

# OpenAI API
OPENAI_API_KEY=sk-...

# Application settings
PORT=8000
LOG_LEVEL=info
```

**`requirements.txt`** - Python dependencies
- FastAPI (web framework)
- OpenAI (LLM API)
- SentenceTransformers (embeddings)
- Qdrant-client (vector database)
- Redis (cache/queue)
- See [Technology Stack](PROJECT_ARCHITECTURE.md#technology-stack)

**`docker-compose.yml`** - Services orchestration
- Qdrant (vector database)
- Redis (cache/queue)
- intent-api (main application)

**`Dockerfile`** - Application container
- Python 3.9-slim base
- Non-root user for security
- Health check endpoint
- See [Deployment](DEPLOYMENT.md)

---

## Application Directory (`app/`)

### Overview
```
app/
├── main.py                         # FastAPI application entry point
├── api/                            # API endpoints
├── ai/                             # AI/ML components
├── core/                           # Core utilities
├── queue/                          # Queue infrastructure
├── schemas/                        # Data models
├── templates/                      # HTML templates
└── utils/                          # Helper functions
```

---

### `app/main.py` - Application Entry Point

**Purpose**: FastAPI application initialization and configuration

**What It Does**:
- Creates FastAPI app instance
- Configures middleware (CORS, logging)
- Registers API routers
- Implements health check endpoints
- Defines classification endpoint `/classify`
- Handles application lifecycle (startup/shutdown)

**Key Sections**:
```python
# 1. Imports and setup
from fastapi import FastAPI
from app.ai.intent_classification.decision_engine import get_intent_classification

# 2. Configuration loading
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")

# 3. FastAPI app creation
app = FastAPI(title="Intent Classification API")

# 4. Endpoints
@app.post("/classify")
async def classify_intent(user_input: ClassificationInput):
    result = get_intent_classification(user_input.text)
    return result

# 5. Server startup
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Used By**: Docker, Uvicorn server
**Dependencies**: All other modules

---

### `app/api/` - API Endpoints

#### `api/v1/intent.py` - Intent Classification Routes

**Purpose**: Additional intent-related endpoints

**Endpoints**:
- Debug endpoints (if enabled)
- Batch classification
- Intent statistics

**Example**:
```python
@router.get("/intents")
async def list_intents():
    """List all available intents"""
    return {"intents": list(ActionCode.__dict__.keys())}
```

#### `api/v1/queue.py` - Queue Status Routes

**Purpose**: Check status of async requests

**Endpoints**:
```python
@router.get("/queue/status/{request_id}")
async def get_status(request_id: str):
    status = status_store.get(request_id)
    return {"request_id": request_id, "status": status}
```

#### `api/v1/cache.py` - Cache Management

**Purpose**: Cache statistics and management

**Endpoints**:
- GET `/cache/stats` - Cache hit rate, size
- DELETE `/cache/clear` - Clear cache
- POST `/cache/warm` - Pre-warm cache

#### `api/cost_dashboard_api.py` - Cost Monitoring API

**Purpose**: API usage and cost tracking

**Endpoints**:
```python
@router.get("/cost/daily")
async def get_daily_cost():
    return {"date": today, "cost": $12.34, "calls": 1234}

@router.get("/cost/breakdown")
async def get_cost_breakdown():
    return {"keyword": 0, "embedding": 0, "llm": $12.34}
```

#### `api/cost_dashboard_ui.py` - Cost Dashboard UI

**Purpose**: Serve HTML dashboard for cost visualization

**Route**: `/cost/dashboard`
**Template**: `templates/cost_dashboard.html`

#### `api/ab_testing_api.py` - A/B Testing API

**Purpose**: Manage experiments

**Endpoints**:
- POST `/experiments/create` - Create new experiment
- GET `/experiments/{id}/results` - Get results
- POST `/experiments/{id}/stop` - Stop experiment

#### `api/testing_framework_api.py` - Testing Framework API

**Purpose**: Run tests via API

**Endpoints**:
- POST `/test/accuracy` - Test on dataset
- POST `/test/performance` - Benchmark
- GET `/test/report` - Get test report

---

### `app/ai/` - AI/ML Components

```
ai/
├── config.py                       # AI configuration constants
├── intent_classification/          # Rule-based classification
├── llm_intent/                     # LLM-based classification
├── entity_extraction/              # Entity extraction
├── cost_monitor/                   # Cost tracking
├── vector_search/                  # Vector search utilities
├── protos/                         # Protobuf definitions
└── queue_producer.py               # Queue message publisher (legacy)
```

---

### `app/ai/config.py` - AI Configuration

**Purpose**: Central configuration for AI components

**Contents**:
```python
# Confidence thresholds
PRIORITY_THRESHOLD = 0.85   # Skip embedding if keyword > this
CONFIDENCE_THRESHOLD = 0.6   # Minimum to return result

# Hybrid weights
WEIGHTS = {
    "keyword": 0.6,
    "embedding": 0.4
}

# LLM settings
LLM_ENABLED = True
LLM_MODEL = "gpt-4-turbo"
LLM_TEMPERATURE = 0.3
```

---

### `app/ai/intent_classification/` - Rule-Based System

```
intent_classification/
├── __init__.py
├── decision_engine.py              # Main orchestrator
├── keyword_matcher.py              # Keyword matching
├── embedding_matcher.py            # Embedding similarity
├── hybrid_classifier.py            # Blend results
├── confidence_threshold.py         # Confidence evaluation
├── ambiguity_resolver.py           # Handle ambiguous queries
├── scoring.py                      # Scoring algorithms
├── intents.py                      # Intent definitions (legacy)
├── entity_extraction.py            # Entity extraction (legacy)
├── intents_modular/                # Modular intent taxonomy
│   ├── enums.py                   # IntentCategory, ActionCode
│   ├── models.py                  # Pydantic models
│   ├── taxonomy.py                # Taxonomy utilities
│   └── definitions/               # Intent definitions by category
│       ├── SEARCH_DISCOVERY.py
│       ├── CART_WISHLIST.py
│       ├── PRODUCT_DETAILS.py
│       └── ...                    # 14 category files
├── keywords/                       # Keyword dictionaries
│   ├── loader.py                  # Load keywords from JSON
│   ├── search_keywords.json
│   ├── cart_keywords.json
│   └── ...                        # 15 keyword files
└── cache/
    ├── intent_embeddings.json      # Pre-computed embeddings
    └── status_store.py            # Status tracking (deprecated)
```

#### Key Files Explained

**`decision_engine.py`** - Main Orchestrator (447 lines)

**What It Does**:
1. Receives user query
2. Normalizes text
3. Runs keyword matching
4. Checks if high confidence → Return (fast path)
5. Runs embedding matching
6. Blends results
7. Checks confidence
8. If low → Queue for LLM
9. Returns classification result

**Key Functions**:
```python
class DecisionEngine:
    def __init__(self):
        self.keyword_matcher = KeywordMatcher()
        self.embedding_matcher = EmbeddingMatcher()
        self.hybrid_classifier = HybridClassifier()
    
    def search(self, query: str) -> Dict:
        # Priority rule: keyword first
        kw_results = self.keyword_matcher.search(query)
        if kw_results[0]["score"] >= 0.85:
            return kw_results[0]  # Short-circuit
        
        # Embedding match
        emb_results = self.embedding_matcher.search(query)
        
        # Blend
        blended = self.hybrid_classifier.blend(kw_results, emb_results)
        
        # Check confidence
        if confident(blended):
            return blended[0]
        
        # LLM fallback
        return trigger_llm(query)
```

**Used By**: `main.py` → `get_intent_classification(query)`
**Dependencies**: keyword_matcher, embedding_matcher, hybrid_classifier

---

**`keyword_matcher.py`** - Keyword Matching Engine (300+ lines)

**What It Does**:
- Loads keywords from JSON files
- Normalizes queries (lowercase, remove punctuation)
- Exact matching (fast)
- Partial matching (with stemming)
- Calculates confidence scores

**Key Algorithm**:
```python
def search(self, query: str) -> List[Dict]:
    query = self.normalize(query)
    
    # Try exact match
    for intent, keywords in self.keyword_dict.items():
        for keyword in keywords:
            if keyword in query:
                return [{
                    "id": intent,
                    "score": 0.95,
                    "source": "keyword",
                    "matched_text": keyword
                }]
    
    # Try partial match
    # ... (stem words and check)
    
    return []
```

**Performance**: 30-50ms
**Accuracy**: 78% (on labeled dataset)

---

**`embedding_matcher.py`** - Semantic Similarity (200+ lines)

**What It Does**:
- Lazy-loads SentenceTransformer model
- Encodes query to 384-dim vector
- Computes cosine similarity with reference embeddings
- Returns top matches

**Key Algorithm**:
```python
def search(self, query: str) -> List[Dict]:
    # Encode query
    query_embedding = self.model.encode(query)
    
    # Compute similarities
    similarities = []
    for intent, ref_embedding in self.ref_embeddings.items():
        sim = cosine_similarity(query_embedding, ref_embedding)
        if sim > 0.7:  # Threshold
            similarities.append({
                "id": intent,
                "score": sim,
                "source": "embedding"
            })
    
    return sorted(similarities, key=lambda x: x["score"], reverse=True)
```

**Model**: sentence-transformers/all-MiniLM-L6-v2
**Performance**: 50-80ms
**Accuracy**: 82%

---

**`hybrid_classifier.py`** - Weighted Blending (150 lines)

**What It Does**:
- Combines keyword and embedding results
- Applies weighted scores (0.6 keyword + 0.4 embedding)
- Resolves conflicts
- Returns sorted list

**Key Algorithm**:
```python
def blend(self, kw_results, emb_results):
    intent_scores = {}
    
    # Merge scores
    for r in kw_results:
        intent_scores[r["id"]] = {"kw": r["score"], "emb": 0}
    
    for r in emb_results:
        if r["id"] in intent_scores:
            intent_scores[r["id"]]["emb"] = r["score"]
        else:
            intent_scores[r["id"]] = {"kw": 0, "emb": r["score"]}
    
    # Calculate blended scores
    blended = []
    for intent, scores in intent_scores.items():
        blended_score = (
            scores["kw"] * self.kw_weight +
            scores["emb"] * self.emb_weight
        )
        blended.append({
            "id": intent,
            "score": blended_score,
            "source": "blended"
        })
    
    return sorted(blended, key=lambda x: x["score"], reverse=True)
```

**Performance**: < 1ms
**Accuracy**: 87% (best of both)

---

**`intents_modular/enums.py`** - Intent Definitions (500+ lines)

**What It Does**:
- Defines all 150+ action codes
- Organizes into 14 categories
- Provides intent metadata

**Structure**:
```python
class IntentCategory(Enum):
    SEARCH_DISCOVERY = "SEARCH_DISCOVERY"
    CART_WISHLIST = "CART_WISHLIST"
    # ... 14 categories

class ActionCode:
    # Search & Discovery (25 codes)
    SEARCH_PRODUCT = "SEARCH_PRODUCT"
    SEARCH_CATEGORY = "SEARCH_CATEGORY"
    # ...
    
    # Cart & Wishlist (20 codes)
    ADD_TO_CART = "ADD_TO_CART"
    REMOVE_FROM_CART = "REMOVE_FROM_CART"
    # ...
    
    # ... 150+ total codes
```

---

**`keywords/search_keywords.json`** - Search Intent Keywords

**Structure**:
```json
{
  "SEARCH_PRODUCT": {
    "priority": 1,
    "keywords": [
      "search product", "find product", "look for product",
      "show me", "display", "browse", "shop for",
      "looking for", "need", "want to buy"
      // ... 30+ keywords
    ]
  },
  "SEARCH_CATEGORY": {
    "priority": 2,
    "keywords": [
      "search category", "find category", "browse category"
      // ... 30+ keywords
    ]
  }
  // ... 25 search intents
}
```

**Total**: 15 keyword files, 4500+ keywords across all intents

---

### `app/ai/llm_intent/` - LLM Classification System

```
llm_intent/
├── request_handler.py              # Main LLM orchestrator
├── openai_client.py                # OpenAI API client
├── resilient_openai_client.py      # Retry/fallback wrapper
├── prompt_loader.py                # Load versioned prompts
├── response_parser.py              # Parse LLM JSON response
├── entity_extractor.py             # Extract entities from response
├── context_collector.py            # Collect conversation context
├── context_summarizer.py           # Summarize long context
├── confidence_calibration.py       # Calibrate confidence scores
├── confidence_calibrator.py        # Confidence adjustment logic
├── response_cache.py               # Cache LLM responses
├── qdrant_cache.py                 # Semantic cache in Qdrant
├── cache_metrics.py                # Cache performance metrics
├── query_normalizer.py             # Normalize queries
├── error_handler.py                # Error handling logic
├── fallback_manager.py             # Fallback strategies
└── prompts/
    ├── system_prompt_v1.0.0.txt    # System prompt v1.0
    ├── system_prompt_v1.1.0.txt    # System prompt v1.1
    ├── few_shot_examples_v1.0.0.json  # Example queries
    ├── changelog.md                # Prompt version history
    └── README.md                   # Prompt documentation
```

#### Key Files Explained

**`request_handler.py`** - LLM Orchestrator (571 lines)

**What It Does**:
1. Receives query + context
2. Checks cache first
3. Loads prompt template
4. Calls OpenAI API
5. Parses response
6. Extracts entities
7. Caches result
8. Returns classification

**Key Function**:
```python
class RequestHandler:
    def classify(self, query: str, context: Dict = None):
        # 1. Check cache
        cached = self.cache.get(query)
        if cached:
            return cached
        
        # 2. Load prompt
        prompt = self.prompt_loader.load("v1.1.0")
        
        # 3. Add context
        if context:
            prompt = self.add_context(prompt, context)
        
        # 4. Call LLM
        response = self.openai_client.classify_intent(query, prompt)
        
        # 5. Parse response
        result = self.response_parser.parse(response)
        
        # 6. Extract entities
        result["entities"] = self.entity_extractor.extract(result)
        
        # 7. Cache
        self.cache.set(query, result)
        
        return result
```

**Performance**: 200ms-2s
**Cost**: ~$0.008 per query
**Accuracy**: 92%

---

**`openai_client.py`** - OpenAI API Integration (250+ lines)

**What It Does**:
- Manages OpenAI API connection
- Implements retry logic (exponential backoff)
- Handles rate limiting
- Tracks token usage and cost

**Key Features**:
```python
class OpenAIClient:
    @retry(stop_after_attempt(3), wait=wait_exponential())
    def classify_intent(self, query, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        
        # Track usage
        self.track_usage(response.usage)
        
        return response.choices[0].message.content
```

**Retry Strategy**:
- Attempt 1 fails → Wait 2s, retry
- Attempt 2 fails → Wait 4s, retry
- Attempt 3 fails → Give up, use fallback

---

**`response_cache.py`** - LLM Response Caching (200+ lines)

**What It Does**:
- Caches LLM responses in Redis
- Semantic similarity cache (Qdrant)
- TTL-based expiration (24 hours)
- Tracks cache hit rate

**Caching Strategy**:
```python
def get(self, query: str):
    # 1. Exact match cache (Redis)
    cached = self.redis.get(f"llm:{query}")
    if cached:
        return json.loads(cached)
    
    # 2. Semantic similarity cache (Qdrant)
    similar = self.qdrant_cache.find_similar(query, threshold=0.95)
    if similar:
        return similar["result"]
    
    return None

def set(self, query: str, result: Dict):
    # Store in both caches
    self.redis.setex(f"llm:{query}", 86400, json.dumps(result))
    self.qdrant_cache.store(query, result)
```

**Performance**:
- Exact cache hit: 5ms
- Semantic cache hit: 20ms
- Cache miss → API call: 500ms+

**Cache Hit Rate**: 30-40% (saves ~$300/month)

---

### `app/ai/entity_extraction/` - Entity Extraction

```
entity_extraction/
├── extractor.py                    # Main extractor
├── normalizer.py                   # Normalize extracted entities
├── validator.py                    # Validate entities
├── resources.py                    # Entity resources (brands, etc.)
└── schema.py                       # Entity schemas
```

**`extractor.py`** - Entity Extractor (300+ lines)

**What It Does**:
- Extracts entities from queries
- Identifies: products, brands, colors, sizes, prices
- Uses regex patterns and dictionaries

**Example**:
```python
query = "red Nike running shoes under $100"

entities = extractor.extract_entities(query)

# Result:
{
  "product_type": "shoes",
  "category": "running",
  "brand": "Nike",
  "color": "red",
  "price_range": {"max": 100, "currency": "USD"}
}
```

---

### `app/ai/cost_monitor/` - Cost Tracking

```
cost_monitor/
├── usage_tracker.py                # Track API usage
├── openai_cost_wrapper.py          # Wrap OpenAI calls with tracking
├── cost_alerts.py                  # Alert on cost spikes
├── cost_spike_detector.py          # Detect unusual spending
├── alert_manager.py                # Send alerts
├── rate_limiter.py                 # Rate limiting
├── scheduler.py                    # Scheduled jobs
└── ab_testing.py                   # Cost-based A/B testing
```

**Purpose**: Monitor and optimize API costs

**Features**:
- Track every API call (tokens + cost)
- Daily/monthly cost reports
- Alert on spikes (> $100/day)
- Rate limiting (500 calls/min)
- Cost breakdown by intent

---

### `app/queue/` - Queue Infrastructure

```
queue/
├── config.py                       # Queue configuration
├── queue_manager.py                # Redis queue manager
├── queue_producer.py               # Publish messages
├── worker.py                       # Process queued requests
├── monitor.py                      # Queue monitoring
├── integration.py                  # Integration helpers
└── README.md                       # Queue documentation
```

**Purpose**: Async processing of ambiguous queries

**Flow**:
1. Rule-based classifies query
2. Low confidence → Publish to queue
3. Worker picks up message
4. Calls LLM
5. Updates status store
6. Client polls for result

---

### `app/core/` - Core Utilities

```
core/
├── config_manager.py               # Configuration hot-reload
├── status_store.py                 # Request status tracking
├── redis_client.py                 # Redis connection pool
├── session_store.py                # User session management
├── resilient_openai_client.py      # Resilient API client
├── alert_notifier.py               # Alert notifications
├── ab_testing_integration.py       # A/B testing integration
└── ab_testing/                     # A/B testing framework
    ├── ab_framework.py             # Experiment management
    ├── bandit.py                   # Multi-armed bandit
    ├── analysis.py                 # Results analysis
    ├── config_cli.py               # CLI for experiments
    ├── rollback_manager.py         # Rollback experiments
    └── experiment_config.json      # Experiment definitions
```

**Purpose**: Shared utilities across the application

---

### `app/schemas/` - Data Models

```
schemas/
├── llm_intent.py                   # LLM request/response models
└── request_status.py               # Status tracking models
```

**`llm_intent.py`**:
```python
class LLMIntentRequest(BaseModel):
    query: str
    context: Optional[Dict] = None
    session_id: Optional[str] = None

class LLMIntentResponse(BaseModel):
    action_code: str
    confidence: float
    entities: Dict[str, Any]
    reasoning: str
```

---

## Tests Directory (`tests/`)

```
tests/
├── entity_extraction_tests/        # Entity extraction tests
│   ├── unit/                       # Unit tests (2 files)
│   └── integration/                # Integration tests (1 file)
│
├── intent_classification_tests/    # Rule-based system tests
│   ├── unit/                       # Unit tests (7 files)
│   ├── integration/                # Integration tests (3 files)
│   ├── perf/                       # Performance tests (3 files)
│   ├── api/                        # API tests (1 file)
│   ├── config/                     # Config tests (1 file)
│   ├── status_store/               # Status store tests (1 file)
│   └── data/                       # Test data
│
├── llm_intent_tests/               # LLM system tests
│   ├── unit/                       # Unit tests (10 files)
│   ├── integration/                # Integration tests (4 files)
│   ├── perf/                       # Performance tests (1 file)
│   ├── api/                        # API tests (1 file)
│   ├── prompts/                    # Prompt tests (2 files)
│   ├── tools/                      # Test utilities (2 files)
│   └── README.md                   # Testing guide
│
└── integration/                    # Cross-system integration
    └── test_tasks_15_16_17_integration.py  # Queue integration
```

**Total**: 40+ test files, 800+ test cases

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for details.

---

## Configuration (`config/`)

```
config/
├── rules.json                      # Main configuration
└── versions/                       # Config version history
    ├── rules_v1.0.0.json
    ├── rules_v1.1.0.json
    └── rules_v1.2.0.json
```

**`rules.json`** - Main Configuration:
```json
{
  "version": "1.2.0",
  "active_variant": "A",
  "rules": {
    "rule_sets": {
      "A": {
        "kw_weight": 0.6,
        "emb_weight": 0.4,
        "confidence_threshold": 0.6
      }
    }
  }
}
```

**Hot-Reload**: Changes take effect immediately (no restart)

---

## Documentation (`docs/`)

```
docs/
├── PROJECT_ARCHITECTURE.md         # This architecture overview
├── DIRECTORY_STRUCTURE.md          # This file
├── DATA_FLOW.md                    # Request flows
├── CONFIGURATION_GUIDE.md          # Configuration reference
├── API_REFERENCE.md                # API documentation
├── TESTING_GUIDE.md                # Testing guide
├── DEPLOYMENT.md                   # Deployment guide
├── intent_classification/          # Rule-based docs
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── KEYWORD_MAINTENANCE.md
│   ├── TROUBLESHOOTING.md
│   └── ...                         # 13 doc files
├── llm_intent/
│   └── CACHING.md                  # LLM caching guide
├── cost_monitoring/
│   ├── COST_MONITORING.md
│   └── PROMPT_OPTIMIZATION.md
└── ...
```

---

## Supporting Files

### `data/` - Runtime Data (Ignored by Git)

```
data/
├── ab_testing.db                   # A/B test results (SQLite)
├── cost_usage.db                   # Cost tracking (SQLite)
└── usage_log.json                  # Usage logs
```

### `logs/` - Application Logs (Ignored by Git)

```
logs/
└── ambiguous.jsonl                 # Ambiguous queries log
```

### `Results/` - Test Results (Ignored by Git)

```
Results/
├── prompt_test_results.csv         # Prompt testing results
└── prompt_test_results.json        # Prompt testing results
```

### `tools/` - Utility Scripts

```
tools/
└── ambiguity_report.py             # Generate ambiguity reports
```

**Usage**:
```bash
python tools/ambiguity_report.py
# Analyzes logs/ambiguous.jsonl
# Generates report of common ambiguous queries
```

---

## File Interaction Diagram

```
                      User Request
                           │
                           ▼
                      main.py (FastAPI)
                           │
                           ▼
                   decision_engine.py
                    ┌──────┴──────┐
                    ▼              ▼
          keyword_matcher    embedding_matcher
                    │              │
                    └──────┬───────┘
                           ▼
                  hybrid_classifier.py
                           │
                    ┌──────┴──────┐
                    ▼              ▼
              Confident?       Not Confident
                    │              │
                    ▼              ▼
               Return         queue_producer.py
                                   │
                                   ▼
                              Redis Queue
                                   │
                                   ▼
                              worker.py
                                   │
                                   ▼
                          request_handler.py
                                   │
                                   ▼
                          openai_client.py
                                   │
                                   ▼
                              OpenAI API
                                   │
                                   ▼
                          response_parser.py
                                   │
                                   ▼
                           status_store.py
                                   │
                                   ▼
                          Client polls for result
```

---

## Module Dependencies

### High-Level Dependencies

```
main.py
  ├─> api/v1/intent.py
  ├─> api/v1/queue.py
  ├─> ai/intent_classification/decision_engine.py
  │     ├─> keyword_matcher.py
  │     ├─> embedding_matcher.py
  │     ├─> hybrid_classifier.py
  │     └─> confidence_threshold.py
  ├─> ai/llm_intent/request_handler.py
  │     ├─> openai_client.py
  │     ├─> prompt_loader.py
  │     ├─> response_parser.py
  │     └─> response_cache.py
  ├─> queue/queue_manager.py
  ├─> core/config_manager.py
  └─> schemas/llm_intent.py
```

### External Dependencies

```
FastAPI → Web framework
Uvicorn → ASGI server
Pydantic → Data validation
OpenAI → LLM API
SentenceTransformers → Embeddings
Qdrant → Vector database
Redis → Cache/Queue
```

---

