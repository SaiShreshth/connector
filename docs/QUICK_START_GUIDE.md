# ChatNShop - Quick Start Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the System](#running-the-system)
4. [Testing the System](#testing-the-system)
5. [Understanding the Flow](#understanding-the-flow)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Python 3.11.9** - Programming language (3.11.9 recommended for stable ML library support across all platforms)
- **Docker & Docker Compose** - Container orchestration
- **Git** - Version control

### Optional (for development)
- **Postman** or **curl** - API testing
- **Redis Desktop Manager** - View Redis data
- **Qdrant Dashboard** - http://localhost:6333/dashboard

### API Keys
- **OpenAI API Key** - Get from https://platform.openai.com/

---

## Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd chatNShop
```

### Step 2: Set Up Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your values
nano .env
```

**Required Variables** in `.env`:
```bash
# OpenAI API (Required for LLM fallback)
OPENAI_API_KEY=sk-your-key-here

# Redis (Default works for local)
REDIS_HOST=localhost
REDIS_PORT=6379

# Qdrant (Default works for local)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_URL=http://localhost:6333

# Application
PORT=8000
LOG_LEVEL=info
```

### Step 3: Start Dependencies with Docker
```bash
# Start Qdrant and Redis
docker-compose up -d qdrant redis

# Verify services are running
docker-compose ps
```

You should see:
```
NAME                 STATUS
chatnshop-qdrant     Up (healthy)
chatns-redis         Up (healthy)
```

### Step 4: Create Python Virtual Environment

**Note**: Python 3.11.9 is recommended for stable ML library compatibility. If you don't have it installed:
```bash
# Install Python 3.11 via Homebrew (Mac)
brew install python@3.11

# Or download from: https://www.python.org/downloads/
```

```bash
# Create venv with Python 3.11
python3.11 -m venv venv

# Activate venv
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate     # On Windows

# Verify Python version
python --version  # Should show Python 3.11.9

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

This will take 5-10 minutes to install all packages.

---

## Running the System

### Option 1: Run Locally (Development)

**Start the application**:
```bash
# Make sure venv is activated
source venv/bin/activate

# Run with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
🚀 Starting Intent Classification API...
✅ Connected to Qdrant at http://localhost:6333
✅ Queue infrastructure ready (Redis connected)
✅ Models loaded and Decision Engine is warm.
✅ Intent Classification API started successfully!
```

**Access the API**:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (Interactive API documentation)
- **Health**: http://localhost:8000/health

### Option 2: Run with Docker (Production-like)

**Build and start all services**:
```bash
docker-compose up --build
```

This starts:
- Qdrant (vector database)
- Redis (cache/queue)
- intent-api (FastAPI application)

**Access the API**:
- Same URLs as Option 1

---

## Testing the System

### 1. Health Check

**Using curl**:
```bash
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-01T10:30:00Z",
  "services": {
    "qdrant": "connected",
    "redis": "connected",
    "openai": "configured"
  },
  "queue": {
    "ambiguous_queue_size": 0,
    "result_queue_size": 0
  },
  "version": "1.0.0"
}
```

### 2. Test Intent Classification

**Test Case 1: Simple Intent (Keyword Match)**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "add to cart"}'
```

**Expected Response** (~30-50ms):
```json
{
  "action_code": "ADD_TO_CART",
  "confidence_score": 0.95,
  "matched_keywords": ["add to cart"],
  "original_text": "add to cart",
  "status": "CONFIDENT_KEYWORD",
  "intent": {
    "id": "ADD_TO_CART",
    "score": 0.95,
    "source": "keyword",
    "matched_text": "add to cart"
  },
  "entities": null
}
```

**Why Fast?** Keyword match - direct dictionary lookup.

---

**Test Case 2: Semantic Intent (Embedding Match)**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "show me running kicks"}'
```

**Expected Response** (~80-120ms):
```json
{
  "action_code": "SEARCH_PRODUCT",
  "confidence_score": 0.82,
  "matched_keywords": [],
  "original_text": "show me running kicks",
  "status": "BLENDED_TOP_CONFIDENT",
  "intent": {
    "id": "SEARCH_PRODUCT",
    "score": 0.82,
    "source": "blended",
    "keyword_score": 0.45,
    "embedding_score": 0.88
  },
  "entities": null
}
```

**Why Slower?** Embedding generation + similarity calculation.
**Why It Works?** "kicks" (slang) isn't in keywords, but embedding understands semantics.

---

**Test Case 3: Complex Query (May Trigger LLM)**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I need something for my trip"}'
```

**Possible Response 1** (Rule-based low confidence):
```json
{
  "action_code": "SEARCH_PRODUCT",
  "confidence_score": 0.45,
  "matched_keywords": [],
  "original_text": "I need something for my trip",
  "status": "FALLBACK_GENERIC",
  "intent": {...}
}
```

**Possible Response 2** (LLM queued):
```json
{
  "request_id": "a1b2c3d4-...",
  "status": "QUEUED",
  "message": "Processing ambiguous query asynchronously"
}
```

Then poll for result:
```bash
curl http://localhost:8000/status/a1b2c3d4-...
```

---

### 3. Interactive API Testing (Swagger UI)

**Open in browser**: http://localhost:8000/docs

You'll see:
- All available endpoints
- Request/response schemas
- "Try it out" buttons for testing

**Steps**:
1. Click on `POST /classify`
2. Click "Try it out"
3. Edit the request body:
   ```json
   {
     "text": "your query here"
   }
   ```
4. Click "Execute"
5. See response below

---

### 4. Run Automated Tests

**Run all tests**:
```bash
pytest tests/ -v
```

**Run specific test suites**:
```bash
# Intent classification tests
pytest tests/intent_classification_tests/ -v

# LLM intent tests
pytest tests/llm_intent_tests/ -v

# Entity extraction tests
pytest tests/entity_extraction_tests/ -v

# Performance tests
pytest tests/intent_classification_tests/perf/ -v
```

**Expected Output**:
```
==================== test session starts ====================
tests/intent_classification_tests/unit/test_keyword_matcher_unit.py::test_exact_match PASSED
tests/intent_classification_tests/unit/test_keyword_matcher_unit.py::test_partial_match PASSED
...
==================== 156 passed in 45.23s ====================
```

---

## Understanding the Flow

### Flow 1: Fast Path (Keyword Match)

```
User Query: "add to cart"
    ↓
FastAPI receives request
    ↓
DecisionEngine.search()
    ↓
KeywordMatcher.search()
    ↓
Found: "add to cart" → ADD_TO_CART (score: 0.95)
    ↓
High confidence (≥ 0.85) → Skip embedding
    ↓
Return: ADD_TO_CART (30-50ms)
```

**When Used**: 85% of queries (common intents)
**Speed**: 30-50ms
**Cost**: $0 (no API calls)

---

### Flow 2: Hybrid Path (Keyword + Embedding)

```
User Query: "show me running kicks"
    ↓
KeywordMatcher.search()
    ↓
Low confidence: "kicks" not in dictionary (score: 0.45)
    ↓
EmbeddingMatcher.search()
    ↓
Semantic similarity: "kicks" ≈ "shoes" (score: 0.88)
    ↓
HybridClassifier.blend()
    ↓
Blended score: 0.45 * 0.6 + 0.88 * 0.4 = 0.622
    ↓
Confidence check: 0.622 > 0.6 → Accept
    ↓
Return: SEARCH_PRODUCT (80-120ms)
```

**When Used**: 14% of queries (paraphrases, slang)
**Speed**: 80-120ms
**Cost**: $0 (local embeddings)

---

### Flow 3: LLM Fallback (Async)

```
User Query: "I need something"
    ↓
Rule-based classification
    ↓
Low confidence: Too vague (score: 0.35)
    ↓
QueueProducer.publish()
    ↓
Return request_id immediately
    ↓
Worker picks up from queue
    ↓
RequestHandler.classify()
    ↓
OpenAI GPT-4 API call
    ↓
Parse response + extract entities
    ↓
Update StatusStore
    ↓
Client polls for result
```

**When Used**: <1% of queries (truly ambiguous)
**Speed**: 200ms-2s (async, doesn't block)
**Cost**: ~$0.008 per query

---

## Configuration

### Key Configuration Files

**1. `.env`** - Environment variables
- API keys
- Service URLs
- Port settings

**2. `config/rules.json`** - Classification rules
```json
{
  "active_variant": "A",
  "rules": {
    "rule_sets": {
      "A": {
        "kw_weight": 0.6,
        "emb_weight": 0.4,
        "confidence_threshold": 0.6,
        "priority_threshold": 0.85
      }
    }
  }
}
```

**3. `app/ai/config.py`** - AI configuration
```python
PRIORITY_THRESHOLD = 0.85
CONFIDENCE_THRESHOLD = 0.6
WEIGHTS = {"keyword": 0.6, "embedding": 0.4}
```

### Tuning Performance

**To make it faster** (trade accuracy):
```json
{
  "priority_threshold": 0.80,  # Lower (more short-circuits)
  "kw_weight": 0.7,            # Higher (prefer fast keywords)
  "emb_weight": 0.3            # Lower
}
```

**To make it more accurate** (trade speed):
```json
{
  "priority_threshold": 0.90,  # Higher (fewer short-circuits)
  "kw_weight": 0.5,            # Equal weight
  "emb_weight": 0.5,
  "confidence_threshold": 0.7  # Higher (more LLM fallbacks)
}
```

---

## Troubleshooting

### Issue 1: "Could not connect to Qdrant"

**Symptoms**:
```
❌ FAILED to initialize Qdrant client after 5 attempts
```

**Solutions**:
```bash
# Check if Qdrant is running
docker-compose ps qdrant

# Restart Qdrant
docker-compose restart qdrant

# Check logs
docker-compose logs qdrant

# Verify connection
curl http://localhost:6333/collections
```

---

### Issue 2: "Queue infrastructure not available"

**Symptoms**:
```
⚠️ Queue infrastructure not available (continuing without async processing)
```

**Solutions**:
```bash
# Check if Redis is running
docker-compose ps redis

# Test Redis connection
redis-cli -h localhost -p 6379 ping
# Should return: PONG

# Restart Redis
docker-compose restart redis
```

---

### Issue 3: "OpenAI API Error"

**Symptoms**:
```
ERROR: OpenAI API authentication failed
```

**Solutions**:
```bash
# 1. Check API key in .env
cat .env | grep OPENAI_API_KEY

# 2. Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# 3. Regenerate key at https://platform.openai.com/api-keys
```

---

### Issue 4: Slow Response Times

**Symptoms**:
- Responses taking > 500ms
- Embedding matches very slow

**Solutions**:
```bash
# 1. Check if model is loaded
# (First request loads model, takes ~5s)

# 2. Increase priority threshold (more short-circuits)
# Edit config/rules.json:
"priority_threshold": 0.80

# 3. Reduce embedding calculations
# Edit config/rules.json:
"use_embedding": false  # Temporarily disable

# 4. Check system resources
docker stats  # Look for high CPU/memory
```

---

### Issue 5: Classification Accuracy Issues

**Symptoms**:
- Wrong intents returned
- Low confidence scores

**Solutions**:
```bash
# 1. Check keyword coverage
python -c "from app.ai.intent_classification.keyword_matcher import KeywordMatcher; m = KeywordMatcher(); print(f'Loaded {len(m.keyword_dict)} intents')"

# 2. Add missing keywords
# Edit: app/ai/intent_classification/keywords/*.json

# 3. Test specific query
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "your failing query"}' \
  | jq .

# 4. Check logs
tail -f logs/ambiguous.jsonl  # See ambiguous queries
```

---

## Frontend Integration Examples

### React Component

```jsx
import React, { useState } from 'react';

function IntentClassifier() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const classifyIntent = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Classification error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input 
        value={text} 
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter your query..."
      />
      <button onClick={classifyIntent} disabled={loading}>
        {loading ? 'Classifying...' : 'Classify'}
      </button>
      {result && (
        <div>
          <h3>Intent: {result.action_code}</h3>
          <p>Confidence: {result.confidence_score}</p>
          <p>Status: {result.status}</p>
        </div>
      )}
    </div>
  );
}

export default IntentClassifier;
```

### Vue.js Component

```vue
<template>
  <div class="intent-classifier">
    <input 
      v-model="text" 
      placeholder="Enter your query..." 
      @keyup.enter="classifyIntent"
    />
    <button @click="classifyIntent" :disabled="loading">
      {{ loading ? 'Classifying...' : 'Classify' }}
    </button>
    <div v-if="result" class="result">
      <h3>Intent: {{ result.action_code }}</h3>
      <p>Confidence: {{ result.confidence_score }}</p>
      <p>Status: {{ result.status }}</p>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      text: '',
      result: null,
      loading: false
    };
  },
  methods: {
    async classifyIntent() {
      this.loading = true;
      try {
        const response = await fetch('http://localhost:8000/classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: this.text })
        });
        this.result = await response.json();
      } catch (error) {
        console.error('Classification error:', error);
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>
```

### Python Client

```python
import requests

def classify_intent(text):
    """Classify intent using the API"""
    try:
        response = requests.post(
            'http://localhost:8000/classify',
            json={'text': text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

# Usage
result = classify_intent('search for laptop')
if result:
    print(f"Intent: {result['action_code']}")
    print(f"Confidence: {result['confidence_score']}")
```

---

## Production Deployment

### Production Checklist

**Before deploying to production**:

- [ ] **Environment Variables**
  - [ ] Set `OPENAI_API_KEY` with valid key
  - [ ] Configure `ALLOWED_ORIGINS` for your domain
  - [ ] Set `LOG_LEVEL=warning` (reduce log noise)
  - [ ] Disable `RELOAD=false` (no auto-reload in prod)

- [ ] **Services**
  - [ ] Qdrant is running and accessible
  - [ ] Redis is running and accessible
  - [ ] All health checks passing

- [ ] **Testing**
  - [ ] All API endpoints tested
  - [ ] Error handling verified
  - [ ] Load testing completed
  - [ ] Edge cases tested

- [ ] **Security**
  - [ ] CORS configured for your domain only
  - [ ] API rate limiting enabled
  - [ ] Sensitive data not logged
  - [ ] HTTPS enabled

- [ ] **Monitoring**
  - [ ] Logging configured
  - [ ] Metrics collection enabled
  - [ ] Alerts set up
  - [ ] Cost monitoring active

### Production Environment Setup

```bash
# .env for production
OPENAI_API_KEY=sk-your-production-key
HOST=0.0.0.0
PORT=8000
RELOAD=false
LOG_LEVEL=warning
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
WORKERS=4

# Redis
REDIS_HOST=your-redis-host
REDIS_PORT=6379

# Qdrant
QDRANT_HOST=your-qdrant-host
QDRANT_PORT=6333
```

### Running in Production

**With multiple workers** (recommended):
```bash
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

**With Docker Compose** (easiest):
```bash
# Production mode
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f intent-api

# Scale workers
docker-compose up -d --scale intent-api=4
```

---

## Daily Development Workflow

### Starting Your Day

```bash
# 1. Activate virtual environment
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# 2. Start services
docker-compose up -d

# 3. Verify services are running
docker-compose ps
curl http://localhost:8000/health

# 4. Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### During Development

```bash
# Run tests frequently
pytest tests/ -v

# Run specific test file
pytest tests/intent_classification_tests/unit/test_keyword_matcher_unit.py -v

# Check code quality
black app/ tests/           # Format code
isort app/ tests/           # Sort imports
flake8 app/                 # Check style

# View logs
docker-compose logs -f redis
docker-compose logs -f qdrant
```

### Ending Your Day

```bash
# Stop development server (Ctrl+C)

# Stop services
docker-compose down

# Deactivate virtual environment
deactivate
```

---

## Next Steps

**Now that the system is running**:

1. **Read the architecture**: [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)
2. **Understand file structure**: [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)
3. **Test thoroughly**: [TESTING_AND_COVERAGE.md](TESTING_AND_COVERAGE.md)
4. **Monitor costs**: [cost_monitoring/COST_MONITORING.md](cost_monitoring/COST_MONITORING.md)
5. **Review documentation**: [README.md](README.md)

---

## Common Test Queries

**Try these queries to see the system in action**:

```bash
# Simple intents (keyword match, ~30ms)
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"text": "add to cart"}'
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"text": "checkout"}'
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"text": "track my order"}'

# Semantic intents (embedding match, ~80ms)
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"text": "put in basket"}'
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"text": "show me kicks"}'
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"text": "where is my package"}'

# Complex queries (may need LLM)
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"text": "I need something"}'
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"text": "help"}'
```

---

**Last Updated**: November 2025  
**Version**: 1.0.0  
**Maintained By**: ChatNShop Development Team


