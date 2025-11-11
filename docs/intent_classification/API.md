# API

Base
- FastAPI app with OpenAPI at /docs and /redoc

Endpoint
- POST /classify
  - Request: { "text": "Add to cart" }
  - Response (200):
    {
      "action_code": "ADD_TO_CART",
      "confidence_score": 0.92,
      "matched_keywords": ["add to cart"],
      "original_text": "Add to cart",
      "status": "CONFIDENT_KEYWORD",
      "intent": { "id": "ADD_TO_CART", "score": 0.92, "source": "keyword" }
    }

Errors
- 422: invalid payload
- 500: JSON error body (error, message, detail)

Export OpenAPI
- Programmatic: from fastapi import FastAPI; app.openapi()
- CLI example: curl http://localhost:8000/openapi.json > openapi.json

Performance
- p95 < 100ms validated via tests/test_api_perf.py
