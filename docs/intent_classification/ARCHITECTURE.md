# Architecture

Components
- Keyword Matcher: Loads keyword JSON, normalizes, matches (exact/regex/partial), scores, returns top-N.
- Embedding Matcher: Loads SentenceTransformer model, builds taxonomy-backed reference embeddings, computes cosine similarity.
- DecisionEngine (Hybrid): Orchestrates keyword/embedding flows, applies priority short-circuit, blends results (kw_weight/emb_weight), deterministic low-confidence selection, optional LLM fallback.
- Config Manager: Loads config/variants, hot-reloads on changes, versioned backups, validates weights.
- API: FastAPI app exposing /classify (response includes action_code, confidence_score, matched_keywords), OpenAPI docs.
- Ambiguity Resolver: Detects UNCLEAR/AMBIGUOUS cases, logs to JSONL, reporting tool.

Data Flow
1) User text → DecisionEngine
2) Keyword matcher (priority rule). If confident → return result
3) Embedding matcher → compute similarities
4) Blend keyword + embedding results (weights) → evaluate confidence
5) If not confident → deterministic top selection; log ambiguity; (optional) LLM fallback
6) API formats output for client

Artifacts
- Keywords: app/ai/intent_classification/keywords/*.json
- Taxonomy: app/ai/intent_classification/intents_modular/definitions/*
- Config: config/*.json (+ versions/)
- Logs: logs/ambiguous.jsonl
