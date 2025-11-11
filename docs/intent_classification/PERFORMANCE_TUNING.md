# Performance Tuning

Targets
- Keyword matcher: p95 < 50ms
- API /classify: p95 < 100ms

Knobs
- DecisionEngine: priority_threshold, kw_weight, emb_weight, use_keywords, use_embedding
- Embeddings: cache query encodings; precompute taxonomy references

Commands
- Keyword p95: pytest tests/intent_classification_tests/unit/test_keyword_benchmark.py
- API p95: pytest tests/intent_classification_tests/perf/test_load_1000_qps.py tests/intent_classification_tests/perf/test_api_perf.py

Tips
- Warm up models on startup
- Avoid heavy logging in hot paths
- Use smaller models if needed (MiniLM variants)
