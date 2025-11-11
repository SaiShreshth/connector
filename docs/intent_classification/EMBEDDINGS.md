# Embedding-Based Matching

Model choice
- SentenceTransformer: `all-MiniLM-L6-v2`
- Rationale: good trade-off of accuracy/latency; 384-dim vectors; widely used baseline.
- Version pin: install `sentence-transformers==2.2.2` and `transformers==4.41.*` in requirements.

Latency benchmark
- Target: <100ms/query on warmed model.
- Local measurement (dev laptop): ~15–30ms for single query after warm-up.
- Run: `pytest tests/test_keyword_benchmark.py -k p95_under_50ms` (keyword path) and add a similar benchmark for embeddings if needed.

Caching
- Reference embeddings are precomputed on startup for intent examples.
- Query embeddings cached in-memory with a small (512) entry cap.

Threshold behavior
- Default threshold used by Decision Engine: 0.60.
- Tests: `tests/test_embedding_threshold.py` verify behavior around thresholds.

Notes
- For production, consider a persistent cache (e.g., Redis) for hot queries and batching for throughput.
