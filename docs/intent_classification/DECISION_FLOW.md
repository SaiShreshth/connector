# Hybrid Matching Decision Flow

High-level steps
1) Keyword matcher → top results (score, matched_text)
2) If top keyword score ≥ priority_threshold → return CONFIDENT_KEYWORD
3) Embedding matcher → top results (score)
4) Blend results if both enabled: blended_score = kw_score*kw_weight + emb_score*emb_weight
5) If confident per confidence_threshold → return top blended
6) Else fallbacks (lower thresholds, generic SEARCH_PRODUCT)

Conflict resolution
- If same intent appears in both, use weighted blended_score; base result is whichever had the higher individual score.
- If only one method returns, that method’s results win.
- Exact/regex beats partial for ties at equal score (via rank in keyword matcher).

Outputs
- status: CONFIDENT_KEYWORD | FALLBACK_* | blended reasons
- intent: { id, score, source, keyword_score?, embedding_score? }

Diagram (ASCII)

[User Query]
   │
   ├─► KeywordMatcher ── top_kw (score)
   │        │
   │        ├─ if score ≥ priority_threshold → CONFIDENT_KEYWORD
   │        │
   │        ▼
   ├─► EmbeddingMatcher ─ top_emb (score)
   │
   └─► Blend(kw, emb) → confidence check → result or fallback

Config knobs
- priority_threshold (keyword short-circuit)
- kw_weight, emb_weight (blending)
- confidence thresholds (absolute and gap)
