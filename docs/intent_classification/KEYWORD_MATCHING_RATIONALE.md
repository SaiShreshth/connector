# Keyword Matching Rationale

Current approach
- Data structures: hash maps + precompiled regex + pre-tokenized phrase tuples
- Normalization: lightweight symbol folding and tokenization with caching (lru_cache)
- Scoring: exact > regex > partial token-overlap with priority-weighting

Why not tries (yet)?
- Our dictionaries are phrase-heavy (multi-token) and include many variants (misspellings, UK/US), which favors normalized token tuples over char-level traversal.
- Regex handles flexible patterns better than trie-only lookups.
- Current latency meets target (p95 < 50ms) with simpler code and easier maintenance.

When to move to tries
- If we onboard much larger single-token dictionaries or need streaming prefix matches at scale.
- If p95 degrades with significantly more phrases and regex patterns.

Possible future hybrid
- Trie for fast single-token/prefix candidates → regex/phrase confirmation step.
- Keep normalization and caching as-is to control GC and allocations.

Operational notes
- Keep priorities in 1–9 to bound weight ranges.
- Re-precompile only on config reload; matching is read-only and cached.
