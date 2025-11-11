# Configuration Examples (Hybrid Matching)

Variant A (balanced)
```json
{
  "rules": {
    "rule_sets": {
      "A": {
        "use_keywords": true,
        "use_embedding": true,
        "priority_threshold": 0.85,
        "kw_weight": 0.6,
        "emb_weight": 0.4
      }
    }
  }
}
```

Variant B (keyword-first)
```json
{
  "rules": {
    "rule_sets": {
      "B": {
        "use_keywords": true,
        "use_embedding": true,
        "priority_threshold": 0.80,
        "kw_weight": 0.8,
        "emb_weight": 0.2
      }
    }
  }
}
```

Notes
- Switch variants via `ACTIVE_VARIANT` in config manager.
- Keep `kw_weight + emb_weight = 1.0` for clarity.
- Use higher `priority_threshold` to short-circuit earlier on clear keyword matches.
