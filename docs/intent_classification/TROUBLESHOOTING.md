# Troubleshooting

No matches
- Ensure keywords exist and include multi-word phrases
- Check normalization (special chars) via unit tests

Low accuracy
- Expand taxonomy example_phrases for underperforming intents
- Adjust kw_weight/emb_weight and priority_threshold
- Check that embeddings loaded from taxonomy (log shows count)

High latency
- Verify model warm-up and caching
- Reduce embeddings size or batch encode on startup
- Check p95 tests

Config reload errors
- See config/versions for last good version
- Ensure rules.rule_sets present and weights sum to 1.0
