# Keyword Maintenance

Where
- app/ai/intent_classification/keywords/*.json

Guidelines
- 20–30 keywords per intent; include multi-word phrases, misspellings, UK/US variants
- Use priority 1–9 per file; lower number = higher priority
- Avoid duplicates; include hyphenated/spaced variants when common

Workflow
1) Edit keyword JSONs
2) Run tests: pytest tests/intent_classification_tests/unit/test_keyword_matcher_unit.py tests/intent_classification_tests/unit/test_keyword_special_chars.py tests/intent_classification_tests/unit/test_keywords_quality.py
3) Commit changes; hot-reload will pick up updates if watcher is running

Review Cadence
- Monthly review + on-demand from ambiguity logs (tools/ambiguity_report.py)
