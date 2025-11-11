"""
⚠️  FALLBACK CONFIGURATION ONLY ⚠️

This file provides SAFE DEFAULT values that are ONLY used when:
- Config Manager fails to load config/rules.json
- Running in emergency/degraded mode
- Config Manager is not available

🎯 PRIMARY CONFIG SOURCE: config/rules.json (via config_manager)
   ↳ Supports hot-reload, A/B testing, and dynamic updates

DO NOT modify this file for production configuration changes.
Instead, update config/rules.json.
"""

# ==============================================================================
# FALLBACK DEFAULTS - Only used when config manager fails
# ==============================================================================

# --- Hybrid Classifier Weights ---
# How to blend keyword vs embedding scores
PRIORITY_THRESHOLD = 0.80  # Skip embeddings if keyword score >= this
WEIGHTS = {
    "keyword": 0.6,      # Weight for keyword matcher
    "embedding": 0.4     # Weight for embedding matcher
}

# --- Confidence Thresholds ---
# Used by confidence_threshold.py to determine if result is confident
MIN_ABSOLUTE_CONFIDENCE = 0.30   # Minimum score for any result
MIN_DIFFERENCE_THRESHOLD = 0.05  # Minimum gap between top 2 results

# --- Ambiguity Resolution ---
# Used by ambiguity_resolver.py for unclear/ambiguous intent handling
UNCLEAR_THRESHOLD = 0.40  # Below this = unclear intent
MIN_CONFIDENCE = 0.60     # Above this = valid intent

# ==============================================================================
# ⚠️  REMINDER: These are FALLBACK values only!
# For production config, update: config/rules.json
# ==============================================================================
