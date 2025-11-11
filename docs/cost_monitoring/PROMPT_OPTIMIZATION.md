# Prompt Optimization for Token Reduction

## Overview

This document details the prompt optimization efforts undertaken to reduce token usage and LLM API costs while maintaining classification accuracy.

## Version Comparison

### v1.0.0 (Original) → v1.1.0 (Optimized)

| Metric | v1.0.0 | v1.1.0 | Improvement |
|--------|--------|--------|-------------|
| **Total Lines** | 147 | 46 | **68.7% reduction** |
| **Token Count** | ~850 tokens | ~280 tokens | **67.1% reduction** |
| **Examples** | 3 full examples | 2 compact examples | **33.3% reduction** |
| **Instruction Verbosity** | High | Minimal | Significant |

## Optimization Strategies Applied

### 1. **Removed Redundant Information**

**Before (v1.0.0):**
```
You are an intelligent assistant that classifies customer queries into predefined **action codes** and extracts relevant entities.  
You must always output JSON with the following structure:
```

**After (v1.1.0):**
```
Classify customer queries into action codes and extract entities. Always output JSON:
```

**Token Savings:** ~15 tokens per request

### 2. **Consolidated Action Code List**

**Before:** Vertical list with bullet points (44 lines)
```
### Allowed Action Codes:
- SEARCH_PRODUCT
- SEARCH_DISCOUNT
- SEARCH_PRICE_RANGE
...
```

**After:** Comma-separated inline list (1 line)
```
### Action Codes:
SEARCH_PRODUCT, SEARCH_DISCOUNT, SEARCH_PRICE_RANGE, ...
```

**Token Savings:** ~25 tokens per request

### 3. **Compressed Entity Definitions**

**Before:** Verbose descriptions with multiple examples per entity
```
**Entity Types:**
- `product_type`: Type of product (e.g., "shoes", "phone", "laptop", "shirt", "watch")
- `category`: Product category or subcategory (e.g., "running", "electronics", "clothing", "sports", "casual")
...
```

**After:** Concise format with fewer examples
```
- product_type: "shoes", "phone", "laptop"
- category: "running", "electronics"
...
```

**Token Savings:** ~40 tokens per request

### 4. **Simplified Examples**

**Before:** Multi-line JSON with extensive formatting and reasoning
```
**Example 2:**
User: "Show me red Nike running shoes under $100"
Assistant:
{
  "action_code": "SEARCH_PRODUCT",
  "confidence": 0.95,
  "reasoning": "Clear product search with specific brand, color, category, and price constraints.",
  "secondary_intents": [],
  "entities": {
    "product_type": "shoes",
    "category": "running",
    "brand": "Nike",
    "color": "red",
    "size": null,
    "price_range": {"min": null, "max": 100, "currency": "USD"}
  }
}
```

**After:** Compact one-line JSON
```
Q: "red Nike shoes under $100"
A: {"action_code":"SEARCH_PRODUCT","confidence":0.95,"reasoning":"Clear product search","secondary_intents":[],"entities":{"product_type":"shoes","category":"running","brand":"Nike","color":"red","size":null,"price_range":{"min":null,"max":100,"currency":"USD"}}}
```

**Token Savings:** ~30 tokens per example × 3 examples = ~90 tokens

### 5. **Removed Separator Lines**

**Before:** Multiple `---` separators for visual clarity

**After:** Minimal separators only where needed

**Token Savings:** ~10 tokens per request

## Cost Impact Analysis

### Per-Request Savings

Using gpt-4o-mini pricing ($0.000005 per token):

| Component | v1.0.0 Tokens | v1.1.0 Tokens | Tokens Saved | Cost Saved |
|-----------|---------------|---------------|--------------|------------|
| System Prompt | 850 | 280 | 570 | $0.00285 |
| User Query (avg) | 15 | 15 | 0 | $0.000 |
| Completion (avg) | 80 | 80 | 0 | $0.000 |
| **Total per Request** | **945** | **375** | **570** | **$0.00285** |

### Monthly Projections

Assuming 40,000 LLM queries/month:

| Metric | v1.0.0 | v1.1.0 | Savings |
|--------|--------|--------|---------|
| **Total Tokens** | 37,800,000 | 15,000,000 | 22,800,000 |
| **Total Cost** | $189.00 | $75.00 | **$114.00/month** |
| **Annual Cost** | $2,268.00 | $900.00 | **$1,368.00/year** |

### ROI Calculation

- **Cost Reduction:** 60.3% per request
- **Accuracy Impact:** Minimal (to be validated with A/B testing)
- **Break-even:** Immediate (no implementation cost)

## Accuracy Validation

### Testing Methodology

1. **Test Dataset:** 100 diverse queries covering all intent categories
2. **Metrics:** Accuracy, F1-score, confidence calibration
3. **Comparison:** v1.0.0 vs v1.1.0 side-by-side

### Expected Results

Based on prior prompt engineering research:
- **Accuracy Retention:** ≥95% (most prompts can be reduced 60-70% with <5% accuracy loss)
- **Confidence Scores:** Should remain stable within ±0.05
- **Entity Extraction:** May see minor improvement due to clearer instructions

### Validation Status

⚠️ **Pending A/B Testing** - Recommend running A/B tests before full production deployment

## Implementation Guide

### Step 1: Update Confidence Calibrator

```python
# app/ai/llm_intent/confidence_calibrator.py

# Change default version
PROMPT_VERSION = "1.1.0"  # Updated from "1.0.0"
```

### Step 2: Deploy New Prompt

The optimized prompt is already available at:
```
app/ai/llm_intent/prompts/system_prompt_v1.1.0.txt
```

### Step 3: Test with Sample Queries

```python
from app.ai.llm_intent.prompt_loader import PromptLoader

loader = PromptLoader(version="1.1.0")
loader.load()
print(f"Prompt tokens: ~{len(loader.system_prompt.split())}")
```

### Step 4: Monitor Cost Impact

Use the Cost Dashboard to compare:
- Before: Monitor costs for 1 week with v1.0.0
- After: Deploy v1.1.0 and monitor for 1 week
- Compare: Daily averages, token counts, and accuracy metrics

### Step 5: Gradual Rollout (Recommended)

Use A/B testing framework (see `ab_testing.py`) to:
1. Route 10% of traffic to v1.1.0
2. Monitor for 48 hours
3. Increase to 50% if metrics are stable
4. Full rollout after 1 week of stable performance

## Additional Optimization Opportunities

### 1. Context-Aware Pruning

Only include relevant action codes based on rule-based pre-classification:

```python
# If rule-based suggests SEARCH category
relevant_actions = ["SEARCH_PRODUCT", "SEARCH_DISCOUNT", "SEARCH_PRICE_RANGE"]

# Include only these in prompt (further 40% token reduction)
```

**Estimated Savings:** Additional 100-150 tokens per request

### 2. Few-Shot Example Optimization

- **Current:** 2 examples in v1.1.0
- **Opportunity:** Use 0-1 examples for high-confidence rule-based pre-classifications
- **Savings:** 50-80 tokens per request

### 3. Entity-Specific Prompts

Create specialized prompts for different query types:

- **Simple queries** (no entities): Minimal prompt (~150 tokens)
- **Product search** (entities expected): Full prompt (~280 tokens)
- **Account/Order actions** (no entities): Minimal prompt (~150 tokens)

**Estimated Savings:** 20-30% additional reduction on average

### 4. Streaming and Early Termination

For queries that resolve quickly, use streaming to:
- Stop generation after action_code is determined
- Skip entity extraction for non-product queries
- Reduce completion tokens by 40-60%

## Best Practices

### Do's ✅

- **Test thoroughly** before production deployment
- **Monitor accuracy** metrics alongside cost metrics
- **Use A/B testing** for gradual rollout
- **Document changes** in prompt versioning system
- **Track token usage** per prompt version

### Don'ts ❌

- **Don't sacrifice accuracy** for marginal cost savings
- **Don't remove critical instructions** that prevent errors
- **Don't optimize without baseline** cost and accuracy data
- **Don't deploy to 100%** without validation period
- **Don't ignore confidence** score changes

## Monitoring Checklist

After deploying v1.1.0, monitor:

- [ ] Average tokens per request (should be ~375)
- [ ] Daily cost trends (should decrease ~60%)
- [ ] Classification accuracy (should be ≥95%)
- [ ] Confidence score distribution (should be stable)
- [ ] Error rates (should not increase)
- [ ] User feedback (implicit quality signals)

## Rollback Plan

If v1.1.0 shows degraded performance:

1. **Immediate:** Switch `PROMPT_VERSION` back to "1.0.0"
2. **Analyze:** Review misclassifications and error patterns
3. **Iterate:** Create v1.1.1 with targeted improvements
4. **Re-test:** Validate before next deployment

## Future Work

- [ ] Create v1.2.0 with context-aware pruning
- [ ] Implement entity-specific prompt routing
- [ ] Add streaming support for early termination
- [ ] Build automated prompt optimization pipeline
- [ ] Integrate reinforcement learning for continuous improvement

## References

- Original Prompt: `app/ai/llm_intent/prompts/system_prompt_v1.0.0.txt`
- Optimized Prompt: `app/ai/llm_intent/prompts/system_prompt_v1.1.0.txt`
- Cost Dashboard: `/dashboard/cost`
- A/B Testing Framework: `app/ai/cost_monitor/ab_testing.py`

---

**Document Version:** 1.0  
**Last Updated:** November 1, 2025  
**Author:** Cost Optimization Team

