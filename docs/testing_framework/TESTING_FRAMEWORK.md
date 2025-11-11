# Testing and Validation Framework - Complete Guide

## Overview

The Testing and Validation Framework (TASK-23) provides comprehensive capability to test different prompts, models, and strategies in production. It combines two complementary A/B testing systems for maximum flexibility.

**Key Features:**
- Traffic splitting with adaptive bandit algorithms
- Model comparison (GPT-4, GPT-4-mini, GPT-3.5, custom)
- Prompt template testing
- Statistical significance testing
- Rollback mechanisms with safety checks
- Unified management across both systems

---

## Architecture

### Two Complementary Systems

#### 1. **Cost Monitoring A/B (TASK-22)**
- **Location:** `app/ai/cost_monitor/ab_testing.py`
- **Storage:** SQLite database
- **Interface:** REST API
- **Focus:** Production cost tracking, budget optimization
- **Best For:** Long-running experiments, cost analysis

#### 2. **Testing Framework (TASK-23)**
- **Location:** `app/core/ab_testing/`
- **Storage:** CSV/JSONL files
- **Interface:** CLI + REST API
- **Focus:** Model/prompt validation, accuracy testing
- **Best For:** Pre-production testing, rapid iteration

#### 3. **Integration Layer**
- **Location:** `app/core/ab_testing_integration.py`
- **Purpose:** Unified interface to both systems
- **Features:** Cross-system metrics, shared reporting

---

## Quick Start

### 1. Create Your First Experiment

Using the CLI:

```bash
# Create from template
python -m app.core.ab_testing.config_cli from-template model_comparison \
  --experiment-id exp_my_first_test \
  --activate

# Or create custom experiment
python -m app.core.ab_testing.config_cli create \
  --name "My A/B Test" \
  --variants "variant_a,variant_b" \
  --use-bandit \
  --activate
```

Using the API:

```bash
curl -X POST http://localhost:8000/api/testing/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "exp_my_test",
    "name": "Model Comparison",
    "description": "Compare GPT-4 vs GPT-4-mini",
    "use_bandit": true,
    "variants": [
      {
        "name": "gpt4",
        "traffic_pct": 50,
        "model": "gpt-4",
        "prompt_template": "Use GPT-4"
      },
      {
        "name": "gpt4_mini",
        "traffic_pct": 50,
        "model": "gpt-4o-mini",
        "prompt_template": "Use GPT-4-mini"
      }
    ]
  }'
```

### 2. Run the Experiment

The framework automatically routes traffic based on your configuration:

```python
from app.core.ab_testing.ab_framework import choose_variant, run_llm_inference, log_event

# Choose variant for user
variant = choose_variant(user_id=12345)

# Run inference
response, latency, cost, prompt_tokens, completion_tokens = run_llm_inference(
    prompt="Find red shoes under $100",
    model_name=variant["model"]
)

# Log result
log_event(
    variant_name=variant["name"],
    user_id=12345,
    latency_ms=latency,
    cost_usd=cost,
    result_label="success"  # or "failure"
)
```

### 3. Analyze Results

```bash
# Run statistical analysis
python -m app.core.ab_testing.analysis

# Or via API
curl http://localhost:8000/api/testing/experiments/exp_my_test/results
```

### 4. Rollback if Needed

```bash
# CLI
python -m app.core.ab_testing.rollback_manager rollback-variant \
  exp_my_test variant_a --traffic 1.0

# API
curl -X POST http://localhost:8000/api/testing/experiments/exp_my_test/rollback \
  -H "Content-Type: application/json" \
  -d '{
    "target_variant": "variant_a",
    "traffic_percentage": 1.0,
    "gradual": true
  }'
```

---

## Detailed Features

### Traffic Splitting

#### Static Split
```json
{
  "use_bandit": false,
  "variants": [
    {"name": "control", "traffic_pct": 50},
    {"name": "treatment", "traffic_pct": 50}
  ]
}
```

#### Adaptive Bandit (Epsilon-Greedy)
```json
{
  "use_bandit": true,
  "bandit_config": {
    "epsilon": 0.1
  },
  "variants": [
    {"name": "var_a", "traffic_pct": 33},
    {"name": "var_b", "traffic_pct": 33},
    {"name": "var_c", "traffic_pct": 34}
  ]
}
```

The bandit algorithm automatically shifts traffic toward better-performing variants.

### Model Comparison

Test different models simultaneously:

```python
variants = {
    "gpt4": {"model": "gpt-4", "description": "High accuracy"},
    "gpt4_mini": {"model": "gpt-4o-mini", "description": "Fast & efficient"},
    "gpt35": {"model": "gpt-3.5-turbo", "description": "Baseline"}
}
```

Track:
- **Accuracy:** Success rate per variant
- **Latency:** P50, P95 response times
- **Cost:** Per-request and total cost
- **Tokens:** Prompt + completion tokens

### Prompt Testing

Compare different prompts with the same model:

```json
{
  "variants": [
    {
      "name": "verbose",
      "model": "gpt-4o-mini",
      "prompt_template": "You are an intelligent assistant... [long prompt]"
    },
    {
      "name": "compact",
      "model": "gpt-4o-mini",
      "prompt_template": "Classify query into action codes..."
    }
  ]
}
```

### Statistical Analysis

The framework provides:

1. **Omnibus Chi-Square Test**
   - Tests if any variant differs significantly
   - H0: All variants have equal success rates
   - Reject if p < 0.05

2. **Pairwise Comparisons**
   - Fisher's Exact Test for small samples
   - Bonferroni correction for multiple comparisons

3. **Winner Determination**
   - Highest success rate
   - Statistical significance considered
   - Minimum sample size requirements

Example output:

```
=== A/B Test Summary ===

 - control: 45/100 successes, success_rate=0.450
 - treatment: 62/100 successes, success_rate=0.620

Omnibus chi-square test:
  chi2: 5.847
  p_value: 0.0156  ✓ Significant!
  
Winner: treatment (62.0% vs 45.0%, p=0.0156)
```

---

## CLI Reference

### Configuration Management

```bash
# List templates
python -m app.core.ab_testing.config_cli templates

# Create from template
python -m app.core.ab_testing.config_cli from-template <template_name>

# Create custom
python -m app.core.ab_testing.config_cli create \
  --name "My Test" \
  --variants "a,b,c" \
  --model "gpt-4o-mini" \
  --use-bandit

# Validate config
python -m app.core.ab_testing.config_cli validate config.json

# List experiments
python -m app.core.ab_testing.config_cli list

# Activate experiment
python -m app.core.ab_testing.config_cli activate exp_001
```

### Rollback Operations

```bash
# Create backup
python -m app.core.ab_testing.rollback_manager backup exp_001

# Rollback to variant
python -m app.core.ab_testing.rollback_manager rollback-variant \
  exp_001 control --traffic 1.0

# Rollback to backup
python -m app.core.ab_testing.rollback_manager rollback-backup exp_001

# List backups
python -m app.core.ab_testing.rollback_manager list --experiment-id exp_001

# View history
python -m app.core.ab_testing.rollback_manager history --limit 10
```

### Analysis

```bash
# Run full analysis
python -m app.core.ab_testing.analysis

# Generate report
python -m app.core.ab_testing.analysis --output report.md
```

---

## API Reference

### Experiment Management

**Create Experiment**
```
POST /api/testing/experiments
Body: {experiment_id, name, description, variants, use_bandit}
```

**List Experiments**
```
GET /api/testing/experiments
```

**Get Results**
```
GET /api/testing/experiments/{experiment_id}/results
```

### Bandit Control

**Get State**
```
GET /api/testing/bandit/state?experiment_id=exp_001
```

**Reset State**
```
POST /api/testing/bandit/reset?experiment_id=exp_001
```

### Rollback

**Rollback to Variant**
```
POST /api/testing/experiments/{experiment_id}/rollback
Body: {target_variant, traffic_percentage, gradual}
```

**Create Backup**
```
POST /api/testing/experiments/{experiment_id}/backup?reason=manual
```

**List Backups**
```
GET /api/testing/experiments/{experiment_id}/backups
```

### System Status

**Get Status**
```
GET /api/testing/status
```

**Health Check**
```
GET /api/testing/health
```

---

## Best Practices

### 1. Experiment Design

**Start Small**
- Begin with 2 variants (A/B test)
- Add more variants once comfortable

**Define Success Metrics**
- Accuracy (primary)
- Latency (secondary)
- Cost (constraint)

**Set Minimum Sample Size**
- At least 100 samples per variant
- More for detecting small effects

### 2. Traffic Allocation

**Use Bandit for**
- Exploratory testing
- Quick optimization
- Adaptive allocation

**Use Static Split for**
- Confirmatory testing
- Regulatory compliance
- Fixed comparisons

### 3. Rollback Strategy

**Always Create Backups**
- Before major changes
- Before rollback operations

**Monitor Actively**
- Check metrics daily
- Set up alerts for anomalies

**Gradual Rollout**
- Start with 10% traffic
- Increase gradually if successful
- Full rollback if issues detected

### 4. Statistical Rigor

**Avoid Peeking**
- Don't stop experiments early based on trends
- Wait for statistical significance

**Multiple Testing Correction**
- Use Bonferroni correction for multiple variants
- Adjust alpha level accordingly

**Check Assumptions**
- Independent samples
- Sufficient sample size
- Similar baseline conditions

---

## Integration Examples

### With Decision Engine

```python
from app.ai.intent_classification.decision_engine import get_intent_classification
from app.core.ab_testing_integration import get_unified_ab_manager

# Initialize
unified_mgr = get_unified_ab_manager()

# In your classification function
def classify_with_ab_test(user_input: str, experiment_id: str):
    # Get variant assignment
    variant = unified_mgr.assign_variant(experiment_id, f"req_{uuid.uuid4()}")
    
    # Run classification with variant config
    result = get_intent_classification(
        user_input=user_input,
        model=variant.config.get("model"),
        prompt_version=variant.config.get("prompt_version")
    )
    
    # Record result
    unified_mgr.record_unified_result(
        experiment_id=experiment_id,
        variant_id=variant.variant_id,
        request_id=result.get("request_id"),
        user_input=user_input,
        predicted_action=result.get("action_code"),
        confidence=result.get("confidence"),
        latency_ms=result.get("latency_ms"),
        token_count=result.get("total_tokens"),
        cost_usd=result.get("cost"),
        is_correct=result.get("is_correct")
    )
    
    return result
```

### With Cost Monitoring

```python
from app.ai.cost_monitor.ab_testing import get_ab_test_manager, ExperimentResult

# Both systems track the same experiment
cost_mgr = get_ab_test_manager()
testing_mgr = get_unified_ab_manager()

# Record to both
cost_mgr.record_result(ExperimentResult(...))
testing_mgr.record_unified_result(...)
```

---

## Troubleshooting

### Issue: No variants being assigned

**Symptoms:** All requests get same variant

**Solutions:**
1. Check if experiment is activated
2. Verify `use_bandit` setting
3. Check traffic percentages sum to 100
4. Review bandit state

### Issue: Statistical tests inconclusive

**Symptoms:** High p-values, no clear winner

**Solutions:**
1. Collect more samples
2. Check if variants are truly different
3. Verify effect size is detectable
4. Increase traffic to variants

### Issue: Rollback doesn't work

**Symptoms:** Traffic still split after rollback

**Solutions:**
1. Check if active config was updated
2. Verify experiment_id matches
3. Restart application to reload config
4. Check rollback history

---

## Acceptance Criteria Status

| Criteria | Status | Implementation |
|----------|--------|----------------|
| ✅ Traffic splitting for A/B tests | Complete | Bandit + static split |
| ✅ Test different prompts simultaneously | Complete | Variant configs |
| ✅ Test GPT-4 vs GPT-4-turbo vs models | Complete | Model variants |
| ✅ Track accuracy, latency, cost | Complete | CSV/JSONL logging |
| ✅ Statistical significance testing | Complete | Chi-square, Fisher's |
| ✅ Easy configuration | Complete | CLI + templates |
| ✅ Rollback mechanism | Complete | Manager + API |
| ✅ Document results | Complete | This guide |

---

## Additional Resources

- **Cost Monitoring:** `docs/cost_monitoring/COST_MONITORING.md`
- **Experiment Design:** `docs/testing_framework/EXPERIMENT_DESIGN.md`
- **API Reference:** `docs/testing_framework/API_REFERENCE.md`
- **Integration Guide:** `docs/testing_framework/INTEGRATION_GUIDE.md`

---

**Last Updated:** November 1, 2025  
**Version:** 1.0.0  
**Status:** Production Ready

