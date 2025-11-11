# Testing and Coverage Verification Guide

## Overview

This guide provides instructions for running all tests and verifying test coverage meets the >80% requirement (TASK-8).

---

## Test Structure

```
tests/
├── intent_classification_tests/     # Rule-based classifier tests
│   ├── unit/                        # Unit tests
│   │   ├── test_keyword_matcher_unit.py
│   │   ├── test_embedding_unit.py
│   │   ├── test_hybrid_classifier.py
│   │   ├── test_hybrid_decision.py
│   │   ├── test_embedding_threshold.py
│   │   ├── test_keyword_special_chars.py
│   │   └── test_keywords_quality.py
│   ├── integration/                 # Integration tests
│   │   ├── test_integration_100_cases.py
│   │   ├── test_accuracy_dataset.py
│   │   └── test_ambiguity_resolver.py
│   ├── perf/                        # Performance tests
│   │   ├── test_keyword_benchmark.py
│   │   ├── test_api_perf.py
│   │   └── test_load_1000_qps.py
│   ├── api/                         # API tests
│   │   └── test_api_error_handling.py
│   ├── config/                      # Config tests
│   │   └── test_config_manager.py
│   └── status_store/                # Status store tests
│       └── test_status_store.py
├── llm_intent_tests/                # LLM classifier tests
│   ├── unit/                        # Unit tests
│   │   ├── test_openai_client.py
│   │   ├── test_response_parser.py
│   │   ├── test_prompt_loader.py
│   │   ├── test_prompt_integration.py
│   │   ├── test_context_collector.py
│   │   └── test_context_summarizer.py
│   ├── integration/                 # Integration tests
│   │   ├── test_request_handler.py
│   │   └── test_context_enhancement.py
│   ├── prompts/                     # Prompt tests
│   │   └── test_prompt_validation.py
│   └── tools/                       # Testing tools
│       └── prompt_tester.py
└── integration/                     # End-to-end tests
    └── test_tasks_15_16_17_integration.py
```

---

## Running All Tests

### 1. Install Test Dependencies

```bash
cd /Users/nagavardhanreddy/Documents/AI-Shopping/chatNShop

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio
```

---

### 2. Run All Tests

```bash
# Run all tests with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/intent_classification_tests/ -v
pytest tests/llm_intent_tests/ -v
pytest tests/integration/ -v
```

---

### 3. View Coverage Report

After running tests with coverage, open the HTML report:

```bash
# Open coverage report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## Coverage Requirements

### TASK-8: Unit and Integration Tests
- **Requirement**: >80% code coverage
- **Target Modules**:
  - `app/ai/intent_classification/` - Rule-based classifier
  - `app/ai/llm_intent/` - LLM classifier
  - `app/queue/` - Queue infrastructure
  - `app/core/` - Core utilities

---

## Running Tests by Category

### Unit Tests

```bash
# Rule-based classifier unit tests
pytest tests/intent_classification_tests/unit/ -v

# LLM classifier unit tests
pytest tests/llm_intent_tests/unit/ -v

# Specific test file
pytest tests/intent_classification_tests/unit/test_keyword_matcher_unit.py -v
```

---

### Integration Tests

```bash
# Rule-based integration tests
pytest tests/intent_classification_tests/integration/ -v

# LLM integration tests
pytest tests/llm_intent_tests/integration/ -v

# End-to-end integration (Tasks 15, 16, 17)
pytest tests/integration/test_tasks_15_16_17_integration.py -v
```

---

### Performance Tests

```bash
# All performance tests
pytest tests/intent_classification_tests/perf/ -v

# Individual performance tests
pytest tests/intent_classification_tests/perf/test_keyword_benchmark.py -v
pytest tests/intent_classification_tests/perf/test_api_perf.py -v
pytest tests/intent_classification_tests/perf/test_load_1000_qps.py -v
```

---

## Coverage Report Interpretation

### Good Coverage (>80%)

```
Name                                           Stmts   Miss  Cover
------------------------------------------------------------------
app/ai/intent_classification/keyword_matcher    250     30    88%
app/ai/intent_classification/embedding_matcher  180     20    89%
app/ai/llm_intent/request_handler              150     25    83%
app/queue/queue_manager                        200     35    82%
------------------------------------------------------------------
TOTAL                                          2500    350    86%
```

### What to Look For
- ✅ Overall coverage > 80%
- ✅ Critical modules (keyword_matcher, request_handler) > 85%
- ✅ Core logic (decision_engine, queue_manager) > 80%
- ⚠️ Test utilities and tools can have lower coverage
- ⚠️ Configuration files and constants can have lower coverage

---

## Running Specific Test Scenarios

### Test Keyword Matching

```bash
# Unit tests
pytest tests/intent_classification_tests/unit/test_keyword_matcher_unit.py -v -k "test_exact_match"

# Edge cases
pytest tests/intent_classification_tests/unit/test_keyword_special_chars.py -v

# Quality checks
pytest tests/intent_classification_tests/unit/test_keywords_quality.py -v
```

---

### Test Embedding Matching

```bash
# Unit tests
pytest tests/intent_classification_tests/unit/test_embedding_unit.py -v

# Threshold tests
pytest tests/intent_classification_tests/unit/test_embedding_threshold.py -v
```

---

### Test Hybrid Classification

```bash
# Hybrid classifier tests
pytest tests/intent_classification_tests/unit/test_hybrid_classifier.py -v

# Decision engine tests
pytest tests/intent_classification_tests/unit/test_hybrid_decision.py -v
```

---

### Test Ambiguity Resolution

```bash
# Ambiguity resolver tests
pytest tests/intent_classification_tests/integration/test_ambiguity_resolver.py -v
```

---

### Test LLM Components

```bash
# OpenAI client tests
pytest tests/llm_intent_tests/unit/test_openai_client.py -v

# Response parser tests
pytest tests/llm_intent_tests/unit/test_response_parser.py -v

# Prompt loader tests
pytest tests/llm_intent_tests/unit/test_prompt_loader.py -v

# Request handler tests
pytest tests/llm_intent_tests/integration/test_request_handler.py -v
```

---

### Test Context Enhancement (TASK-17)

```bash
# Context collector tests
pytest tests/llm_intent_tests/unit/test_context_collector.py -v

# Context summarizer tests
pytest tests/llm_intent_tests/unit/test_context_summarizer.py -v

# Integration tests
pytest tests/llm_intent_tests/integration/test_context_enhancement.py -v
```

---

### Test Queue Infrastructure (TASKS 14-16)

```bash
# Queue producer tests
pytest tests/test_queue_producer.py -v

# Status store tests
pytest tests/test_status_store.py -v
pytest tests/intent_classification_tests/status_store/test_status_store.py -v

# End-to-end integration
pytest tests/integration/test_tasks_15_16_17_integration.py -v
```

---

## Test Accuracy Requirements

### TASK-8: Integration Tests
- **Requirement**: Minimum 100 test cases
- **Test**: `tests/intent_classification_tests/integration/test_integration_100_cases.py`
- **Expected**: All 100+ cases pass

### TASK-12: Prompt Engineering
- **Requirement**: >90% accuracy on test dataset
- **Test**: `tests/llm_intent_tests/prompts/test_prompt_validation.py`
- **Dataset**: 50+ labeled queries

---

## Running Accuracy Tests

### Test on 100+ Integration Cases

```bash
pytest tests/intent_classification_tests/integration/test_integration_100_cases.py -v -s
```

**Expected Output**:
```
test_integration_100_cases ... ok
Passed: 98/100 (98% accuracy)
```

---

### Test on Labeled Accuracy Dataset

```bash
pytest tests/intent_classification_tests/integration/test_accuracy_dataset.py -v -s
```

**Expected Output**:
```
test_accuracy_dataset ... ok
Accuracy: 92.5% (should be >= 90%)
```

---

## Continuous Integration Setup

### GitHub Actions Example

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests with coverage
      run: |
        pytest --cov=app --cov-report=xml --cov-report=term
    
    - name: Check coverage threshold
      run: |
        coverage report --fail-under=80
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

---

## Test Debugging

### Run Tests with Debug Output

```bash
# Show print statements
pytest -v -s

# Show logs
pytest -v --log-cli-level=DEBUG

# Stop at first failure
pytest -x

# Run last failed tests only
pytest --lf

# Run specific test
pytest tests/intent_classification_tests/unit/test_keyword_matcher_unit.py::TestKeywordMatcher::test_exact_match -v
```

---

### Profile Slow Tests

```bash
# Show slowest 10 tests
pytest --durations=10

# Show all test durations
pytest --durations=0
```

---

## Test Coverage Goals by Module

### Rule-Based Classifier (app/ai/intent_classification/)

| Module | Target Coverage | Priority |
|--------|----------------|----------|
| keyword_matcher.py | >90% | Critical |
| embedding_matcher.py | >85% | High |
| hybrid_classifier.py | >85% | High |
| decision_engine.py | >80% | High |
| ambiguity_resolver.py | >80% | High |
| confidence_threshold.py | >75% | Medium |

### LLM Classifier (app/ai/llm_intent/)

| Module | Target Coverage | Priority |
|--------|----------------|----------|
| request_handler.py | >85% | Critical |
| openai_client.py | >80% | High |
| response_parser.py | >85% | High |
| prompt_loader.py | >80% | High |
| context_collector.py | >75% | Medium |
| context_summarizer.py | >75% | Medium |
| confidence_calibration.py | >70% | Medium |

### Queue Infrastructure (app/queue/)

| Module | Target Coverage | Priority |
|--------|----------------|----------|
| queue_manager.py | >80% | High |
| queue_producer.py | >80% | High |
| worker.py | >75% | High |
| monitor.py | >70% | Medium |

### Core Utilities (app/core/)

| Module | Target Coverage | Priority |
|--------|----------------|----------|
| status_store.py | >80% | High |
| session_store.py | >75% | Medium |
| config_manager.py | >70% | Medium |

---

## Missing Coverage Analysis

### Generate Missing Coverage Report

```bash
# Run with missing lines report
pytest --cov=app --cov-report=term-missing

# Filter to show only uncovered lines
pytest --cov=app --cov-report=term-missing | grep "MISS"
```

**Example Output**:
```
app/ai/intent_classification/keyword_matcher.py    88%   45-52, 78-82
app/ai/llm_intent/request_handler.py               83%   102-110, 145
```

Lines 45-52, 78-82 in keyword_matcher.py are not covered by tests.

---

## Adding New Tests

### Test Template

```python
import unittest
from app.ai.intent_classification.keyword_matcher import match_keywords

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = []
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def test_feature_basic(self):
        """Test basic feature functionality"""
        result = match_keywords("test query")
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
    
    def test_feature_edge_case(self):
        """Test edge case"""
        result = match_keywords("")
        self.assertEqual(result, [])

if __name__ == "__main__":
    unittest.main()
```

---

## Test Checklist

Before marking testing as complete:

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All performance tests pass
- [ ] Overall coverage > 80%
- [ ] Critical modules coverage > 85%
- [ ] 100+ integration test cases exist and pass
- [ ] Accuracy on labeled dataset > 90%
- [ ] Load test passes (1000 QPS)
- [ ] No failing tests in CI/CD pipeline
- [ ] Coverage report generated and reviewed
- [ ] Missing coverage areas documented
- [ ] Plan to improve coverage < 80% modules

---

## Commands Quick Reference

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term

# Run specific category
pytest tests/intent_classification_tests/ -v
pytest tests/llm_intent_tests/ -v

# Run performance tests
pytest tests/intent_classification_tests/perf/ -v

# Check coverage threshold
coverage report --fail-under=80

# View HTML coverage report
open htmlcov/index.html

# Run with debug output
pytest -v -s --log-cli-level=DEBUG

# Run only failed tests
pytest --lf

# Show slowest tests
pytest --durations=10
```

---

## Next Steps

1. ✅ Run all tests: `pytest -v`
2. ✅ Generate coverage report: `pytest --cov=app --cov-report=html`
3. ✅ Review coverage: Open `htmlcov/index.html`
4. ✅ Verify >80% coverage
5. ✅ Add tests for uncovered code if needed
6. ✅ Set up CI/CD pipeline with coverage checks
7. ✅ Document test results in `TESTING_RESULTS.md`

