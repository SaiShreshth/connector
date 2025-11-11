# LLM Intent Classification Tests

Organized test suite for the LLM intent classification module, following the same structure as `intent_classification_tests`.

## 📁 Directory Structure

```
tests/llm_intent_tests/
├── unit/              # Unit tests for individual components
│   ├── test_prompt_loader.py
│   ├── test_response_parser.py
│   ├── test_openai_client.py
│   └── test_prompt_integration.py
├── integration/       # Integration tests for end-to-end flows
│   └── test_request_handler.py
├── api/              # API-level tests
│   └── (to be added)
├── prompts/          # Prompt validation and testing
│   └── test_prompt_validation.py
└── tools/            # Test utilities
    └── prompt_tester.py
```

## 🧪 Test Categories

### Unit Tests (`unit/`)
- **test_prompt_loader.py**: Prompt loading, caching, versioning
- **test_response_parser.py**: JSON parsing, confidence clamping, field inference
- **test_openai_client.py**: Client initialization, error handling, circuit breaker
- **test_prompt_integration.py**: Prompt integration with request handler

### Integration Tests (`integration/`)
- **test_request_handler.py**: End-to-end flow with prompts and LLM responses

### Prompt Tests (`prompts/`)
- **test_prompt_validation.py**: Schema validation, versioning checks

### Tools (`tools/`)
- **prompt_tester.py**: Utility for testing prompts with real/mock OpenAI API

## 🚀 Running Tests

### Run all LLM intent tests:
```bash
pytest tests/llm_intent_tests/ -v
```

### Run specific test category:
```bash
# Unit tests only
pytest tests/llm_intent_tests/unit/ -v

# Integration tests only
pytest tests/llm_intent_tests/integration/ -v

# Prompt tests only
pytest tests/llm_intent_tests/prompts/ -v
```

### Run specific test file:
```bash
pytest tests/llm_intent_tests/unit/test_prompt_loader.py -v
```

## 📝 Test Coverage

- ✅ Prompt loading and caching
- ✅ Message building with system prompt + few-shot examples
- ✅ Response parsing (JSON, code blocks, dict formats)
- ✅ Error handling and fallbacks
- ✅ OpenAI client initialization
- ✅ Circuit breaker functionality
- ✅ Request handler integration
- ✅ Prompt validation

## 🔧 Prerequisites

- OpenAI API key in `.env` (for tests that make real API calls)
- All dependencies installed (`pip install -r requirements.txt`)

## 📊 Test Status

All tests should pass with the current implementation. If any tests fail, check:
1. OpenAI API key is set (for API tests)
2. Prompt files exist in `app/ai/llm_intent/prompts/`
3. All dependencies are installed

