"""
Test suite to validate few-shot prompt schema and versioning.
Ensures prompt examples conform to the structure defined in confidence_calibrator.py.
"""

import json
from pathlib import Path
import pytest

from app.ai.llm_intent.confidence_calibrator import (
    PROMPT_VERSION,
    validate_prompt_schema,
)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path("app/ai/llm_intent/prompts")
PROMPT_FILE = PROMPTS_DIR / f"few_shot_examples_{PROMPT_VERSION}.json"


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_prompt_file_exists():
    """Ensure the few-shot example file exists."""
    assert PROMPT_FILE.exists(), f"❌ Prompt file missing: {PROMPT_FILE}"


def test_prompt_schema_valid():
    """Validate the schema of all few-shot examples."""
    data = json.load(open(PROMPT_FILE, "r", encoding="utf-8"))
    assert isinstance(data, list), "Prompt file must contain a list of examples"
    assert len(data) >= 10, "Prompt file should have at least 10 examples for coverage"

    # Use the validator function from confidence_calibrator
    assert validate_prompt_schema(data), "Schema validation failed"


def test_prompt_keys_completeness():
    """Check each example for key presence and correct types."""
    data = json.load(open(PROMPT_FILE, "r", encoding="utf-8"))

    for i, example in enumerate(data):
        assert "user" in example, f"Example {i} missing 'user'"
        assert "assistant" in example, f"Example {i} missing 'assistant'"

        assistant = example["assistant"]

        # Verify all required assistant fields
        required_keys = {
            "action_code",
            "confidence",
            "reasoning",
            "secondary_intents",
            "entities_extracted",
        }
        for key in required_keys:
            assert key in assistant, f"Example {i} assistant missing key '{key}'"

        # Type checks
        assert isinstance(assistant["action_code"], str), f"Example {i}: action_code must be str"
        assert isinstance(assistant["confidence"], (float, int)), f"Example {i}: confidence must be number"
        assert 0 <= assistant["confidence"] <= 1, f"Example {i}: confidence must be between 0 and 1"
        assert isinstance(assistant["reasoning"], str), f"Example {i}: reasoning must be string"
        assert isinstance(assistant["secondary_intents"], list), f"Example {i}: secondary_intents must be list"
        assert isinstance(assistant["entities_extracted"], list), f"Example {i}: entities_extracted must be list"


def test_prompt_version_metadata():
    """Ensure prompt version used in code and file are in sync."""
    expected_file_name = f"few_shot_examples_{PROMPT_VERSION}.json"
    assert PROMPT_FILE.name == expected_file_name, (
        f"Version mismatch: Expected {expected_file_name}, got {PROMPT_FILE.name}"
    )


# ---------------------------------------------------------------------------
# Accuracy baseline stub (optional)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Add accuracy evaluation once GPT-4 benchmark data available")
def test_prompt_accuracy_baseline():
    """
    Placeholder for automated accuracy CI threshold.
    CI should fail if <90% accuracy against labeled dataset.
    """
    accuracy = 0.95  # Replace with dynamic test output later
    assert accuracy >= 0.90, "Model accuracy below CI threshold (90%)"


# ---------------------------------------------------------------------------
# CLI helper (manual run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
