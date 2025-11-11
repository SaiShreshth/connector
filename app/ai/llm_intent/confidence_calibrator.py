"""Confidence calibration rules governing when the LLM should activate."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Set
import json
from pathlib import Path
from loguru import logger

# ---------------------------------------------------------------------------
# Prompt versioning & schema validation
# ---------------------------------------------------------------------------

PROMPT_VERSION = "v1.0.0"
"""Current prompt template and few-shot example version used in LLM classification."""

def validate_prompt_schema(prompt_json: list[dict]) -> bool:
    """
    Basic runtime JSON schema validation for few-shot examples.
    Ensures consistent fields for CI integration and downstream analytics.
    """
    required_keys = {"user", "assistant"}
    required_assistant_keys = {
        "action_code",
        "confidence",
        "reasoning",
        "secondary_intents",
        "entities_extracted",
    }

    for i, example in enumerate(prompt_json):
        if not required_keys.issubset(example.keys()):
            raise ValueError(f"[SchemaError] Example {i} missing required keys: {example}")
        assistant = example["assistant"]
        if not required_assistant_keys.issubset(assistant.keys()):
            missing = required_assistant_keys - assistant.keys()
            raise ValueError(f"[SchemaError] Example {i} missing assistant fields: {missing}")
    return True


# ---------------------------------------------------------------------------
# Trigger configuration
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_THRESHOLD: float = 0.50
"""Invoke LLM when rule-based confidence falls below this value."""

AMBIGUITY_DELTA: float = 0.10
"""Escalate if top-two rule-based scores differ by less than this delta."""

RULE_BASED_ESCALATION_CODES: Set[str] = {
    "UNKNOWN_INTENT",
    "PAYMENT_FAILED_HELP",
    "ORDER_CANCEL",
    "REPORT_FRAUD",
    "PERSONALIZED_RECOMMENDATIONS",
    "INITIATE_RETURN",
    "CONTACT_SUPPORT",
}
"""Action codes that always require LLM review due to high business risk."""


@dataclass(frozen=True)
class TriggerContext:
    """Snapshot of rule-based classifier metrics for escalation checks."""

    top_confidence: float
    next_best_confidence: float
    action_code: Optional[str]
    is_fallback: bool


def should_trigger_llm(ctx: TriggerContext) -> tuple[bool, str]:
    """Evaluate whether the LLM classifier must be activated.

    Returns:
        tuple[bool, str]: (should_trigger, reason)
        where reason describes the trigger for observability and analytics.
    """

    if ctx.action_code in RULE_BASED_ESCALATION_CODES:
        return True, "escalation_action_code"

    if ctx.is_fallback:
        return True, "rule_based_fallback"

    if ctx.top_confidence < LOW_CONFIDENCE_THRESHOLD:
        return True, "low_confidence_threshold"

    if (ctx.top_confidence - ctx.next_best_confidence) < AMBIGUITY_DELTA:
        return True, "ambiguous_ranking"

    return False, "sufficient_confidence"


# ---------------------------------------------------------------------------
# CLI entry point for schema + prompt validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prompt_path = Path(__file__).parent / "prompts" / f"few_shot_examples_{PROMPT_VERSION}.json"
    if not prompt_path.exists():
        logger.warning(f"Prompt file not found at: {prompt_path}")
    else:
        data = json.load(open(prompt_path, "r", encoding="utf-8"))
        try:
            validate_prompt_schema(data)
            logger.info(f"Schema validation passed for {prompt_path.name} ({len(data)} examples)")
            logger.info(f"Prompt version: {PROMPT_VERSION}")
        except ValueError as e:
            logger.error(f"Schema validation failed: {e}")


__all__ = [
    "PROMPT_VERSION",
    "validate_prompt_schema",
    "LOW_CONFIDENCE_THRESHOLD",
    "AMBIGUITY_DELTA",
    "RULE_BASED_ESCALATION_CODES",
    "TriggerContext",
    "should_trigger_llm",
]
