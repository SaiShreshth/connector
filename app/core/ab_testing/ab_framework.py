"""
A/B Testing Framework for chatNShop intent classification system.
NOW: Variants compare real models (GPT-4o vs GPT-4o-mini vs GPT-3.5 baseline).
Uses config_manager.py for centralized configuration.
"""

import json
import random
import uuid
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

# === Import from centralized config manager ===
from app.core.config_manager import (
    OPENAI_API_KEY,
    GPT4_MODEL,
    GPT4_TURBO_MODEL,
    GPT35_MODEL,
    DRY_RUN,
)

# === Initialize OpenAI client ===
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# === Event log file ===
LOG_FILE = Path("app/core/ab_testing/ab_events.jsonl")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# === Variants configuration (A/B/C) ===
VARIANTS = {
    "gpt4": {"model": GPT4_MODEL, "description": "GPT-4o (high accuracy, balanced speed)"},
    "gpt4_turbo": {"model": GPT4_TURBO_MODEL, "description": "GPT-4o-mini (fast and cost-efficient)"},
    "gpt35": {"model": GPT35_MODEL, "description": "GPT-3.5 baseline model"},
}

# === Updated cost estimation (realistic pricing 2025) ===
def estimate_cost(model_name: str, tokens: int) -> float:
    """
    Estimate approximate cost per 1K tokens (input + output).
    These are approximate OpenAI 2025 rates.
    """
    pricing = {
        GPT4_MODEL: 0.005,        # GPT-4o
        GPT4_TURBO_MODEL: 0.0006, # GPT-4o-mini
        GPT35_MODEL: 0.0006,      # GPT-3.5 baseline (similar to mini)
    }
    rate = pricing.get(model_name, 0.001)
    return round((tokens / 1000) * rate, 6)

# === Model calling or dry-run simulation ===
def run_llm_inference(prompt: str, model_name: str):
    """
    Run OpenAI model inference or simulate it (if DRY_RUN=True).
    Returns: (response_text, latency_ms, cost_usd, prompt_tokens, completion_tokens)
    """
    start = time.perf_counter()

    if DRY_RUN:
        # Simulate latency and cost
        time.sleep(random.uniform(0.05, 0.25))
        latency_ms = (time.perf_counter() - start) * 1000
        prompt_tokens = max(1, min(50, len(prompt.split())))
        completion_tokens = max(1, int(prompt_tokens * 0.6))
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = estimate_cost(model_name, total_tokens)
        fake_response = f"[DRY-RUN] Simulated response from {model_name}: '{prompt}' processed."
        return fake_response, latency_ms, cost_usd, prompt_tokens, completion_tokens

    if not client:
        raise RuntimeError("❌ OpenAI client not initialized. Ensure OPENAI_API_KEY is set in your .env file.")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Extract response text safely
        message = response.choices[0].message.content.strip() if response.choices else ""
        usage = getattr(response, "usage", {})
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = estimate_cost(model_name, total_tokens)

        return message, latency_ms, cost_usd, prompt_tokens, completion_tokens

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return f"[Error: {e}]", latency_ms, 0.0, 0, 0

# === Variant chooser ===
def choose_variant(user_id: int) -> str:
    """
    Weighted random selection between models.
    Default weights: GPT-4o (0.6), GPT-4o-mini (0.25), GPT-3.5 (0.15)
    """
    keys = list(VARIANTS.keys())
    weights = [0.6, 0.25, 0.15]
    return random.choices(keys, weights=weights, k=1)[0]

# === Log utility ===
def log_event(event: dict):
    """Append experiment log to JSONL file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        json.dump(event, f)
        f.write("\n")

# === Core simulation ===
def run_simulation(user_input: str, user_id: int, label: str):
    """Run one inference or dry-run and log results."""
    variant = choose_variant(user_id)
    model_name = VARIANTS[variant]["model"]

    request_id = str(uuid.uuid4())
    ts_utc = datetime.now(timezone.utc).isoformat()

    response_text, latency_ms, cost_usd, prompt_tokens, completion_tokens = run_llm_inference(
        user_input, model_name
    )

    total_tokens = prompt_tokens + completion_tokens
    event = {
        "request_id": request_id,
        "ts_utc": ts_utc,
        "user_id": user_id,
        "variant": variant,
        "model": model_name,
        "prompt": user_input,
        "response": response_text,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "result_label": label,
    }

    log_event(event)

    print("\n✅ Logged event:")
    for k, v in event.items():
        print(f"{k}: {v}")

# === CLI interface ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A/B test between GPT variants.")
    parser.add_argument("--simulate", type=str, help="User query to simulate.")
    parser.add_argument("--user-id", type=int, help="User ID.")
    parser.add_argument("--label", type=str, choices=["success", "fail"], help="Result label.")

    args = parser.parse_args()

    if args.simulate and args.user_id is not None and args.label:
        run_simulation(args.simulate, args.user_id, args.label)
    else:
        print("Usage:")
        print('  python -m app.core.ab_testing.ab_framework --simulate "Find best running shoes" --user-id 101 --label success')
