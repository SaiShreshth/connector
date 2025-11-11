import os
import json
import time
import csv
import random
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------------------------
# ✅ Load environment variables
# -------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_REAL_API = OPENAI_API_KEY and not OPENAI_API_KEY.startswith("dummy") and not OPENAI_API_KEY.startswith("sk-your")

# -------------------------------------------------------------------
# ✅ OpenAI setup (only if real API key is available)
# -------------------------------------------------------------------
if USE_REAL_API:
    from openai import OpenAI
    client = OpenAI()
    print("🔗 Using real OpenAI API connection.")
else:
    print("⚙️  Running in OFFLINE MOCK MODE (no valid API key detected).")

# -------------------------------------------------------------------
# ✅ Load system prompt & few-shot examples
# -------------------------------------------------------------------
SYSTEM_PROMPT_PATH = "app/ai/llm_intent/prompts/system_prompt_v1.0.0.txt"
FEW_SHOT_PATH = "app/ai/llm_intent/prompts/few_shot_examples_v1.0.0.json"

if not os.path.exists(SYSTEM_PROMPT_PATH):
    raise FileNotFoundError(f"❌ Missing system prompt: {SYSTEM_PROMPT_PATH}")

if not os.path.exists(FEW_SHOT_PATH):
    raise FileNotFoundError(f"❌ Missing few-shot examples: {FEW_SHOT_PATH}")

SYSTEM_PROMPT = open(SYSTEM_PROMPT_PATH, encoding="utf-8").read()
FEW_SHOT_EXAMPLES = json.load(open(FEW_SHOT_PATH, encoding="utf-8"))

# -------------------------------------------------------------------
# ✅ Define Test Cases
# -------------------------------------------------------------------
TEST_CASES = [
    {"q": "Show me running shoes under $100", "expected": "SEARCH_PRICE_RANGE"},
    {"q": "Add this to my cart", "expected": "ADD_TO_CART"},
    {"q": "Apply coupon SAVE20", "expected": "APPLY_COUPON"},
    {"q": "Where is my order ORD123?", "expected": "ORDER_STATUS"},
    {"q": "I want to return this item", "expected": "INITIATE_RETURN"},
    {"q": "Notify me when this is back in stock", "expected": "SUBSCRIBE_PRODUCT"},
    {"q": "There was a suspicious charge on my card", "expected": "REPORT_FRAUD"},
    {"q": "What do you recommend for runners?", "expected": "PERSONALIZED_RECOMMENDATIONS"},
    {"q": "Help me reset my password", "expected": "PASSWORD_RESET"},
    {"q": "Show me my cart", "expected": "ADD_TO_CART"}
]

# -------------------------------------------------------------------
# ✅ Build few-shot message structure
# -------------------------------------------------------------------
def build_messages(query):
    """Constructs messages with system + few-shot + query."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["assistant"])})

    messages.append({"role": "user", "content": query})
    return messages

# -------------------------------------------------------------------
# ✅ Main GPT handler (real or mock)
# -------------------------------------------------------------------
def call_gpt(messages):
    """Switch between GPT-4 API and offline mock classifier."""
    if USE_REAL_API:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
            messages=messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.0)),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 400))
        )
        return response.choices[0].message.content

    # -------------------------------------------
    # Mock mode – offline local intent detection
    # -------------------------------------------
    query = messages[-1]["content"].lower()
    intents = {
        # Cart & Checkout
        "cart": "ADD_TO_CART",
        "basket": "ADD_TO_CART",
        "checkout": "CHECKOUT_INITIATE",
        "buy now": "CHECKOUT_INITIATE",
        "purchase": "CHECKOUT_INITIATE",

        # Orders & Returns
        "order": "ORDER_STATUS",
        "track": "ORDER_STATUS",
        "delivery": "ORDER_STATUS",
        "return": "INITIATE_RETURN",
        "refund": "INITIATE_RETURN",
        "cancel": "ORDER_CANCEL",

        # Coupons & Discounts
        "coupon": "APPLY_COUPON",
        "discount": "APPLY_COUPON",
        "offer": "APPLY_COUPON",
        "promo": "APPLY_COUPON",
        "code": "APPLY_COUPON",

        # Notifications
        "notify": "SUBSCRIBE_PRODUCT",
        "stock": "SUBSCRIBE_PRODUCT",
        "back in stock": "SUBSCRIBE_PRODUCT",
        "alert": "SUBSCRIBE_PRODUCT",

        # Security & Fraud
        "fraud": "REPORT_FRAUD",
        "charge": "REPORT_FRAUD",
        "unauthorized": "REPORT_FRAUD",
        "suspicious": "REPORT_FRAUD",

        # Recommendations
        "recommend": "PERSONALIZED_RECOMMENDATIONS",
        "suggest": "PERSONALIZED_RECOMMENDATIONS",
        "best": "PERSONALIZED_RECOMMENDATIONS",

        # Account
        "password": "PASSWORD_RESET",
        "login": "PASSWORD_RESET",
        "sign in": "PASSWORD_RESET",
        "profile": "ACCOUNT_PROFILE",

        # Support
        "support": "CONTACT_SUPPORT",
        "help": "CONTACT_SUPPORT",
        "customer service": "CONTACT_SUPPORT",

        # Search & Discovery
        "find": "SEARCH_PRICE_RANGE",
        "show": "SEARCH_PRICE_RANGE",
        "search": "SEARCH_PRICE_RANGE",
        "look for": "SEARCH_PRICE_RANGE",
        "browse": "SEARCH_PRICE_RANGE"
    }

    predicted = next((v for k, v in intents.items() if k in query), "HELP_GENERAL")
    mock = {
        "action_code": predicted,
        "confidence": round(random.uniform(0.75, 0.99), 2),
        "reasoning": f"Offline mock — detected intent: {predicted}",
        "secondary_intents": [],
        "entities_extracted": []
    }
    return json.dumps(mock)

# -------------------------------------------------------------------
# ✅ JSON parsing helper
# -------------------------------------------------------------------
def safe_parse_json(raw):
    """Extracts JSON safely from model output."""
    try:
        return json.loads(raw)
    except Exception:
        import re
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

# -------------------------------------------------------------------
# ✅ Test runner
# -------------------------------------------------------------------
def run_tests():
    """Run all test cases and generate reports."""
    correct = 0
    results = []

    print("\n🧠 Running Intent Classification Prompt Tests...\n")

    for i, case in enumerate(TEST_CASES, 1):
        query = case["q"]
        expected = case["expected"]

        try:
            msgs = build_messages(query)
            raw = call_gpt(msgs)
            parsed = safe_parse_json(raw)

            pred = parsed.get("action_code", "PARSE_ERROR") if parsed else "PARSE_ERROR"
            conf = parsed.get("confidence", None) if parsed else None

            ok = pred.strip().upper() == expected.strip().upper()
            if ok:
                correct += 1

            result = {
                "id": i,
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "expected": expected,
                "predicted": pred,
                "confidence": conf,
                "correct": ok,
                "reasoning": parsed.get("reasoning") if parsed else None
            }
            results.append(result)

            print(f"[{i:02}] {query[:40]:40} → {pred:25} | {'✅' if ok else '❌'}")

        except Exception as e:
            print(f"[{i:02}] ERROR for query: {query} → {e}")
            results.append({
                "id": i,
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "expected": expected,
                "predicted": "ERROR",
                "confidence": None,
                "correct": False,
                "reasoning": str(e)
            })

        time.sleep(0.2)

    accuracy = correct / len(TEST_CASES)
    print(f"\n🎯 Accuracy: {accuracy:.2%}")

    # ----------------------------------------------------------------
    # ✅ Save results inside 'results' folder (only change made here)
    # ----------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "prompt_test_results.csv")
    json_path = os.path.join("results", "prompt_test_results.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📁 Results saved to:\n - {csv_path}\n - {json_path}\n")

# -------------------------------------------------------------------
# ✅ Main entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_tests()
