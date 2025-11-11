import json
import unittest
from app.ai.intent_classification.decision_engine import get_intent_classification

class TestIntegration100Cases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("tests/intent_classification_tests/data/integration_cases.json", encoding="utf-8") as f:
            cls.cases = json.load(f)

    def test_integration_accuracy(self):
        correct = 0
        for c in self.cases:
            out = get_intent_classification(c["q"]) or {}
            intent = (out.get("intent") or {}).get("id") if isinstance(out, dict) else None
            pred = intent or "UNKNOWN"
            if pred == c["expected"]:
                correct += 1
        acc = correct / len(self.cases)
        # Production hybrid accuracy threshold
        self.assertGreaterEqual(acc, 0.85, f"Hybrid accuracy too low: {acc:.2%}")

if __name__ == "__main__":
    unittest.main()
