import json
import unittest
from app.ai.intent_classification.decision_engine import get_intent_classification

class TestAccuracyDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("tests/intent_classification_tests/data/integration_cases.json", encoding="utf-8") as f:
            cls.dataset = json.load(f)

    def test_accuracy_min_threshold(self):
        correct = 0
        for row in self.dataset:
            out = get_intent_classification(row["q"]) or {}
            intent = (out.get("intent") or {}).get("id") if isinstance(out, dict) else None
            pred = intent or "UNKNOWN"
            if pred == row["expected"]:
                correct += 1
        acc = correct / len(self.dataset)
        # Hybrid accuracy threshold for production
        self.assertGreaterEqual(acc, 0.85, f"Hybrid accuracy too low on dataset: {acc:.2%}")

if __name__ == "__main__":
    unittest.main()
