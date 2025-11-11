import os
import json
import unittest
from app.ai.intent_classification.ambiguity_resolver import detect_intent, LOG_FILE

class TestAmbiguityResolver(unittest.TestCase):
    def setUp(self):
        # Clean prior log file
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

    def test_returns_ambiguous_with_multiple_high_confidence(self):
        # Phrase likely to trigger PRODUCT_INFO and ADD_TO_CART
        q = "show specs of iPhone and add to cart"
        out = detect_intent(q)
        self.assertEqual(out.get("action"), "AMBIGUOUS")
        poss = out.get("possible_intents", {})
        self.assertTrue(isinstance(poss, dict) and len(poss) >= 2)

    def test_logs_persisted_as_jsonl(self):
        q = "help me and add to cart"
        _ = detect_intent(q)
        self.assertTrue(os.path.exists(LOG_FILE))
        with open(LOG_FILE, "r", encoding="utf-8") as fh:
            line = fh.readline().strip()
            self.assertTrue(line)
            entry = json.loads(line)
            self.assertIn("user_input", entry)
            self.assertIn("intent_scores", entry)

if __name__ == "__main__":
    unittest.main()
