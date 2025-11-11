import unittest
from app.ai.intent_classification.keyword_matcher import match_keywords

class TestKeywordSpecialChars(unittest.TestCase):
    def test_punctuation_ignored(self):
        text = "Add to cart!!!"
        matches = match_keywords(text)
        intents = [m.get("intent") for m in matches]
        self.assertIn("ADD_TO_CART", intents)

    def test_hyphenated_and_spaced_variants(self):
        text1 = "e-mail receipt"
        text2 = "email receipt"
        m1 = match_keywords(text1)
        m2 = match_keywords(text2)
        self.assertTrue(m1 or m2)

    def test_unicode_and_accents(self):
        text = "add to cart – now"  # en-dash
        matches = match_keywords(text)
        intents = [m.get("intent") for m in matches]
        self.assertIn("ADD_TO_CART", intents)

    def test_symbols_and_noise(self):
        text = "$$$ apply coupon SAVE20 ###"
        matches = match_keywords(text)
        intents = [m.get("intent") for m in matches]
        self.assertIn("APPLY_COUPON", intents)

if __name__ == "__main__":
    unittest.main()
