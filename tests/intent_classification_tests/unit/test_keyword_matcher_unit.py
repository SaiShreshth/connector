import unittest
from app.ai.intent_classification.keyword_matcher import match_keywords

class TestKeywordMatcherUnit(unittest.TestCase):
    def test_basic_add_to_cart(self):
        r = match_keywords("Add to cart")
        intents = [m.get("intent") for m in r]
        self.assertIn("ADD_TO_CART", intents)

    def test_empty_input(self):
        self.assertEqual(match_keywords("") , [])

    def test_special_characters(self):
        intents = [m.get("intent") for m in match_keywords("@@@ add ## to $$ cart !!!")]
        self.assertIn("ADD_TO_CART", intents)

    def test_long_input(self):
        text = ("show me running shoes under $100 and add to cart now ") * 100
        r = match_keywords(text)
        self.assertTrue(r)

    def test_scoring_order(self):
        r = match_keywords("add to cart and checkout")
        self.assertTrue(r[0]["score"] >= r[-1]["score"]) if len(r) > 1 else None

if __name__ == "__main__":
    unittest.main()
