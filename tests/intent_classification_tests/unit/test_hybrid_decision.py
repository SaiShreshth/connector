import unittest
from app.ai.intent_classification.decision_engine import DecisionEngine

class TestHybridDecision(unittest.TestCase):
    def test_priority_rule_short_circuit(self):
        engine = DecisionEngine()
        engine.priority_threshold = 0.8
        # Prevent config manager from overriding our flags during search
        engine._load_config_from_manager = lambda: None  # type: ignore
        # Force keyword path for this test regardless of config manager state
        engine.use_keywords = True
        engine.use_embedding = False
        class KWStub:
            def search(self, q):
                return [{"id": "ADD_TO_CART", "intent": "ADD_TO_CART", "score": 0.9, "source": "keyword"}]
        class EMBStub:
            def search(self, q):
                return []
        engine.keyword_matcher = KWStub()
        engine.embedding_matcher = EMBStub()
        out = engine.search("add to cart")
        self.assertEqual(out.get("status"), "CONFIDENT_KEYWORD")
        self.assertEqual(out.get("intent", {}).get("id"), "ADD_TO_CART")

    def test_blending_conflict_resolution(self):
        engine = DecisionEngine()
        engine.kw_weight = 0.7
        engine.emb_weight = 0.3
        # Update classifier weights
        engine.hybrid_classifier.update_weights(0.7, 0.3)
        # same intent in both lists with different scores
        kw = [{"id": "PRODUCT_INFO", "intent": "PRODUCT_INFO", "score": 0.6, "source": "keyword"}]
        emb = [{"id": "PRODUCT_INFO", "intent": "PRODUCT_INFO", "score": 0.9, "source": "embedding"}]
        blended = engine.hybrid_classifier.blend(kw, emb)
        self.assertTrue(blended)
        top = blended[0]
        expected = 0.6*engine.kw_weight + 0.9*engine.emb_weight
        self.assertAlmostEqual(top.get("score"), expected, places=4)
        self.assertEqual(top.get("id"), "PRODUCT_INFO")

if __name__ == "__main__":
    unittest.main()
