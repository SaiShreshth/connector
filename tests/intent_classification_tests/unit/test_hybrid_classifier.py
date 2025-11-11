"""
Unit tests for HybridClassifier
Tests the blending logic in isolation.
"""
import unittest
from app.ai.intent_classification.hybrid_classifier import HybridClassifier


class TestHybridClassifier(unittest.TestCase):
    """Test HybridClassifier blending logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = HybridClassifier(kw_weight=0.6, emb_weight=0.4)
    
    def test_blend_empty_results(self):
        """Test blending when both inputs are empty."""
        result = self.classifier.blend([], [])
        self.assertEqual(result, [])
    
    def test_blend_keyword_only(self):
        """Test blending when only keyword results exist."""
        kw_results = [
            {"id": "ADD_TO_CART", "score": 0.9, "source": "keyword"}
        ]
        result = self.classifier.blend(kw_results, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "ADD_TO_CART")
        self.assertEqual(result[0]["score"], 0.9)
    
    def test_blend_embedding_only(self):
        """Test blending when only embedding results exist."""
        emb_results = [
            {"id": "PRODUCT_INFO", "score": 0.85, "source": "embedding"}
        ]
        result = self.classifier.blend([], emb_results)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "PRODUCT_INFO")
        self.assertEqual(result[0]["score"], 0.85)
    
    def test_blend_same_intent_both_methods(self):
        """Test blending when same intent appears in both methods."""
        kw_results = [
            {"id": "ADD_TO_CART", "score": 0.8, "source": "keyword", "matched_text": "add to cart"}
        ]
        emb_results = [
            {"id": "ADD_TO_CART", "score": 0.9, "source": "embedding"}
        ]
        result = self.classifier.blend(kw_results, emb_results)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "ADD_TO_CART")
        # Blended score: 0.8 * 0.6 + 0.9 * 0.4 = 0.48 + 0.36 = 0.84
        expected_score = 0.8 * 0.6 + 0.9 * 0.4
        self.assertAlmostEqual(result[0]["score"], expected_score, places=4)
        self.assertEqual(result[0]["source"], "blended")
        self.assertEqual(result[0]["keyword_score"], 0.8)
        self.assertEqual(result[0]["embedding_score"], 0.9)
        # Base result comes from embedding (higher score 0.9 > 0.8), so matched_text won't be present
        # This is expected behavior - higher-scoring method's fields are preserved
    
    def test_blend_different_intents(self):
        """Test blending when different intents appear in each method."""
        kw_results = [
            {"id": "ADD_TO_CART", "score": 0.8, "source": "keyword"}
        ]
        emb_results = [
            {"id": "PRODUCT_INFO", "score": 0.7, "source": "embedding"}
        ]
        result = self.classifier.blend(kw_results, emb_results)
        
        self.assertEqual(len(result), 2)
        # Should be sorted by blended score (descending)
        # ADD_TO_CART: 0.8 * 0.6 + 0.0 * 0.4 = 0.48
        # PRODUCT_INFO: 0.0 * 0.6 + 0.7 * 0.4 = 0.28
        self.assertEqual(result[0]["id"], "ADD_TO_CART")
        self.assertEqual(result[1]["id"], "PRODUCT_INFO")
    
    def test_blend_multiple_results_sorted(self):
        """Test that blended results are sorted by score descending."""
        kw_results = [
            {"id": "INTENT_A", "score": 0.9, "source": "keyword"},
            {"id": "INTENT_B", "score": 0.7, "source": "keyword"}
        ]
        emb_results = [
            {"id": "INTENT_A", "score": 0.8, "source": "embedding"},
            {"id": "INTENT_C", "score": 0.6, "source": "embedding"}
        ]
        result = self.classifier.blend(kw_results, emb_results)
        
        # INTENT_A: 0.9 * 0.6 + 0.8 * 0.4 = 0.54 + 0.32 = 0.86 (highest)
        # INTENT_B: 0.7 * 0.6 + 0.0 * 0.4 = 0.42
        # INTENT_C: 0.0 * 0.6 + 0.6 * 0.4 = 0.24
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["id"], "INTENT_A")
        self.assertGreater(result[0]["score"], result[1]["score"])
        self.assertGreater(result[1]["score"], result[2]["score"])
    
    def test_blend_conflict_resolution_higher_score_wins(self):
        """Test that when same intent appears in both, base result comes from higher-scoring method."""
        kw_results = [
            {"id": "PRODUCT_INFO", "score": 0.6, "source": "keyword", "match_type": "exact"}
        ]
        emb_results = [
            {"id": "PRODUCT_INFO", "score": 0.9, "source": "embedding", "match_type": "semantic"}
        ]
        result = self.classifier.blend(kw_results, emb_results)
        
        # Embedding score (0.9) > keyword score (0.6), so base should be from embedding
        self.assertEqual(result[0].get("match_type"), "semantic")  # From embedding result
    
    def test_update_weights(self):
        """Test hot-reload weight updating."""
        self.classifier.update_weights(0.8, 0.2)
        self.assertEqual(self.classifier.kw_weight, 0.8)
        self.assertEqual(self.classifier.emb_weight, 0.2)
        
        # Verify new weights are used
        kw_results = [{"id": "TEST", "score": 0.8, "source": "keyword"}]
        emb_results = [{"id": "TEST", "score": 0.6, "source": "embedding"}]
        result = self.classifier.blend(kw_results, emb_results)
        # New blended: 0.8 * 0.8 + 0.6 * 0.2 = 0.64 + 0.12 = 0.76
        expected = 0.8 * 0.8 + 0.6 * 0.2
        self.assertAlmostEqual(result[0]["score"], expected, places=4)


if __name__ == "__main__":
    unittest.main()

