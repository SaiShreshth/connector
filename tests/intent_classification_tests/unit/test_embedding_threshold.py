import unittest
import os
import platform
from dotenv import load_dotenv

# Skip on macOS due to NumPy crash
if platform.system() == "Darwin":
    import pytest
    pytest.skip("Skipping embedding tests on macOS due to NumPy compatibility issues", allow_module_level=True)

from app.ai.intent_classification.embedding_matcher import EmbeddingMatcher

class TestEmbeddingThreshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize matcher with Qdrant if available, otherwise use mock."""
        # Load environment variables
        load_dotenv()
        
        # Get Qdrant config from env
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        # Try to connect to Qdrant
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=qdrant_host, port=qdrant_port)
            # Test connection
            client.get_collections()
            cls.matcher = EmbeddingMatcher(client=client)
            cls.has_qdrant = True
            print(f"✅ Connected to Qdrant at {qdrant_host}:{qdrant_port} - using real embeddings")
        except Exception as e:
            # Graceful degradation: use mock
            cls.matcher = EmbeddingMatcher()
            cls.has_qdrant = False
            print(f"⚠️  Qdrant not available ({e}) - using mock results")

    def test_above_threshold_returns_results(self):
        results = self.matcher.search("add this to my cart", threshold=0.5)
        self.assertTrue(results, "Expected results above threshold")
        self.assertGreaterEqual(results[0]["score"], 0.1)

    def test_below_threshold_filters_out(self):
        """Test that low-quality queries return low scores (requires Qdrant)."""
        if not self.has_qdrant:
            # Graceful degradation: skip this test when using mock
            self.skipTest("Skipping: Qdrant not available (mock always returns 0.95)")
        
        # Use a gibberish query unlikely to match
        results = self.matcher.search("zzzzzz qwerty asdfgh", threshold=0.95)
        # With real embeddings, gibberish should score low
        if results:
            self.assertLess(results[0]["score"], 0.95)

if __name__ == "__main__":
    unittest.main()
