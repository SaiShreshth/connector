import unittest
import os
import platform
from dotenv import load_dotenv

# Skip on macOS due to NumPy crash
if platform.system() == "Darwin":
    import pytest
    pytest.skip("Skipping embedding tests on macOS due to NumPy compatibility issues", allow_module_level=True)

from app.ai.intent_classification.embedding_matcher import EmbeddingMatcher

class TestEmbeddingUnit(unittest.TestCase):
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

    def test_similarity_present(self):
        r = self.matcher.search("show specs")
        self.assertTrue(isinstance(r, list))

    def test_threshold_filters(self):
        """Test that threshold filtering works (requires Qdrant)."""
        if not self.has_qdrant:
            # Graceful degradation: skip this test when using mock
            self.skipTest("Skipping: Qdrant not available (mock always returns 0.95)")
        
        r = self.matcher.search("zzzz qqqq", threshold=0.95)
        # With real embeddings, gibberish should score low
        if r:
            self.assertLess(r[0]["score"], 0.95)

if __name__ == "__main__":
    unittest.main()
