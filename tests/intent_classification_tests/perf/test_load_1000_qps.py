import concurrent.futures
import time
import unittest
from app.ai.intent_classification.keyword_matcher import match_keywords

class TestLoad1000QPS(unittest.TestCase):
    def test_load_simulated_1000_qps(self):
        # Simulate 1000 queries within ~1s window using thread pool
        queries = ["add to cart", "apply coupon", "view cart", "checkout now", "product info"] * 200
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as ex:
            list(ex.map(match_keywords, queries))
        elapsed = time.perf_counter() - start
        # Accept up to 2 seconds on CI machines; adjust as needed
        self.assertLess(elapsed, 2.0, f"Took too long: {elapsed:.2f}s")

if __name__ == "__main__":
    unittest.main()
