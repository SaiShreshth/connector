import time
import statistics
import unittest
from app.ai.intent_classification.keyword_matcher import match_keywords

class TestKeywordBenchmark(unittest.TestCase):
    def test_p95_under_50ms(self):
        queries = [
            "add to cart",
            "apply coupon SAVE20",
            "where is my order ORD123?",
            "show me running shoes under $100",
            "compare iphone 14 and galaxy s23",
            "view my cart",
            "remove from cart",
            "checkout now",
            "track my return",
            "notify me when back in stock",
        ] * 20  # 200 invocations

        measurements = []
        # Warm-up
        for _ in range(10):
            match_keywords("warm up")
        for q in queries:
            start = time.perf_counter()
            match_keywords(q)
            end = time.perf_counter()
            measurements.append((end - start) * 1000.0)

        p95 = statistics.quantiles(measurements, n=100)[94]
        self.assertLess(p95, 50.0, f"p95 latency too high: {p95:.2f}ms")

if __name__ == "__main__":
    unittest.main()
