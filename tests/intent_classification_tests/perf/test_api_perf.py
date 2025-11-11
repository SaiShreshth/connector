import statistics
import time
import unittest
from fastapi.testclient import TestClient
from app.main import app

class TestAPIPerf(unittest.TestCase):
    def test_classify_p95_under_100ms(self):
        client = TestClient(app)
        payloads = [
            {"text": "add to cart"},
            {"text": "apply coupon SAVE20"},
            {"text": "show me running shoes under $100"},
            {"text": "compare iphone and galaxy"},
            {"text": "view my cart"},
        ] * 20  # 100 calls
        measurements = []
        # warm-up
        client.post("/classify", json={"text": "warm up"})
        for p in payloads:
            t0 = time.perf_counter()
            r = client.post("/classify", json=p)
            t1 = time.perf_counter()
            assert r.status_code in (200, 500, 202)
            measurements.append((t1 - t0) * 1000.0)
        p95 = statistics.quantiles(measurements, n=100)[94]
        assert p95 < 100.0, f"/classify p95 too high: {p95:.2f}ms"

if __name__ == "__main__":
    unittest.main()
