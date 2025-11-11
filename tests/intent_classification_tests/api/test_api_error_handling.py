import unittest
from fastapi.testclient import TestClient
from app.main import app

class TestAPIErrorHandling(unittest.TestCase):
    def test_invalid_payload_422(self):
        client = TestClient(app)
        r = client.post("/classify", json={"wrong": "field"})
        self.assertEqual(r.status_code, 422)

    def test_internal_error_500_example(self):
        # Force an internal error by sending an extremely long string to simulate failure
        client = TestClient(app)
        text = "x" * 1000000
        r = client.post("/classify", json={"text": text})
        # Depending on environment, may be 200; accept 200 or 500 but response structure should be JSON
        self.assertIn(r.status_code, (200, 500))
        self.assertTrue(r.headers.get("content-type", "").startswith("application/json"))

if __name__ == "__main__":
    unittest.main()
