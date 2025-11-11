"""
Tests for Status Store (TASK-16)

Tests both InMemoryStatusStore and RedisStatusStore implementations.
"""

import unittest
import uuid
from datetime import datetime, timezone
from app.core.status_store import InMemoryStatusStore, RedisStatusStore
from app.schemas.request_status import RequestStatus, ResultSchema


class TestInMemoryStatusStore(unittest.TestCase):
    """Test InMemoryStatusStore implementation."""
    
    def setUp(self):
        self.store = InMemoryStatusStore()

    def test_add_and_get_request_status(self):
        rid = str(uuid.uuid4())
        status_data = RequestStatus(
            request_id=rid,
            status="QUEUED",
            queued_at=datetime.now(timezone.utc)
        )
        self.store.save(status_data)
        fetched = self.store.get(rid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.request_id, rid)
        self.assertEqual(fetched.status, "QUEUED")

    def test_update_status(self):
        rid = str(uuid.uuid4())
        status_data = RequestStatus(
            request_id=rid,
            status="QUEUED",
            queued_at=datetime.now(timezone.utc)
        )
        self.store.save(status_data)
        self.store.update_status(rid, "PROCESSING")
        updated = self.store.get(rid)
        self.assertEqual(updated.status, "PROCESSING")
        self.assertIsNotNone(updated.started_at)

    def test_complete_request_with_result(self):
        rid = str(uuid.uuid4())
        status_data = RequestStatus(
            request_id=rid,
            status="PROCESSING",
            started_at=datetime.now(timezone.utc)
        )
        self.store.save(status_data)

        result = ResultSchema(action_code="SEARCH", confidence=0.95, entities={"query": "laptop"})
        self.store.complete_request(rid, result)
        completed = self.store.get(rid)
        self.assertEqual(completed.status, "COMPLETED")
        self.assertIsNotNone(completed.completed_at)
        self.assertIsNotNone(completed.result)
        self.assertEqual(completed.result.action_code, "SEARCH")
        self.assertAlmostEqual(completed.result.confidence, 0.95)

    def test_fail_request(self):
        rid = str(uuid.uuid4())
        status_data = RequestStatus(
            request_id=rid,
            status="PROCESSING",
            started_at=datetime.now(timezone.utc)
        )
        self.store.save(status_data)
        
        error = {"type": "LLM_ERROR", "message": "API timeout"}
        self.store.fail_request(rid, error)
        failed = self.store.get(rid)
        self.assertEqual(failed.status, "FAILED")
        self.assertIsNotNone(failed.error)
        self.assertEqual(failed.error["type"], "LLM_ERROR")


class TestRedisStatusStore(unittest.TestCase):
    """Test RedisStatusStore implementation."""
    
    def setUp(self):
        try:
            self.store = RedisStatusStore(fail_silently=True)
        except Exception as e:
            self.skipTest(f"Redis not available: {e}")

    def test_add_and_get_request_status(self):
        rid = str(uuid.uuid4())
        status_data = RequestStatus(
            request_id=rid,
            status="QUEUED",
            queued_at=datetime.now(timezone.utc)
        )
        self.store.save(status_data)
        fetched = self.store.get(rid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.request_id, rid)
        self.assertEqual(fetched.status, "QUEUED")

    def test_update_status_timestamps(self):
        rid = str(uuid.uuid4())
        status_data = RequestStatus(
            request_id=rid,
            status="QUEUED",
            queued_at=datetime.now(timezone.utc)
        )
        self.store.save(status_data)
        
        # Update to PROCESSING - should set started_at
        self.store.update_status(rid, "PROCESSING")
        processing = self.store.get(rid)
        self.assertEqual(processing.status, "PROCESSING")
        self.assertIsNotNone(processing.started_at)
        
        # Update to COMPLETED - should set completed_at
        result = ResultSchema(action_code="SEARCH", confidence=0.9)
        self.store.complete_request(rid, result)
        completed = self.store.get(rid)
        self.assertEqual(completed.status, "COMPLETED")
        self.assertIsNotNone(completed.completed_at)
        self.assertIsNotNone(completed.result)

    def test_status_lifecycle(self):
        """Test complete status lifecycle: QUEUED → PROCESSING → COMPLETED"""
        rid = str(uuid.uuid4())
        
        # Create QUEUED status
        queued = RequestStatus(
            request_id=rid,
            status="QUEUED",
            queued_at=datetime.now(timezone.utc)
        )
        self.store.save(queued)
        
        # Update to PROCESSING
        self.store.update_status(rid, "PROCESSING")
        processing = self.store.get(rid)
        self.assertEqual(processing.status, "PROCESSING")
        self.assertIsNotNone(processing.started_at)
        
        # Complete with result
        result = ResultSchema(action_code="ADD_TO_CART", confidence=0.87, entities={"product": "shoes"})
        self.store.complete_request(rid, result)
        completed = self.store.get(rid)
        self.assertEqual(completed.status, "COMPLETED")
        self.assertEqual(completed.result.action_code, "ADD_TO_CART")
        self.assertIsNotNone(completed.completed_at)


if __name__ == "__main__":
    print("=== Running Status Store Tests ===")
    unittest.main()

