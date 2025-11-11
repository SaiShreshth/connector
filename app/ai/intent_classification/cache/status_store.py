# app/ai/intent_classification/cache/status_store.py

import sqlite3
import os
from threading import Lock
from typing import Dict
from datetime import datetime

DB_PATH = os.path.join("data", "cost_usage.db")

class StatusStore:
    """
    Thread-safe status + cost tracker.
    Tracks:
        - Request status (pending/completed/error)
        - Token usage (prompt + completion)
        - Cost per request
        - Daily/monthly summaries
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}
        self._lock = Lock()

        # ✅ ensure persistent directory + SQLite connection
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_table()

    def create_table(self):
        """Create SQLite table for request tracking"""
        query = """
        CREATE TABLE IF NOT EXISTS request_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE,
            status TEXT,
            message TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            created_at TEXT
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    # ------------------------------------------------------------------
    # 🔹 BASIC STATUS TRACKING (same as your version)
    # ------------------------------------------------------------------
    def set_status(self, request_id: str, status: str, message: str = "", result: dict = None):
        """Update or insert a request's status (both in-memory + DB)."""
        with self._lock:
            # update in-memory
            self._store[request_id] = {
                "status": status,
                "message": message,
                "result": result or {}
            }

            # persist in SQLite
            now = datetime.utcnow().isoformat()
            query = """
            INSERT INTO request_status (request_id, status, message, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(request_id)
            DO UPDATE SET status=excluded.status, message=excluded.message;
            """
            self.conn.execute(query, (request_id, status, message, now))
            self.conn.commit()

    def get_status(self, request_id: str) -> dict:
        """Retrieve from in-memory first, fallback to DB."""
        with self._lock:
            if request_id in self._store:
                return self._store[request_id]

        # fallback from DB
        cursor = self.conn.execute(
            "SELECT request_id, status, message, cost, total_tokens, created_at FROM request_status WHERE request_id=?",
            (request_id,),
        )
        row = cursor.fetchone()
        if not row:
            return {"status": "not_found", "message": "Request ID not found"}

        return {
            "request_id": row[0],
            "status": row[1],
            "message": row[2],
            "cost": row[3],
            "total_tokens": row[4],
            "created_at": row[5],
        }

    def delete_status(self, request_id: str):
        """Delete from memory + DB"""
        with self._lock:
            if request_id in self._store:
                del self._store[request_id]
        self.conn.execute("DELETE FROM request_status WHERE request_id=?", (request_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # 🔹 COST + TOKEN TRACKING
    # ------------------------------------------------------------------
    def log_usage(self, request_id: str, prompt_tokens: int, completion_tokens: int, cost: float):
        """Store token and cost metrics for a given request."""
        total = prompt_tokens + completion_tokens
        query = """
        UPDATE request_status
        SET prompt_tokens=?, completion_tokens=?, total_tokens=?, cost=?
        WHERE request_id=?
        """
        self.conn.execute(query, (prompt_tokens, completion_tokens, total, cost, request_id))
        self.conn.commit()

    # ------------------------------------------------------------------
    # 🔹 COST SUMMARIES
    # ------------------------------------------------------------------
    def get_daily_total(self):
        """Total cost today"""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        query = "SELECT SUM(cost) FROM request_status WHERE DATE(created_at) = ?"
        result = self.conn.execute(query, (today,)).fetchone()
        return round(result[0] or 0.0, 6)

    def get_monthly_total(self):
        """Total cost this month"""
        month = datetime.utcnow().strftime("%Y-%m")
        query = "SELECT SUM(cost) FROM request_status WHERE strftime('%Y-%m', created_at) = ?"
        result = self.conn.execute(query, (month,)).fetchone()
        return round(result[0] or 0.0, 6)
