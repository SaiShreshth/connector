import asyncio
import json
import os
import uuid
from typing import Any, Dict, Optional

import asyncpg
# from loguru import logger

import logging
from urllib.parse import urlparse
logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.pool.Pool] = None


async def init_pool(dsn: str) -> None:
    global _pool
    if _pool is not None:
        return
    try:
        _log_masked_dsn = _mask_dsn(dsn)
        logger.info(f"Initializing PostgreSQL pool for DSN={_log_masked_dsn}")
        _pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
        await ensure_tables()
        logger.info("PostgreSQL pool initialized")
    except Exception as exc:
        logger.warning(f"PostgreSQL init failed, will use file fallback. Error: {exc}")
        _pool = None


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
        finally:
            _pool = None


async def ensure_tables() -> None:
    if _pool is None:
        return
    create_sql = """
    CREATE TABLE IF NOT EXISTS shopify_api_calls (
        id UUID PRIMARY KEY,
        request_ts TIMESTAMPTZ,
        response_ts TIMESTAMPTZ,
        latency_ms INTEGER,
        method TEXT,
        url TEXT,
        path TEXT,
        status_code INTEGER,
        request_headers JSONB,
        request_body TEXT,
        response_headers JSONB,
        response_body TEXT,
        error TEXT,
        correlation_id UUID
    );
    """
    async with _pool.acquire() as conn:
        await conn.execute(create_sql)


async def insert_shopify_call(log: Dict[str, Any]) -> None:
    """
    Inserts a Shopify call log into Postgres or falls back to JSONL file if PG is unavailable.
    """
    global _pool
    # Make sure we always have an id
    log = dict(log)
    log["id"] = log.get("id") or str(uuid.uuid4())
    
    corr_id = log.get("correlation_id", "unknown")
    logger.info(f"insert_shopify_call called | corr_id={corr_id} pool_exists={_pool is not None}")

    if _pool is None:
        logger.warning(f"Postgres pool is None, using JSONL fallback | corr_id={corr_id}")
        _append_jsonl_fallback(log)
        return

    insert_sql = """
    INSERT INTO shopify_api_calls (
        id, request_ts, response_ts, latency_ms, method, url, path, status_code,
        request_headers, request_body, response_headers, response_body, error, correlation_id
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11::jsonb, $12, $13, $14
    );
    """
    try:
        # Ensure headers are JSON-serializable dicts
        req_headers = log.get("request_headers") or {}
        resp_headers = log.get("response_headers") or {}
        
        # Convert to plain dict and ensure all values are strings/JSON-serializable
        def sanitize_headers(hdrs):
            if not isinstance(hdrs, dict):
                return {}
            return {str(k): str(v) if v is not None else "" for k, v in hdrs.items()}
        
        req_headers_json = json.dumps(sanitize_headers(req_headers), ensure_ascii=False)
        resp_headers_json = json.dumps(sanitize_headers(resp_headers), ensure_ascii=False)
        
        logger.info(f"About to execute INSERT | corr_id={corr_id} status={log.get('status_code')}")
        try:
            async with _pool.acquire() as conn:
                logger.info(f"Acquired connection from pool | corr_id={corr_id}")
                await conn.execute(
                    insert_sql,
                    log.get("id"),
                    log.get("request_ts"),
                    log.get("response_ts"),
                    log.get("latency_ms"),
                    log.get("method"),
                    log.get("url"),
                    log.get("path"),
                    log.get("status_code"),
                    req_headers_json,
                    log.get("request_body"),
                    resp_headers_json,
                    log.get("response_body"),
                    log.get("error"),
                    log.get("correlation_id"),
                )
                logger.info(f"INSERT executed successfully | corr_id={corr_id}")
        except Exception as db_exc:
            logger.error(f"Database INSERT failed | corr_id={corr_id} error={db_exc}", exc_info=True)
            raise
        logger.info(
            "Shopify call logged to Postgres | corr_id=%s method=%s path=%s status=%s latency_ms=%s",
            log.get("correlation_id"),
            log.get("method"),
            log.get("path"),
            log.get("status_code"),
            log.get("latency_ms"),
        )
    except Exception as exc:
        logger.warning(f"Failed to insert Shopify log into PG, using file fallback. Error: {exc}")
        _append_jsonl_fallback(log)


def _append_jsonl_fallback(record: Dict[str, Any]) -> None:
    try:
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", "shopify_api_fallback.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Shopify call logged to JSONL fallback | file=%s corr_id=%s method=%s path=%s status=%s latency_ms=%s",
            path,
            record.get("correlation_id"),
            record.get("method"),
            record.get("path"),
            record.get("status_code"),
            record.get("latency_ms"),
        )
    except Exception as exc:
        logger.error(f"Failed to write Shopify log to JSONL fallback: {exc}")


def _mask_dsn(dsn: str) -> str:
    try:
        parsed = urlparse(dsn)
        # Mask credentials in netloc
        netloc = parsed.netloc
        if "@" in netloc:
            userinfo, hostinfo = netloc.split("@", 1)
            if ":" in userinfo:
                user, _ = userinfo.split(":", 1)
            else:
                user = userinfo
            netloc = f"{user}:***@{hostinfo}"
        return parsed._replace(netloc=netloc).geturl()
    except Exception:
        return "postgresql://***:***@***:***/***"


