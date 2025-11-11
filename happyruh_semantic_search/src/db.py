import os, psycopg2
from psycopg2.extras import Json

DSN = os.getenv("POSTGRES_DSN", "postgresql://ruh:ruhpass@localhost:5432/happyruh")

DDL = """
CREATE TABLE IF NOT EXISTS shopify_products_raw (
  product_id BIGINT PRIMARY KEY,
  handle TEXT,
  title TEXT,
  url TEXT,
  product_json JSONB NOT NULL,
  fetched_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
  run_id SERIAL PRIMARY KEY,
  started_at TIMESTAMP DEFAULT NOW(),
  finished_at TIMESTAMP,
  status TEXT,                        -- 'running' | 'ok' | 'error'
  products_raw_count INTEGER,
  qdrant_upserted_count INTEGER,
  error TEXT
);
"""

def get_conn():
    return psycopg2.connect(DSN)

def ensure_tables():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)

def upsert_raw(product):
    """
    product = {
      'id': <shopify_id>,
      'handle': str|None,
      'title': str|None,
      'url': str|None,
      'json': <full shopify product object>
    }
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
        INSERT INTO shopify_products_raw (product_id, handle, title, url, product_json)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (product_id)
        DO UPDATE SET product_json = EXCLUDED.product_json,
                      handle = EXCLUDED.handle,
                      title  = EXCLUDED.title,
                      url    = EXCLUDED.url,
                      fetched_at = NOW();
        """, (
            product["id"],
            product.get("handle"),
            product.get("title"),
            product.get("url"),
            Json(product["json"]),
        ))

def fetch_all_raw():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
        SELECT product_id, handle, title, url, product_json
        FROM shopify_products_raw
        """)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

# ----- pipeline run logs -----
def run_start() -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO pipeline_runs (status) VALUES ('running') RETURNING run_id;")
        return cur.fetchone()[0]

def run_finish(run_id: int, status: str, products_raw_count: int | None, qdrant_upserted_count: int | None, error: str | None):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE pipeline_runs
               SET finished_at = NOW(),
                   status = %s,
                   products_raw_count = %s,
                   qdrant_upserted_count = %s,
                   error = %s
             WHERE run_id = %s;
        """, (status, products_raw_count, qdrant_upserted_count, error, run_id))

def last_runs(limit: int = 10):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT run_id, started_at, finished_at, status, products_raw_count, qdrant_upserted_count, error
              FROM pipeline_runs
             ORDER BY run_id DESC
             LIMIT %s;
        """, (limit,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
