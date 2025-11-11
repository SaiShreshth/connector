import os, time, schedule, subprocess, sys, traceback
from db import ensure_tables, run_start, run_finish

CRON_TIME = os.getenv("CRON_TIME", "03:10")

def run_pipeline_once():
    print("⏰ Daily pipeline: RAW scrape → Postgres → clean+embed → Qdrant")
    ensure_tables()
    run_id = run_start()
    products_raw_count = None
    qdrant_count = None
    err_txt = None

    try:
        r1 = subprocess.run([sys.executable, "src/happyruh_scraper.py"], capture_output=True, text=True)
        if r1.returncode != 0:
            raise RuntimeError(f"scraper failed: {r1.stderr or r1.stdout}")

        # count rows after scrape
        import psycopg2
        from db import get_conn
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM shopify_products_raw;")
            products_raw_count = cur.fetchone()[0]

        r2 = subprocess.run([sys.executable, "src/semantic_search_loader.py"], capture_output=True, text=True)
        if r2.returncode != 0:
            raise RuntimeError(f"loader failed: {r2.stderr or r2.stdout}")

        qdrant_count = products_raw_count  # approximate; loader currently uploads all

        run_finish(run_id, "ok", products_raw_count, qdrant_count, None)
        print(f"🎉 Pipeline done. run_id={run_id} raw={products_raw_count} qdrant≈{qdrant_count}")
    except Exception as e:
        err_txt = f"{e}\n{traceback.format_exc()}"
        run_finish(run_id, "error", products_raw_count, qdrant_count, err_txt)
        print("❌ Pipeline error:", err_txt)

if __name__ == "__main__":
    print(f"🗓️ Worker started. Daily at {CRON_TIME}")
    schedule.every().day.at(CRON_TIME).do(run_pipeline_once)
    # Run once on start so you don't wait for midnight
    run_pipeline_once()
    while True:
        schedule.run_pending()
        time.sleep(1)
