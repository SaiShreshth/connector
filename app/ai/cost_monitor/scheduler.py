# app/ai/cost_monitor/scheduler.py
import logging
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from app.ai.cost_monitor.usage_tracker import UsageTracker
from app.ai.cost_monitor.cost_spike_detector import CostSpikeDetector
from app.ai.cost_monitor.alert_manager import AlertManager


def run_spike_check():
    """Performs periodic cost spike detection and sends alerts if anomalies are found."""
    logger.info("Running automatic cost spike check...")

    try:
        usage = UsageTracker()
        detector = CostSpikeDetector()
        alert = AlertManager()

        # Get the latest daily summary
        summary = usage.get_daily_summary()

        # Detect potential spikes
        spike_detected, report = detector.detect_spike(summary)

        if spike_detected:
            alert.send_alert("⚠️ Cost Spike Detected", report)
            logger.warning("Spike detected and notification sent.")
        else:
            logger.debug("No spike detected today.")

    except Exception as e:
        logger.error(f"Spike check failed: {e}")
        logging.exception("Spike check failed.")


def start_scheduler():
    """Starts the background scheduler for periodic cost spike checks."""
    try:
        scheduler = BackgroundScheduler()
        # Schedule the job every 6 hours (can be changed to daily via 'cron')
        scheduler.add_job(run_spike_check, "interval", minutes=1)
        scheduler.start()

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Started automatic cost spike monitoring at {now} (runs every 6 hours).")
        logging.info(f"[SCHEDULER] Automatic cost spike monitoring started at {now}")

    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        logging.exception("Failed to start scheduler.")
if __name__ == "__main__":
    print("🔁 Manual Scheduler Test — starting for 3 minutes...")
    start_scheduler()

    import time
    try:
        # Let it run for 3 minutes to observe multiple runs
        time.sleep(180)
    except KeyboardInterrupt:
        print("🛑 Scheduler test interrupted manually.")
