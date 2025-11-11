"""
Review Scheduler

Implements weekly/monthly review process with automated reports.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any
from loguru import logger

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

from app.ai.feedback.feedback_store import get_feedback_store
from app.ai.feedback.pattern_analyzer import PatternAnalyzer
from app.ai.feedback.accuracy_tracker import AccuracyTracker


def generate_review_report(report_type: str = "weekly") -> Dict[str, Any]:
    """
    Generate a review report.
    
    Args:
        report_type: "weekly" or "monthly"
    
    Returns:
        Dict with report data
    """
    feedback_store = get_feedback_store()
    pattern_analyzer = PatternAnalyzer()
    accuracy_tracker = AccuracyTracker()
    
    stats = feedback_store.get_stats()
    patterns = pattern_analyzer.analyze_patterns()
    accuracy_summary = accuracy_tracker.get_accuracy_summary()
    
    report = {
        "report_type": report_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_feedback": stats.get("total_feedback", 0),
            "total_misclassifications": stats.get("total_misclassifications", 0),
            "accuracy_rate": stats.get("accuracy_rate", 0.0),
            "current_accuracy": accuracy_summary.get("current_accuracy", 0.0),
            "accuracy_trend": accuracy_summary.get("trend", "no_data"),
        },
        "top_misclassified_codes": stats.get("misclassifications_by_action_code", {}),
        "patterns": {
            "total_misclassifications": patterns.get("total_misclassifications", 0),
            "avg_confidence": patterns.get("avg_confidence", 0.0),
            "insights": patterns.get("insights", []),
        },
        "recommendations": [],
    }
    
    # Generate recommendations
    if patterns.get("insights"):
        for insight in patterns.get("insights", []):
            report["recommendations"].append({
                "priority": insight.get("severity", "medium"),
                "action": insight.get("recommendation", ""),
            })
    
    return report


def save_report(report: Dict[str, Any]) -> str:
    """Save report to file."""
    os.makedirs("data/reports", exist_ok=True)
    report_type = report.get("report_type", "weekly")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"data/reports/{report_type}_report_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    return filename


def weekly_review_job():
    """Weekly review job - generates and saves weekly report."""
    logger.info("Running weekly review...")
    try:
        # Record daily accuracy before generating report
        accuracy_tracker = AccuracyTracker()
        accuracy_tracker.record_daily_accuracy()
        
        report = generate_review_report("weekly")
        filename = save_report(report)
        logger.info(f"Weekly report saved: {filename}")
        logger.info(f"  - Total misclassifications: {report['summary']['total_misclassifications']}")
        logger.info(f"  - Accuracy rate: {report['summary']['accuracy_rate']:.2%}")
    except Exception as e:
        logger.error(f"Error in weekly review: {e}")


def monthly_review_job():
    """Monthly review job - generates and saves monthly report."""
    logger.info("Running monthly review...")
    try:
        report = generate_review_report("monthly")
        filename = save_report(report)
        logger.info(f"Monthly report saved: {filename}")
        logger.info(f"  - Total misclassifications: {report['summary']['total_misclassifications']}")
        logger.info(f"  - Accuracy rate: {report['summary']['accuracy_rate']:.2%}")
    except Exception as e:
        logger.error(f"Error in monthly review: {e}")


def start_review_scheduler():
    """Start the background scheduler for review reports."""
    if not APSCHEDULER_AVAILABLE:
        logger.warning("APScheduler not available, review scheduler not started")
        return None
    
    scheduler = BackgroundScheduler()
    
    # Weekly review: Every Monday at 9 AM
    scheduler.add_job(
        weekly_review_job,
        trigger=CronTrigger(day_of_week="mon", hour=9, minute=0),
        id="weekly_review",
        name="Weekly Review Report",
    )
    
    # Monthly review: 1st of every month at 9 AM
    scheduler.add_job(
        monthly_review_job,
        trigger=CronTrigger(day=1, hour=9, minute=0),
        id="monthly_review",
        name="Monthly Review Report",
    )
    
    scheduler.start()
    logger.info("Review scheduler started (weekly: Mondays 9 AM, monthly: 1st of month 9 AM)")
    return scheduler
