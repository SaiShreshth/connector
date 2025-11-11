"""
Handles alerting when unusual cost spikes occur.
"""

import logging
from loguru import logger as loguru_logger

class AlertManager:
    def __init__(self):
        self.logger = logging.getLogger("cost_alerts")

    def send_alert(self, message: str):
        """
        For now: simple console + log alert.
        Later: integrate email/Slack/webhook.
        """
        self.logger.warning(message)
        loguru_logger.warning(f"ALERT: {message}")
