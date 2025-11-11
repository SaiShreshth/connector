"""
Queue Infrastructure Module

Provides message queue infrastructure for asynchronous LLM intent classification.
Handles ambiguous queries from rule-based classifier and routes them to LLM processing.
"""

from .queue_manager import QueueManager
from .config import QueueConfig

__all__ = ["QueueManager", "QueueConfig"]
