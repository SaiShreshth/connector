"""
Queue Configuration

Technical decisions:
- Technology: Redis (already in requirements.txt)
- Message retention: 24 hours
- Max retry attempts: 3
- Priority queue: Supported (premium users get higher priority)
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class QueueConfig(BaseSettings):
    """Queue infrastructure configuration"""
    
    # Redis Connection
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # Queue Names
    AMBIGUOUS_QUERY_QUEUE: str = "chatns:queue:ambiguous_queries"
    CLASSIFICATION_RESULT_QUEUE: str = "chatns:queue:classification_results"
    DEAD_LETTER_QUEUE: str = "chatns:queue:dead_letter"
    
    # Queue Configuration
    MESSAGE_TTL: int = 86400  # 24 hours in seconds
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: int = 5  # seconds between retries
    
    # Connection Pooling
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    
    # Priority Queue Settings
    ENABLE_PRIORITY_QUEUE: bool = True
    PRIORITY_HIGH: int = 1  # Premium users
    PRIORITY_NORMAL: int = 5  # Regular users
    PRIORITY_LOW: int = 10  # Batch processing
    
    # Monitoring
    ENABLE_QUEUE_METRICS: bool = True
    METRICS_INTERVAL: int = 60  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    class Config:
        extra="ignore"


# Global config instance
queue_config = QueueConfig()
