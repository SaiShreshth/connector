"""
Environment Variable Validation
Validates and provides type-safe access to environment variables using Pydantic.
Fails fast at startup if critical variables are missing or invalid.
"""

import os
from typing import Optional
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings
from loguru import logger


class EnvironmentSettings(BaseSettings):
    """
    Environment variable settings with validation.
    
    Required variables will cause startup failure if missing.
    Optional variables have defaults.
    """
    
    # Application Configuration
    app_name: str = Field(default="Intent Classification API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT", gt=0, le=65535)
    workers: int = Field(default=1, env="WORKERS", gt=0, le=32)
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_dir: str = Field(default="logs", env="LOG_DIR")
    
    # OpenAI Configuration (Required for LLM calls unless DRY_RUN=true)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    gpt4_model: str = Field(default="gpt-4", env="GPT4_MODEL")
    gpt4_turbo_model: str = Field(default="gpt-4-turbo", env="GPT4_TURBO_MODEL")
    gpt35_model: str = Field(default="gpt-3.5-turbo", env="GPT35_MODEL")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE", ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=400, env="OPENAI_MAX_TOKENS", gt=0)
    openai_timeout_secs: float = Field(default=30.0, env="OPENAI_TIMEOUT_SECS", gt=0.0)
    dry_run: bool = Field(default=True, env="DRY_RUN")
    
    # Redis Configuration (Optional - has defaults for local dev)
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_host: Optional[str] = Field(default=None, env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT", gt=0, le=65535)
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB", ge=0)
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS", gt=0)
    
    # Qdrant Configuration (Optional - has defaults for local dev)
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT", gt=0, le=65535)
    
    # JWT Authentication (Required for production, has insecure default)
    jwt_secret_key: str = Field(default="super-secret-key", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES", gt=0)
    
    # CORS Configuration
    allowed_origins: str = Field(default="http://localhost:3000", env="ALLOWED_ORIGINS")
    
    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_ip_per_minute: int = Field(default=500, env="RATE_LIMIT_IP_PER_MINUTE", gt=0)
    rate_limit_ip_per_day: int = Field(default=10000, env="RATE_LIMIT_IP_PER_DAY", gt=0)
    rate_limit_user_per_minute: int = Field(default=1000, env="RATE_LIMIT_USER_PER_MINUTE", gt=0)
    rate_limit_user_per_day: int = Field(default=100000, env="RATE_LIMIT_USER_PER_DAY", gt=0)
    
    # Application Environment
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    # LLM Cache Configuration
    enable_llm_cache: bool = Field(default=True, env="ENABLE_LLM_CACHE")
    llm_cache_similarity_threshold: float = Field(default=0.95, env="LLM_CACHE_SIMILARITY_THRESHOLD", ge=0.0, le=1.0)
    llm_cache_ttl: int = Field(default=86400, env="LLM_CACHE_TTL", gt=0)  # 24 hours in seconds
    llm_cache_max_size: int = Field(default=10000, env="LLM_CACHE_MAX_SIZE", gt=0)
    llm_cache_min_query_length: int = Field(default=3, env="LLM_CACHE_MIN_QUERY_LENGTH", ge=0)
    
    # Session Context Configuration
    enable_session_context: bool = Field(default=True, env="ENABLE_SESSION_CONTEXT")
    context_token_limit: int = Field(default=2000, env="CONTEXT_TOKEN_LIMIT", gt=0)
    context_history_limit: int = Field(default=5, env="CONTEXT_HISTORY_LIMIT", gt=0)
    
    # LLM Queue Configuration
    enable_llm_queue: bool = Field(default=True, env="ENABLE_LLM_QUEUE")
    
    # Cost Monitoring & Alerting Configuration
    cost_alert_daily_threshold_usd: float = Field(default=2.0, env="COST_ALERT_DAILY_THRESHOLD_USD", ge=0.0)
    accuracy_drop_threshold: float = Field(default=0.05, env="ACCURACY_DROP_THRESHOLD", ge=0.0, le=1.0)
    latency_spike_threshold_ms: float = Field(default=3000.0, env="LATENCY_SPIKE_THRESHOLD_MS", gt=0.0)
    error_rate_spike_threshold: float = Field(default=0.05, env="ERROR_RATE_SPIKE_THRESHOLD", ge=0.0, le=1.0)
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is one of the standard levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            logger.warning(f"Invalid LOG_LEVEL '{v}', defaulting to 'INFO'")
            return "INFO"
        return v_upper
    
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v):
        """Warn if using default insecure JWT secret in production."""
        if v == "super-secret-key":
            logger.warning(
                "⚠️  Using default JWT_SECRET_KEY! This is insecure for production. "
                "Set JWT_SECRET_KEY environment variable to a secure random string."
            )
        elif len(v) < 32:
            logger.warning(
                f"⚠️  JWT_SECRET_KEY is too short ({len(v)} chars). "
                "Recommend at least 32 characters for production security."
            )
        return v
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v):
        """Log warning if OpenAI API key is missing."""
        if not v:
            logger.warning(
                "OPENAI_API_KEY not set. LLM functionality will be disabled. "
                "Set DRY_RUN=true to suppress this warning."
            )
        return v
    
    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError(
                f"Invalid REDIS_URL format: '{v}'. "
                "Must start with 'redis://' or 'rediss://'"
            )
        return v
    
    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v):
        """Validate Qdrant URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid QDRANT_URL format: '{v}'. "
                "Must start with 'http://' or 'https://'"
            )
        return v
    
    @field_validator("allowed_origins")
    @classmethod
    def validate_cors_origins(cls, v):
        """Validate CORS origins are provided."""
        if not v or not v.strip():
            raise ValueError("ALLOWED_ORIGINS cannot be empty")
        return v
    
    def model_post_init(self, __context):
        """Post-init validation for openai_api_key requiring dry_run."""
        if not self.dry_run and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when DRY_RUN=false. "
                "Either set OPENAI_API_KEY or set DRY_RUN=true to skip LLM calls."
            )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra environment variables not defined in the model
    }


# Global settings instance
_env_settings: Optional[EnvironmentSettings] = None


def validate_environment() -> EnvironmentSettings:
    """
    Validate environment variables at startup.
    
    Returns:
        Validated EnvironmentSettings instance
        
    Raises:
        ValidationError: If required variables are missing or invalid
        SystemExit: If critical validation fails (to prevent startup)
    """
    global _env_settings
    
    if _env_settings is not None:
        return _env_settings
    
    try:
        _env_settings = EnvironmentSettings()
        logger.info("✅ Environment variables validated successfully")
        return _env_settings
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            field = error.get("loc", ["unknown"])[-1]
            msg = error.get("msg", "validation error")
            error_messages.append(f"  - {field}: {msg}")
        
        logger.error("❌ Environment variable validation failed:")
        logger.error("\n".join(error_messages))
        logger.error("\nPlease fix the environment variables and restart the application.")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during environment validation: {e}")
        raise SystemExit(1)


def get_env_settings() -> EnvironmentSettings:
    """
    Get validated environment settings.
    Call validate_environment() first during startup.
    
    Returns:
        EnvironmentSettings instance
    """
    if _env_settings is None:
        logger.warning("Environment not validated yet. Calling validate_environment()...")
        return validate_environment()
    return _env_settings

