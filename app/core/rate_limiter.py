"""
Redis-based Rate Limiting
Implements sliding window algorithm for per-IP and per-user rate limiting.
Supports both sync and async Redis operations.
"""

import os
import time
from typing import Optional, Tuple
from fastapi import HTTPException, status, Request
from loguru import logger

try:
    from app.core.redis_client import get_redis_client
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

try:
    from app.core.async_redis_client import get_async_redis_client
    ASYNC_REDIS_AVAILABLE = True
except Exception:
    ASYNC_REDIS_AVAILABLE = False

# Rate limit configuration
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_IP_PER_MINUTE = int(os.getenv("RATE_LIMIT_IP_PER_MINUTE", "500"))
RATE_LIMIT_IP_PER_DAY = int(os.getenv("RATE_LIMIT_IP_PER_DAY", "10000"))
RATE_LIMIT_USER_PER_MINUTE = int(os.getenv("RATE_LIMIT_USER_PER_MINUTE", "1000"))
RATE_LIMIT_USER_PER_DAY = int(os.getenv("RATE_LIMIT_USER_PER_DAY", "100000"))


class RateLimiter:
    """Redis-based rate limiter using sliding window algorithm."""
    
    def __init__(self):
        self.redis_client = None
        self.async_redis_client = None
        self._async_client_initialized = False
        
        if REDIS_AVAILABLE and RATE_LIMIT_ENABLED:
            try:
                self.redis_client = get_redis_client()
            except Exception as e:
                logger.warning(f"Rate limiter: Redis not available: {e}")
                self.redis_client = None
    
    async def _ensure_async_redis(self):
        """Lazy initialization of async Redis client."""
        if self._async_client_initialized:
            return self.async_redis_client
        
        if ASYNC_REDIS_AVAILABLE and RATE_LIMIT_ENABLED:
            try:
                self.async_redis_client = await get_async_redis_client()
                self._async_client_initialized = True
            except Exception as e:
                logger.debug(f"Rate limiter: Async Redis not available: {e}")
                self.async_redis_client = None
        
        return self.async_redis_client
    
    def _get_key(self, identifier: str, window: str) -> str:
        """Generate Redis key for rate limit tracking."""
        return f"chatns:ratelimit:{window}:{identifier}"
    
    def _check_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
        window_name: str
    ) -> Tuple[bool, int, int]:
        """
        Check if identifier is within rate limit.
        
        Args:
            identifier: IP address or user_id
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            window_name: Name of window (e.g., "minute", "day")
            
        Returns:
            Tuple of (is_allowed, remaining, reset_after_seconds)
        """
        if not self.redis_client or not RATE_LIMIT_ENABLED:
            # Rate limiting disabled or Redis unavailable - allow all requests
            return True, limit, window_seconds
        
        key = self._get_key(identifier, window_name)
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        try:
            # Remove expired entries (outside window)
            self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count requests in window
            count = self.redis_client.zcard(key)
            
            if count >= limit:
                # Rate limit exceeded
                # Get oldest entry to calculate reset time
                oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = int(oldest[0][1])
                    reset_after = oldest_time + window_seconds - current_time
                else:
                    reset_after = window_seconds
                
                return False, 0, max(1, reset_after)
            
            # Add current request to window
            self.redis_client.zadd(key, {str(current_time): current_time})
            self.redis_client.expire(key, window_seconds)
            
            remaining = limit - count - 1
            return True, remaining, window_seconds
            
        except Exception as e:
            logger.error(f"Rate limit check failed for {identifier}: {e}", exc_info=True)
            # On error, allow request (fail open)
            return True, limit, window_seconds
    
    async def check_rate_limit_async(
        self,
        request: Request,
        user_id: Optional[str] = None
    ) -> None:
        """
        Async version of check_rate_limit for use in async FastAPI endpoints.
        
        Args:
            request: FastAPI request object
            user_id: Optional authenticated user ID
            
        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        if not RATE_LIMIT_ENABLED:
            return
        
        await self._ensure_async_redis()
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check IP-based limits (async)
        allowed_minute, remaining_minute, reset_minute = await self._acheck_limit(
            client_ip,
            RATE_LIMIT_IP_PER_MINUTE,
            60,
            "ip_minute"
        )
        
        if not allowed_minute:
            reset_after = reset_minute
            logger.warning(
                f"Rate limit exceeded: IP {client_ip} exceeded {RATE_LIMIT_IP_PER_MINUTE} req/min",
                extra={"ip": client_ip, "limit": RATE_LIMIT_IP_PER_MINUTE, "window": "minute"}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {RATE_LIMIT_IP_PER_MINUTE} requests per minute",
                headers={"Retry-After": str(reset_after)},
            )
        
        allowed_day, remaining_day, reset_day = await self._acheck_limit(
            client_ip,
            RATE_LIMIT_IP_PER_DAY,
            86400,  # 24 hours
            "ip_day"
        )
        
        if not allowed_day:
            reset_after = reset_day
            logger.warning(
                f"Rate limit exceeded: IP {client_ip} exceeded {RATE_LIMIT_IP_PER_DAY} req/day",
                extra={"ip": client_ip, "limit": RATE_LIMIT_IP_PER_DAY, "window": "day"}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {RATE_LIMIT_IP_PER_DAY} requests per day",
                headers={"Retry-After": str(reset_after)},
            )
        
        # If user is authenticated, check user-based limits
        if user_id:
            allowed_user_minute, remaining_user_minute, reset_user_minute = await self._acheck_limit(
                f"user:{user_id}",
                RATE_LIMIT_USER_PER_MINUTE,
                60,
                "user_minute"
            )
            
            if not allowed_user_minute:
                reset_after = reset_user_minute
                logger.warning(
                    f"Rate limit exceeded: User {user_id} exceeded {RATE_LIMIT_USER_PER_MINUTE} req/min",
                    extra={"user_id": user_id, "limit": RATE_LIMIT_USER_PER_MINUTE, "window": "minute"}
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {RATE_LIMIT_USER_PER_MINUTE} requests per minute",
                    headers={"Retry-After": str(reset_after)},
                )
            
            allowed_user_day, remaining_user_day, reset_user_day = await self._acheck_limit(
                f"user:{user_id}",
                RATE_LIMIT_USER_PER_DAY,
                86400,  # 24 hours
                "user_day"
            )
            
            if not allowed_user_day:
                reset_after = reset_user_day
                logger.warning(
                    f"Rate limit exceeded: User {user_id} exceeded {RATE_LIMIT_USER_PER_DAY} req/day",
                    extra={"user_id": user_id, "limit": RATE_LIMIT_USER_PER_DAY, "window": "day"}
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {RATE_LIMIT_USER_PER_DAY} requests per day",
                    headers={"Retry-After": str(reset_after)},
                )
    
    def check_rate_limit(
        self,
        request: Request,
        user_id: Optional[str] = None
    ) -> None:
        """
        Sync version of check_rate_limit (for backward compatibility).
        
        Args:
            request: FastAPI request object
            user_id: Optional authenticated user ID
            
        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        if not RATE_LIMIT_ENABLED:
            return
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check IP-based limits (sync)
        allowed_minute, remaining_minute, reset_minute = self._check_limit(
            client_ip,
            RATE_LIMIT_IP_PER_MINUTE,
            60,
            "ip_minute"
        )
        
        if not allowed_minute:
            reset_after = reset_minute
            logger.warning(
                f"Rate limit exceeded: IP {client_ip} exceeded {RATE_LIMIT_IP_PER_MINUTE} req/min",
                extra={"ip": client_ip, "limit": RATE_LIMIT_IP_PER_MINUTE, "window": "minute"}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {RATE_LIMIT_IP_PER_MINUTE} requests per minute",
                headers={"Retry-After": str(reset_after)},
            )
        
        allowed_day, remaining_day, reset_day = self._check_limit(
            client_ip,
            RATE_LIMIT_IP_PER_DAY,
            86400,
            "ip_day"
        )
        
        if not allowed_day:
            reset_after = reset_day
            logger.warning(
                f"Rate limit exceeded: IP {client_ip} exceeded {RATE_LIMIT_IP_PER_DAY} req/day",
                extra={"ip": client_ip, "limit": RATE_LIMIT_IP_PER_DAY, "window": "day"}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {RATE_LIMIT_IP_PER_DAY} requests per day",
                headers={"Retry-After": str(reset_after)},
            )
        
        # If user is authenticated, check user-based limits
        if user_id:
            allowed_user_minute, remaining_user_minute, reset_user_minute = self._check_limit(
                f"user:{user_id}",
                RATE_LIMIT_USER_PER_MINUTE,
                60,
                "user_minute"
            )
            
            if not allowed_user_minute:
                reset_after = reset_user_minute
                logger.warning(
                    f"Rate limit exceeded: User {user_id} exceeded {RATE_LIMIT_USER_PER_MINUTE} req/min",
                    extra={"user_id": user_id, "limit": RATE_LIMIT_USER_PER_MINUTE, "window": "minute"}
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {RATE_LIMIT_USER_PER_MINUTE} requests per minute",
                    headers={"Retry-After": str(reset_after)},
                )
            
            allowed_user_day, remaining_user_day, reset_user_day = self._check_limit(
                f"user:{user_id}",
                RATE_LIMIT_USER_PER_DAY,
                86400,
                "user_day"
            )
            
            if not allowed_user_day:
                reset_after = reset_user_day
                logger.warning(
                    f"Rate limit exceeded: User {user_id} exceeded {RATE_LIMIT_USER_PER_DAY} req/day",
                    extra={"user_id": user_id, "limit": RATE_LIMIT_USER_PER_DAY, "window": "day"}
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {RATE_LIMIT_USER_PER_DAY} requests per day",
                    headers={"Retry-After": str(reset_after)},
                )
    
    async def _acheck_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
        window_name: str
    ) -> Tuple[bool, int, int]:
        """
        Async version of _check_limit using async Redis.
        
        Args:
            identifier: IP address or user_id
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            window_name: Name of window
            
        Returns:
            Tuple of (is_allowed, remaining, reset_after_seconds)
        """
        if not self.async_redis_client or not RATE_LIMIT_ENABLED:
            return True, limit, window_seconds
        
        key = self._get_key(identifier, window_name)
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        try:
            # Remove expired entries (async)
            await self.async_redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count requests in window (async)
            count = await self.async_redis_client.zcard(key)
            
            if count >= limit:
                # Rate limit exceeded
                oldest = await self.async_redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = int(oldest[0][1])
                    reset_after = oldest_time + window_seconds - current_time
                else:
                    reset_after = window_seconds
                
                return False, 0, max(1, reset_after)
            
            # Add current request to window (async)
            await self.async_redis_client.zadd(key, {str(current_time): current_time})
            await self.async_redis_client.expire(key, window_seconds)
            
            remaining = limit - count - 1
            return True, remaining, window_seconds
            
        except Exception as e:
            logger.error(f"Rate limit check failed (async) for {identifier}: {e}", exc_info=True)
            # On error, allow request (fail open)
            return True, limit, window_seconds


# Singleton instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create singleton RateLimiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware for rate limiting.
    Can be added to app or used as dependency.
    """
    if not RATE_LIMIT_ENABLED:
        return await call_next(request)
    
    # Extract user_id from request state if available (set by auth)
    user_id = None
    if hasattr(request.state, "user") and hasattr(request.state.user, "user_id"):
        user_id = request.state.user.user_id
    
    limiter = get_rate_limiter()
    limiter.check_rate_limit(request, user_id)
    
    response = await call_next(request)
    return response

