"""
JWT Authentication System
Supports both authenticated users (with user_id in token) and anonymous users.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "1440"))  # 24 hours

# Password hashing context (for future use if needed)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


class UserContext:
    """User context from JWT token."""
    def __init__(self, user_id: Optional[str] = None, is_authenticated: bool = False):
        self.user_id = user_id
        self.is_authenticated = is_authenticated


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary containing user data (e.g., {"sub": user_id})
        expires_delta: Optional expiration time delta
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.debug(f"JWT verification failed: {e}")
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> UserContext:
    """
    Dependency to extract user context from JWT token.
    Returns anonymous user if no token provided (optional authentication).
    
    Args:
        credentials: HTTP Bearer credentials (optional)
        
    Returns:
        UserContext object with user_id and is_authenticated flag
    """
    if not credentials:
        # Anonymous user - no token provided
        return UserContext(user_id=None, is_authenticated=False)
    
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        # Invalid token - treat as anonymous
        logger.warning("Invalid JWT token provided, treating as anonymous user")
        return UserContext(user_id=None, is_authenticated=False)
    
    # Extract user_id from token (sub is standard JWT claim for subject/user_id)
    user_id = payload.get("sub") or payload.get("user_id")
    
    if not user_id:
        logger.warning("JWT token missing user_id, treating as anonymous")
        return UserContext(user_id=None, is_authenticated=False)
    
    return UserContext(user_id=str(user_id), is_authenticated=True)


async def require_auth(
    current_user: UserContext = Depends(get_current_user)
) -> UserContext:
    """
    Dependency to require authenticated user.
    Raises 401 if user is not authenticated.
    
    Args:
        current_user: User context from get_current_user
        
    Returns:
        UserContext object
        
    Raises:
        HTTPException: 401 if not authenticated
    """
    if not current_user.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


async def optional_auth(
    current_user: UserContext = Depends(get_current_user)
) -> UserContext:
    """
    Dependency for optional authentication.
    Returns user context (authenticated or anonymous).
    
    Args:
        current_user: User context from get_current_user
        
    Returns:
        UserContext object (may be anonymous)
    """
    return current_user


def generate_test_token(user_id: str) -> str:
    """
    Generate a test token for a user (useful for testing/development).
    
    Args:
        user_id: User identifier
        
    Returns:
        JWT token string
    """
    data = {"sub": user_id, "user_id": user_id}
    return create_access_token(data)

