#!/usr/bin/env python3
"""
Researcher Authentication API Endpoints

FastAPI router for user authentication, registration, and session management.
Provides JWT-based authentication with role-based access control.

Endpoints:
- POST /auth/register - Register new user account
- POST /auth/login - User login with JWT token generation
- POST /auth/refresh - Refresh access token using refresh token
- POST /auth/logout - Logout and revoke refresh token
- POST /auth/password-reset - Request password reset
- POST /auth/password-reset/confirm - Confirm password reset
- GET /auth/me - Get current user information
- GET /auth/users - Admin endpoint to list users
- PUT /auth/users/{user_id}/activate - Admin endpoint to activate user
- PUT /auth/users/{user_id}/deactivate - Admin endpoint to deactivate user
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from agents.common.auth_models import (
    UserCreate, UserLogin, Token, PasswordResetRequest, PasswordReset,
    UserRole, UserStatus, create_user, get_user_by_username_or_email,
    hash_password, verify_password, create_access_token, create_refresh_token,
    verify_token, update_user_login, increment_login_attempts, reset_login_attempts,
    activate_user, deactivate_user, store_refresh_token, validate_refresh_token,
    revoke_refresh_token, revoke_all_user_sessions, create_password_reset_token,
    validate_password_reset_token, mark_password_reset_token_used, update_user_password,
    get_all_users, get_user_by_id, create_user_tables
)

logger = logging.getLogger("auth_api")

# Security scheme
security = HTTPBearer()

# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])

class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class UserResponse(BaseModel):
    """User information response"""
    user_id: int
    email: str
    username: str
    full_name: str
    role: UserRole
    status: UserStatus
    created_at: datetime
    last_login: Optional[datetime] = None

class RegisterResponse(BaseModel):
    """Registration response model"""
    message: str
    user_id: int
    requires_activation: bool = True

class PasswordResetResponse(BaseModel):
    """Password reset response model"""
    message: str
    email_sent: bool = True

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    payload = verify_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(payload.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if user['status'] != UserStatus.ACTIVE.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not active",
        )

    return user

async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency to ensure user has admin role"""
    if current_user['role'] != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user

@router.post("/register", response_model=RegisterResponse)
async def register_user(user_data: UserCreate, background_tasks: BackgroundTasks):
    """Register a new user account"""
    try:
        # Check if user already exists
        existing_user = get_user_by_username_or_email(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )

        existing_email = get_user_by_username_or_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create user
        user_id = create_user(user_data)
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )

        # TODO: Send activation email in background
        # background_tasks.add_task(send_activation_email, user_data.email, activation_token)

        logger.info(f"New user registered: {user_data.username} (ID: {user_id})")

        return RegisterResponse(
            message="User registered successfully. Please check your email for activation instructions.",
            user_id=user_id,
            requires_activation=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=LoginResponse)
async def login_user(login_data: UserLogin):
    """Authenticate user and return JWT tokens"""
    try:
        # Get user by username or email
        user = get_user_by_username_or_email(login_data.username_or_email)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Check if account is locked
        if user.get('locked_until') and user['locked_until'] > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked due to too many failed login attempts"
            )

        # Check if account is active
        if user['status'] != UserStatus.ACTIVE.value:
            if user['status'] == UserStatus.PENDING.value:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is pending activation. Please check your email."
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is not active"
                )

        # Verify password
        if not verify_password(login_data.password, user['hashed_password'], user['salt']):
            increment_login_attempts(user['user_id'])
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Reset login attempts and update last login
        reset_login_attempts(user['user_id'])
        update_user_login(user['user_id'])

        # Create tokens
        token_data = {
            "user_id": user['user_id'],
            "username": user['username'],
            "email": user['email'],
            "role": user['role']
        }

        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        # Store refresh token
        if not store_refresh_token(user['user_id'], refresh_token):
            logger.warning(f"Failed to store refresh token for user {user['user_id']}")

        # Prepare user response data
        user_response = {
            "user_id": user['user_id'],
            "username": user['username'],
            "email": user['email'],
            "full_name": user['full_name'],
            "role": user['role'],
            "last_login": user['last_login'].isoformat() if user['last_login'] else None
        }

        logger.info(f"User logged in: {user['username']} (ID: {user['user_id']})")

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=30 * 60,  # 30 minutes
            user=user_response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh")
async def refresh_access_token(refresh_data: Dict[str, str]):
    """Refresh access token using refresh token"""
    try:
        refresh_token = refresh_data.get("refresh_token")
        if not refresh_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Refresh token required"
            )

        # Validate refresh token
        user_id = validate_refresh_token(refresh_token)
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        # Get user
        user = get_user_by_id(user_id)
        if user is None or user['status'] != UserStatus.ACTIVE.value:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        # Create new access token
        token_data = {
            "user_id": user['user_id'],
            "username": user['username'],
            "email": user['email'],
            "role": user['role']
        }

        access_token = create_access_token(token_data)

        logger.info(f"Access token refreshed for user: {user['username']}")

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 30 * 60
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout_user(refresh_data: Dict[str, str], current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout user and revoke refresh token"""
    try:
        refresh_token = refresh_data.get("refresh_token")
        if refresh_token:
            revoke_refresh_token(refresh_token)

        logger.info(f"User logged out: {current_user['username']}")
        return {"message": "Logged out successfully"}

    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        user_id=current_user['user_id'],
        email=current_user['email'],
        username=current_user['username'],
        full_name=current_user['full_name'],
        role=UserRole(current_user['role']),
        status=UserStatus(current_user['status']),
        created_at=current_user['created_at'],
        last_login=current_user['last_login']
    )

@router.post("/password-reset", response_model=PasswordResetResponse)
async def request_password_reset(request: PasswordResetRequest, background_tasks: BackgroundTasks):
    """Request password reset"""
    try:
        user = get_user_by_username_or_email(request.email)
        if user is None:
            # Don't reveal if email exists or not for security
            return PasswordResetResponse(
                message="If an account with this email exists, a password reset link has been sent."
            )

        # Create password reset token
        reset_token = create_password_reset_token(user['user_id'])

        # TODO: Send password reset email in background
        # background_tasks.add_task(send_password_reset_email, request.email, reset_token)

        logger.info(f"Password reset requested for: {request.email}")

        return PasswordResetResponse(
            message="If an account with this email exists, a password reset link has been sent."
        )

    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )

@router.post("/password-reset/confirm")
async def confirm_password_reset(reset_data: PasswordReset):
    """Confirm password reset with token"""
    try:
        # Validate reset token
        user_id = validate_password_reset_token(reset_data.token)
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )

        # Update password
        if not update_user_password(user_id, reset_data.new_password):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )

        # Mark token as used
        mark_password_reset_token_used(reset_data.token)

        # Revoke all user sessions for security
        revoke_all_user_sessions(user_id)

        logger.info(f"Password reset confirmed for user ID: {user_id}")

        return {"message": "Password reset successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset confirmation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset confirmation failed"
        )

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Admin endpoint to list users"""
    try:
        users = get_all_users(limit=limit, offset=offset)
        return [
            UserResponse(
                user_id=user['user_id'],
                email=user['email'],
                username=user['username'],
                full_name=user['full_name'],
                role=UserRole(user['role']),
                status=UserStatus(user['status']),
                created_at=user['created_at'],
                last_login=user['last_login']
            )
            for user in users
        ]

    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

@router.put("/users/{user_id}/activate")
async def activate_user_account(
    user_id: int,
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Admin endpoint to activate user account"""
    try:
        if not activate_user(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        logger.info(f"User account activated by admin {current_user['username']}: User ID {user_id}")
        return {"message": "User account activated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Activate user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate user"
        )

@router.put("/users/{user_id}/deactivate")
async def deactivate_user_account(
    user_id: int,
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Admin endpoint to deactivate user account"""
    try:
        if not deactivate_user(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Revoke all sessions for security
        revoke_all_user_sessions(user_id)

        logger.info(f"User account deactivated by admin {current_user['username']}: User ID {user_id}")
        return {"message": "User account deactivated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deactivate user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate user"
        )

# Initialize database tables on import
try:
    create_user_tables()
    logger.info("Authentication database tables initialized")
except Exception as e:
    logger.error(f"Failed to initialize authentication tables: {e}")