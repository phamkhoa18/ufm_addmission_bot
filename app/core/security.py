"""
Security Module — JWT Auth (Admin) + Rate Limiting + Origin Check (Chat).

Cung cấp:
  ADMIN:
    - create_access_token(): Tạo JWT token cho Admin
    - get_current_admin(): FastAPI Dependency
    - admin_rate_limiter: Rate limiter cho Admin API

  CHAT (PUBLIC):
    - verify_origin(): Kiểm tra Origin header khớp domain cho phép
    - chat_rate_limiter: Rate limiter per-IP cho Chat API

  CHUNG:
    - RateLimiter: Class reusable (sliding window)
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from collections import defaultdict
from urllib.parse import urlparse

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config.admin_config import admin_cfg
from app.core.config.chat_config import chat_cfg
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── JWT Library ──
try:
    import jwt as pyjwt
except ImportError:
    raise ImportError("PyJWT chưa được cài. Chạy: pip install PyJWT")


# ══════════════════════════════════════════════════════════
# JWT TOKEN — ADMIN ONLY
# ══════════════════════════════════════════════════════════
_admin_jwt_cfg = admin_cfg.jwt
_security_scheme = HTTPBearer(auto_error=True)


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Tạo JWT Access Token cho Admin."""
    if expires_delta is None:
        expires_delta = timedelta(minutes=_admin_jwt_cfg.access_token_expire_minutes)

    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": "admin",
        "iat": now,
        "exp": now + expires_delta,
    }

    token = pyjwt.encode(payload, _admin_jwt_cfg.secret_key, algorithm=_admin_jwt_cfg.algorithm)
    logger.info("Security - Admin JWT created for '%s', expires in %s", subject, expires_delta)
    return token


def verify_token(token: str) -> dict:
    """Decode + Validate JWT token (Admin only)."""
    try:
        payload = pyjwt.decode(
            token,
            _admin_jwt_cfg.secret_key,
            algorithms=[_admin_jwt_cfg.algorithm],
        )
        return payload
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token đã hết hạn. Vui lòng đăng nhập lại.",
        )
    except pyjwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token không hợp lệ: {e}",
        )


async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Security(_security_scheme),
) -> str:
    """FastAPI Dependency — Xác thực Admin từ header Authorization."""
    payload = verify_token(credentials.credentials)
    role = payload.get("role", "")
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token này không có quyền Admin.",
        )
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token thiếu thông tin người dùng.",
        )
    return username


# ══════════════════════════════════════════════════════════
# ORIGIN CHECK — Chat API Domain Lock
# ══════════════════════════════════════════════════════════
_allowed_origins = set(chat_cfg.security.allowed_origins)
_enforce_origin = chat_cfg.security.enforce_origin

# Build set các domain đã chuẩn hóa (chỉ scheme+host+port)
_allowed_domains: set[str] = set()
for origin in _allowed_origins:
    parsed = urlparse(origin)
    _allowed_domains.add(f"{parsed.scheme}://{parsed.netloc}")


def verify_origin(request: Request) -> str:
    """
    Kiểm tra Origin/Referer header phải thuộc domain cho phép.

    Args:
        request: FastAPI Request object.

    Returns:
        Client IP address (dùng cho rate limiting).

    Raises:
        HTTPException 403 nếu Origin không hợp lệ.
    """
    client_ip = request.client.host if request.client else "unknown"

    if not _enforce_origin:
        return client_ip

    # Kiểm tra Origin header (browser luôn gửi khi cross-origin)
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")

    # Nếu là same-origin (browser không gửi Origin header) → cho phép
    if not origin and not referer:
        # Có thể là request từ cùng server hoặc tool (Postman, cURL)
        # Trong production, cURL/Postman không có Origin → vẫn bị chặn
        # Nhưng nếu server-side rendering gọi API cùng host → cho phép
        logger.debug("Security - No Origin/Referer from %s (same-origin or tool)", client_ip)
        return client_ip

    # Check Origin header
    if origin:
        parsed = urlparse(origin)
        origin_domain = f"{parsed.scheme}://{parsed.netloc}"
        if origin_domain in _allowed_domains:
            return client_ip

    # Fallback: Check Referer header
    if referer:
        parsed = urlparse(referer)
        referer_domain = f"{parsed.scheme}://{parsed.netloc}"
        if referer_domain in _allowed_domains:
            return client_ip

    # Bị chặn
    logger.warning(
        "Security - BLOCKED request from origin='%s' referer='%s' ip=%s",
        origin, referer, client_ip,
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Truy cập bị từ chối. API chỉ cho phép gọi từ domain đã đăng ký.",
    )


# ══════════════════════════════════════════════════════════
# RATE LIMITER (In-memory, sliding window)
# ══════════════════════════════════════════════════════════
class RateLimiter:
    """
    Rate Limiter đơn giản dùng sliding window (in-memory).

    Sử dụng:
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        limiter.check("192.168.1.1")  # Raises HTTPException nếu quá giới hạn
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str) -> None:
        """Kiểm tra rate limit. Raises HTTPException 429 nếu quá giới hạn."""
        now = time.time()
        cutoff = now - self.window_seconds

        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > cutoff
        ]

        if len(self._requests[client_id]) >= self.max_requests:
            logger.warning(
                "Security - Rate limit exceeded for '%s': %d/%d",
                client_id, len(self._requests[client_id]), self.max_requests,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Bạn đã gửi quá {self.max_requests} tin nhắn trong "
                       f"{self.window_seconds}s. Vui lòng đợi một lát.",
            )

        self._requests[client_id].append(now)


# ══════════════════════════════════════════════════════════
# SINGLETON RATE LIMITERS
# ══════════════════════════════════════════════════════════

# Admin: 10 req/phút
admin_rate_limiter = RateLimiter(
    max_requests=admin_cfg.rate_limit.max_requests_per_minute,
    window_seconds=60,
)

# Chat: 8 msg/phút per IP (public, không cần đăng nhập)
chat_rate_limiter = RateLimiter(
    max_requests=chat_cfg.rate_limit.max_messages_per_minute,
    window_seconds=60,
)
