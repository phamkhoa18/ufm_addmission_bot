"""
Admin Config — Pydantic models cho admin_config.yaml.
"""

import os
from pydantic import BaseModel, Field
from app.core.config import _load_yaml

_admin_data = _load_yaml("admin_config.yaml").get("admin", {})

# ── JWT ──
_jwt = _admin_data.get("jwt", {})

class JWTConfig(BaseModel):
    secret_key: str = os.getenv(
        _jwt.get("secret_key_env", "ADMIN_JWT_SECRET"),
        "dev-only-secret-change-in-production-2026"
    )
    algorithm: str = _jwt.get("algorithm", "HS256")
    access_token_expire_minutes: int = _jwt.get("access_token_expire_minutes", 480)


# ── Credentials ──
_cred = _admin_data.get("credentials", {})

class AdminCredentialsConfig(BaseModel):
    username: str = os.getenv(
        _cred.get("username_env", "ADMIN_USERNAME"),
        _cred.get("default_username", "ufm_admin"),
    )
    password: str = os.getenv(
        _cred.get("password_env", "ADMIN_PASSWORD"),
        _cred.get("default_password", "ufm_admin_2026"),
    )


# ── Rate Limit ──
_rl = _admin_data.get("rate_limit", {})

class RateLimitConfig(BaseModel):
    max_requests_per_minute: int = _rl.get("max_requests_per_minute", 10)
    max_file_size_mb: int = _rl.get("max_file_size_mb", 10)
    allowed_extensions: list[str] = Field(
        default=_rl.get("allowed_extensions", [".md", ".markdown"])
    )


# ── Ingestion Pipeline ──
_ing = _admin_data.get("ingestion", {})

class IngestionPipelineConfig(BaseModel):
    embedding_model: str = _ing.get("embedding_model", "baai/bge-m3")
    embedding_dimensions: int = _ing.get("embedding_dimensions", 1024)
    embedding_batch_size: int = _ing.get("embedding_batch_size", 30)
    max_concurrent_tasks: int = _ing.get("max_concurrent_tasks", 3)


# ── Config tổng ──
class AdminConfig(BaseModel):
    jwt: JWTConfig = JWTConfig()
    credentials: AdminCredentialsConfig = AdminCredentialsConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    ingestion: IngestionPipelineConfig = IngestionPipelineConfig()


# Singleton
admin_cfg = AdminConfig()
