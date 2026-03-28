"""
Chat Config — Pydantic models cho chat_config.yaml.
"""

from pydantic import BaseModel, Field
from app.core.config import _load_yaml

_chat_data = _load_yaml("chat_config.yaml").get("chat", {})


# ── Security (Domain-Lock) ──
_sec = _chat_data.get("security", {})

class ChatSecurityConfig(BaseModel):
    allowed_origins: list[str] = Field(
        default=_sec.get("allowed_origins", ["http://localhost:3000"])
    )
    enforce_origin: bool = _sec.get("enforce_origin", True)


# ── Rate Limit (IP-based) ──
_rl = _chat_data.get("rate_limit", {})

class ChatRateLimitConfig(BaseModel):
    max_messages_per_minute: int = _rl.get("max_messages_per_minute", 8)
    max_messages_per_hour: int = _rl.get("max_messages_per_hour", 120)


# ── History ──
_hist = _chat_data.get("history", {})

class ChatHistoryConfig(BaseModel):
    max_history_messages: int = _hist.get("max_history_messages", 20)
    max_query_length: int = _hist.get("max_query_length", 2000)


# ── Pipeline ──
_pipe = _chat_data.get("pipeline", {})

class ChatPipelineConfig(BaseModel):
    timeout_seconds: int = _pipe.get("timeout_seconds", 60)


# ── Config tổng ──
class ChatConfig(BaseModel):
    security: ChatSecurityConfig = ChatSecurityConfig()
    rate_limit: ChatRateLimitConfig = ChatRateLimitConfig()
    history: ChatHistoryConfig = ChatHistoryConfig()
    pipeline: ChatPipelineConfig = ChatPipelineConfig()


# Singleton
chat_cfg = ChatConfig()
