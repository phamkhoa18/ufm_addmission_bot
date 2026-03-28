# app/core/config/query_context.py
# Cấu hình cho Context Node (Memory, Reformulation, Multi-Query)
# Đọc từ: query_context_config.yaml

from pydantic import BaseModel, Field
from typing import Optional
from app.core.config import query_context_yaml_data, models_yaml_data


# ============================================================
# BỘ NHỚ HỘI THOẠI (Sliding Window Memory)
# ============================================================
_mem = query_context_yaml_data.get("memory", {})
_auto_sum_model = models_yaml_data.get("auto_summarize", {})

class AutoSummarizeConfig(BaseModel):
    enabled: bool = _mem.get("auto_summarize", {}).get("enabled", True)
    trigger_length: int = _mem.get("auto_summarize", {}).get("trigger_length", 300)
    target_length: int = _mem.get("auto_summarize", {}).get("target_length", 120)
    provider: str = _auto_sum_model.get("provider", "openrouter")
    model: str = _auto_sum_model.get("model", "google/gemini-2.0-flash-001")
    temperature: float = _auto_sum_model.get("temperature", 0.0)
    max_tokens: int = _auto_sum_model.get("max_tokens", 80)
    timeout_seconds: int = _auto_sum_model.get("timeout_seconds", 6)

class MemoryConfig(BaseModel):
    max_history_turns: int = _mem.get("max_history_turns", 10)
    max_tokens_per_message: int = _mem.get("max_tokens_per_message", 400)
    include_system_summary: bool = _mem.get("include_system_summary", False)
    auto_summarize: AutoSummarizeConfig = AutoSummarizeConfig()


# ============================================================
# QUERY REFORMULATION (Tái tạo câu hỏi)
# ============================================================
_qr = models_yaml_data.get("query_reformulation", {})

class QueryReformulationConfig(BaseModel):
    enabled: bool = _qr.get("enabled", True)
    provider: str = _qr.get("provider", "openrouter")
    model: str = _qr.get("model", "google/gemini-3.1-flash-lite-preview")
    temperature: float = _qr.get("temperature", 0.0)
    max_tokens: int = _qr.get("max_tokens", 250)
    timeout_seconds: int = _qr.get("timeout_seconds", 8)
    skip_if_no_history: bool = _qr.get("skip_if_no_history", True)
    system_prompt: str = "" # Note: Moved to prompts_config.yaml


# ============================================================
# MULTI-QUERY EXPANSION (Sinh biến thể câu hỏi)
# ============================================================
_mq = models_yaml_data.get("multi_query", {})

class MultiQueryConfig(BaseModel):
    enabled: bool = _mq.get("enabled", True)
    provider: str = _mq.get("provider", "openrouter")
    model: str = _mq.get("model", "google/gemini-3.1-flash-lite-preview")
    temperature: float = _mq.get("temperature", 0.3)
    max_tokens: int = _mq.get("max_tokens", 300)
    timeout_seconds: int = _mq.get("timeout_seconds", 8)
    num_variants: int = _mq.get("num_variants", 3)
    merge_strategy: str = _mq.get("merge_strategy", "rrf")
    top_k_per_query: int = _mq.get("top_k_per_query", 5)
    top_k_final: int = _mq.get("top_k_final", 7)
    system_prompt: str = "" # Note: Moved to prompts_config.yaml


# ============================================================
# EMBEDDING CONFIG
# ============================================================
_emb = models_yaml_data.get("embedding", {})

class EmbeddingConfig(BaseModel):
    provider: str = _emb.get("provider", "openrouter")
    model: str = _emb.get("model", "baai/bge-m3")
    dimensions: int = _emb.get("dimensions", 1024)
    batch_size: int = _emb.get("batch_size", 4)
