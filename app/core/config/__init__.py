# app/core/config/__init__.py
# Phần dùng chung: Load .env, Load YAML (4 files), API Keys, Main Bot, Pipeline tổng

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

# ============================================================
# 1. Load .env (API Keys & Secrets)
# ============================================================
load_dotenv()

# ============================================================
# 2. Load YAML Config Files (Non-sensitive settings)
#    Tách riêng vào thư mục yaml/ cho gọn gàng
# ============================================================
_CONFIG_DIR = Path(__file__).parent / "yaml"

def _load_yaml(filename: str) -> dict:
    """Đọc file YAML config. Trả về dict rỗng nếu file không tồn tại."""
    filepath = _CONFIG_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

# Load YAML tách biệt theo domain
guardian_yaml_data      = _load_yaml("guardian_config.yaml")         # Layer 0-2: Bảo vệ
intent_yaml_data        = _load_yaml("intent_config.yaml")           # Layer 3:   Phân loại ý định (prompt/allowed)
intent_routing_yaml_data = _load_yaml("intent_routing_config.yaml")  # Layer 3:   Ngưỡng + Anchor + Action map
query_context_yaml_data = _load_yaml("query_context_config.yaml")    # Context: Memory + Reformulation + Multi-Query

# ── 2 file hợp nhất mới (nguồn sự thật chính) ──
models_yaml_data        = _load_yaml("models_config.yaml")           # Toàn bộ model primary + fallback theo thứ tự pipeline
prompts_yaml_data       = _load_yaml("prompts_config.yaml")          # Toàn bộ prompt + fallback messages

# Backward-compatible: giữ yaml_data trỏ tới guardian (các module cũ có thể dùng)
yaml_data = guardian_yaml_data

# ============================================================
# 3. API Key Config (từ .env)
# ============================================================
class APIKeyConfig(BaseModel):
    """Quản lý API keys cho tất cả các cloud provider."""
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )
    groq_base_url: str = Field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    google_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )
    google_base_url: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    )
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openrouter_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )

    def get_key(self, provider: str) -> Optional[str]:
        """Lấy API key theo tên provider."""
        return getattr(self, f"{provider}_api_key", None)

    def get_base_url(self, provider: str) -> str:
        """Lấy base URL theo tên provider."""
        return getattr(self, f"{provider}_base_url", "")


# ============================================================
# LỚP 4: Main Bot (RAG / LLM Generation)
# ============================================================
_mb = models_yaml_data.get("main_bot", {})

class MainBotConfig(BaseModel):
    enabled: bool = _mb.get("enabled", True)
    provider: str = _mb.get("provider", "groq")
    model: str = _mb.get("model", "llama-3.1-70b-versatile")
    temperature: float = _mb.get("temperature", 0.2)
    max_tokens: int = _mb.get("max_tokens", 800)
    timeout_seconds: int = _mb.get("timeout_seconds", 15)


# ============================================================
# CẤU HÌNH TỔNG (Pipeline Orchestrator)
# ============================================================
from app.core.config.guardian import InputValidationConfig, KeywordFilterConfig, PromptGuardFastConfig, PromptGuardDeepConfig, LongQuerySummarizerConfig
from app.core.config.intent import VectorRouterConfig, IntentValidatorConfig, SemanticRouterConfig
from app.core.config.query_context import MemoryConfig, QueryReformulationConfig, MultiQueryConfig, EmbeddingConfig
from app.core.config.intent_routing import IntentThresholdConfig, IntentActionConfig, ResponseTemplateConfig
from app.core.config.rag_search import (
    ProceedRagSearchConfig,
    PRQueryConfig, UFMQueryConfig, WebSearchConfig,
    InfoSynthesizerConfig, PRSynthesizerConfig, SanitizerConfig,
    SearchCacheConfig, EvaluatorConfig
)
from app.core.config.fallback_models import FallbackModelsConfig
from app.core.config.retriever import RetrieverConfig

class QueryFlowConfig(BaseModel):
    api_keys: APIKeyConfig = APIKeyConfig()
    input_validation: InputValidationConfig = InputValidationConfig()
    keyword_filter: KeywordFilterConfig = KeywordFilterConfig()
    prompt_guard_fast: PromptGuardFastConfig = PromptGuardFastConfig()
    prompt_guard_deep: PromptGuardDeepConfig = PromptGuardDeepConfig()
    long_query_summarizer: LongQuerySummarizerConfig = LongQuerySummarizerConfig()
    vector_router: VectorRouterConfig = VectorRouterConfig()
    intent_validator: IntentValidatorConfig = IntentValidatorConfig()
    semantic_router: SemanticRouterConfig = SemanticRouterConfig()
    intent_threshold: IntentThresholdConfig = IntentThresholdConfig()
    intent_actions: IntentActionConfig = IntentActionConfig()
    response_templates: ResponseTemplateConfig = ResponseTemplateConfig()
    memory: MemoryConfig = MemoryConfig()
    query_reformulation: QueryReformulationConfig = QueryReformulationConfig()
    multi_query: MultiQueryConfig = MultiQueryConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    main_bot: MainBotConfig = MainBotConfig()
    # RAG Search Pipeline
    proceed_rag_search: ProceedRagSearchConfig = ProceedRagSearchConfig()
    pr_query: PRQueryConfig = PRQueryConfig()
    ufm_query: UFMQueryConfig = UFMQueryConfig()
    web_search: WebSearchConfig = WebSearchConfig()
    info_synthesizer: InfoSynthesizerConfig = InfoSynthesizerConfig()
    pr_synthesizer: PRSynthesizerConfig = PRSynthesizerConfig()
    sanitizer: SanitizerConfig = SanitizerConfig()
    search_cache: SearchCacheConfig = SearchCacheConfig()
    context_evaluator: EvaluatorConfig = EvaluatorConfig()
    # Hybrid Retriever (Vector + BM25 + RRF + Parent)
    retriever: RetrieverConfig = RetrieverConfig()
    # Fallback Models
    fallback_models: FallbackModelsConfig = FallbackModelsConfig()


# Khởi tạo instance config chung
query_flow_config = QueryFlowConfig()
