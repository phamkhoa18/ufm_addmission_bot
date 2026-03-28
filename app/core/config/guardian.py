# app/core/config/guardian.py
# Cấu hình các lớp BẢO VỆ (Security Layers)
# Lớp 0: Input Validation (Chống DoS)
# Lớp 1: Keyword Filter & Normalization (Chống từ khóa cấm)
# Lớp 2: Prompt Guard (Chống Injection / Jailbreak)
# Đọc từ: guardian_config.yaml

from pydantic import BaseModel, Field
from typing import List, Dict
from app.core.config import guardian_yaml_data, models_yaml_data, prompts_yaml_data

_iv = guardian_yaml_data.get("input_validation", {})
_iv_prompts = prompts_yaml_data.get("fallback_messages", {})

class InputValidationConfig(BaseModel):
    max_input_chars: int = _iv.get("max_input_chars", 2000)
    summarize_threshold: int = _iv.get("summarize_threshold", 1999)
    fallback_too_long: str = _iv_prompts.get(
        "too_long",
        "Câu hỏi của bạn quá dài. Vui lòng tóm tắt lại."
    )


# ============================================================
# LỚP 0b: Long Query Summarizer (Tóm tắt query dài)
# ============================================================
_lqs = models_yaml_data.get("long_query_summarizer", {})

class LongQuerySummarizerConfig(BaseModel):
    provider: str = _lqs.get("provider", "openrouter")
    model: str = _lqs.get("model", "google/gemini-2.5-flash-lite")
    temperature: float = _lqs.get("temperature", 0.0)
    max_tokens: int = _lqs.get("max_tokens", 300)
    timeout_seconds: int = _lqs.get("timeout_seconds", 10)


# ============================================================
# LỚP 1: Keyword Filter & Normalization
# ============================================================
_kf = guardian_yaml_data.get("keyword_filter", {})
_kf_prompts = prompts_yaml_data.get("fallback_messages", {})

class KeywordFilterConfig(BaseModel):
    banned_regex_patterns: List[str] = _kf.get("banned_regex_patterns", [])
    injection_regex_patterns: List[str] = _kf.get("injection_regex_patterns", [])
    teencode_map: Dict[str, str] = _kf.get("teencode_map", {})
    fallback_message: str = _kf_prompts.get(
        "banned_content",
        "Câu hỏi chứa nội dung không phù hợp."
    )
    fallback_injection: str = _kf_prompts.get(
        "injection",
        "Hệ thống phát hiện dấu hiệu can thiệp bất thường."
    )


# ============================================================
# LỚP 2a: Prompt Guard Fast (Llama 86M - Score-based)
# ============================================================
_pgf = models_yaml_data.get("prompt_guard_fast", {})

class PromptGuardFastConfig(BaseModel):
    provider: str = _pgf.get("provider", "groq")
    model: str = _pgf.get("model", "meta-llama/llama-prompt-guard-2-86m")
    max_tokens_per_chunk: int = _pgf.get("max_tokens_per_chunk", 512)
    score_threshold: float = Field(
        default=_pgf.get("score_threshold", 0.9),
        ge=0.0, le=1.0
    )
    fallback_unsafe: str = _kf_prompts.get(
        "guard_fast_unsafe",
        "Phát hiện dấu hiệu tấn công rõ ràng."
    )


# ============================================================
# LỚP 2b: Prompt Guard Deep (Qwen 7B - Vietnamese SAFE/UNSAFE)
# ============================================================
_pgd = models_yaml_data.get("prompt_guard_deep", {})

class PromptGuardDeepConfig(BaseModel):
    provider: str = _pgd.get("provider", "openrouter")
    model: str = _pgd.get("model", "qwen/qwen-2.5-7b-instruct")
    temperature: float = _pgd.get("temperature", 0.0)
    max_tokens: int = _pgd.get("max_tokens", 50)
    timeout_seconds: int = _pgd.get("timeout_seconds", 5)
    response_format: str = _pgd.get("response_format", "json_object")
    system_prompt: str = _pgd.get(
        "system_prompt",
        'Bạn là hệ thống bảo mật. Trả về JSON: {"status": "SAFE"} hoặc {"status": "UNSAFE"}'
    )
    fallback_unsafe: str = _kf_prompts.get(
        "guard_deep_unsafe",
        "Phát hiện dấu hiệu bất thường. Vui lòng diễn đạt lại."
    )