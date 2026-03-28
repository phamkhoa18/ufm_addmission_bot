# app/core/config/fallback_models.py
# Pydantic config cho Fallback Settings (đọc từ models_config.yaml)
#
# REFACTORED: Không còn dùng model_group ("light"/"medium"/"search").
# Mỗi node tự quản lý fallback riêng qua node_key trong models_config.yaml.
# File này chỉ còn giữ FallbackSettingsConfig (max_retries, retry_delay_ms...).

from pydantic import BaseModel
from app.core.config import models_yaml_data


class FallbackSettingsConfig(BaseModel):
    max_retries: int = 2
    retry_delay_ms: int = 500
    log_fallback: bool = True


# ── Khởi tạo từ YAML ──
_settings = models_yaml_data.get("fallback_settings", {})


class FallbackModelsConfig(BaseModel):
    """
    Config cho hệ thống fallback.

    Cách dùng mới (node_key trực tiếp):
        # Trong _call_gemini_api_with_fallback:
        node_cfg = models_yaml_data.get(node_key, {})
        fallbacks_raw = node_cfg.get("fallbacks", [])

    File này chỉ còn chứa settings chung (max_retries, retry_delay).
    """
    settings: FallbackSettingsConfig = FallbackSettingsConfig(
        max_retries=_settings.get("max_retries", 2),
        retry_delay_ms=_settings.get("retry_delay_ms", 500),
        log_fallback=_settings.get("log_fallback", True),
    )
