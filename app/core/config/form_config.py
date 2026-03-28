"""
Cấu trúc Pydantic cho Form Config.

- FormSettings: Đọc model config từ models_config.yaml (section "form:")
- FormFieldDef: Định nghĩa key trích xuất (từ form_config.yaml)
- form_cfg: Singleton instance
"""

from typing import List
from pydantic import BaseModel
import yaml
import os
from app.utils.logger import get_logger
from app.core.config import models_yaml_data

_logger = get_logger(__name__)

_fm_settings = models_yaml_data.get("form", {})


class FormSettings(BaseModel):
    # Selector (Gemini 001 chọn mẫu đơn)
    selector_model: str = _fm_settings.get("selector", {}).get("model", "google/gemini-2.0-flash-001")
    selector_temperature: float = _fm_settings.get("selector", {}).get("temperature", 0.0)
    selector_max_tokens: int = _fm_settings.get("selector", {}).get("max_tokens", 50)
    selector_timeout: int = _fm_settings.get("selector", {}).get("timeout_seconds", 5)

    # Extractor (trích xuất thông tin cá nhân)
    extractor_model: str = _fm_settings.get("extractor", {}).get("model", "google/gemini-2.5-flash")
    extractor_temperature: float = _fm_settings.get("extractor", {}).get("temperature", 0.0)
    extractor_max_tokens: int = _fm_settings.get("extractor", {}).get("max_tokens", 800)
    extractor_timeout: int = _fm_settings.get("extractor", {}).get("timeout_seconds", 10)

    # Drafter (soạn thảo văn bản)
    drafter_model: str = _fm_settings.get("drafter", {}).get("model", "google/gemini-2.5-flash")
    drafter_temperature: float = _fm_settings.get("drafter", {}).get("temperature", 0.4)
    drafter_temperature_no_template: float = _fm_settings.get("drafter", {}).get("temperature_no_template", 0.3)
    drafter_max_tokens: int = _fm_settings.get("drafter", {}).get("max_tokens", 4000)
    drafter_timeout: int = _fm_settings.get("drafter", {}).get("timeout_seconds", 20)

    provider: str = _fm_settings.get("provider", "openrouter")


class FormFieldDef(BaseModel):
    """Định nghĩa 1 key trích xuất cho Extractor."""
    key: str
    label: str
    extract_hint: str = ""


class FormConfig(BaseModel):
    settings: FormSettings
    fields: List[FormFieldDef]

    @classmethod
    def load(cls, yaml_path: str) -> "FormConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Khong tim thay {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        fields = data.get("fields", [])

        return cls(
            settings=FormSettings(),
            fields=[FormFieldDef(**f) for f in fields],
        )


# Singleton — Load 1 lần khi import
_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "yaml", "form_config.yaml"
)

try:
    form_cfg = FormConfig.load(_YAML_PATH)
except Exception as e:
    _logger.warning("FormConfig - Loi khi load form_config.yaml: %s", e)
    form_cfg = FormConfig(
        settings=FormSettings(),
        fields=[],
    )
