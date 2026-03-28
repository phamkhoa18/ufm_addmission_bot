# app/core/config/care.py
# Pydantic config cho Care Node
# Model config → models_config.yaml section "care:"
# Contact info → care_config.yaml
# Prompts      → prompts_config.yaml section "care_node:"

from pydantic import BaseModel, Field
from typing import Optional
from app.core.config import _load_yaml, models_yaml_data


_care_data = _load_yaml("care_config.yaml")
_care = models_yaml_data.get("care", {})
_contacts = _care_data.get("care", {}).get("contacts", {})


class CareContactInfo(BaseModel):
    """Thong tin lien he cua 1 don vi."""
    don_vi: str = ""
    hotline: str = ""
    email: str = ""
    dia_chi: str = ""
    gio_lam_viec: str = ""
    ghi_chu: str = ""

    def to_text(self) -> str:
        """Chuyen sang dang text de truyen vao System Prompt."""
        lines = []
        if self.don_vi:
            lines.append(f"Don vi: {self.don_vi}")
        if self.hotline:
            lines.append(f"Hotline: {self.hotline}")
        if self.email:
            lines.append(f"Email: {self.email}")
        if self.dia_chi:
            lines.append(f"Dia chi: {self.dia_chi}")
        if self.gio_lam_viec:
            lines.append(f"Gio lam viec: {self.gio_lam_viec}")
        if self.ghi_chu:
            lines.append(f"Luu y: {self.ghi_chu}")
        return "\n".join(lines)


class CareConfig(BaseModel):
    """Config cho Care Node (Qwen LLM + Contact Info)."""
    provider: str = _care.get("provider", "openrouter")
    model: str = _care.get("model", "qwen/qwen3.5-flash-02-23")
    temperature: float = Field(
        default=_care.get("temperature", 0.6),
        ge=0.0, le=2.0,
    )
    max_tokens: int = _care.get("max_tokens", 300)

    # Contact info per intent
    ho_tro_sinh_vien: CareContactInfo = CareContactInfo(
        **_contacts.get("ho_tro_sinh_vien", {})
    )
    khieu_nai_gop_y: CareContactInfo = CareContactInfo(
        **_contacts.get("khieu_nai_gop_y", {})
    )
    default_contact: CareContactInfo = CareContactInfo(
        **_contacts.get("default", {})
    )

    def get_contact(self, intent: str) -> CareContactInfo:
        """Lay thong tin lien he theo intent."""
        intent_lower = intent.lower()
        if intent_lower == "ho_tro_sinh_vien":
            return self.ho_tro_sinh_vien
        elif intent_lower == "khieu_nai_gop_y":
            return self.khieu_nai_gop_y
        return self.default_contact
