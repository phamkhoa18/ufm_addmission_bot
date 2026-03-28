# app/core/config/intent.py
# Cấu hình các lớp PHÂN LOẠI Ý ĐỊNH (Intent Classification)
# Đọc cấu hình chi tiết từ: intent_config.yaml

from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from app.core.config import intent_yaml_data, models_yaml_data, prompts_yaml_data


# ============================================================
# LỚP 3.1: Vector Intent Router (Fast Semantic Search)
# ============================================================
_vr = models_yaml_data.get("vector_router", {})
_vr_cfg = intent_yaml_data.get("vector_router", {})

class VectorRouterConfig(BaseModel):
    enabled: bool = _vr_cfg.get("enabled", True)
    provider: str = _vr.get("provider", "openrouter")
    model: str = _vr.get("model", "baai/bge-m3")
    dimensions: int = _vr.get("dimensions", 1024)
    similarity_threshold: float = Field(
        default=_vr.get("similarity_threshold", 0.82),
        ge=0.0, le=1.0
    )


# ============================================================
# LỚP 3 - Validator: Chống LLM sai chính tả Intent
# ============================================================
_iv_val = intent_yaml_data.get("intent_validator", {})

class IntentValidatorConfig(BaseModel):
    enabled: bool = _iv_val.get("enabled", True)
    fallback_intent: str = _iv_val.get("fallback_intent", "KHONG_XAC_DINH")


# ============================================================
# LỚP 3.2: LLM Semantic Router (Deep Intent Classification)
# Source of Truth: intent_config.yaml
# ============================================================
_sr = models_yaml_data.get("semantic_router", {})
_sr_cfg = intent_yaml_data.get("semantic_router", {})
_sr_fallbacks = prompts_yaml_data.get("intent_classification", {}).get("fallbacks", {})
_block_fallbacks = prompts_yaml_data.get("fallback_messages", {})

class SemanticRouterConfig(BaseModel):
    provider: str = _sr.get("provider", "openrouter")
    model: str = _sr.get("model", "qwen/qwen-2.5-7b-instruct")
    temperature: float = _sr.get("temperature", 0.0)
    max_tokens: int = _sr.get("max_tokens", 150)
    timeout_seconds: int = _sr.get("timeout_seconds", 12)
    response_format: Literal["json_object"] = _sr.get("response_format", "json_object")
    
    # System Prompt hướng dẫn LLM phân loại intent
    system_prompt: str = "" # Note: Moved to prompts_config.yaml
    
    allowed_intents: List[str] = _sr_cfg.get(
        "allowed_intents",
        [
            # Nhóm 1: Thông tin cốt lõi
            "THONG_TIN_TUYEN_SINH",
            "CHUONG_TRINH_DAO_TAO",
            "HOC_PHI_HOC_BONG",
            "THU_TUC_HANH_CHINH",
            # Nhóm 1.5: Yêu cầu mẫu đơn (→ FormAgent)
            "TAO_MAU_DON",
            # Nhóm 2: Truyền thông & Thương hiệu
            "THANH_TICH_UFM",
            "DOI_SONG_SINH_VIEN",
            "SO_SANH_TRUONG",
            "CO_HOI_VIEC_LAM",
            # Nhóm 3: Chăm sóc & Hỗ trợ sinh viên
            "HO_TRO_SINH_VIEN",
            "KHIEU_NAI_GOP_Y",
            # Nhóm 4: Bảo vệ hệ thống
            "BOI_NHO_DOI_THU",
            "DOI_HOI_CAM_KET",
            "TAN_CONG_HE_THONG",
            "CAU_HOI_LAC_DE",
            # Nhóm 5: Giao tiếp & Ngoại lệ
            "CHAO_HOI",
            "KHONG_XAC_DINH",
        ]
    )
    
    # Fallback messages cho các intent nhóm 4 (trả về thẳng, không gọi RAG)
    fallbacks: Dict[str, str] = _sr_fallbacks
    
    fallback_out_of_scope: str = _block_fallbacks.get(
        "out_of_scope",
        "Câu hỏi nằm ngoài phạm vi hỗ trợ tuyển sinh UFM."
    )
