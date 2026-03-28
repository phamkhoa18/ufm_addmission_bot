# app/core/config/intent_routing.py
# Action mapping, response templates, edge case threshold
# Đọc từ: intent_routing_config.yaml

from pydantic import BaseModel, Field
from typing import Dict, List
from app.core.config import intent_routing_yaml_data, prompts_yaml_data


# ============================================================
# EDGE CASE THRESHOLD (Pre-check trước khi gọi Qwen)
# ============================================================
_thr = intent_routing_yaml_data.get("thresholds", {})

class IntentThresholdConfig(BaseModel):
    min_query_length: int = Field(
        default=_thr.get("min_query_length", 5),
        ge=1,
        description=(
            "Câu < min_query_length ký tự → CHAO_HOI ngay, không gọi Qwen. "
            "Ví dụ: 'Alo' (3), 'ok' (2), 'Dạ' (2)."
        )
    )



# ============================================================
# INTENT → ACTION MAPPING
# ============================================================
_actions_raw = intent_routing_yaml_data.get("intent_actions", {})

VALID_ACTIONS = {
    "PROCEED_RAG",
    "PROCEED_RAG_UFM_SEARCH",
    "PROCEED_RAG_PR_SEARCH",
    "PROCEED_FORM",
    "PROCEED_PR",
    "PROCEED_CARE",
    "GREET",
    "CLARIFY",
    "BLOCK_FALLBACK",
}

class IntentActionConfig(BaseModel):
    mapping: Dict[str, str] = Field(
        default=_actions_raw,
        description="intent_name → action_string"
    )

    def get_action(self, intent: str) -> str:
        """Trả về action cho 1 intent. Mặc định: CLARIFY."""
        return self.mapping.get(intent, "CLARIFY")


# ============================================================
# RESPONSE TEMPLATES (GREET & CLARIFY — Không cần gọi LLM)
# ============================================================
_tmpl = prompts_yaml_data.get("response_templates", {})

class ResponseTemplateConfig(BaseModel):
    greet_messages: List[str] = Field(
        default=_tmpl.get("GREET", [
            "Chào bạn! 😊 Tôi là trợ lý tư vấn tuyển sinh UFM. Bạn muốn hỏi gì?"
        ])
    )
    clarify_messages: List[str] = Field(
        default=_tmpl.get("CLARIFY", [
            "Xin lỗi bạn, bạn có thể nói rõ hơn về câu hỏi không ạ?"
        ])
    )

    def get_greet(self) -> str:
        """Trả về 1 template chào hỏi ngẫu nhiên."""
        import random
        return random.choice(self.greet_messages)

    def get_clarify(self) -> str:
        """Trả về 1 template hỏi lại ngẫu nhiên."""
        import random
        return random.choice(self.clarify_messages)
