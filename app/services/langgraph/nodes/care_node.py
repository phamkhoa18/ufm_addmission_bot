"""
Care Node — Xử lý nhóm intent Chăm sóc / Khiếu nại / Tâm lý.

Vị trí trong Graph:
  [intent_node] → [care_node] → [response_node] → END

Nhiệm vụ:
  Nhận intent từ Intent Node (HO_TRO_SINH_VIEN hoặc KHIEU_NAI_GOP_Y).
  Gọi Qwen Flash (qua OpenRouter) để sinh câu trả lời đồng cảm,
  truyền thông tin liên hệ từ care_config.yaml vào system prompt.

Config sources:
  - Model config  → models_config.yaml section "care:"
  - System prompt → prompts_config.yaml section "care_node:"
  - Contact info  → care_config.yaml
  - API keys      → .env (qua query_flow_config.api_keys)

Chi phí: Cực thấp (~0.001$/query, model nhỏ nhất)
Latency: ~500ms-1s
Fallback: Nếu LLM lỗi, trả thẳng contact info từ config (bypass cứng)
"""

import json
import time
import urllib.request
import urllib.error

from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config, prompts_yaml_data
from app.core.config.care import CareConfig
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Khởi tạo config 1 lần duy nhất khi module load
_care_config = CareConfig()

# Tone guides từ prompts_config.yaml
_tone_guides = prompts_yaml_data.get("care_node", {}).get("tone_guides", {})


def _get_tone_guide(intent: str) -> str:
    """Lấy hướng dẫn giọng điệu theo intent từ prompts_config.yaml."""
    intent_lower = intent.lower()
    return str(
        _tone_guides.get(intent_lower, _tone_guides.get("default", ""))
    ).strip()


def _call_care_llm(system_prompt: str, user_query: str) -> str:
    """Gọi Care LLM qua OpenRouter/Groq API — dùng config hợp nhất."""
    provider = _care_config.provider
    api_key = query_flow_config.api_keys.get_key(provider)
    base_url = query_flow_config.api_keys.get_base_url(provider)

    if not api_key:
        raise ValueError(f"Chưa cấu hình API Key cho provider '{provider}'")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "UFM-Admission-Bot/1.0",
        "HTTP-Referer": "https://ufm.edu.vn",
    }
    payload = {
        "model": _care_config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": _care_config.temperature,
        "max_tokens": _care_config.max_tokens,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    return body["choices"][0]["message"]["content"].strip()


def care_node(state: GraphState) -> GraphState:
    """
    Care Node — Gọi Qwen Flash sinh câu trả lời đồng cảm.

    Input:
      - state["intent"]: Tên intent (HO_TRO_SINH_VIEN, KHIEU_NAI_GOP_Y)
      - state["standalone_query"]: Câu hỏi của sinh viên

    Output:
      - state["final_response"]: Câu trả lời đồng cảm + thông tin liên hệ
      - state["response_source"]: "care_template"
      - state["next_node"]: "response"
    """
    intent = state.get("intent", "")
    intent_action = state.get("intent_action", "")
    user_query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    # ── Guard: Chỉ chạy khi intent đúng là PROCEED_CARE ──
    if intent_action != "PROCEED_CARE":
        logger.info("Care Node - SKIP (intent_action='%s' != 'PROCEED_CARE')", intent_action)
        return state

    # Lấy thông tin liên hệ theo intent (từ care_config.yaml)
    contact = _care_config.get_contact(intent)
    contact_text = contact.to_text()

    # Lấy tone guide theo intent (từ prompts_config.yaml)
    tone_guide = _get_tone_guide(intent)

    # Gọi Qwen LLM
    try:
        # Render system prompt từ prompts_config.yaml (biến Jinja2)
        sys_prompt_raw = prompt_manager.get_system("care_node")

        # System prompt có {{ contact_text }} và {{ tone_guide }}
        # prompt_manager.get_system() trả raw string, cần render thủ công
        from jinja2 import Environment, BaseLoader
        _env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
        sys_template = _env.from_string(sys_prompt_raw)
        system_prompt = sys_template.render(
            contact_text=contact_text,
            tone_guide=tone_guide,
        )

        care_response = _call_care_llm(system_prompt, user_query)
        elapsed = time.time() - start_time
        logger.info(
            "Care Node [%.3fs] Qwen OK, intent='%s', %d ky tu",
            elapsed, intent, len(care_response)
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "Care Node [%.3fs] Qwen loi: %s -> Fallback",
            elapsed, e, exc_info=True
        )
        # Fallback: render fallback_message từ prompts_config.yaml
        fallback_raw = prompts_yaml_data.get("care_node", {}).get("fallback_message", "")
        if fallback_raw and "{{ contact_text }}" in str(fallback_raw):
            from jinja2 import Environment, BaseLoader
            _env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
            care_response = _env.from_string(str(fallback_raw)).render(
                contact_text=contact_text
            )
        else:
            care_response = (
                "Mình hiểu bạn đang cần hỗ trợ. "
                "Bạn có thể liên hệ trực tiếp qua các kênh sau:\n\n"
                f"{contact_text}\n\n"
                "Đừng ngại liên hệ nhé, đội ngũ UFM luôn sẵn sàng hỗ trợ bạn!"
            )

    return {
        **state,
        "final_response": care_response,
        "response_source": "care_template",
        "next_node": "response",
    }
