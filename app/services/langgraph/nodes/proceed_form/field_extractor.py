"""
Field Extractor — Trích xuất thông tin người dùng từ chat_history.

CHIẾN LƯỢC MỚI (Template-Driven):
  Thay vì dùng danh sách field chung (form_config.yaml),
  bước Extractor giờ chỉ trích xuất THÔNG TIN THÔ (free-form)
  mà user đã cung cấp trong chat history.
  
  Form Drafter sẽ tự đối chiếu thông tin thô này với template thực tế.
"""

import json
import re
import time
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config.form_config import form_cfg
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def extract_fields(chat_history: list, user_query: str) -> dict:
    """
    Quét lịch sử hội thoại để thu thập MỌI thông tin cá nhân user đã cung cấp.

    Trả về Dict dạng: { "ho_ten": "Nguyen Van A", "nganh_du_tuyen": "QTKD", ... }
    Giá trị nào không tìm thấy trả về None.
    """
    start_time = time.time()

    # ── Xây dựng context từ history ──
    context_str = ""
    if chat_history:
        for msg in chat_history[-8:]:  # Lấy 8 lượt gần nhất
            role = "Tư vấn viên" if msg.get("role") == "assistant" else "Người dùng"
            content = msg.get("content", "").replace("\n", " ")
            context_str += f"{role}: {content}\n"
    context_str += f"Người dùng (hiện tại): {user_query}\n"

    # ── Xây dựng danh sách fields hint từ config ──
    fields_hint = "\n".join(
        f"- {f.key} ({f.extract_hint})" for f in form_cfg.fields
    ) if form_cfg.fields else "- ho_ten, ngay_sinh, dia_chi, so_dien_thoai, email, cccd, nganh_hoc, ly_do"

    # ── Render System Prompt ──
    sys_prompt_raw = prompt_manager.get_system("form_extractor")
    sys_prompt = sys_prompt_raw.replace("{{ fields_config }}", fields_hint)

    # ── Render User Prompt (Jinja2 qua PromptManager) ──
    user_content = prompt_manager.render_user(
        "form_extractor",
        context=context_str
    )

    # ── Config API calls ──
    class _ExtractorConfig:
        model = form_cfg.settings.extractor_model
        provider = form_cfg.settings.provider
        temperature = form_cfg.settings.extractor_temperature
        max_tokens = form_cfg.settings.extractor_max_tokens
        timeout_seconds = form_cfg.settings.extractor_timeout

    try:
        raw_output = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=_ExtractorConfig(),
            node_key="form",
        )

        # ── Parse JSON — strip markdown code fences nếu LLM trả về ──
        cleaned = raw_output.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        # Fallback: tìm object JSON đầu tiên bằng regex
        if not cleaned.startswith("{"):
            match = re.search(r'\{[^{}]*\}', cleaned)
            if match:
                cleaned = match.group(0)

        extracted_data = json.loads(cleaned)

        # ── Chuẩn hoá: bỏ các giá trị null/empty ──
        result = {}
        for key, val in extracted_data.items():
            if val and str(val).lower() not in ("null", "none", ""):
                result[key] = val

        elapsed = time.time() - start_time
        logger.info("Form Extractor [%.3fs] OK -> %d fields co du lieu", elapsed, len(result))
        logger.debug("Form Extractor - Data: %s", result)
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "Form Extractor [%.3fs] Loi: %s -> Fallback Return Empty",
            elapsed, e, exc_info=True,
        )
        return {}
