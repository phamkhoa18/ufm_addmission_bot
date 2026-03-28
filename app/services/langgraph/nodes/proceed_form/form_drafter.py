"""
Form Drafter — Sinh file Markdown mẫu đơn cuối cùng bằng LLM.

HAI CHẾ ĐỘ:
  CHẾ ĐỘ 1 (Có template): Bám sát template gốc, điền thông tin user vào.
  CHẾ ĐỘ 2 (Không template): Truyền standalone_query cho Gemini Flash,
    AI tự soạn đơn hành chính phù hợp với yêu cầu người dùng.
    Kèm disclaimer "Đây chỉ là mẫu tham khảo".
"""

import time
import os
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config.form_config import form_cfg
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Câu disclaimer khi KHÔNG có template chuẩn ──
_NO_TEMPLATE_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Lưu ý:** Hiện tại hệ thống chưa được cung cấp mẫu đơn chuẩn cho loại yêu cầu này. "
    "Đây chỉ là **mẫu tham khảo** được soạn tự động dựa trên yêu cầu của bạn. "
    "**Bạn hãy đọc kỹ và chỉnh sửa nội dung** cho phù hợp trước khi nộp. "
    "Để có mẫu đơn chính thức, vui lòng liên hệ Phòng Đào tạo hoặc truy cập cổng thông tin sinh viên UFM."
)


def _load_template_content(filename: str) -> tuple[str, str]:
    """
    Đọc file biểu mẫu markdown trên ổ đĩa.
    
    Luôn strip YAML frontmatter (---...---) ra khỏi nội dung đơn.
    Trả về (metadata_text, template_body).
    """
    if not filename:
        return "", ""

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )))
    )
    template_path = os.path.join(
        base_dir, "data", "unstructured", "markdown", "maudon", filename
    )

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = ""
        body = content

        # Bước 1: Strip YAML frontmatter (---...---)
        if content.lstrip().startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                metadata = parts[1].strip()   # YAML metadata (chỉ dùng nội bộ)
                body = parts[2].strip()        # Nội dung sau frontmatter

        # Bước 2: Nếu có marker -start-, chỉ lấy phần sau
        if "-start-" in body:
            body = body.split("-start-", 1)[1].strip()

        return metadata, body

    except Exception as e:
        logger.warning("Form Drafter - Khong the doc file mau %s: %s", template_path, e)
        return "", ""


def _build_extracted_info_block(extracted_fields: dict) -> str:
    """
    Xây dựng block thông tin user đã cung cấp — dạng đơn giản, liệt kê key-value.
    """
    if not extracted_fields:
        return "THÔNG TIN NGƯỜI DÙNG: Chưa cung cấp thông tin cá nhân nào.\n"

    lines = []
    for key, val in extracted_fields.items():
        if val:
            label = key.replace("_", " ").capitalize()
            lines.append(f"- {label}: {val}")

    if not lines:
        return "THÔNG TIN NGƯỜI DÙNG: Chưa cung cấp thông tin cá nhân nào.\n"

    result = "THONG TIN NGUOI DUNG DA CUNG CAP (DIEN NGUYEN VAN VAO MAU DON, KHONG DUOC SUA):\n"
    result += "\n".join(lines) + "\n"
    return result


def generate_form(form_metadata: dict, extracted_fields: dict, standalone_query: str = "") -> str:
    """
    Soạn thảo văn bản hành chính.

    Chế độ 1 (Có template): Bám sát template, dùng temperature từ config (0.4).
    Chế độ 2 (Không template): Gemini Flash tự soạn từ standalone_query,
        dùng temperature 0.3, kèm disclaimer.
    """
    start_time = time.time()

    form_name = form_metadata.get("name", "Mẫu đơn")
    template_file = form_metadata.get("template_file")

    # ── Đọc nội dung mẫu ──
    metadata, template_content = _load_template_content(template_file)
    has_template = bool(template_content)

    if not template_content:
        template_content = ""

    # ── Xây dựng block thông tin user ──
    info_str = _build_extracted_info_block(extracted_fields)

    # ── Render Prompts ──
    sys_prompt_raw = prompt_manager.get_system("form_drafter")
    sys_prompt = sys_prompt_raw.replace("{{ form_name }}", form_name)

    user_content = prompt_manager.render_user(
        "form_drafter",
        template_metadata=metadata,
        template_content=template_content,
        extracted_info=info_str,
        form_name=form_name,
        standalone_query=standalone_query,
        has_template=has_template,
    )

    # ── Chọn Temperature theo chế độ ──
    if has_template:
        # Có template → dùng config gốc (0.4)
        drafter_temp = form_cfg.settings.drafter_temperature
    else:
        # Không template → dùng config riêng (0.3)
        drafter_temp = form_cfg.settings.drafter_temperature_no_template

    class _DrafterConfig:
        model = form_cfg.settings.drafter_model
        provider = form_cfg.settings.provider
        temperature = drafter_temp
        max_tokens = form_cfg.settings.drafter_max_tokens
        timeout_seconds = form_cfg.settings.drafter_timeout

    try:
        draft = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=_DrafterConfig(),
            node_key="form",
        )

        elapsed = time.time() - start_time

        # ── Kèm disclaimer nếu không có template chuẩn ──
        if not has_template:
            draft = draft.strip() + _NO_TEMPLATE_DISCLAIMER
            logger.info(
                "Form Drafter [%.3fs] NO TEMPLATE — AI soạn từ query: '%s' → %d ký tự",
                elapsed, standalone_query[:60], len(draft),
            )
        else:
            logger.info("Form Drafter [%.3fs] TEMPLATE OK → %d ký tự", elapsed, len(draft))

        return draft.strip()

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Form Drafter [%.3fs] Loi: %s", elapsed, e, exc_info=True)
        return "Xin lỗi bạn, hiện tại hệ thống soạn thảo biểu mẫu đang bị lỗi. Vui lòng thử lại sau."
