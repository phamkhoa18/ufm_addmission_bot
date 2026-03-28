"""
Form Node — Pipeline 3 bước cho PROCEED_FORM (hỗ trợ multi-form).

Input: standalone_query, chat_history
Output: final_response chứa 1-3 biểu mẫu markdown.

Pipeline:
  Bước 1: Selector  — Gemini 001 chọn 1-3 mẫu đơn theo title metadata
  Bước 2: Extractor — Gemini Flash trích xuất thông tin cá nhân (1 lần duy nhất)
  Bước 3: Drafter   — Gemini Flash soạn từng văn bản hành chính (loop max 3)

Ghost History:
  Response trả về cho Frontend đầy đủ nội dung đơn.
  Nhưng state lưu `form_history_summary` để Frontend biết
  chỉ lưu câu tóm tắt vào chat_history thay vì nội dung form đầy đủ.
"""

import time
from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.proceed_form.form_selector import select_forms
from app.services.langgraph.nodes.proceed_form.field_extractor import extract_fields
from app.services.langgraph.nodes.proceed_form.form_drafter import generate_form
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Giới hạn tối đa ──
_MAX_FORMS = 3


def form_node(state: GraphState) -> GraphState:
    """
    FORM NODE — Tạo 1-3 biểu mẫu hành chính.

    - Extractor chạy 1 lần → lấy thông tin cá nhân chung.
    - Drafter chạy N lần (N = số đơn, max 3).
    - Ghost History: lưu tóm tắt thay vì toàn bộ nội dung đơn.
    """
    start_time = time.time()
    logger.info("========== BAT DAU FORM NODE ==========")

    # ── Guard: Chỉ chạy khi intent đúng là PROCEED_FORM ──
    intent_action = state.get("intent_action", "")
    if intent_action != "PROCEED_FORM":
        logger.info("Form Node - SKIP (intent_action='%s' != 'PROCEED_FORM')", intent_action)
        return state

    query = state.get("standalone_query", state.get("user_query", ""))
    chat_history = state.get("chat_history", [])

    # ══════════════════════════════════════════════════
    # Bước 1: Chọn 1-3 mẫu đơn phù hợp (Multi-Selector)
    # ══════════════════════════════════════════════════
    target_forms = select_forms(query)
    target_forms = target_forms[:_MAX_FORMS]

    form_names = [f.get("name", "N/A") for f in target_forms]
    logger.info(
        "Form Node - %d đơn được chọn: %s",
        len(target_forms), form_names,
    )

    # ══════════════════════════════════════════════════
    # Bước 2: Trích xuất thông tin cá nhân (1 LẦN DUY NHẤT)
    # ══════════════════════════════════════════════════
    extracted = extract_fields(chat_history, query)
    logger.info(
        "Form Node - Extracted %d fields: %s",
        len(extracted), list(extracted.keys()),
    )

    # ══════════════════════════════════════════════════
    # Bước 3: Draft từng đơn (loop max 3)
    # ══════════════════════════════════════════════════
    drafts = []
    draft_summaries = []

    for idx, t_form in enumerate(target_forms, 1):
        form_name = t_form.get("name", "Mẫu đơn")
        has_template = bool(t_form.get("template_file"))

        logger.info(
            "Form Node - Drafting [%d/%d]: '%s' (template=%s)",
            idx, len(target_forms), form_name, "CÓ" if has_template else "KHÔNG",
        )

        draft = generate_form(t_form, extracted, standalone_query=query)
        drafts.append(draft)

        # Tóm tắt cho Ghost History
        mode = "từ mẫu chuẩn" if has_template else "tự soạn (mẫu tham khảo)"
        draft_summaries.append(f"• {form_name} ({mode})")

    # ══════════════════════════════════════════════════
    # Ghép kết quả
    # ══════════════════════════════════════════════════
    if len(drafts) == 1:
        final_draft = drafts[0]
    else:
        # Nhiều đơn → ngăn cách bằng divider
        final_draft = "\n\n---\n\n".join(drafts)

    # ══════════════════════════════════════════════════
    # Ghost History: Câu tóm tắt cho chat_history
    # (Frontend sẽ lưu câu này thay vì toàn bộ nội dung đơn)
    # ══════════════════════════════════════════════════
    summary_lines = "\n".join(draft_summaries)
    history_summary = (
        f"[Hệ thống đã tạo {len(drafts)} biểu mẫu theo yêu cầu:\n"
        f"{summary_lines}\n"
        f"Nội dung chi tiết đã hiển thị cho người dùng. "
        f"Ẩn khỏi lịch sử chat để tối ưu bộ nhớ.]"
    )

    elapsed = time.time() - start_time
    logger.info(
        "========== HOAN TAT FORM NODE (%.3fs) | %d đơn | %d ký tự ==========",
        elapsed, len(drafts), len(final_draft),
    )

    return {
        **state,
        "final_response": final_draft,
        "response_source": "form_template",
        # Ghost History: Frontend dùng trường này để lưu vào chat_history
        "form_history_summary": history_summary,
    }
