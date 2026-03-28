"""
Form Selector — Chọn mẫu đơn phù hợp bằng LLM Semantic Matching.

CHIẾN LƯỢC:
  1. Quét thư mục maudon/, đọc YAML frontmatter lấy `title`.
  2. Gửi standalone_query + danh sách TITLE cho Gemini Flash.
  3. AI trả về 1-3 filename phù hợp (hỗ trợ multi-form cùng lúc).
  4. Nếu yêu cầu nào không có mẫu sẵn → trả "NONE" kèm mô tả.
"""

import os
import json
import yaml
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config.form_config import form_cfg
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Đường dẫn thư mục chứa mẫu đơn ──
_MAUDON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ))),
    "data", "unstructured", "markdown", "maudon",
)

# ── System prompt cho Form Selector ──
_SELECTOR_SYSTEM_PROMPT = """Bạn là module phân loại mẫu đơn hành chính cho Đại học Tài chính - Marketing (UFM).

NHIỆM VỤ:
Đọc CÂU HỎI của người dùng và DANH SÁCH MẪU ĐƠN CÓ SẴN.
Xác định người dùng cần BAO NHIÊU mẫu đơn (tối đa 3) và chọn file phù hợp cho từng yêu cầu.

QUY TẮC:
1. Phân tích NGỮ NGHĨA câu hỏi, đối chiếu với TITLE từng mẫu đơn.
2. Nếu user yêu cầu NHIỀU đơn (VD: "cho em đơn tiến sĩ và giấy cam đoan") → trả về nhiều item.
3. Nếu CÓ mẫu phù hợp → dùng đúng filename.
4. Nếu KHÔNG có mẫu phù hợp nhưng user vẫn yêu cầu → dùng "NONE" làm filename, kèm mô tả loại đơn.
5. Tối đa 3 item. Trả JSON thuần túy, KHÔNG giải thích.

VÍ DỤ:
- User: "Cho em đơn tiến sĩ" → {"forms": [{"filename": "dondangkitiensi.md", "description": "Đơn đăng ký tiến sĩ"}]}
- User: "Đơn tiến sĩ và giấy cam đoan" → {"forms": [{"filename": "dondangkitiensi.md", "description": "Đơn đăng ký tiến sĩ"}, {"filename": "Giấy cam đoan.md", "description": "Giấy cam đoan"}]}
- User: "Đơn hủy môn" → {"forms": [{"filename": "NONE", "description": "Đơn xin hủy đăng ký môn học"}]}
- User: "Đơn thạc sĩ, đề cương và đơn bảo lưu" → {"forms": [{"filename": "mẪU ĐƠN...md", ...}, {"filename": "Mẫu đề cương.md", ...}, {"filename": "NONE", "description": "Đơn xin bảo lưu kết quả học tập"}]}

ĐỊNH DẠNG: {"forms": [{"filename": "xxx.md", "description": "yyy"}, ...]}"""


def _scan_templates() -> list[dict]:
    """
    Quét thư mục maudon/, đọc YAML frontmatter lấy title + filename.
    Chỉ lấy file .md, bỏ qua .bak.
    """
    templates = []
    if not os.path.isdir(_MAUDON_DIR):
        logger.warning("Form Selector — Không tìm thấy thư mục: %s", _MAUDON_DIR)
        return templates

    for fname in os.listdir(_MAUDON_DIR):
        if not fname.endswith(".md") or fname.endswith(".bak"):
            continue

        fpath = os.path.join(_MAUDON_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    meta = yaml.safe_load(parts[1]) or {}
                    title = meta.get("title", "")
                    if title:
                        templates.append({"filename": fname, "title": title})
                        continue

            templates.append({"filename": fname, "title": fname.replace(".md", "")})
        except Exception as e:
            logger.warning("Form Selector — Lỗi đọc %s: %s", fname, e)

    logger.info("Form Selector — Quét được %d mẫu đơn", len(templates))
    return templates


def _build_catalog_text(templates: list[dict]) -> str:
    """Xây danh sách title dễ đọc cho LLM."""
    lines = []
    for i, t in enumerate(templates, 1):
        lines.append(f"{i}. File: {t['filename']} → Title: {t['title']}")
    return "\n".join(lines)


def select_forms(standalone_query: str) -> list[dict]:
    """
    Chọn 1-3 mẫu đơn phù hợp bằng LLM Semantic Matching.

    Returns:
        List[dict] — Mỗi item có:
          - id: str (tên file hoặc "don_chung")
          - name: str (title hoặc description)
          - template_file: str|None (filename hoặc None nếu NONE)
    """
    templates = _scan_templates()

    if not templates:
        logger.warning("Form Selector — Không có mẫu đơn nào → fallback")
        return [_fallback_result("Mẫu đơn theo yêu cầu")]

    catalog_text = _build_catalog_text(templates)

    user_prompt = (
        f"CÂU HỎI CỦA NGƯỜI DÙNG:\n{standalone_query}\n\n"
        f"DANH SÁCH MẪU ĐƠN CÓ SẴN (chỉ chọn từ đây):\n{catalog_text}\n\n"
        f"Xác định user cần bao nhiêu đơn, chọn file phù hợp cho từng yêu cầu. "
        f"Nếu không có mẫu → dùng filename \"NONE\" kèm description."
    )

    class _SelectorConfig:
        model = form_cfg.settings.selector_model
        provider = form_cfg.settings.provider
        temperature = form_cfg.settings.selector_temperature
        max_tokens = form_cfg.settings.selector_max_tokens
        timeout_seconds = form_cfg.settings.selector_timeout

    try:
        raw = _call_gemini_api_with_fallback(
            system_prompt=_SELECTOR_SYSTEM_PROMPT,
            user_content=user_prompt,
            config_section=_SelectorConfig(),
            node_key="form",
        )

        # Parse JSON
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw_clean)
        form_items = result.get("forms", [])

        if not form_items:
            return [_fallback_result("Mẫu đơn theo yêu cầu")]

        # Giới hạn tối đa 3
        form_items = form_items[:3]

        # Build kết quả
        results = []
        valid_filenames = {t["filename"] for t in templates}

        for item in form_items:
            fname = item.get("filename", "NONE")
            desc = item.get("description", "Mẫu đơn theo yêu cầu")

            if fname != "NONE" and fname in valid_filenames:
                # Có template sẵn
                title = next(
                    (t["title"] for t in templates if t["filename"] == fname), desc
                )
                results.append({
                    "id": fname.replace(".md", ""),
                    "name": title,
                    "template_file": fname,
                })
                logger.info("Form Selector — ✅ Match: '%s' → '%s'", title, fname)
            else:
                # Không có template → AI tự soạn
                results.append(_fallback_result(desc))
                logger.info("Form Selector — 📝 No template: '%s' → AI sẽ tự soạn", desc)

        return results

    except Exception as e:
        logger.warning("Form Selector — LLM lỗi: %s → fallback", e)
        return [_fallback_result("Mẫu đơn theo yêu cầu")]


def _fallback_result(description: str) -> dict:
    """Trả về kết quả fallback khi không match mẫu đơn nào."""
    return {
        "id": "don_chung",
        "name": description,
        "template_file": None,
    }


# ── Backward-compatible: select_form() trả 1 kết quả ──
def select_form(standalone_query: str) -> dict:
    """Chọn 1 mẫu đơn (backward-compatible). Dùng select_forms() thay thế."""
    results = select_forms(standalone_query)
    return results[0] if results else _fallback_result("Mẫu đơn theo yêu cầu")
