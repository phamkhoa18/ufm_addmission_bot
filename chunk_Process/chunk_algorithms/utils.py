"""
Tiện ích chung cho các thuật toán Chunking.

Tập trung tại đây để tránh trùng lặp (DRY) giữa:
  - semantic_chunker.py
  - hierarchical_chunker.py
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path


# ================================================================
# TEXT NORMALIZATION
# ================================================================
def normalize_vietnamese(text: str) -> str:
    """Chuẩn hóa Unicode NFC cho tiếng Việt."""
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    """Xử lý khoảng trắng thừa (không phá cấu trúc bảng Markdown)."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ================================================================
# TOKEN ESTIMATION — SEGMENT-AWARE MIXER
# ================================================================
def estimate_tokens(text: str, chars_per_token: float = 0) -> int:
    """
    Ước tính số token — SEGMENT-AWARE MIXER tách riêng ASCII và Non-ASCII.

    Chiến lược Mixer:
      1. Non-ASCII (dấu tiếng Việt):    1.5 chars/token  ← BPE tách mạnh
      2. ASCII chữ/số:                   4.0 chars/token  ← BPE xử lý tốt
      3. Dấu câu / ký tự đặc biệt:     2.0 chars/token  ← thường 1 token riêng
      4. Khoảng trắng:                   5.0 chars/token  ← merge vào token liền kề
    Cộng tổng + safety buffer 5%.

    Args:
        text: Văn bản cần ước tính.
        chars_per_token: Nếu > 0 → dùng hệ số cố định thay Mixer (backward-compatible).
    """
    if not text:
        return 1

    if chars_per_token > 0:
        return max(1, int(len(text) / chars_per_token))

    non_ascii_chars = 0
    ascii_alpha_num = 0
    punctuation_special = 0
    whitespace_chars = 0

    for ch in text:
        if not ch.isascii():
            non_ascii_chars += 1
        elif ch.isspace():
            whitespace_chars += 1
        elif ch.isalnum():
            ascii_alpha_num += 1
        else:
            punctuation_special += 1

    subtotal = (
        non_ascii_chars / 1.5
        + ascii_alpha_num / 4.0
        + punctuation_special / 2.0
        + whitespace_chars / 5.0
    )
    total = subtotal + max(1, subtotal * 0.05)
    return max(1, int(total))


# ================================================================
# SENTENCE SPLITTING — TIẾNG VIỆT
# ================================================================
def split_sentences_vietnamese(text: str) -> list[str]:
    """
    Tách câu tiếng Việt theo dấu câu và xuống dòng.

    ⚠️ Không dùng cho bảng Markdown — bảng không có dấu chấm câu cuối câu.
    Dùng is_markdown_table() để kiểm tra trước.
    """
    sentences = re.split(r'(?<=[.!?;])\s+(?=[A-ZÀ-Ỹa-zà-ỹ\d])', text)
    result = []
    for sent in sentences:
        parts = sent.split("\n\n")
        result.extend(p.strip() for p in parts if p.strip())
    return result


# ================================================================
# MARKDOWN TABLE DETECTION
# ================================================================
def is_markdown_table(text: str) -> bool:
    """
    Kiểm tra đoạn văn bản có phải Markdown table không.

    Tiêu chí:
      - Có ít nhất 2 dòng bắt đầu bằng '|'
      - Có dòng separator chứa '---'
    """
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    pipe_lines = [ln for ln in lines if ln.startswith("|")]
    has_separator = any(re.match(r'^\|[\s\-:|]+\|', ln) for ln in lines)
    return len(pipe_lines) >= 2 and has_separator


# ================================================================
# DOCUMENT HEADER PARSING — Trích xuất metadata từ Header
# ================================================================

# Level display names
LEVEL_DISPLAY = {
    "thac_si": "Thạc sĩ",
    "tien_si": "Tiến sĩ",
    "dai_hoc": "Đại học",
    "unknown": "",
}

# ── Mã ngành mapping — load từ file JSON (tránh hardcode) ──
_MAPPING_JSON = (
    Path(__file__).resolve().parents[2]
    / "app" / "core" / "config" / "yaml" / "admissions_mapping.json"
)

# Cache module-level (chỉ đọc JSON 1 lần)
_MA_NGANH_MAP_CACHE: dict | None = None


def _load_ma_nganh_map() -> dict:
    """
    Đọc admissions_mapping.json và chuyển đổi key
    từ dạng '<level>|<name>' → (level, name) tuple.
    Load 1 lần duy nhất, cache lại cho các lần sau.
    """
    global _MA_NGANH_MAP_CACHE
    if _MA_NGANH_MAP_CACHE is not None:
        return _MA_NGANH_MAP_CACHE

    if not _MAPPING_JSON.exists():
        # File không tồn tại → log cảnh báo nhẹ, trả dict rỗng
        import warnings
        warnings.warn(
            f"[utils] admissions_mapping.json không tìm thấy: {_MAPPING_JSON}. "
            "lookup_ma_nganh sẽ luôn trả None.",
            stacklevel=2,
        )
        _MA_NGANH_MAP_CACHE = {}
        return _MA_NGANH_MAP_CACHE

    with open(_MAPPING_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ma_nganh_raw: dict = raw.get("ma_nganh", {})
    # Chuyển key dạng str "level|name" → tuple (level, name)
    _MA_NGANH_MAP_CACHE = {
        tuple(k.split("|", 1)): v
        for k, v in ma_nganh_raw.items()
    }
    return _MA_NGANH_MAP_CACHE


# ── Pattern nhận dạng trình độ từ header ──
_LEVEL_PATTERNS = [
    (r"trình\s+độ\s+tiến\s+sĩ", "tien_si"),
    (r"trình\s+độ\s+thạc\s+sĩ", "thac_si"),
    (r"tiến\s+sĩ", "tien_si"),
    (r"thạc\s+sĩ", "thac_si"),
    (r"đại\s+học", "dai_hoc"),
]


def lookup_ma_nganh(program_level: str, program_name: str) -> str | None:
    """
    Tra cứu mã ngành từ admissions_mapping.json.
    Nếu không tìm thấy → trả None (an toàn cho DB).
    """
    if not program_level or not program_name:
        return None
    ma_nganh_map = _load_ma_nganh_map()
    key = (program_level, program_name.strip().lower())
    return ma_nganh_map.get(key, None)



def parse_document_header(raw_text: str) -> dict:
    """
    Phân tích Header trước '-start-' để trích xuất metadata tự động.

    Hỗ trợ 2 format:
      1. YAML Frontmatter (chuẩn mới) — ưu tiên
      2. Legacy text header (Ngày hiệu lực:...) — tương thích ngược

    Wrapper gọi sang header_parser.parse_header() rồi trả về dict
    với cùng shape cũ để KHÔNG phá vỡ callers hiện tại.

    Returns:
        dict với các key:
            - "content":        str   — Nội dung chính SAU `-start-`
            - "valid_from":     datetime | None
            - "program_level":  str | None  ("thac_si", "tien_si", "dai_hoc")
            - "header_context": str | None  — Phần header (tiêu đề + mô tả)
            - "academic_year":  str | None  (VD: "2026")
            - (Bổ sung các key mới từ YAML nếu có)
    """
    from chunk_Process.chunk_algorithms.header_parser import parse_header
    return parse_header(raw_text)

def build_context_prefix(breadcrumb: str, source: str, extra: dict = None) -> str:
    """
    Tạo tiền tố ngữ cảnh thống nhất bổ sung vào đầu mỗi chunk để bơm metadata.
    Giúp LLM/Vector Search nhận diện rõ chunk này thuộc bối cảnh nào.

    Args:
        breadcrumb: Chuỗi thư mục cấu trúc, VD: "1. Ngành tuyển sinh"
        source: Tên file gốc
        extra: Tuỳ chọn dict (có thể chứa 'extra': {'header_context': '...'})
    """
    parts = []
    # 1. Ưu tiên Header Context (Thông báo/Phụ lục chung chung)
    header_ctx = None
    if extra:
        header_ctx = extra.get("header_context")
        if not header_ctx and isinstance(extra.get("extra"), dict):
            header_ctx = extra["extra"].get("header_context")
            
    if header_ctx:
        parts.append(header_ctx)

    # 2. Source & Mục cụ thể
    parts.append(f"[Nguồn: {source}]")
    if breadcrumb:
        parts.append(f"[Mục: {breadcrumb}]")
        
    return "\n".join(parts) + "\n\n"

