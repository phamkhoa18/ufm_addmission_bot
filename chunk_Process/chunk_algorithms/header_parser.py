"""
Header Parser — Parse YAML Frontmatter + Legacy text header.

Module chuyên trách đọc metadata header từ file Markdown UFM.
Hỗ trợ 2 định dạng:

  1. YAML Frontmatter (chuẩn mới):
     ---
     doc_type: thongtinchung
     doc_id: TS-THS-2026-D1
     title: "Tuyển sinh Thạc sĩ đợt 1 năm 2026"
     effective_date: 2026-01-15
     program_level: thac_si
     academic_year: "2026"
     doc_number: "186/TB-ĐHTCM"
     ---
     -start-
     (Nội dung chính)

  2. Legacy text header (tương thích ngược):
     Ngày hiệu lực: 15/01/2026
     Văn bản số: 186 /TB-ĐHTCM
     # THÔNG BÁO
     -start-
     (Nội dung chính)

Luồng gọi:
  from chunk_Process.chunk_algorithms.header_parser import parse_header
  result = parse_header(raw_text)
  body = result["content"]
  meta = result  # valid_from, program_level, academic_year, ...
"""

import re
import unicodedata
from datetime import datetime
from typing import Optional

import yaml


# ================================================================
# SECURITY PATTERNS — Chặn injection qua YAML field values
# ================================================================
#
# Chiến lược NỚI LỎNG hợp lý cho văn bản tuyển sinh UFM:
#   - YAML header fields: Vẫn REJECT injection (Jinja2, Shell) vì header
#     sẽ đi thẳng vào DB metadata / Jinja2 template.
#   - Body content: Chỉ SANITIZE (xóa tag nguy hiểm), KHÔNG reject cả file.
#     Vì body sẽ qua Chunking → Embedding, không chạy code.
#   - Field length: Nới rộng vì title văn bản hành chính VN có thể rất dài.
#   - File size: KHÔNG giới hạn ở module này (ingestion pipeline lo).
#
_INJECTION_PATTERNS = [
    re.compile(r"\{\{.*\}\}"),       # Jinja2 template injection
    re.compile(r"\{%.*%\}"),         # Jinja2 block injection
    re.compile(r"\$\(.*\)"),         # Shell command injection
    re.compile(r"\$\{.*\}"),         # Shell variable injection
]

_DANGEROUS_HTML_TAGS = re.compile(
    r"<\s*(script|iframe|object|embed|form|meta|link|base)\b",
    re.IGNORECASE,
)

# Giới hạn độ dài field — NỚI RỘNG cho văn bản hành chính dài
# (Phụ lục tuyển sinh UFM có title 200+ ký tự là bình thường)
_MAX_FIELD_LENGTH = 2000
_MAX_TITLE_LENGTH = 3000

# Allowed values cho validation
_ALLOWED_DOC_TYPES = {"thongtinchung", "maudon"}
_ALLOWED_PROGRAM_LEVELS = {"dai_hoc", "thac_si", "tien_si", "chung"}

# Level detection patterns (cho legacy header)
_LEVEL_PATTERNS = [
    (r"trình\s+độ\s+tiến\s+sĩ", "tien_si"),
    (r"trình\s+độ\s+thạc\s+sĩ", "thac_si"),
    (r"tiến\s+sĩ", "tien_si"),
    (r"thạc\s+sĩ", "thac_si"),
    (r"đại\s+học", "dai_hoc"),
]


# ================================================================
# SECURITY HELPERS
# ================================================================
def _check_injection(value: str) -> bool:
    """
    Trả True nếu value chứa injection pattern nguy hiểm.

    Lưu ý: Chỉ kiểm tra YAML header fields (ngắn, metadata).
    Body content KHÔNG đi qua hàm này — body chỉ bị sanitize HTML.
    """
    if not isinstance(value, str):
        return False
    for pat in _INJECTION_PATTERNS:
        if pat.search(value):
            return True
    return False


def _sanitize_html(text: str) -> str:
    """
    Xóa các HTML tag nguy hiểm khỏi body content.

    CHỈ xóa tag thực sự nguy hiểm (<script>, <iframe>, ...).
    Giữ nguyên các tag vô hại (<br/>, <b>, <table>, ...) vì
    file Markdown tuyển sinh UFM hay có <br/> trong bảng.
    """
    return _DANGEROUS_HTML_TAGS.sub("", text)


def _truncate_field(value, max_len: int = _MAX_FIELD_LENGTH) -> str:
    """
    Cắt giá trị field quá dài — NỚI LỎNG.

    Chỉ áp dụng cho YAML header fields (metadata ngắn).
    Body content không bị truncate — Pipeline chunking sẽ lo.
    """
    if value is None:
        return value
    value = str(value)
    if len(value) > max_len:
        return value[:max_len]
    return value


def _safe_str(value) -> Optional[str]:
    """Chuyển value về string an toàn, trả None nếu rỗng."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


# ================================================================
# MAIN PARSER
# ================================================================
def parse_header(raw_text: str) -> dict:
    """
    Parse metadata header từ file Markdown. Tự động nhận dạng format.

    Returns:
        dict chứa:
            content:        str   — Nội dung chính SAU -start-
            doc_type:       str | None  — "thongtinchung" | "maudon"
            doc_id:         str | None  — Mã tài liệu
            title:          str | None  — Tiêu đề
            valid_from:     datetime | None
            program_level:  str | None  — "thac_si" | "tien_si" | "dai_hoc" | "chung"
            academic_year:  str | None  — "2026" hoặc "2025-2026"
            doc_number:     str | None  — Số văn bản
            parent_doc_id:  str | None  — Mã tài liệu cha
            keywords:       list[str]   — Từ khóa bổ sung
            source_url:     str | None
            header_context: str | None  — Header text (cho context prefix)
            extra:          dict        — Dict mở rộng cho DB column extra
            warnings:       list[str]   — Danh sách cảnh báo (nếu có)
            errors:         list[str]   — Danh sách lỗi nghiêm trọng (nếu có)
    """
    # Chuẩn hóa Unicode
    raw_text = unicodedata.normalize("NFC", raw_text)

    result = {
        "content": raw_text,
        "doc_type": None,
        "doc_id": None,
        "title": None,
        "valid_from": None,
        "program_level": None,
        "academic_year": None,
        "doc_number": None,
        "parent_doc_id": None,
        "keywords": [],
        "source_url": None,
        "header_context": None,
        "extra": {},
        "warnings": [],
        "errors": [],
    }

    # ── Thử parse YAML Frontmatter trước ──
    yaml_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw_text, re.DOTALL)
    if yaml_match:
        result = _parse_yaml_frontmatter(raw_text, yaml_match, result)
    else:
        # ── Fallback: Legacy text header ──
        result = _parse_legacy_header(raw_text, result)

    # ── Security: Sanitize body content ──
    result["content"] = _sanitize_html(result["content"])

    return result


# ================================================================
# YAML FRONTMATTER PARSER
# ================================================================
def _parse_yaml_frontmatter(
    raw_text: str,
    yaml_match: re.Match,
    result: dict,
) -> dict:
    """Parse YAML Frontmatter format."""
    yaml_block = yaml_match.group(1)
    after_frontmatter = raw_text[yaml_match.end():]

    # ── Security: YAML Bomb check (S2-01) ──
    if "&" in yaml_block or "*" in yaml_block:
        # Kiểm tra nghi vấn YAML anchor/alias
        if re.search(r"[&*]\w+", yaml_block):
            result["errors"].append(
                "S2-01: Frontmatter chua YAML anchor/alias (&/*). "
                "Bi REJECT vi ly do bao mat."
            )
            return result

    # ── Parse YAML ──
    try:
        meta = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as e:
        result["errors"].append(f"YAML parse error: {e}")
        return result

    if not isinstance(meta, dict):
        result["errors"].append("Frontmatter YAML khong phai dict.")
        return result

    # ── Security: Check injection trong YAML header field values (S2-02) ──
    # NỚI LỎNG: Chỉ WARN + strip ký tự nguy hiểm, KHÔNG reject cả file.
    # Lý do: Đôi khi title văn bản hành chính VN có chứa ký tự đặc biệt
    # hợp lệ mà regex bắt nhầm. Chỉ reject khi pattern rõ ràng nguy hiểm.
    for key, value in meta.items():
        if isinstance(value, str) and _check_injection(value):
            result["warnings"].append(
                f"S2-02: Field '{key}' nghi ngo injection pattern. "
                f"Da strip ky tu nguy hiem."
            )
            # Strip injection patterns thay vì reject
            cleaned = value
            for pat in _INJECTION_PATTERNS:
                cleaned = pat.sub("", cleaned)
            meta[key] = cleaned.strip()

    # ── Tách content body ──
    if "-start-" in after_frontmatter:
        parts = after_frontmatter.split("-start-", 1)
        # Phần giữa frontmatter và -start- (nếu có text)
        between_text = parts[0].strip()
        result["content"] = parts[1].strip()
        # Header context = phần giữa frontmatter và -start-
        if between_text:
            result["header_context"] = between_text
    else:
        result["content"] = after_frontmatter.strip()
        result["warnings"].append(
            "V0-02: Khong tim thay marker '-start-'. "
            "Toan bo phan sau frontmatter la body."
        )

    # ── Map fields ──
    # Required fields
    doc_type = _safe_str(meta.get("doc_type"))
    if doc_type and doc_type in _ALLOWED_DOC_TYPES:
        result["doc_type"] = doc_type
    elif doc_type:
        result["errors"].append(
            f"V1-10: doc_type '{doc_type}' khong hop le. "
            "Chi chap nhan: thongtinchung, maudon."
        )
        return result

    doc_id = _safe_str(meta.get("doc_id"))
    if doc_id:
        # V1-13: Chỉ ASCII safe chars
        if re.match(r"^[A-Za-z0-9_-]+$", doc_id):
            result["doc_id"] = _truncate_field(doc_id)
        else:
            result["errors"].append(
                f"V1-13: doc_id '{doc_id}' chua ky tu khong hop le. "
                "Chi dung A-Z, 0-9, gach ngang, gach duoi."
            )
            return result

    result["title"] = _truncate_field(
        _safe_str(meta.get("title")), _MAX_TITLE_LENGTH
    )

    # effective_date → valid_from
    eff_date = meta.get("effective_date")
    if eff_date:
        if isinstance(eff_date, datetime):
            result["valid_from"] = eff_date
        elif isinstance(eff_date, str):
            # ISO 8601
            try:
                result["valid_from"] = datetime.fromisoformat(eff_date)
            except ValueError:
                # Fallback DD/MM/YYYY
                try:
                    result["valid_from"] = datetime.strptime(eff_date, "%d/%m/%Y")
                except ValueError:
                    result["errors"].append(
                        f"V1-11: effective_date '{eff_date}' sai dinh dang. "
                        "Dung: YYYY-MM-DD hoac DD/MM/YYYY."
                    )
                    return result
        elif hasattr(eff_date, "year"):
            # YAML auto-parses date objects
            result["valid_from"] = datetime(eff_date.year, eff_date.month, eff_date.day)

    # Optional fields
    program_level = _safe_str(meta.get("program_level"))
    if program_level:
        if program_level in _ALLOWED_PROGRAM_LEVELS:
            result["program_level"] = program_level
        else:
            result["warnings"].append(
                f"V1-12: program_level '{program_level}' khong hop le. Set null."
            )

    result["academic_year"] = _truncate_field(
        _safe_str(meta.get("academic_year"))
    )
    result["doc_number"] = _truncate_field(
        _safe_str(meta.get("doc_number"))
    )
    result["parent_doc_id"] = _truncate_field(
        _safe_str(meta.get("parent_doc_id"))
    )

    keywords = meta.get("keywords")
    if isinstance(keywords, list):
        result["keywords"] = [
            _truncate_field(str(k)) for k in keywords if k
        ]

    # source_url validation (S2-04) — NỚI LỎNG cho phép http:// (nội bộ UFM)
    source_url = _safe_str(meta.get("source_url"))
    if source_url:
        is_valid_url = (
            (source_url.startswith("https://") or source_url.startswith("http://"))
            and ".." not in source_url
            and not source_url.startswith("file://")
            and not source_url.startswith("data:")
        )
        if is_valid_url:
            result["source_url"] = _truncate_field(source_url)
        else:
            result["warnings"].append(
                "S2-04: source_url khong hop le (phai la http/https, "
                "khong chua '../', 'file://', 'data:'). Dat ve null."
            )

    # ── Build extra dict cho DB column `extra` (JSONB) ──
    result["extra"] = {
        k: v for k, v in {
            "doc_type": result["doc_type"],
            "doc_id": result["doc_id"],
            "doc_number": result["doc_number"],
            "parent_doc_id": result["parent_doc_id"],
            "keywords": result["keywords"] if result["keywords"] else None,
            "source_url": result["source_url"],
            "header_context": result["header_context"],
        }.items() if v is not None
    }

    return result


# ================================================================
# LEGACY HEADER PARSER (tương thích ngược)
# ================================================================
def _parse_legacy_header(raw_text: str, result: dict) -> dict:
    """
    Parse Legacy text header (format cũ trước YAML Frontmatter).

    Format:
        Ngày hiệu lực: 15/01/2026
        Văn bản số: 186 /TB-ĐHTCM
        # THÔNG BÁO
        ## Về việc tuyển sinh trình độ thạc sĩ đợt 1 năm 2026
        -start-
        (Nội dung chính)
    """
    if "-start-" not in raw_text:
        return result

    parts = raw_text.split("-start-", 1)
    header = parts[0].strip()
    result["content"] = parts[1].strip()

    # 1. Trích xuất Ngày hiệu lực
    date_match = re.search(r"Ngày hiệu lực:\s*(\d{2}/\d{2}/\d{4})", header)
    if date_match:
        try:
            result["valid_from"] = datetime.strptime(
                date_match.group(1), "%d/%m/%Y"
            )
        except ValueError:
            pass

    # 2. Nhận dạng Trình độ (program_level)
    header_lower = header.lower()
    for pattern, level in _LEVEL_PATTERNS:
        if re.search(pattern, header_lower):
            result["program_level"] = level
            break

    # 3. Trích xuất Năm tuyển sinh (academic_year)
    year_match = re.search(r"năm\s+(\d{4})", header_lower)
    if year_match:
        result["academic_year"] = year_match.group(1)

    # 4. Văn bản số
    doc_num_match = re.search(r"Văn bản số:\s*(.+)", header)
    if doc_num_match:
        result["doc_number"] = doc_num_match.group(1).strip()

    # 5. Header context
    context_lines = []
    for line in header.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("ngày hiệu lực:"):
            continue
        if stripped.lower().startswith("văn bản số:"):
            continue
        context_lines.append(stripped)

    if context_lines:
        result["header_context"] = "\n".join(context_lines)

    # 6. doc_type tự suy luận từ path (nếu có thể)
    # Sẽ được set bên ngoài bởi ingestion pipeline dựa trên thư mục

    # Build extra dict
    result["extra"] = {
        k: v for k, v in {
            "doc_number": result["doc_number"],
            "header_context": result["header_context"],
        }.items() if v is not None
    }

    return result
