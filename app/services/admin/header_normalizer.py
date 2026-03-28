"""
Header Normalizer — Chuẩn hóa YAML front-matter cho file Markdown.

Đảm bảo mọi file nạp vào VectorDB đều có header đầy đủ:
  - program_name (chuẩn hóa từ programs_config.yaml)
  - program_level (thac_si / tien_si / dai_hoc / null)
  - source (tên file gốc)
  - academic_year (nếu có)
"""

import re
from typing import Optional
from app.core.config import _load_yaml
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Load danh sách ngành từ programs_config.yaml ──
_programs_data = _load_yaml("programs_config.yaml")

# Build lookup dict: {"qtkd": "Quản Trị Kinh Doanh", ...}
_PROGRAM_NAME_LOOKUP: dict[str, str] = {}
for entry in _programs_data.get("program_names", []):
    canonical = entry.get("name", "")
    for kw_pattern in entry.get("keywords", []):
        # Dùng bản text gốc (không regex) để build lookup nhanh
        # Chỉ lấy keyword đơn giản (không có ký tự regex phức tạp)
        simple_key = kw_pattern.replace("\\s*", " ").replace("\\b", "").lower().strip()
        if simple_key:
            _PROGRAM_NAME_LOOKUP[simple_key] = canonical

_VALID_LEVELS = {"thac_si", "tien_si", "dai_hoc"}

# Regex parse YAML front-matter
_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Tách YAML front-matter ra khỏi nội dung Markdown.

    Returns:
        (metadata_dict, body_content)
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    try:
        import yaml
        meta = yaml.safe_load(match.group(1)) or {}
    except Exception:
        meta = {}

    body = content[match.end():]
    return meta, body


def _normalize_program_name(raw_name: Optional[str]) -> Optional[str]:
    """
    Chuẩn hóa tên ngành từ giá trị thô (viết tắt, viết hoa, etc.)
    về dạng canonical (khớp DB).

    Examples:
        "QTKD" → "Quản Trị Kinh Doanh"
        "marketing" → "Marketing"
        "xyz" → None (không nhận dạng được)
    """
    if not raw_name:
        return None

    # Check trực tiếp
    lower = raw_name.strip().lower()
    if lower in _PROGRAM_NAME_LOOKUP:
        return _PROGRAM_NAME_LOOKUP[lower]

    # Check partial match (chứa keyword)
    for key, canonical in _PROGRAM_NAME_LOOKUP.items():
        if key in lower or lower in key:
            return canonical

    # Không match → trả về nguyên bản (có thể là ngành mới)
    return raw_name.strip()


def _normalize_program_level(raw_level: Optional[str]) -> Optional[str]:
    """
    Chuẩn hóa bậc đào tạo.

    Examples:
        "Thạc sĩ" → "thac_si"
        "tien_si" → "tien_si"
        "ĐH" → "dai_hoc"
    """
    if not raw_level:
        return None

    lower = raw_level.strip().lower()

    # Đã chuẩn
    if lower in _VALID_LEVELS:
        return lower

    # Map text → key
    _LEVEL_MAP = {
        "thạc sĩ": "thac_si", "thac si": "thac_si",
        "cao học": "thac_si", "cao hoc": "thac_si",
        "master": "thac_si", "ths": "thac_si",
        "tiến sĩ": "tien_si", "tien si": "tien_si",
        "phd": "tien_si", "ncs": "tien_si",
        "nghiên cứu sinh": "tien_si",
        "đại học": "dai_hoc", "dai hoc": "dai_hoc",
        "cử nhân": "dai_hoc", "cu nhan": "dai_hoc",
        "đh": "dai_hoc", "dh": "dai_hoc",
    }

    return _LEVEL_MAP.get(lower)


def normalize_header(
    content: str,
    filename: str,
    override_level: Optional[str] = None,
    override_program: Optional[str] = None,
    override_year: Optional[str] = None,
) -> tuple[dict, str]:
    """
    Chuẩn hóa header file Markdown.

    Args:
        content: Nội dung file Markdown đầy đủ.
        filename: Tên file gốc (dùng làm source).
        override_level: Bậc học do Admin chỉ định (nếu có).
        override_program: Ngành do Admin chỉ định (nếu có).
        override_year: Năm học do Admin chỉ định (nếu có).

    Returns:
        (normalized_metadata, body_content)
        - normalized_metadata: dict chứa program_name, program_level, source...
        - body_content: Nội dung Markdown (không có front-matter)
    """
    meta, body = _parse_frontmatter(content)

    # ── Chuẩn hóa từng field (Admin override > file front-matter > null) ──
    program_name = _normalize_program_name(
        override_program or meta.get("program_name") or meta.get("nganh")
    )
    program_level = _normalize_program_level(
        override_level or meta.get("program_level") or meta.get("bac_hoc")
    )
    academic_year = (
        override_year
        or meta.get("academic_year")
        or meta.get("nam_hoc")
    )

    normalized = {
        "program_name": program_name,
        "program_level": program_level,
        "source": filename,
        "academic_year": academic_year,
        # Giữ lại các field khác từ front-matter gốc
        **{k: v for k, v in meta.items()
           if k not in ("program_name", "program_level", "nganh",
                        "bac_hoc", "academic_year", "nam_hoc")},
    }

    logger.info(
        "HeaderNormalizer - file='%s' program='%s' level='%s'",
        filename, program_name, program_level,
    )

    return normalized, body
