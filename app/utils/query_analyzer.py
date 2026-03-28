"""
Query Analyzer — Trích xuất metadata từ câu hỏi bằng Regex (0ms, $0).

Đọc danh sách ngành + bậc đào tạo từ programs_config.yaml.
Admin thêm/xóa ngành → chỉ cần sửa YAML, KHÔNG cần sửa code.

Hỗ trợ:
  - program_level: thac_si | tien_si | dai_hoc | None
  - program_name: tên ngành (QTKD, Marketing, ...) | None
"""

import re
from typing import Optional
from app.core.config import _load_yaml
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# LOAD PATTERNS TỪ YAML (chỉ chạy 1 lần khi import)
# ══════════════════════════════════════════════════════════
_programs_data = _load_yaml("programs_config.yaml")


def _compile_level_patterns(data: dict) -> list:
    """
    Đọc program_levels từ YAML → compile thành list[(level, compiled_regex)].
    Giữ nguyên thứ tự YAML (tien_si > thac_si > dai_hoc).
    """
    levels_cfg = data.get("program_levels", {})
    patterns = []

    for level_name, keyword_list in levels_cfg.items():
        if not keyword_list:
            continue
        # Gộp tất cả keywords thành 1 regex OR
        combined = "|".join(keyword_list)
        try:
            compiled = re.compile(combined, re.IGNORECASE)
            patterns.append((level_name, compiled))
        except re.error as e:
            logger.error("Regex lỗi cho program_level '%s': %s", level_name, e)

    logger.info("Query Analyzer - Loaded %d program_levels từ YAML", len(patterns))
    return patterns


def _compile_name_patterns(data: dict) -> list:
    """
    Đọc program_names từ YAML → compile thành list[(canonical_name, compiled_regex)].
    """
    names_cfg = data.get("program_names", [])
    patterns = []

    for entry in names_cfg:
        name = entry.get("name", "")
        keywords = entry.get("keywords", [])
        if not name or not keywords:
            continue
        combined = "|".join(keywords)
        try:
            compiled = re.compile(combined, re.IGNORECASE)
            patterns.append((name, compiled))
        except re.error as e:
            logger.error("Regex lỗi cho program_name '%s': %s", name, e)

    logger.info("Query Analyzer - Loaded %d program_names từ YAML", len(patterns))
    return patterns


# Compile 1 lần duy nhất khi module được import
_PROGRAM_LEVEL_PATTERNS = _compile_level_patterns(_programs_data)
_PROGRAM_NAME_PATTERNS = _compile_name_patterns(_programs_data)


# ══════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════
def extract_program_level(query: str) -> Optional[str]:
    """
    Trích xuất bậc đào tạo từ câu hỏi.
    Returns: "thac_si" | "tien_si" | "dai_hoc" | None
    """
    if not query:
        return None
    for level, pattern in _PROGRAM_LEVEL_PATTERNS:
        if pattern.search(query):
            return level
    return None


def extract_program_name(query: str) -> Optional[str]:
    """
    Trích xuất tên ngành đào tạo từ câu hỏi.
    Returns: canonical program name (VD: "Marketing") hoặc None.
    """
    if not query:
        return None
    for name, pattern in _PROGRAM_NAME_PATTERNS:
        if pattern.search(query):
            return name
    return None


def extract_all(query: str) -> dict:
    """
    Trích xuất tất cả metadata từ câu hỏi trong 1 lần gọi.
    Returns: {"program_level": str|None, "program_name": str|None}
    """
    return {
        "program_level": extract_program_level(query),
        "program_name": extract_program_name(query),
    }
