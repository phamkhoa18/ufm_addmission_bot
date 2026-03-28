# app/utils/query_summarizer.py
"""
Long Query Summarizer — Tóm tắt query dài trước khi đi vào Guardian Pipeline.

Khi user_query >= summarize_threshold (1999 chars):
  1. Gọi LLM nhẹ (Gemini Flash Lite) để nén thông tin chính + ý định
  2. Trả về câu hỏi ngắn gọn (<500 ký tự)
  3. Nếu lỗi API → fallback cắt cứng (fail-safe, không block user)

Model: google/gemini-2.5-flash-lite (OpenRouter)
Chi phí: ~$0.000005/query
Latency: ~300-500ms
"""

import time
from typing import Tuple

from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.services.langgraph.nodes.context_node import _call_gemini_api
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Fallback: cắt cứng nếu LLM tóm tắt thất bại
_HARD_CUT_LENGTH = 1200


def summarize_long_query(query: str) -> Tuple[str, bool]:
    """
    Tóm tắt query dài bằng LLM.

    Args:
        query: Câu hỏi gốc (>= 1999 ký tự)

    Returns:
        (summarized_query, success)
        - success=True:  summarized_query là bản tóm tắt từ LLM
        - success=False: summarized_query là bản cắt cứng (fallback)
    """
    config = query_flow_config.long_query_summarizer
    start_time = time.time()

    try:
        # Tạo config object tương thích với _call_gemini_api
        class _SummarizerConfig:
            pass

        cfg = _SummarizerConfig()
        cfg.provider = config.provider
        cfg.model = config.model
        cfg.temperature = config.temperature
        cfg.max_tokens = config.max_tokens
        cfg.timeout_seconds = config.timeout_seconds

        summarized = _call_gemini_api(
            system_prompt=prompt_manager.get_system("long_query_summarizer"),
            user_content=query,
            config_section=cfg,
        )

        elapsed = time.time() - start_time

        # Validate: bản tóm tắt phải ngắn hơn bản gốc đáng kể
        if not summarized or not summarized.strip():
            logger.warning(
                "QuerySummarizer [%.3fs] LLM trả rỗng → fallback cắt cứng",
                elapsed,
            )
            return query[:_HARD_CUT_LENGTH] + "...", False

        summarized = summarized.strip()

        # Nếu LLM trả về quá dài (>= 1200 chars), trim lại
        if len(summarized) >= 1200:
            summarized = summarized[:1197] + "..."

        logger.info(
            "QuerySummarizer [%.3fs] %s/%s | %d chars → %d chars",
            elapsed,
            config.provider,
            config.model,
            len(query),
            len(summarized),
        )

        return summarized, True

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "QuerySummarizer [%.3fs] LỖI: %s → fallback cắt cứng",
            elapsed,
            e,
            exc_info=True,
        )
        return query[:_HARD_CUT_LENGTH] + "...", False
