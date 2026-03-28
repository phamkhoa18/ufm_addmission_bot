"""
Fast-Scan Node — Chốt 1: Chặn thô TRƯỚC khi gọi Gemini.

Vị trí trong Graph:
  [START] → [fast_scan_node] → [context_node] → ...
                              ↘ [END] (nếu bị chặn)

Nhiệm vụ:
  Quét user_query thô bằng Regex + kiểm tra độ dài.
  KHÔNG GỌI API (trừ trường hợp query dài cần tóm tắt).
  Chi phí: $0 (bình thường) | ~$0.000005 (khi tóm tắt query dài)

Layers:
  Layer 0: Input Validation — Max 2000 ký tự (chống DoS/spam)
           + Auto-summarize nếu >= 1999 chars (LLM tóm tắt)
  Layer 1a: Keyword Filter — Từ cấm nhạy cảm (bạo lực, ma tuý, cờ bạc)
  Layer 1b: Injection Filter — Prompt Injection/Jailbreak pattern

Lợi ích:
  - Query quá dài (> 2000) → CHẶN NGAY, không tốn tiền
  - Query dài (1999-2000) → Tóm tắt bằng LLM nhẹ, giữ thông tin quan trọng
  - Query rác/tấn công → CHẶN NGAY, không tốn tiền gọi Gemini
"""

import time
from app.services.langgraph.state import GraphState
from app.utils.guardian_utils import GuardianService
from app.utils.query_summarizer import summarize_long_query
from app.core.config import query_flow_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def fast_scan_node(state: GraphState) -> GraphState:
    """
    🟢 CHỐT 1: FAST-SCAN — Chặn thô trên user_query gốc.

    Input:  state["user_query"]
    Output: state["fast_scan_passed"], state["fast_scan_blocked_layer"],
            state["fast_scan_message"], state["normalized_query"],
            state["original_query"], state["query_was_summarized"],
    """
    query = state.get("user_query", "")
    start_time = time.time()

    iv_config = query_flow_config.input_validation
    original_query = ""
    query_was_summarized = False

    # ════════════════════════════════════════════════════════
    # LAYER 0a: Hard Limit — Chống DoS (> 2000 ký tự → CHẶN)
    # ════════════════════════════════════════════════════════
    if len(query) > iv_config.max_input_chars:  
        elapsed = time.time() - start_time
        logger.info(
            "FastScan L0a BLOCKED: %d chars > %d max (%.3fs)",
            len(query), iv_config.max_input_chars, elapsed,
        )
        return {
            **state,
            "normalized_query": query[:200].lower(),  # Chỉ normalize mẫu nhỏ cho log
            "original_query": "",
            "query_was_summarized": False,
            "fast_scan_passed": False,
            "fast_scan_blocked_layer": 0,
            "fast_scan_message": f"[Fast-Scan L0a — {elapsed:.3f}s] Chặn DoS: {len(query)} chars",
            "final_response": iv_config.fallback_too_long,
            "response_source": "fast_scan",
        }

    # ════════════════════════════════════════════════════════
    # LAYER 0b: Long Query Summarizer (>= 1999 chars → LLM tóm tắt)
    # ════════════════════════════════════════════════════════
    if len(query) >= iv_config.summarize_threshold:
        logger.info(
            "FastScan L0b: Query dài %d chars (>= %d) → gọi LLM tóm tắt...",
            len(query), iv_config.summarize_threshold,
        )
        summarized, success = summarize_long_query(query)

        original_query = query       # Lưu bản gốc
        query = summarized           # Thay bằng bản tóm tắt
        query_was_summarized = True

        elapsed_sum = time.time() - start_time
        if success:
            logger.info(
                "FastScan L0b: Tóm tắt OK (%.3fs) | %d → %d chars | Tóm tắt: '%s'",
                elapsed_sum, len(original_query), len(summarized), summarized[:120],
            )
        else:
            logger.warning(
                "FastScan L0b: LLM lỗi → fallback cắt cứng (%.3fs) | %d → %d chars",
                elapsed_sum, len(original_query), len(summarized),
            )

    # ── Chuẩn hóa teencode (lowercase + thay teencode) ──
    normalized = GuardianService.normalize_text(query)

    # ════════════════════════════════════════════════════════
    # LAYER 1a: Keyword Filter — Từ cấm nhạy cảm
    # ════════════════════════════════════════════════════════
    is_valid, msg = GuardianService.check_layer_1_keyword_filter(normalized)
    if not is_valid:
        elapsed = time.time() - start_time
        return {
            **state,
            "user_query": query,
            "normalized_query": normalized,
            "original_query": original_query,
            "query_was_summarized": query_was_summarized,
            "fast_scan_passed": False,
            "fast_scan_blocked_layer": 1,
            "fast_scan_message": f"[Fast-Scan L1a — {elapsed:.3f}s] {msg}",
            "final_response": msg,
            "response_source": "fast_scan",
        }

    # ════════════════════════════════════════════════════════
    # LAYER 1b: Injection Filter — Prompt Injection/Jailbreak
    # ════════════════════════════════════════════════════════
    is_valid, msg = GuardianService.check_layer_1b_injection_filter(normalized)
    if not is_valid:
        elapsed = time.time() - start_time
        return {
            **state,
            "user_query": query,
            "normalized_query": normalized,
            "original_query": original_query,
            "query_was_summarized": query_was_summarized,
            "fast_scan_passed": False,
            "fast_scan_blocked_layer": 1,
            "fast_scan_message": f"[Fast-Scan L1b — {elapsed:.3f}s] {msg}",
            "final_response": msg,
            "response_source": "fast_scan",
        }

    # ════════════════════════════════════════════════════════
    # ✅ PASS — Cho qua Context Node (Gemini Lite reformulate)
    # ════════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    # Cập nhật user_query nếu đã summarize (để các node sau dùng bản ngắn)
    result_state = {
        **state,
        "user_query": query,
        "normalized_query": normalized,
        "original_query": original_query,
        "query_was_summarized": query_was_summarized,
        "fast_scan_passed": True,
        "fast_scan_blocked_layer": None,
        "fast_scan_message": f"[Fast-Scan PASS — {elapsed:.3f}s] Sạch, cho qua Context Node",
    }

    if query_was_summarized:
        result_state["fast_scan_message"] = (
            f"[Fast-Scan PASS — {elapsed:.3f}s] "
            f"Query đã tóm tắt ({len(original_query)} → {len(query)} chars), cho qua Context Node"
        )

    return result_state


