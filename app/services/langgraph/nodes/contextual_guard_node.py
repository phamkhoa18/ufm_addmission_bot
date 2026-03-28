"""
Contextual-Guard Node — Chốt 2: Chặn tinh SAU khi có ngữ cảnh.

Vị trí trong Graph:
  [context_node] → [contextual_guard_node] → [intent_node] → ...
                                            ↘ [END] (nếu bị chặn)

Nhiệm vụ:
  Quét standalone_query (đã reformulate, có ngữ cảnh đầy đủ) bằng:
    Layer 2a: Llama 86M Score-based (Groq, ~100ms)
    Layer 2b: Gemini 2.0 Flash (OpenRouter) — SAFE/UNSAFE JSON
  Hai model chạy SONG SONG — ai UNSAFE trước thì CHẶN ngay.
"""

import time
import concurrent.futures
from app.services.langgraph.state import GraphState
from app.utils.guardian_utils import GuardianService
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _timed_check_2a(query: str):
    """Wrap Layer 2a với timing chi tiết."""
    t0 = time.time()
    try:
        is_valid, msg = GuardianService.check_layer_2a_prompt_guard_fast(query)
        elapsed = time.time() - t0
        status = "SAFE" if is_valid else "UNSAFE"
        logger.info("  ⏱️ Layer 2a (Groq Llama 86M): %.3fs → %s %s", elapsed, status, f"| {msg}" if msg else "")
        return is_valid, msg, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("  ⏱️ Layer 2a (Groq Llama 86M): %.3fs → ERROR: %s", elapsed, e)
        return True, f"Bỏ qua 2a ({e})", elapsed


def _timed_check_2b(query: str):
    """Wrap Layer 2b với timing chi tiết."""
    t0 = time.time()
    try:
        is_valid, msg = GuardianService.check_layer_2b_prompt_guard_deep(query)
        elapsed = time.time() - t0
        status = "SAFE" if is_valid else "UNSAFE"
        logger.info("  ⏱️ Layer 2b (Gemini Flash):    %.3fs → %s %s", elapsed, status, f"| {msg}" if msg else "")
        return is_valid, msg, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("  ⏱️ Layer 2b (Gemini Flash):    %.3fs → ERROR: %s", elapsed, e)
        return True, f"Bỏ qua 2b ({e})", elapsed


def contextual_guard_node(state: GraphState) -> GraphState:
    """
    🔴 CHỐT 2: CONTEXTUAL-GUARD — Chặn tinh trên standalone_query.

    Chạy Layer 2a + 2b SONG SONG.
    - UNSAFE → block ngay
    - Cả 2 SAFE → PASS
    - Timeout/Error → cho qua (SAFE)
    """
    query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    logger.info("CONTEXTUAL GUARD — Bắt đầu quét song song 2a + 2b...")

    # ════════════════════════════════════════════════════════
    # LAYER 2: Chạy 2a + 2b SONG SONG bằng ThreadPool
    # ════════════════════════════════════════════════════════
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        future_2a = pool.submit(_timed_check_2a, query)
        future_2b = pool.submit(_timed_check_2b, query)

        results = {}
        for future in concurrent.futures.as_completed([future_2a, future_2b], timeout=5):
            is_valid, msg, layer_time = future.result()
            layer_name = "2a" if future is future_2a else "2b"
            results[layer_name] = (is_valid, msg, layer_time)

            if not is_valid:
                # UNSAFE → block ngay, hủy task còn lại
                pool.shutdown(wait=False, cancel_futures=True)
                elapsed = time.time() - start_time
                logger.warning(
                    "CONTEXTUAL GUARD [%.3fs] BLOCKED by L%s (layer_time=%.3fs)",
                    elapsed, layer_name, layer_time,
                )
                return {
                    **state,
                    "contextual_guard_passed": False,
                    "contextual_guard_blocked_layer": layer_name,
                    "contextual_guard_message": f"[Contextual-Guard L{layer_name} — {elapsed:.3f}s] {msg}",
                    "final_response": msg,
                    "response_source": "contextual_guard",
                }

        pool.shutdown(wait=False, cancel_futures=True)

    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False, cancel_futures=True)
        elapsed = time.time() - start_time
        logger.warning("CONTEXTUAL GUARD [%.3fs] TIMEOUT → cho qua (SAFE)", elapsed)
    except Exception as e:
        pool.shutdown(wait=False, cancel_futures=True)
        elapsed = time.time() - start_time
        logger.error("CONTEXTUAL GUARD [%.3fs] ERROR: %s → cho qua (SAFE)", elapsed, e, exc_info=True)

    # ════════════════════════════════════════════════════════
    # ✅ PASS — standalone_query an toàn
    # ════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("CONTEXTUAL GUARD [%.3fs] PASS", elapsed)
    return {
        **state,
        "contextual_guard_passed": True,
        "contextual_guard_blocked_layer": None,
        "contextual_guard_message": f"[Contextual-Guard PASS — {elapsed:.3f}s] standalone_query an toàn",
    }


