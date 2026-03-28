"""
Context Evaluator (Self-RAG Gate) — Quyết định có cần Web Search không.

Vị trí trong Graph:
  [rag_node] → [evaluator] → {YES → response | NO → proceed_rag_search}

Nhiệm vụ:
  1. Nhận rag_context (5 Parent Chunks từ DB) + standalone_query.
  2. Gọi Gemini 3.0 Flash Preview phán đoán: "YES" (đủ) hoặc "NO" (thiếu).
  3. Trả về Boolean.

Quy tắc rẽ nhánh (nằm trong graph.py):
  - Intent là PR/Uy tín  → LUÔN chạy Web Search (bỏ qua Evaluator).
  - Intent thông tin đào tạo → Chạy Evaluator:
      YES → Trả thẳng context từ DB, bỏ qua Web Search.
      NO  → Chạy Web Search bổ sung.

Model: google/gemini-3.0-flash-preview (OpenRouter)
Chi phí: ~$0.00001 (chỉ nhả 1 token YES/NO)
Latency: ~200-400ms
"""

import time
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_rag_context(
    standalone_query: str,
    rag_context: str,
) -> bool:
    """
    🧠 CONTEXT EVALUATOR — Đánh giá ngữ cảnh DB có đủ trả lời không.

    Args:
        standalone_query: Câu hỏi đã reformulate.
        rag_context: Chuỗi text từ 5 Parent Chunks gộp lại.

    Returns:
        True  → DB đủ thông tin, BỎ QUA Web Search.
        False → DB thiếu, CẦN Web Search bổ sung.
    """
    config = query_flow_config.context_evaluator
    start_time = time.time()

    # Nếu evaluator bị tắt → mặc định NO (luôn chạy Web Search)
    if not config.enabled:
        logger.info("Evaluator - bi tat -> mac dinh chay Web Search")
        return False

    # Nếu rag_context rỗng → chắc chắn NO
    if not rag_context or len(rag_context.strip()) < 50:
        elapsed = time.time() - start_time
        logger.info("Evaluator [%.3fs] Context rong/qua ngan -> can Web Search", elapsed)
        return False

    try:
        # ── Render prompt ──
        sys_prompt = prompt_manager.get_system("context_evaluator")
        user_content = prompt_manager.render_user(
            "context_evaluator",
            standalone_query=standalone_query,
            rag_context=rag_context,
        )

        # ── Gọi API ──
        raw_output = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=config,
            node_key="context_evaluator",
        )

        # ── Parse kết quả ──
        verdict = raw_output.strip().upper()
        is_sufficient = verdict.startswith("YES")

        elapsed = time.time() - start_time

        if is_sufficient:
            logger.info("Evaluator [%.3fs] YES -> DB du thong tin", elapsed)
        else:
            logger.info("Evaluator [%.3fs] NO -> DB thieu, can Web Search", elapsed)

        return is_sufficient

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Evaluator [%.3fs] Loi: %s -> Fallback Web Search", elapsed, e, exc_info=True)
        return False
