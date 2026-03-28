"""
Response Node — LLM Phản Hồi Chính (Node Chốt Sale).

Vị trí trong Graph (Node cuối cùng):
  [rag_search / rag_node / intent_node] → [response_node] → END

Nhiệm vụ:
  1. Kiểm tra response_source: Nếu đã có final_response từ các node
     trước (GREET / CLARIFY / BLOCK_FALLBACK / Guard layers) → Bypass.
  2. Nếu response_source là context từ RAG/Web → Gọi Gemini 3.0 Flash
     Preview sinh câu trả lời cuối cùng cho người dùng.
  3. Ghi kết quả vào state["final_response"].

Logic Bypass (KHÔNG gọi LLM — $0, ~0.001s):
  - "greet_template"       → Lời chào sẵn
  - "clarify_template"     → Hỏi lại sẵn
  - "intent_block"         → Fallback block sẵn
  - "fast_scan_block"      → Prompt Guard block sẵn
  - "contextual_guard"     → Deep Guard block sẵn

Logic Generate (GỌI LLM — ~$0.001, ~1-3s):
  - "rag_db_only"          → Context chỉ từ DB (Evaluator phán YES)
  - "rag_search_context"   → Context từ DB + Web Search
  - Mọi trường hợp khác   → Sinh câu trả lời

Model: google/gemini-3.0-flash-preview (via OpenRouter)
Fallback: Tự động lùi model qua _call_gemini_api_with_fallback.
"""

import time
from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config import query_flow_config
from app.core.config.contact_loader import get_hotline_short
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# DANH SÁCH RESPONSE_SOURCE KHÔNG CẦN GỌI LLM
# ═══════════════════════════════════════════════════════════
_BYPASS_SOURCES = {
    "greet_template",
    "clarify_template",
    "care_template",
    "form_template",              # Form Node đã soạn xong → bypass
    "rag_search_synthesized",     # Sanitizer đã duyệt xong → bypass
    "intent_block",
    "fast_scan",                  # Fast Scan chặn → bypass (matched node output)
    "contextual_guard",
    "keyword_filter",
    "input_validation",
}


def response_node(state: GraphState) -> GraphState:
    """
    🎯 RESPONSE NODE — Node Cuối Cùng của LangGraph.

    Input:
      - state["standalone_query"]:   Câu hỏi đã reformulate
      - state["final_response"]:     Context / Template đã chuẩn bị từ node trước
      - state["response_source"]:    Nguồn gốc (để quyết định Bypass hay Generate)
      - state["rag_context"]:        Context thuần RAG (dự phòng)
      - state["chat_history_text"]:  Lich sử hội thoại (để LLM hiểu ngữ cảnh)

    Output:
      - state["final_response"]:     Câu trả lời cuối cùng cho người dùng
      - state["response_source"]:    Ghi lại nguồn (thêm "_generated" nếu gọi LLM)
    """
    start_time = time.time()
    response_source = state.get("response_source", "")
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    existing_response = state.get("final_response", "")

    logger.info("RESPONSE NODE - Source: %s, Existing: %d ky tu", response_source, len(existing_response))

    # ══════════════════════════════════════════════════════════
    # NHÁNH A: BYPASS — Đã có câu trả lời sẵn, không cần LLM
    # ══════════════════════════════════════════════════════════
    if response_source in _BYPASS_SOURCES:
        elapsed = time.time() - start_time
        logger.info("RESPONSE NODE [%.3fs] BYPASS -> %s", elapsed, response_source)
        return {
            **state,
        }

    # ══════════════════════════════════════════════════════════
    # NHÁNH B: GENERATE — Gọi Gemini 3.0 Flash sinh câu trả lời
    # ══════════════════════════════════════════════════════════
    config = query_flow_config.main_bot

    if not config.enabled:
        elapsed = time.time() - start_time
        logger.info("RESPONSE NODE [%.3fs] Main Bot disabled -> raw context", elapsed)
        return {
            **state,
            "final_response": existing_response or state.get("rag_context", ""),
            "response_source": "main_bot_disabled",
        }

    # ── Chuẩn bị biến cho prompt ──
    rag_context = state.get("rag_context") or ""
    chat_history_text = state.get("chat_history_text") or ""
    web_citations = state.get("web_search_citations") or []

    try:
        # ── Render prompt từ prompts.yaml ──
        sys_prompt = prompt_manager.get_system("main_bot")
        user_content = prompt_manager.render_user(
            "main_bot",
            standalone_query=standalone_query,
            final_response=existing_response,
            rag_context=rag_context,
            chat_history_text=chat_history_text,
            web_citations=web_citations,
        )

        logger.info("RESPONSE NODE - Goi LLM %s/%s, query='%s'", config.provider, config.model, standalone_query[:80])

        # ── Gọi API với fallback tự động ──
        generated = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=config,
            node_key="main_bot",
        )

        elapsed = time.time() - start_time

        if generated and generated.strip():
            logger.info("RESPONSE NODE [%.3fs] LLM OK: %d ky tu", elapsed, len(generated))
            return {
                **state,
                "final_response": generated.strip(),
                "response_source": f"{response_source}_generated",
                }
        else:
            # LLM trả về rỗng → fallback về context thô + contact
            logger.warning("RESPONSE NODE [%.3fs] LLM tra rong -> context tho", elapsed)
            empty_fallback = (
                existing_response or rag_context
                or (
                    "Xin lỗi bạn, mình chưa tìm thấy thông tin phù hợp cho câu hỏi này. "
                    "Bạn có thể diễn đạt cụ thể hơn hoặc liên hệ trực tiếp:\n"
                    f"{get_hotline_short()}"
                )
            )
            return {
                **state,
                "final_response": empty_fallback,
                "response_source": f"{response_source}_fallback",
                }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("RESPONSE NODE [%.3fs] Loi LLM: %s", elapsed, e, exc_info=True)

        # Lỗi API → fallback trả context thô kèm contact info
        fallback = existing_response or rag_context
        if not fallback:
            fallback = (
                "Xin lỗi bạn, hệ thống đang bảo trì tạm thời để cập nhật định kì. "
                "Bạn vui lòng thử lại sau giây lát hoặc liên hệ trực tiếp "
                "để được hỗ trợ nhanh nhất nhé!\n\n"
                f"{get_hotline_short()}"
            )

        return {
            **state,
            "final_response": fallback,
            "response_source": "main_bot_error",
        }


def response_router(state: GraphState) -> str:
    """Conditional Edge: response → end (luôn luôn kết thúc)."""
    return "end"
