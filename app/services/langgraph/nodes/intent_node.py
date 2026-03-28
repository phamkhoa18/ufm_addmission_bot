"""
Intent Router Node — Phân loại ý định người dùng bằng Qwen LLM.

Vị trí trong Graph:
  [embedding_node] → [intent_node] → [rag/form/pr/care/response]

Nhiệm vụ:
  Gọi IntentService (Qwen) để xác định intent của standalone_query.
  Ghi kết quả vào State và điều hướng sang Node tương ứng.

Routing:
  PROCEED_RAG    → rag_node   (Tìm kiếm tuyển sinh, học phí, CTĐT...)
  PROCEED_FORM   → form_node  (Yêu cầu mẫu đơn)
  PROCEED_PR     → pr_node    (Thành tích, so sánh, cơ hội việc làm...)
  PROCEED_CARE   → care_node  (Tâm lý, bỏ học, khiếu nại...)
  GREET          → response   (Template chào hỏi — $0, 0ms)
  CLARIFY        → response   (Template hỏi lại — $0, 0ms)
  BLOCK_FALLBACK → response   (Fallback cứng từ YAML — $0, 0ms)
"""

import time
from app.services.langgraph.state import GraphState
from app.services.intent_service import classify_intent
from app.core.config import query_flow_config
from app.utils.query_analyzer import extract_all
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Map intent_action → next_node ──
_ACTION_TO_NODE: dict = {
    "PROCEED_RAG":            "response",    # (FIXED) RAG context đã lấy xong ở đầu, chạy thẳng ra LLM
    "PROCEED_RAG_UFM_SEARCH": "rag_search",  # Chạy qua luồng Web/Evaluator
    "PROCEED_RAG_PR_SEARCH":  "rag_search",  # Chạy qua luồng Web PR
    "PROCEED_FORM":           "form",
    "PROCEED_PR":             "rag_search",   # PR cũ → fallback
    "PROCEED_CARE":           "care",
    "GREET":              "response",
    "CLARIFY":            "response",
    "BLOCK_FALLBACK":     "response",
}


def intent_node(state: GraphState) -> GraphState:
    """
    🧭 INTENT ROUTER NODE

    Input:
      - state["standalone_query"]: Câu hỏi đã reformulate (từ Context Node)

    Output:
      - state["intent"]:          Tên intent ("HOC_PHI_HOC_BONG", ...)
      - state["intent_summary"]:  Tóm tắt câu hỏi (từ Qwen)
      - state["intent_action"]:   Action routing ("PROCEED_RAG", ...)
      - state["next_node"]:       "rag" / "form" / "pr" / "care" / "response"
      - state["final_response"]:  Ghi sẵn nếu GREET / CLARIFY / BLOCK_FALLBA CK
      - state["response_source"]: Nguồn gốc final_response
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    # ── Phân loại intent (Qwen + edge case ngắn) ──
    result = classify_intent(standalone_query=standalone_query)

    intent        = result["intent"]
    intent_summary = result["intent_summary"]
    intent_action  = result["intent_action"]
    next_node      = _ACTION_TO_NODE.get(intent_action, "response")
    elapsed        = time.time() - start_time

    # ── Trích xuất metadata (Regex 0ms) để filter RAG ──
    query_meta = extract_all(standalone_query)
    program_level = query_meta["program_level"]
    program_name = query_meta["program_name"]
    if program_level:
        logger.info("Intent Node - program_level_filter='%s'", program_level)
    if program_name:
        logger.info("Intent Node - program_name_filter='%s'", program_name)

    logger.info("Intent Node [%.3fs] intent='%s' action='%s' -> %s", elapsed, intent, intent_action, next_node)

    # ════════════════════════════════════════════════════════
    # XỬ LÝ CÁC NHÓM KHÔNG CẦN GỌI RAG (Trả ngay lập tức)
    # ════════════════════════════════════════════════════════

    # ── GREET: Chào lại + Mời hỏi ──
    if intent_action == "GREET":
        chat_history = state.get("chat_history", [])
        if chat_history and len(chat_history) > 0:
            # Tránh lặp lại câu chào hỏi tạ bot sau mỗi câu "cảm ơn" của User ở giữa hội thoại
            greet_msg = (
                "Dạ không có chi! Rất vui vì những thông tin vừa rồi đã giúp ích cho bạn. 😊 "
                "Nếu bạn chợt nhớ ra cần hỏi thêm gì về điểm chuẩn, học phí hay ngành nghề, cứ thoải mái nhắn mình nhé!"
            )
        else:
            greet_msg = query_flow_config.response_templates.get_greet()
            
        logger.info("Intent Node - GREET -> Tra template (has_history: %s)", bool(chat_history))
        return {
            **state,
            "intent": intent,
            "intent_summary": intent_summary,
            "intent_action": intent_action,
            "program_level_filter": program_level,
            "program_name_filter": program_name,
            "next_node": "response",
            "final_response": greet_msg,
            "response_source": "greet_template",
        }

    # ── CLARIFY: Hỏi lại nhẹ nhàng ──
    if intent_action == "CLARIFY":
        clarify_msg = query_flow_config.response_templates.get_clarify()
        logger.info("Intent Node - CLARIFY -> Moi user noi ro hon")
        return {
            **state,
            "intent": intent,
            "intent_summary": intent_summary,
            "intent_action": intent_action,
            "program_level_filter": program_level,
            "program_name_filter": program_name,
            "next_node": "response",
            "final_response": clarify_msg,
            "response_source": "clarify_template",
        }

    # ── BLOCK_FALLBACK: Intent nhóm 4 lọt xuống ──
    if intent_action == "BLOCK_FALLBACK":
        semantic_cfg = query_flow_config.semantic_router
        fallback_msg = semantic_cfg.fallbacks.get(
            intent,
            semantic_cfg.fallback_out_of_scope
        ).strip()
        logger.info("Intent Node - BLOCK_FALLBACK -> intent='%s'", intent)
        return {
            **state,
            "intent": intent,
            "intent_summary": intent_summary,
            "intent_action": intent_action,
            "program_level_filter": program_level,
            "program_name_filter": program_name,
            "next_node": "response",
            "final_response": fallback_msg,
            "response_source": "intent_block",
        }

    # ════════════════════════════════════════════════════════
    # PROCEED: Chuyển sang Agent Node (RAG / Form / PR / Care)
    # ════════════════════════════════════════════════════════
    return {
        **state,
        "intent": intent,
        "intent_summary": intent_summary,
        "intent_action": intent_action,
        "program_level_filter": program_level,
        "program_name_filter": program_name,
        "next_node": next_node,
        "final_response": state.get("final_response", ""),
        "response_source": state.get("response_source", ""),
    }

