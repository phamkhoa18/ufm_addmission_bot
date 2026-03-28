"""
LangGraph Graph Builder — Topology TỐI ƯU CHI PHÍ cho UFM Admission Bot.

Nguyên tắc thiết kế:
  Intent phân loại TRƯỚC → chỉ gọi Embedding/RAG khi THỰC SỰ CẦN.
  Câu chào hỏi "Xin chào" → KHÔNG tốn tiền embedding + DB search.

Topology tối ưu:
  fast_scan ─┬─ BLOCKED → END
             └─ context → guard ─┬─ BLOCKED → END
                                  └─ intent (ROUTER TRUNG TÂM)
                                       │
                                       ├─ GREET/CLARIFY/BLOCK → response → END        (💰 $0!)
                                       ├─ PROCEED_FORM → form → response → END        (💰 $0!)
                                       ├─ PROCEED_CARE → care → response → END        (💰 $0!)
                                       └─ PROCEED_RAG* → multi_query → embedding → rag
                                                                                   │
                                                          ┌────────────────────────┘
                                                          ├─ PROCEED_RAG → response → END
                                                          └─ PROCEED_RAG_*_SEARCH → rag_search → response → END

Chi phí so sánh:
  ┌──────────────────────────┬──────────────┬──────────────┐
  │ Luồng                   │ Cũ (9 nodes) │ Mới (tối ưu) │
  ├──────────────────────────┼──────────────┼──────────────┤
  │ "Xin chào" (GREET)      │ 9 nodes, $$$│ 4 nodes, $0! │
  │ "Ok thks" (CLARIFY)     │ 9 nodes, $$$│ 4 nodes, $0! │
  │ "Hack UFM" (BLOCK)      │ 3 nodes, $0 │ 3 nodes, $0  │
  │ "Học phí?" (PROCEED_RAG)│ 9 nodes, $$$│ 9 nodes, $$$ │
  └──────────────────────────┴──────────────┴──────────────┘

Sử dụng:
    from app.services.langgraph.graph_builder import chat_graph
    result = chat_graph.invoke(state, config={"configurable": {"thread_id": "user_123"}})
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.services.langgraph.state import GraphState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# IMPORT TẤT CẢ NODE FUNCTIONS (giữ nguyên 100%)
# ══════════════════════════════════════════════════════════
from app.services.langgraph.nodes.fast_scan_node import fast_scan_node
from app.services.langgraph.nodes.context_node import context_node
from app.services.langgraph.nodes.contextual_guard_node import contextual_guard_node
from app.services.langgraph.nodes.multi_query_node import multi_query_node
from app.services.langgraph.nodes.embedding_node import embedding_node
from app.services.langgraph.nodes.rag_node import rag_node
from app.services.langgraph.nodes.intent_node import intent_node
from app.services.langgraph.nodes.response_node import response_node
from app.services.langgraph.nodes.care_node import care_node

# Sub-graphs
from app.services.langgraph.nodes.proceed_form.graph import form_node
from app.services.langgraph.nodes.proceed_rag_search.graph import proceed_rag_search_pipeline


# ══════════════════════════════════════════════════════════
# ROUTER FUNCTIONS — Quyết định rẽ nhánh
# ══════════════════════════════════════════════════════════

# Các intent_action cần chạy RAG chain (tốn tiền embedding + DB)
_RAG_ACTIONS = {
    "PROCEED_RAG",
    "PROCEED_RAG_UFM_SEARCH",
    "PROCEED_RAG_PR_SEARCH",
    "PROCEED_PR",
}


def _fast_scan_router(state: dict) -> str:
    """FastScan: PASS → context | BLOCKED → END."""
    if state.get("fast_scan_passed", False):
        return "context"
    return END


def _guard_router(state: dict) -> str:
    """
    Guard: PASS → intent | BLOCKED → END.

    ⚡ TỐI ƯU: Đi thẳng tới Intent (KHÔNG qua Embedding/RAG).
    Intent chỉ cần standalone_query để phân loại — không cần vector.
    """
    if state.get("contextual_guard_passed", False):
        return "intent"
    return END


def _intent_router(state: dict) -> str:
    """
    Router trung tâm — Quyết định có cần RAG hay không.

    💰 Tối ưu chi phí:
      - GREET/CLARIFY/BLOCK → response trực tiếp ($0, skip RAG!)
      - PROCEED_FORM → form ($0, skip RAG!)
      - PROCEED_CARE → care ($0, skip RAG!)
      - PROCEED_RAG* → multi_query (bắt đầu RAG chain, tốn tiền)
    """
    intent_action = state.get("intent_action", "")

    # Luồng cần RAG: multi_query → embedding → rag → ...
    if intent_action in _RAG_ACTIONS:
        return "multi_query"

    # Luồng KHÔNG cần RAG
    next_node = state.get("next_node", "response")
    if next_node in ("form", "care"):
        return next_node

    return "response"


def _post_rag_router(state: dict) -> str:
    """
    Router SAU khi RAG xong — Quyết định bước tiếp theo.

      - PROCEED_RAG → response (LLM sinh câu trả lời từ rag_context)
      - PROCEED_RAG_*_SEARCH → rag_search (web search bổ sung)
    """
    intent_action = state.get("intent_action", "")
    if intent_action in ("PROCEED_RAG_UFM_SEARCH", "PROCEED_RAG_PR_SEARCH", "PROCEED_PR"):
        return "rag_search"
    return "response"


# ══════════════════════════════════════════════════════════
# BUILD GRAPH — Topology tối ưu chi phí
# ══════════════════════════════════════════════════════════

def _add_common_edges(graph):
    """Shared edge topology cho cả 2 graph (full + pre-response)."""
    graph.set_entry_point("fast_scan")

    # Fixed Edges
    graph.add_edge("context", "guard")
    graph.add_edge("multi_query", "embedding")
    graph.add_edge("embedding", "rag")
    graph.add_edge("form", "response")
    graph.add_edge("care", "response")
    graph.add_edge("rag_search", "response")
    graph.add_edge("response", END)

    # Conditional Edges
    graph.add_conditional_edges(
        "fast_scan", _fast_scan_router,
        {"context": "context", END: END},
    )
    graph.add_conditional_edges(
        "guard", _guard_router,
        {"intent": "intent", END: END},
    )
    graph.add_conditional_edges(
        "intent", _intent_router,
        {"response": "response", "form": "form", "care": "care", "multi_query": "multi_query"},
    )
    graph.add_conditional_edges(
        "rag", _post_rag_router,
        {"response": "response", "rag_search": "rag_search"},
    )


def _add_common_nodes(graph, response_fn):
    """Shared node registration — only response_fn differs."""
    graph.add_node("fast_scan", fast_scan_node)
    graph.add_node("context", context_node)
    graph.add_node("guard", contextual_guard_node)
    graph.add_node("intent", intent_node)
    graph.add_node("multi_query", multi_query_node)
    graph.add_node("embedding", embedding_node)
    graph.add_node("rag", rag_node)
    graph.add_node("form", form_node)
    graph.add_node("care", care_node)
    graph.add_node("rag_search", proceed_rag_search_pipeline)
    graph.add_node("response", response_fn)


def build_chat_graph():
    """
    Build và compile LangGraph pipeline tối ưu.
    Response node gọi LLM non-streaming (dùng cho /message endpoint).
    """
    graph = StateGraph(GraphState)
    _add_common_nodes(graph, response_node)
    _add_common_edges(graph)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    logger.info("LangGraph - Chat Graph compiled: 11 nodes, 4 conditional edges")
    return compiled


# ── Passthrough Response Node (cho streaming pipeline) ──
def _response_passthrough(state: GraphState) -> GraphState:
    """
    No-op: Giữ nguyên state, KHÔNG gọi LLM.
    Pipeline streaming sẽ tự gọi LLM streaming bên ngoài.
    """
    return {**state}


def build_chat_graph_pre_response():
    """
    Giống chat_graph nhưng response_node là passthrough (no-op).

    Dùng cho streaming pipeline:
      1. Chạy graph này → lấy được RAG context, intent, etc.
      2. Sau đó gọi streaming LLM API trực tiếp → yield từng token thật.
    """
    graph = StateGraph(GraphState)
    _add_common_nodes(graph, _response_passthrough)  # ← PASSTHROUGH!
    _add_common_edges(graph)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    logger.info("LangGraph - Pre-Response Graph compiled (for streaming)")
    return compiled


# ══════════════════════════════════════════════════════════
# SINGLETON — Compile 1 lần khi import, dùng mãi
# ══════════════════════════════════════════════════════════
chat_graph = build_chat_graph()
chat_graph_pre_response = build_chat_graph_pre_response()

