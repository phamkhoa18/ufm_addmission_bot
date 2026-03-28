"""
Chat Workflow — Production Runner sử dụng LangGraph StateGraph chuẩn thư viện.

Đóng gói toàn bộ luồng xử lý tin nhắn thành 1 phương thức reusable:
  Fast Scan → Context → Guard → Multi-Query → Embedding → RAG → Intent → Agent → Response

Sử dụng:
  from app.services.chat_workflow import run_chat_pipeline

  # Thread-safe, gọi bao nhiêu lần cũng được
  result = run_chat_pipeline(
      query="Học phí ngành Marketing?",
      chat_history=[],
      session_id="user_abc123",
  )
  print(result["response"])
"""

import time
from typing import Optional

from app.utils.logger import get_logger
from app.services.langgraph.graph_builder import chat_graph
from app.services.langgraph.nodes.context_node import (
    _stream_gemini_api_with_fallback,
)

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# INITIAL STATE FACTORY — Tạo state mới cho mỗi lần gọi
# ══════════════════════════════════════════════════════════
def _create_initial_state(
    query: str,
    chat_history: list[dict],
) -> dict:
    """
    Tạo state ban đầu cho pipeline.
    Mỗi lần gọi tạo dict MỚI — an toàn cho multi-thread.
    """
    return {
        # ── Input ──
        "user_query": query,
        "chat_history": chat_history,

        # ── Fast-Scan output ──
        "original_query": "",
        "query_was_summarized": False,
        "normalized_query": "",

        # ── Context output ──
        "standalone_query": "",

        # ── Guard output ──
        "fast_scan_passed": None,
        "fast_scan_blocked_layer": None,
        "fast_scan_message": "",
        "contextual_guard_passed": None,
        "contextual_guard_blocked_layer": None,
        "contextual_guard_message": "",

        # ── Multi-Query + Embedding output ──
        "multi_queries": [],
        "query_embeddings": [],

        # ── RAG output ──
        "rag_context": "",
        "retrieved_chunks": [],

        # ── Intent output ──
        "intent": "",
        "intent_summary": "",
        "intent_action": "",
        "program_level_filter": None,
        "program_name_filter": None,

        # ── Response ──
        "final_response": "",
        "response_source": "",

        # ── Form Node: Ghost History ──
        "form_history_summary": "",
    }


# ══════════════════════════════════════════════════════════
# FORMAT RESULT — Đóng gói output cho API layer    
# ══════════════════════════════════════════════════════════
def _format_result(state: dict, elapsed: float) -> dict:
    """
    Chuyển LangGraph state thành dict chuẩn cho API response.

    Xử lý cả 2 trường hợp:
      - Pipeline bị chặn (FastScan/Guard → final_response có sẵn, intent=BLOCKED)
      - Pipeline hoàn thành (Response Node → final_response generated)
    """
    # Kiểm tra pipeline bị chặn bởi Guardian
    blocked = False
    blocked_reason = ""

    if not state.get("fast_scan_passed", True):
        blocked = True
        blocked_reason = state.get("fast_scan_message", "")
    elif state.get("contextual_guard_passed") is False:
        blocked = True
        blocked_reason = state.get("contextual_guard_message", "")

    result = {
        "response": state.get("final_response", ""),
        "source": state.get("response_source", ""),
        "intent": "BLOCKED" if blocked else state.get("intent", ""),
        "intent_action": "BLOCK_FALLBACK" if blocked else state.get("intent_action", ""),
        "blocked": blocked,
        "blocked_reason": blocked_reason,
        "elapsed_seconds": round(elapsed, 3),
    }

    # Ghost History: nếu là form, trả câu tóm tắt để Frontend lưu thay vì nội dung đầy đủ
    form_summary = state.get("form_history_summary", "")
    if form_summary:
        result["history_content"] = form_summary

    return result


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE — Phương thức chính (Production Runner)
# ══════════════════════════════════════════════════════════
def run_chat_pipeline(
    query: str,
    chat_history: Optional[list[dict]] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Chạy toàn bộ LangGraph pipeline từ query → final_response.

    Thread-safe: Mỗi lần gọi tạo state dict MỚI.
    Multi-tenant: Mỗi session_id có bộ nhớ (checkpoint) riêng biệt.

    Args:
        query:        Tin nhắn của người dùng.
        chat_history: Lịch sử chat [{role, content}]. Mặc định [].
        session_id:   ID phiên chat (cho Multi-Tenant Memory + logging).

    Returns:
        {
            "response":        str,   # Câu trả lời cuối cùng
            "source":          str,   # "rag_db_only_generated" | "greet_template" | ...
            "intent":          str,   # "HOC_PHI_HOC_BONG" | "BLOCKED" | ...
            "intent_action":   str,   # "PROCEED_RAG" | "GREET" | "BLOCK_FALLBACK" | ...
            "blocked":         bool,  # True nếu bị Guardian chặn
            "blocked_reason":  str,   # Lý do chặn
            "elapsed_seconds": float, # Thời gian xử lý (giây)
        }
    """
    pipeline_start = time.time()
    log_prefix = f"[{session_id[:8]}]" if session_id else "[chat]"

    logger.info(
        "%s Pipeline START | query='%s' history=%d msgs",
        log_prefix, query[:80], len(chat_history or []),
    )

    # ── Tạo state mới cho lần gọi này ──
    initial_state = _create_initial_state(
        query=query,
        chat_history=chat_history or [],
    )

    # ── Cấu hình Multi-Tenant: mỗi user có session riêng ──
    config = {
        "configurable": {
            "thread_id": session_id or "default_session",
        }
    }

    # ── Gọi LangGraph Pipeline ──
    try:
        result_state = chat_graph.invoke(initial_state, config=config)
    except Exception as e:
        elapsed = time.time() - pipeline_start
        logger.error(
            "%s Pipeline CRASHED after %.3fs: %s",
            log_prefix, elapsed, e, exc_info=True,
        )
        return {
            "response": "Xin lỗi, hệ thống gặp lỗi khi xử lý. Vui lòng thử lại sau.",
            "source": "error",
            "intent": "ERROR",
            "intent_action": "",
            "blocked": False,
            "blocked_reason": "",
            "elapsed_seconds": round(elapsed, 3),
        }

    # ── Đóng gói kết quả ──
    elapsed = time.time() - pipeline_start

    logger.info(
        "%s Pipeline DONE (%.3fs) | intent=%s action=%s source=%s response=%d chars",
        log_prefix, elapsed,
        result_state.get("intent", "?"),
        result_state.get("intent_action", "?"),
        result_state.get("response_source", "?"),
        len(result_state.get("final_response", "")),
    )

    return _format_result(result_state, elapsed)


# ══════════════════════════════════════════════════════════
# STREAM PIPELINE — LangGraph .stream() chuẩn thư viện
# ══════════════════════════════════════════════════════════

# Map tên node → emoji + label đẹp cho UI
_NODE_LABELS = {
    "fast_scan":    ("🛡️", "Kiểm tra nhanh"),
    "context":      ("📝", "Phân tích ngữ cảnh"),
    "guard":        ("🔒", "Bảo mật"),
    "intent":       ("🧭", "Phân loại ý định"),
    "multi_query":  ("🔀", "Sinh biến thể"),
    "embedding":    ("📐", "Nhúng vector"),
    "rag":          ("🗄️", "Tìm kiếm DB"),
    "form":         ("📋", "Tạo mẫu đơn"),
    "care":         ("💚", "Hỗ trợ tâm lý"),
    "rag_search":   ("🌐", "Tìm kiếm Web"),
    "response":     ("💬", "Sinh câu trả lời"),
}


def stream_chat_pipeline(
    query: str,
    chat_history: Optional[list[dict]] = None,
    session_id: Optional[str] = None,
):
    """
    Stream LangGraph pipeline qua generator — mỗi node xong yield 1 event.

    Dùng `chat_graph.stream(state, config, stream_mode="updates")` chuẩn thư viện.

    Yields:
        dict — Mỗi event dạng:
          {"type": "node",  "node": "fast_scan", "label": "🛡️ Kiểm tra nhanh"}
          {"type": "node",  "node": "response",  "label": "💬 Sinh câu trả lời"}
          {"type": "result", "response": "...", "source": "...", ...}
          {"type": "error",  "message": "..."}
    """
    import json as _json

    pipeline_start = time.time()
    log_prefix = f"[{session_id[:8]}]" if session_id else "[stream]"

    logger.info(
        "%s Stream START | query='%s' history=%d msgs",
        log_prefix, query[:80], len(chat_history or []),
    )

    initial_state = _create_initial_state(
        query=query,
        chat_history=chat_history or [],
    )

    config = {
        "configurable": {
            "thread_id": session_id or "default_session",
        }
    }

    last_state = initial_state
    node_timings = []  # [(node_name, elapsed, status)]
    _node_start = time.time()

    try:
        # ── LangGraph .stream() chuẩn thư viện ──
        # stream_mode="updates" → yield {node_name: state_delta} sau mỗi node
        for chunk in chat_graph.stream(
            initial_state, config=config, stream_mode="updates"
        ):
            # chunk là dict: {"fast_scan": {field: value, ...}}
            for node_name, state_delta in chunk.items():
                node_elapsed = time.time() - _node_start

                # Merge delta vào last_state (để có state cuối cùng)
                if isinstance(state_delta, dict):
                    last_state = {**last_state, **state_delta}

                # Track timing
                node_timings.append((node_name, node_elapsed, "PASS"))
                _node_start = time.time()

                # Emit node progress event
                emoji, label = _NODE_LABELS.get(node_name, ("⚙️", node_name))
                yield {
                    "type": "node",
                    "node": node_name,
                    "label": f"{emoji} {label}",
                }

                logger.info("%s Stream - Node '%s' completed (%.3fs)", log_prefix, node_name, node_elapsed)

    except Exception as e:
        elapsed = time.time() - pipeline_start
        logger.error(
            "%s Stream CRASHED after %.3fs: %s",
            log_prefix, elapsed, e, exc_info=True,
        )
        yield {
            "type": "error",
            "message": "Xin lỗi, hệ thống gặp lỗi khi xử lý. Vui lòng thử lại sau.",
        }
        return

    # ── Emit kết quả cuối cùng ──
    elapsed = time.time() - pipeline_start
    result = _format_result(last_state, elapsed)

    logger.info(
        "%s Stream DONE (%.3fs) | intent=%s response=%d chars",
        log_prefix, elapsed, result.get("intent", "?"), len(result.get("response", "")),
    )

    yield {"type": "result", **result}

    # ── Ghi file log giống E2E Test ──
    _write_stream_log(query, last_state, result, node_timings, elapsed, session_id)


def _write_stream_log(
    query: str,
    state: dict,
    result: dict,
    node_timings: list,
    total_elapsed: float,
    session_id: Optional[str],
):
    """Ghi log file chi tiết giống E2E test — để debug production chat."""
    import os
    from datetime import datetime

    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "logs")
        os.makedirs(log_dir, exist_ok=True)

        now = datetime.now()
        slug = query[:50].upper().replace(" ", "_")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{slug}.txt"
        filepath = os.path.join(log_dir, filename)

        lines = []
        w = lines.append

        w("")
        w("╔══════════════════════════════════════════════════════════════════════════╗")
        w("║    UFM ADMISSION BOT — Stream Chat Log                                 ║")
        w(f"║    {now.strftime('%Y-%m-%d %H:%M:%S')}                                                ║")
        w("╚══════════════════════════════════════════════════════════════════════════╝")
        w("")
        w(f"  📝 Input Query  : {query}")
        w(f"  🔑 Session ID   : {session_id or 'N/A'}")
        w(f"  💬 Chat History : {len(state.get('chat_history', []))} messages")
        w("")

        # ── Node details ──
        _NODE_DETAIL_LABELS = {
            "fast_scan":   "🛡️ FAST SCAN NODE (Guard Layer 0-1)",
            "context":     "🔄 CONTEXT NODE (Query Reformulation)",
            "guard":       "🔐 CONTEXTUAL GUARD NODE (Layer 2a+2b)",
            "intent":      "🧭 INTENT NODE (Router Phân Loại Ý Định)",
            "multi_query": "🔀 MULTI-QUERY NODE (Sinh biến thể)",
            "embedding":   "📊 EMBEDDING NODE (BGE-M3 batch)",
            "rag":         "🗄️ RAG NODE (Hybrid Search DB)",
            "rag_search":  "🌐 RAG SEARCH PIPELINE (Evaluator + Web Search)",
            "form":        "📋 FORM NODE (Tạo mẫu đơn)",
            "care":        "💚 CARE NODE (Hỗ trợ sinh viên)",
            "response":    "🎯 RESPONSE NODE (Final Answer)",
        }

        for idx, (node_name, node_time, status) in enumerate(node_timings, 1):
            label = _NODE_DETAIL_LABELS.get(node_name, f"⚙️ {node_name}")
            w("──────────────────────────────────────────────────────────────────────────")
            w(f"  {label}")
            w("──────────────────────────────────────────────────────────────────────────")
            w(f"{'Status':>40s}  │  {'✅' if status == 'PASS' else '❌'} {status}  ({node_time:.3f}s)")

            # Node-specific fields
            if node_name == "fast_scan":
                w(f"{'fast_scan_passed':>40s}  │  {'✓' if state.get('fast_scan_passed') else '✗'} {state.get('fast_scan_passed')}")
                w(f"{'fast_scan_message':>40s}  │  {_trunc(state.get('fast_scan_message', ''))}")
                w(f"{'normalized_query':>40s}  │  {_trunc(state.get('normalized_query', ''))}")
            elif node_name == "context":
                w(f"{'standalone_query':>40s}  │  {_trunc(state.get('standalone_query', ''))}")
            elif node_name == "guard":
                w(f"{'contextual_guard_passed':>40s}  │  {'✓' if state.get('contextual_guard_passed') else '✗'} {state.get('contextual_guard_passed')}")
                w(f"{'contextual_guard_message':>40s}  │  {_trunc(state.get('contextual_guard_message', ''))}")
            elif node_name == "intent":
                w(f"{'intent':>40s}  │  {state.get('intent', '')}")
                w(f"{'intent_action':>40s}  │  {state.get('intent_action', '')}")
                w(f"{'intent_summary':>40s}  │  {_trunc(state.get('intent_summary', ''))}")
                w(f"{'program_level_filter':>40s}  │  {state.get('program_level_filter', '')}")
                w(f"{'program_name_filter':>40s}  │  {state.get('program_name_filter', '')}")
            elif node_name == "multi_query":
                mq = state.get("multi_queries", [])
                w(f"{'multi_queries':>40s}  │  [{len(mq)} items]")
                for i, q in enumerate(mq[:5]):
                    w(f"{'':>42s}       [{i}] \"{_trunc(q, 80)}\"")
            elif node_name == "embedding":
                embs = state.get("query_embeddings", [])
                dim = len(embs[0]) if embs and isinstance(embs[0], list) else "?"
                w(f"{'query_embeddings':>40s}  │  [{len(embs)} vectors × {dim}D]")
            elif node_name == "rag":
                w(f"{'rag_context':>40s}  │  {len(state.get('rag_context', ''))} chars")
                w(f"{'retrieved_chunks':>40s}  │  {len(state.get('retrieved_chunks', []))} chunks")
            elif node_name == "rag_search":
                w(f"{'response_source':>40s}  │  {state.get('response_source', '')}")
                w(f"{'final_response':>40s}  │  {len(state.get('final_response', ''))} chars")
            elif node_name == "response":
                w(f"{'response_source':>40s}  │  {state.get('response_source', '')}")
                w(f"{'final_response':>40s}  │  {len(state.get('final_response', ''))} chars")
            w("")

        # ── Final response ──
        w("╔══════════════════════════════════════════════════════════════════════════╗")
        w("║                         FINAL RESPONSE                                ║")
        w("╚══════════════════════════════════════════════════════════════════════════╝")
        w(f"  Response Source : {result.get('source', '')}")
        w(f"  Intent          : {result.get('intent', '')} → {result.get('intent_action', '')}")
        w(f"  Blocked         : {result.get('blocked', False)}")
        w(f"  Response Length : {len(result.get('response', ''))} chars")
        w(f"  Total Time      : {total_elapsed:.2f}s")
        w(f"  Nodes Executed  : {len(node_timings)}")
        w("──────────────────────────────────────────────────────────────────────────")
        w("")
        w(result.get("response", ""))
        w("")

        # ── Timing table ──
        w("══════════════════════════════════════════════════════════════════════════")
        w("")
        w("  ┌──────────────────────────────────────────────┬──────────┬──────────┐")
        w("  │ Node                                         │   Time   │  Status  │")
        w("  ├──────────────────────────────────────────────┼──────────┼──────────┤")
        for node_name, node_time, status in node_timings:
            label = _NODE_DETAIL_LABELS.get(node_name, node_name).split("(", 1)
            short = label[0].strip()
            detail = f"({label[1]}" if len(label) > 1 else ""
            display = f"{short} {detail}"[:44]
            w(f"  │ {display:<44s} │ {node_time:6.3f}s  │ {status:>6s}   │")
        w("  ├──────────────────────────────────────────────┼──────────┼──────────┤")
        w(f"  │ {'TOTAL':>44s} │ {total_elapsed:6.2f}s  │          │")
        w("  └──────────────────────────────────────────────┴──────────┴──────────┘")
        w("")
        w(f"  📁 Log saved: {filepath}")
        w("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Stream log saved: %s", filepath)

    except Exception as e:
        logger.warning("Failed to write stream log: %s", e)


def _trunc(text, max_len=120):
    """Truncate text for log display."""
    if not text:
        return '""'
    s = str(text).replace("\n", " ")
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


# ══════════════════════════════════════════════════════════
# TOKEN STREAM PIPELINE — ChatGPT-style token-by-token
# ══════════════════════════════════════════════════════════

# Response sources that DON'T need LLM — bypass directly
_BYPASS_SOURCES = {
    "greet_template",
    "clarify_template",
    "care_template",
    "form_template",
    "rag_search_synthesized",
    "intent_block",
    "fast_scan",
    "contextual_guard",
    "keyword_filter",
    "input_validation",
}


def stream_chat_pipeline_tokens_v2(
    query: str,
    chat_history: Optional[list[dict]] = None,
    session_id: Optional[str] = None,
):
    """
    TRUE ChatGPT-style token streaming.

    Cơ chế:
      1. Chạy chat_graph_pre_response (response_node = passthrough, KHÔNG gọi LLM)
         → Yield node events cho frontend hiển thị pipeline steps
      2. Sau khi graph xong:
         - BYPASS (chào hỏi, block, form...) → yield full result ngay
         - Cần LLM → gọi _stream_gemini_api_with_fallback → yield REAL tokens từ API

    Yields:
        {"type": "node",   "node": "fast_scan", "label": "🛡️ Kiểm tra nhanh"}
        {"type": "token",  "content": "Xin "}           ← REAL token từ LLM API!
        {"type": "token",  "content": "chào "}
        {"type": "done",   "source": "rag_db_only_generated", ...}
        {"type": "result", "response": "...", ...}       ← bypass (chào hỏi, block)
        {"type": "error",  "message": "..."}
    """
    from app.core.config import query_flow_config
    from app.core.prompts import prompt_manager
    from app.core.config.contact_loader import get_hotline_short
    from app.services.langgraph.graph_builder import chat_graph_pre_response

    pipeline_start = time.time()
    log_prefix = f"[{session_id[:8]}]" if session_id else "[v2-stream]"

    logger.info(
        "%s V2 Token Stream START | query='%s' history=%d msgs",
        log_prefix, query[:80], len(chat_history or []),
    )

    initial_state = _create_initial_state(
        query=query,
        chat_history=chat_history or [],
    )

    config = {
        "configurable": {
            "thread_id": f"stream_{session_id or 'default_session'}",
        }
    }

    last_state = initial_state

    # ─────────────────────────────────────────────────────
    # PHASE 1: Run graph WITH PASSTHROUGH response node
    # ─────────────────────────────────────────────────────
    try:
        for chunk in chat_graph_pre_response.stream(
            initial_state, config=config, stream_mode="updates"
        ):
            for node_name, state_delta in chunk.items():
                if isinstance(state_delta, dict):
                    last_state = {**last_state, **state_delta}

                emoji, label = _NODE_LABELS.get(node_name, ("⚙️", node_name))
                yield {
                    "type": "node",
                    "node": node_name,
                    "label": f"{emoji} {label}",
                }
                logger.info("%s V2 Stream - Node '%s' completed", log_prefix, node_name)

    except Exception as e:
        elapsed = time.time() - pipeline_start
        logger.error("%s V2 Stream CRASHED: %s", log_prefix, e, exc_info=True)
        yield {
            "type": "error",
            "message": "Xin lỗi, hệ thống gặp lỗi khi xử lý. Vui lòng thử lại sau.",
        }
        return

    # ─────────────────────────────────────────────────────
    # PHASE 2: Response — BYPASS hoặc STREAMING LLM
    # ─────────────────────────────────────────────────────
    response_source = last_state.get("response_source", "")
    final_response = last_state.get("final_response", "")

    # A) BYPASS — response đã sẵn (chào hỏi, block, form, care, rag_search_synthesized)
    if response_source in _BYPASS_SOURCES:
        elapsed = time.time() - pipeline_start
        result = _format_result(last_state, elapsed)
        logger.info("%s V2 BYPASS (%.3fs) | source=%s", log_prefix, elapsed, response_source)
        yield {"type": "result", **result}
        return

    # B) CẦN GỌI LLM → STREAMING TỪ API (REAL tokens!)
    main_bot_config = query_flow_config.main_bot

    if not main_bot_config.enabled:
        # Main Bot disabled → trả raw context
        elapsed = time.time() - pipeline_start
        last_state["final_response"] = final_response or last_state.get("rag_context", "")
        last_state["response_source"] = "main_bot_disabled"
        result = _format_result(last_state, elapsed)
        yield {"type": "result", **result}
        return

    # ── Chuẩn bị prompt (giống response_node.py) ──
    standalone_query = last_state.get("standalone_query", last_state.get("user_query", ""))
    rag_context = last_state.get("rag_context") or ""
    chat_history_text = last_state.get("chat_history_text") or ""
    web_citations = last_state.get("web_search_citations") or []

    try:
        sys_prompt = prompt_manager.get_system("main_bot")
        user_content = prompt_manager.render_user(
            "main_bot",
            standalone_query=standalone_query,
            final_response=final_response,
            rag_context=rag_context,
            chat_history_text=chat_history_text,
            web_citations=web_citations,
        )

        logger.info(
            "%s V2 Stream - Calling STREAMING LLM: %s/%s",
            log_prefix, main_bot_config.provider, main_bot_config.model,
        )

        # ── STREAM REAL TOKENS FROM API ──
        full_response = ""
        token_count = 0

        for token in _stream_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=main_bot_config,
            node_key="main_bot",
        ):
            full_response += token
            token_count += 1
            yield {"type": "token", "content": token}

        elapsed = time.time() - pipeline_start

        logger.info(
            "%s V2 Stream LLM DONE (%.3fs) | %d tokens, %d chars",
            log_prefix, elapsed, token_count, len(full_response),
        )

        # ── Emit done event with metadata ──
        yield {
            "type": "done",
            "source": f"{response_source}_generated",
            "intent": last_state.get("intent", ""),
            "intent_action": last_state.get("intent_action", ""),
            "elapsed_seconds": round(elapsed, 3),
        }

    except Exception as e:
        elapsed = time.time() - pipeline_start
        logger.error("%s V2 Stream LLM ERROR: %s", log_prefix, e, exc_info=True)

        # Fallback: nếu có rag_context thì trả đó, không thì trả lỗi
        fallback = final_response or rag_context
        if not fallback:
            fallback = (
                "Xin lỗi bạn, hệ thống đang bảo trì tạm thời. "
                "Bạn vui lòng thử lại sau giây lát hoặc liên hệ trực tiếp "
                "để được hỗ trợ nhanh nhất nhé!\n\n"
                f"{get_hotline_short()}"
            )

        yield {
            "type": "result",
            "response": fallback,
            "source": "main_bot_error",
            "intent": last_state.get("intent", ""),
            "intent_action": last_state.get("intent_action", ""),
            "blocked": False,
            "blocked_reason": "",
            "elapsed_seconds": round(elapsed, 3),
        }

