"""
Test Pipeline E2E — Chạy thật LangGraph `chat_graph.stream()` chuẩn thư viện.

** GỌI ĐÚNG API LangGraph — KHÔNG gọi node thủ công **

Cách chạy:
  python tests/test_pipeline_e2e.py
  python tests/test_pipeline_e2e.py --query "Học phí ngành Marketing?"
  python tests/test_pipeline_e2e.py --query "Xin chào" --history '[{"role":"user","content":"Hi"}]'

Log file tự động lưu tại tests/logs/
"""

import io
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ══════════════════════════════════════════════════════════
# IMPORT — LangGraph compiled graph (chuẩn thư viện)
# ══════════════════════════════════════════════════════════
from app.core.config import query_flow_config, models_yaml_data
from app.services.langgraph.graph_builder import chat_graph


# ══════════════════════════════════════════════════════════
# FILE LOGGING — Ghi toàn bộ output ra .txt
# ══════════════════════════════════════════════════════════
LOG_DIR = PROJECT_ROOT / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None


def _strip_ansi(text: str) -> str:
    """Xóa ANSI escape codes cho file log."""
    return re.sub(r'\033\[[0-9;]*m', '', text)


def _log(text: str = ""):
    """In ra console VÀ ghi vào file log."""
    print(text)
    if _log_file:
        _log_file.write(_strip_ansi(text) + "\n")


def _open_log(query: str) -> Path:
    global _log_file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r'[^\w\s-]', '', query[:40]).strip().replace(' ', '_')
    log_path = LOG_DIR / f"{ts}_{slug}.txt"
    _log_file = open(log_path, "w", encoding="utf-8")
    return log_path


def _close_log():
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None


# ══════════════════════════════════════════════════════════
# DISPLAY UTILS
# ══════════════════════════════════════════════════════════
COLORS = {
    "HEADER":  "\033[95m",
    "BLUE":    "\033[94m",
    "CYAN":    "\033[96m",
    "GREEN":   "\033[92m",
    "YELLOW":  "\033[93m",
    "RED":     "\033[91m",
    "BOLD":    "\033[1m",
    "DIM":     "\033[2m",
    "RESET":   "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['RESET']}"


def _trunc(text, max_len=120):
    if not text:
        return '""'
    s = str(text).replace("\n", " ")
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _format_value(val) -> str:
    """Format giá trị state cho display."""
    if val is None:
        return _c("DIM", "null")
    if isinstance(val, bool):
        return _c("GREEN", "✓ True") if val else _c("RED", "✗ False")
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, str):
        if len(val) == 0:
            return _c("DIM", '""')
        return f"{val}\n{'>':>55s}({len(val)} chars)"
    if isinstance(val, list):
        if not val:
            return _c("DIM", "[]")
        if all(isinstance(v, list) for v in val):
            dims = len(val[0]) if val[0] else 0
            return f"[{len(val)} vectors × {dims}D]"
        if all(isinstance(v, dict) for v in val):
            lines = [f"[{len(val)} items]"]
            for i, item in enumerate(val):
                preview = json.dumps(item, ensure_ascii=False)
                lines.append(f"{'':>55s}  [{i}] {preview}")
            return "\n".join(lines)
        if all(isinstance(v, str) for v in val):
            lines = [f"[{len(val)} items]"]
            for i, v in enumerate(val):
                lines.append(f"{'':>55s}  [{i}] \"{v}\"")
            return "\n".join(lines)
        return f"[{len(val)} items]"
    return str(val)


# ══════════════════════════════════════════════════════════
# NODE → Label mapping (giống graph_builder 11 nodes)
# ══════════════════════════════════════════════════════════
_NODE_DISPLAY = {
    "fast_scan":    ("🛡️",  "FAST SCAN NODE (Guard Layer 0-1)"),
    "context":      ("🔄",  "CONTEXT NODE (Query Reformulation)"),
    "guard":        ("🔐",  "CONTEXTUAL GUARD NODE (Layer 2a+2b)"),
    "intent":       ("🧭",  "INTENT NODE (Router Phân Loại Ý Định)"),
    "multi_query":  ("🔀",  "MULTI-QUERY NODE (Sinh biến thể)"),
    "embedding":    ("📊",  "EMBEDDING NODE (BGE-M3 batch)"),
    "rag":          ("🗄️",  "RAG NODE (Hybrid Search DB)"),
    "form":         ("📝",  "FORM NODE (Agent Mẫu đơn)"),
    "care":         ("💚",  "CARE NODE (Chăm sóc Sinh viên)"),
    "rag_search":   ("🌐",  "RAG SEARCH PIPELINE (Evaluator + Web Search)"),
    "response":     ("🎯",  "RESPONSE NODE (Final Answer)"),
}

# Node → danh sách key quan trọng cần hiển thị
_NODE_KEYS = {
    "fast_scan": [
        "normalized_query", "original_query", "query_was_summarized",
        "fast_scan_passed", "fast_scan_blocked_layer", "fast_scan_message",
    ],
    "context": ["standalone_query"],
    "guard": [
        "contextual_guard_passed", "contextual_guard_blocked_layer",
        "contextual_guard_message",
    ],
    "intent": [
        "intent", "intent_summary", "intent_action",
        "program_level_filter", "program_name_filter",
        "next_node", "response_source",
    ],
    "multi_query": ["multi_queries"],
    "embedding": ["query_embeddings"],
    "rag": [
        "rag_context", "retrieved_chunks",
    ],
    "form": ["final_response", "response_source"],
    "care": ["final_response", "response_source"],
    "rag_search": [
        "ufm_search_queries", "pr_search_query",
        "web_search_results", "web_search_citations",
        "search_cache_hit", "search_cache_similarity",
        "final_response", "response_source",
    ],
    "response": ["final_response", "response_source"],
}


# ══════════════════════════════════════════════════════════
# BANNER + TABLE HELPERS
# ══════════════════════════════════════════════════════════
def print_banner():
    _log()
    _log(_c("BOLD", "╔══════════════════════════════════════════════════════════════════════════╗"))
    _log(_c("BOLD", "║") + _c("CYAN", "    UFM ADMISSION BOT — Pipeline E2E Test (LangGraph .stream())         ") + _c("BOLD", "║"))
    _log(_c("BOLD", "║") + _c("CYAN", f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                                ") + _c("BOLD", "║"))
    _log(_c("BOLD", "╚══════════════════════════════════════════════════════════════════════════╝"))


def print_active_models():
    """Hiện model đang active cho mỗi node."""
    _log()
    _log(_c("YELLOW", "  ┌─────────────────────────────────────────────────────────────────┐"))
    _log(_c("YELLOW", "  │                    ACTIVE MODELS (from YAML)                    │"))
    _log(_c("YELLOW", "  ├──────────────────────┬──────────────────────────────────────────┤"))

    qfc = query_flow_config
    rows = [
        ("❶ Summarizer",     qfc.long_query_summarizer.provider, qfc.long_query_summarizer.model),
        ("❷ Guard (Fast)",   qfc.prompt_guard_fast.provider,     qfc.prompt_guard_fast.model),
        ("❸ Reformulate",    qfc.query_reformulation.provider,   qfc.query_reformulation.model),
        ("❹ Guard (Deep)",   qfc.prompt_guard_deep.provider,     qfc.prompt_guard_deep.model),
        ("❺ Intent Router",  qfc.semantic_router.provider,       qfc.semantic_router.model),
        ("❻ Multi-Query",    qfc.multi_query.provider,           qfc.multi_query.model),
        ("❼ Embedding",      qfc.embedding.provider,             qfc.embedding.model),
        ("❽ Vector Router",  qfc.vector_router.provider,         qfc.vector_router.model),
        ("❾a PR Query",      qfc.pr_query.provider,              qfc.pr_query.model),
        ("❾b UFM Query",     qfc.ufm_query.provider,             qfc.ufm_query.model),
        ("❾c Web Search",    qfc.web_search.provider,            qfc.web_search.model),
        ("❾d Info Synth",    qfc.info_synthesizer.provider,      qfc.info_synthesizer.model),
        ("❾e PR Synth",      qfc.pr_synthesizer.provider,        qfc.pr_synthesizer.model),
        ("❾f Sanitizer",     qfc.sanitizer.provider,             qfc.sanitizer.model),
        ("❾g Evaluator",     qfc.context_evaluator.provider,     qfc.context_evaluator.model),
        ("⓾ Care",           models_yaml_data.get("care", {}).get("provider", "?"),
                              models_yaml_data.get("care", {}).get("model", "?")),
        ("⓫ Form Extract",   models_yaml_data.get("form", {}).get("provider", "?"),
                              models_yaml_data.get("form", {}).get("extractor", {}).get("model", "?")),
        ("⓫ Form Draft",     models_yaml_data.get("form", {}).get("provider", "?"),
                              models_yaml_data.get("form", {}).get("drafter", {}).get("model", "?")),
        ("⓬ Main Bot",       qfc.main_bot.provider,              qfc.main_bot.model),
    ]

    for label, provider, model in rows:
        _log(f"  │ {label:<20s} │ {provider}/{model:<36s} │")

    _log(_c("YELLOW", "  └──────────────────────┴──────────────────────────────────────────┘"))


def print_graph_topology():
    """In topology thực tế của LangGraph — giúp verify đúng luồng."""
    _log()
    _log(_c("YELLOW", "  ┌─────────────────────────────────────────────────────────────────┐"))
    _log(_c("YELLOW", "  │              LANGGRAPH TOPOLOGY (Tối ưu chi phí)               │"))
    _log(_c("YELLOW", "  ├─────────────────────────────────────────────────────────────────┤"))
    _log(f"  │ fast_scan ─┬─ BLOCKED → END                                    │")
    _log(f"  │            └─ context → guard ─┬─ BLOCKED → END                │")
    _log(f"  │                                └─ intent (ROUTER TRUNG TÂM)    │")
    _log(f"  │                                     │                          │")
    _log(f"  │                   GREET/CLARIFY/BLOCK → response → END ($0!)   │")
    _log(f"  │                   PROCEED_FORM → form → response → END  ($0!) │")
    _log(f"  │                   PROCEED_CARE → care → response → END  ($0!) │")
    _log(f"  │                   PROCEED_RAG* → multi_query → embedding → rag │")
    _log(f"  │                                   ├─ PROCEED_RAG → response    │")
    _log(f"  │                                   └─ *_SEARCH → rag_search     │")
    _log(_c("YELLOW", "  └─────────────────────────────────────────────────────────────────┘"))


def print_node_header(node_name: str, step: int, emoji: str, label: str):
    _log()
    _log(_c("BOLD", f"{'─' * 74}"))
    _log(_c("YELLOW", f"  {emoji} [{step}] {label}"))
    _log(_c("BOLD", f"{'─' * 74}"))


def print_state_delta(state: dict, keys: list):
    """In các key quan trọng từ state delta."""
    for key in keys:
        val = state.get(key)
        display = _format_value(val)
        _log(f"    {_c('BLUE', key):>50s}  │  {display}")


def print_node_status(node_name: str, state: dict, elapsed: float):
    """In status tùy theo loại node."""
    passed = True
    msg = ""

    if node_name == "fast_scan":
        if state.get("fast_scan_passed") is False:
            passed = False
        msg = state.get("fast_scan_message", "")
        if state.get("query_was_summarized"):
            msg += " [Query was summarized by LLM]"

    elif node_name == "guard":
        if state.get("contextual_guard_passed") is False:
            passed = False
        msg = state.get("contextual_guard_message", "")
        blocked_layer = state.get("contextual_guard_blocked_layer")
        if blocked_layer:
            msg += f" [Blocked by Layer {blocked_layer}]"

    elif node_name == "context":
        reformed = state.get("standalone_query", "")
        if reformed:
            msg = f"Reformulated → '{_trunc(reformed, 80)}'"
        else:
            msg = "Giữ nguyên (no history hoặc skip)"

    elif node_name == "intent":
        intent = state.get("intent", "")
        action = state.get("intent_action", "")
        next_n = state.get("next_node", "")
        msg = f"intent='{intent}' → action='{action}' → next='{next_n}'"

    elif node_name == "multi_query":
        mq = state.get("multi_queries", [])
        msg = f"Sinh {len(mq)} biến thể"

    elif node_name == "embedding":
        embs = state.get("query_embeddings", [])
        if embs:
            dim = len(embs[0]) if isinstance(embs[0], list) else "?"
            msg = f"Nhúng {len(embs)} vectors × {dim}D"
        else:
            msg = "Không có embeddings"

    elif node_name == "rag":
        chunks = state.get("retrieved_chunks", [])
        ctx_len = len(state.get("rag_context", ""))
        msg = f"Chunks={len(chunks)}, Context={ctx_len} chars"

    elif node_name == "rag_search":
        cache_hit = state.get("search_cache_hit", False)
        citations = state.get("web_search_citations") or []
        msg = f"cache={'HIT' if cache_hit else 'MISS'}, citations={len(citations)}"

    elif node_name in ("care", "form"):
        resp_len = len(state.get("final_response", ""))
        msg = f"Response={resp_len} chars"

    elif node_name == "response":
        source = state.get("response_source", "")
        resp_len = len(state.get("final_response", ""))
        msg = f"source='{source}', length={resp_len} chars"

    # Print
    if passed is None:
        status = _c("DIM", "⏭️  SKIPPED")
    elif passed:
        status = _c("GREEN", "✅ PASS")
    else:
        status = _c("RED", "⛔ BLOCKED")

    time_str = _c("CYAN", f"{elapsed:.3f}s")
    _log(f"    {'Status':>43s}  │  {status}  ({time_str})")
    if msg:
        _log(f"    {'Message':>43s}  │  {msg[:200]}")

    return passed


def print_final_response(state: dict, total_elapsed: float, nodes_executed: int):
    _log()
    _log(_c("BOLD", "╔══════════════════════════════════════════════════════════════════════════╗"))
    _log(_c("BOLD", "║") + _c("GREEN", "                         FINAL RESPONSE                                ") + _c("BOLD", "║"))
    _log(_c("BOLD", "╚══════════════════════════════════════════════════════════════════════════╝"))

    resp = state.get("final_response", "(trống)")
    source = state.get("response_source", "N/A")
    intent = state.get("intent", "N/A")
    intent_action = state.get("intent_action", "N/A")

    _log(f"  Response Source : {_c('YELLOW', source)}")
    _log(f"  Intent          : {_c('CYAN', intent)} → {_c('CYAN', intent_action)}")
    _log(f"  Response Length : {len(resp)} chars")
    _log(f"  Total Time      : {_c('BOLD', f'{total_elapsed:.2f}s')}")
    _log(f"  Nodes Executed  : {nodes_executed}")
    _log(_c("BOLD", "─" * 74))
    _log()
    _log(resp)
    _log()
    _log(_c("BOLD", "═" * 74))


def print_pipeline_summary(timings: list, total_elapsed: float):
    _log()
    _log(_c("YELLOW", "  ┌──────────────────────────────────────────────┬──────────┬──────────┐"))
    _log(_c("YELLOW", "  │ Node                                         │   Time   │  Status  │"))
    _log(_c("YELLOW", "  ├──────────────────────────────────────────────┼──────────┼──────────┤"))

    for entry in timings:
        name = entry["name"][:44]
        elapsed = entry["elapsed"]
        passed = entry["passed"]

        status_str = "PASS" if passed else "BLOCKED" if passed is False else "SKIP"
        status_color = "GREEN" if passed else "RED" if passed is False else "DIM"

        _log(f"  │ {name:<44s} │ {elapsed:>6.3f}s  │ {_c(status_color, f'{status_str:^8s}')} │")

    _log(_c("YELLOW", "  ├──────────────────────────────────────────────┼──────────┼──────────┤"))
    _log(f"  │ {'TOTAL':>44s} │ {_c('BOLD', f'{total_elapsed:>6.2f}s')}  │          │")
    _log(_c("YELLOW", "  └──────────────────────────────────────────────┴──────────┴──────────┘"))


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE — Gọi chat_graph.stream() chuẩn LangGraph
# ══════════════════════════════════════════════════════════
def run_pipeline(user_query: str, chat_history: list = None):
    """
    Chạy toàn bộ pipeline bằng LangGraph `chat_graph.stream(stream_mode="updates")`.

    Đây là cách chạy ĐÚNG thư viện LangGraph:
      - Không gọi node thủ công
      - Graph quyết định routing (conditional edges)
      - Mỗi node xong → yield state delta → ta log ra
    """
    log_path = _open_log(user_query)

    print_banner()
    _log(f"\n  📁 Log file: {log_path}")
    print_active_models()
    print_graph_topology()

    _log(f"\n  📝 Input Query  : {_c('BOLD', user_query)}")
    if chat_history:
        _log(f"  💬 Chat History : {len(chat_history)} messages")
        for msg in chat_history[-4:]:
            role_icon = "👤" if msg.get("role") == "user" else "🤖"
            content_preview = msg.get("content", "")[:80]
            _log(f"     {role_icon} {content_preview}")
    else:
        _log(f"  💬 Chat History : {_c('DIM', 'Trống (lượt đầu tiên)')}")
    _log()

    # ── Khởi tạo state ban đầu ──
    initial_state = {
        "user_query": user_query,
        "chat_history": chat_history or [],
        "normalized_query": "",
        "standalone_query": "",
        "original_query": "",
        "query_was_summarized": False,
        "fast_scan_passed": None,
        "fast_scan_blocked_layer": None,
        "fast_scan_message": "",
        "contextual_guard_passed": None,
        "contextual_guard_blocked_layer": None,
        "contextual_guard_message": "",
        "multi_queries": [],
        "query_embeddings": [],
        "rag_context": "",
        "retrieved_chunks": [],
        "intent": "",
        "intent_summary": "",
        "intent_action": "",
        "program_level_filter": None,
        "program_name_filter": None,
        "next_node": "",
        "final_response": "",
        "response_source": "",
    }

    # ── Config cho LangGraph (Multi-Tenant MemorySaver) ──
    config = {
        "configurable": {
            "thread_id": f"e2e_test_{datetime.now().strftime('%H%M%S')}",
        }
    }

    # ── Chạy LangGraph stream ──
    step = 0
    timings = []
    merged_state = dict(initial_state)  # Merge tất cả delta vào đây
    pipeline_start = time.time()
    node_start = time.time()

    _log(_c("BOLD", "  🚀 BẮT ĐẦU CHẠY LangGraph chat_graph.stream(stream_mode='updates')"))
    _log()

    try:
        # ═══════════════════════════════════════════════
        # chat_graph.stream() — API chuẩn LangGraph
        #   stream_mode="updates" → yield {node_name: state_delta}
        #   Graph tự quyết định routing qua conditional edges
        # ═══════════════════════════════════════════════
        for chunk in chat_graph.stream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):
            for node_name, state_delta in chunk.items():
                node_elapsed = time.time() - node_start
                step += 1

                # Merge state delta
                if isinstance(state_delta, dict):
                    merged_state = {**merged_state, **state_delta}

                # Display
                emoji, label = _NODE_DISPLAY.get(node_name, ("⚙️", node_name))
                print_node_header(node_name, step, emoji, label)

                # Status + Message
                passed = print_node_status(node_name, merged_state, node_elapsed)

                # State fields
                keys = _NODE_KEYS.get(node_name, [])
                print_state_delta(merged_state, keys)

                # Track timing
                timings.append({
                    "name": f"{emoji} {label}",
                    "elapsed": node_elapsed,
                    "passed": passed,
                })

                node_start = time.time()

    except Exception as e:
        import traceback
        elapsed = time.time() - pipeline_start
        _log()
        _log(_c("RED", f"  ⛔ PIPELINE CRASHED after {elapsed:.3f}s: {type(e).__name__}: {e}"))
        _log(_c("RED", traceback.format_exc()))
        merged_state["final_response"] = f"Pipeline crashed: {e}"
        merged_state["response_source"] = "error"

    # ═══════════════════════════════════════════════
    # KẾT QUẢ
    # ═══════════════════════════════════════════════
    total_elapsed = time.time() - pipeline_start
    print_final_response(merged_state, total_elapsed, step)
    print_pipeline_summary(timings, total_elapsed)
    _log(f"\n  📁 Log saved: {log_path}\n")

    _close_log()
    return merged_state


# ══════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ══════════════════════════════════════════════════════════
def interactive_mode():
    """Chế độ tương tác — nhập câu hỏi liên tục."""
    print(_c("BOLD", "\n╔══════════════════════════════════════════════════════════════╗"))
    print(_c("CYAN",   "║  UFM Pipeline Tester — Interactive Mode (LangGraph Stream)  ║"))
    print(_c("CYAN",   "║  Nhập câu hỏi để test, 'quit' để thoát                     ║"))
    print(_c("CYAN",   "║  'clear' = xóa history | 'history' = xem history            ║"))
    print(_c("BOLD",   "╚══════════════════════════════════════════════════════════════╝\n"))

    chat_history = []

    while True:
        try:
            user_input = input(_c("GREEN", "\n🎤 Nhập câu hỏi: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "clear":
            chat_history = []
            print(_c("YELLOW", "✅ Đã xóa chat history."))
            continue
        if user_input.lower() == "history":
            if not chat_history:
                print(_c("DIM", "(History trống)"))
            else:
                print(json.dumps(chat_history, indent=2, ensure_ascii=False))
            continue

        # Chạy pipeline
        state = run_pipeline(user_input, chat_history)

        # Lưu vào history cho lượt sau
        chat_history.append({"role": "user", "content": user_input})
        final = state.get("final_response", "")
        if final:
            chat_history.append({"role": "assistant", "content": final})

        # Giữ history gọn (10 lượt gần nhất)
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]


# ══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UFM Pipeline E2E Tester (LangGraph Stream)")
    parser.add_argument("--query", "-q", help="Câu hỏi test (nếu không có → interactive mode)")
    parser.add_argument("--history", help="JSON string chat history")
    args = parser.parse_args()

    if args.query:
        history = []
        if args.history:
            try:
                history = json.loads(args.history)
            except json.JSONDecodeError:
                print("⚠️  --history phải là JSON hợp lệ")
                sys.exit(1)
        run_pipeline(args.query, history)
    else:
        interactive_mode()
