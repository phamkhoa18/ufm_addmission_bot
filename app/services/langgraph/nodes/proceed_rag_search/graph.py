"""
Proceed RAG Search — Sub-pipeline Graph (v3: Web Search Toggle).

Nguyên tắc GỐC RỄ:
  ★ RAG (Vector DB) LUÔN là nguồn chính — KHÔNG BAO GIỜ bị bỏ qua.
  ★ Web Search chỉ là BỔ SUNG — có thể bật/tắt linh hoạt cho production.

TOGGLE HIERARCHY:
  ┌─────────────────────────────────────────────────────────────┐
  │ proceed_rag_search.enabled = false                         │
  │   → Tắt TOÀN BỘ pipeline, trả rag_context thô             │
  │                                                             │
  │ web_search.enabled = false (★ MỚI — Production Toggle)     │
  │   → Chỉ tắt internet search, KHÔNG ảnh hưởng:             │
  │     • Evaluator Gate (Self-RAG)                            │
  │     • Synthesizer (tổng hợp từ DB context)                 │
  │     • Sanitizer (kiểm duyệt)                              │
  │   → Các agent node khác (form, care, response) KHÔNG ĐỔI  │
  │                                                             │
  │ web_search.enabled = true (default)                        │
  │   → Chạy đầy đủ: Query Gen → Cache → Web Search → Merge   │
  └─────────────────────────────────────────────────────────────┘

Luồng v3:
  [Evaluator Gate] ── DB đủ?
      ↓ YES                     ↓ NO (hoặc PR)
      Trả rag_context            ┌── web_search.enabled?
      → response                 │  YES: [query_gen] → [cache] → [web_search] → merge
                                 │  NO:  Bỏ qua web, dùng rag_context only
                                 └── [synthesizer] → [sanitizer] → response
"""

import time
from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.proceed_rag_search.pr_query_node import pr_query_node
from app.services.langgraph.nodes.proceed_rag_search.web_search_node import web_search_node
from app.services.langgraph.nodes.proceed_rag_search.synthesizer_node import synthesizer_node
from app.services.langgraph.nodes.proceed_rag_search.sanitizer_node import sanitizer_node
from app.services.langgraph.nodes.proceed_rag_search.search_cache import cache_lookup, cache_save
from app.services.langgraph.nodes.proceed_rag_search.evaluator import evaluate_rag_context
from app.core.config import query_flow_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _merge_context(rag_context: str, web_results: str, web_citations: list) -> str:
    """
    Ghép ngữ cảnh từ DB nội bộ + Web Search thành 1 khối text.
    Không gọi LLM — chỉ format chuỗi Python thuần.
    """
    parts = []

    if rag_context:
        parts.append(f"[DỮ LIỆU NỘI BỘ]\n{rag_context}")

    if web_results:
        parts.append(f"[DỮ LIỆU TỪ WEB]\n{web_results}")
        if web_citations:
            links = "\n".join(f"- [{c['text']}]({c['url']})" for c in web_citations)
            parts.append(f"[NGUỒN TRÍCH DẪN]\n{links}")

    if not parts:
        return ""

    return "\n\n".join(parts)


def _run_web_search_pipeline(state: dict, standalone_query: str, action: str) -> dict:
    """
    Chạy luồng Web Search: Query Gen → Cache → Web Search → Merge.
    Chỉ được gọi khi web_search.enabled = true.
    """
    # ── Bước 1: Sinh Query (PR hoặc UFM) ──
    logger.info("Web Search ON - Buoc 1: Sinh query...")
    state = pr_query_node(state)

    # ── Bước 2: SEMANTIC CACHE CHECK ──
    logger.info("Web Search ON - Buoc 2: Kiem tra Semantic Cache...")
    cache_hit, cache_sim, cached_results, cached_citations, query_vector = cache_lookup(
        query_text=standalone_query,
        intent_action=action,
    )
    state["search_cache_hit"] = cache_hit
    state["search_cache_similarity"] = cache_sim
    state["search_query_vector"] = query_vector

    if cache_hit:
        state["web_search_results"] = cached_results
        state["web_search_citations"] = cached_citations
        logger.info("CACHE HIT -> Bo qua Web Search API")
    else:
        # ── Bước 3: Web Search API ──
        logger.info("Web Search ON - Buoc 3: Goi Web Search API...")
        state = web_search_node(state)

        # Lưu cache nếu có kết quả
        if state.get("web_search_results"):
            cache_save(
                query_text=standalone_query,
                intent_action=action,
                web_results=state["web_search_results"],
                web_citations=state.get("web_search_citations") or [],
                query_vector=state.get("search_query_vector"),
            )

    return state


def _run_synthesizer_sanitizer_loop(state: dict) -> dict:
    """
    Chạy luồng Synthesizer → Sanitizer (loop nếu cần).
    Pipeline này LUÔN chạy bất kể web_search bật/tắt.
    """
    config = query_flow_config.sanitizer
    max_loops = config.max_loops

    for loop_idx in range(max_loops + 1):
        # ── Synthesizer: Tổng hợp từ rag_context + web_results ──
        state = synthesizer_node(state)

        # ── Sanitizer: Kiểm duyệt bản nháp ──
        state = sanitizer_node(state)

        if state.get("sanitizer_passed", True):
            logger.info("Synthesizer-Sanitizer PASSED (loop %d)", loop_idx + 1)
            break
        else:
            logger.info("Synthesizer-Sanitizer REJECTED (loop %d/%d)", loop_idx + 1, max_loops)

    return state


def proceed_rag_search_pipeline(state: GraphState) -> GraphState:
    """
    🔍 PROCEED RAG SEARCH PIPELINE v3 — RAG-First + Web Search Toggle.

    Toggle Hierarchy:
      1. proceed_rag_search.enabled = false → Bypass toàn bộ, trả rag_context thô
      2. web_search.enabled = false → Chỉ skip internet, vẫn chạy synthesizer+sanitizer
      3. web_search.enabled = true  → Full pipeline (query gen → cache → web → synth → sanit)
    """
    pipeline_start = time.time()
    action = state.get("intent_action", "PROCEED_RAG_PR_SEARCH")
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    rag_context = state.get("rag_context") or ""

    # ══════════════════════════════════════════════════════════
    # Guard: Chỉ chạy khi intent thuộc nhánh cần search
    # ══════════════════════════════════════════════════════════
    if not action.startswith("PROCEED_RAG_") and action != "PROCEED_PR":
        logger.info("RAG Search Pipeline - SKIP (intent_action='%s')", action)
        return state

    # ══════════════════════════════════════════════════════════
    # TOGGLE 1: Master Toggle — Tắt toàn bộ pipeline
    # ══════════════════════════════════════════════════════════
    if not query_flow_config.proceed_rag_search.enabled:
        logger.info(
            "RAG Search Pipeline - DISABLED (proceed_rag_search.enabled=false). "
            "Tra context DB truc tiep, bo qua toan bo pipeline."
        )
        state["final_response"] = rag_context
        state["response_source"] = "rag_db_only"
        return state

    # ══════════════════════════════════════════════════════════
    # TOGGLE 2: Web Search Toggle — Đọc từ web_search.enabled
    # ══════════════════════════════════════════════════════════
    web_search_enabled = query_flow_config.web_search.enabled

    logger.info(
        "PROCEED RAG SEARCH v3 - Action: %s | RAG ctx: %d chars | web_search: %s",
        action, len(rag_context), "ON" if web_search_enabled else "OFF"
    )

    # ── Khởi tạo state bổ sung ──
    state = {
        **state,
        "ufm_search_queries": None,
        "pr_search_query": None,
        "web_search_results": None,
        "web_search_citations": None,
        "search_cache_hit": False,
        "search_cache_similarity": 0.0,
        "sanitizer_loop_count": 0,
        "sanitizer_critique": None,
    }

    # ══════════════════════════════════════════════════════════
    # Bước 0: SELF-RAG EVALUATOR GATE
    # Chỉ áp dụng khi web_search_enabled=true VÀ nhánh UFM_SEARCH.
    # Khi web_search tắt → bỏ qua evaluator (không cần đánh giá)
    # ══════════════════════════════════════════════════════════
    skip_web_search = False

    if web_search_enabled:
        if action == "PROCEED_RAG_UFM_SEARCH" and rag_context:
            logger.info("Buoc 0: Self-RAG Evaluator...")
            is_sufficient = evaluate_rag_context(
                standalone_query=standalone_query,
                rag_context=rag_context,
            )

            if is_sufficient:
                skip_web_search = True
                logger.info("EVALUATOR: YES -> DB du, bo qua Web Search")
            else:
                logger.info("EVALUATOR: NO -> DB thieu, tiep tuc Web Search")
        elif action in ("PROCEED_RAG_PR_SEARCH", "PROCEED_PR"):
            logger.info("Buoc 0: Nhanh PR -> LUON chay Web Search")
        elif not rag_context:
            logger.info("Buoc 0: Context DB rong -> bat buoc Web Search")
    else:
        skip_web_search = True
        logger.info(
            "Buoc 0: web_search.enabled=false -> SKIP toan bo internet search. "
            "Chi dung RAG context lam nguon chinh."
        )

    # ══════════════════════════════════════════════════════════
    # NHÁNH A: DB ĐỦ + Evaluator YES → Trả rag_context thô
    # (Chỉ khi evaluator quyết DB đủ, KHÔNG phải khi web tắt)
    # ══════════════════════════════════════════════════════════
    if skip_web_search and web_search_enabled:
        # Evaluator phán DB đủ → trả thẳng, không cần synthesizer
        state["final_response"] = rag_context
        state["response_source"] = "rag_db_only"

        elapsed = time.time() - pipeline_start
        logger.info(
            "PROCEED RAG SEARCH - Ket thuc som (%.2fs), Evaluator YES → DB only, %d ky tu",
            elapsed, len(rag_context)
        )
        return state

    # ══════════════════════════════════════════════════════════
    # NHÁNH B: CẦN WEB SEARCH (web_search=ON + DB thiếu/PR)
    # ══════════════════════════════════════════════════════════
    cache_hit = False
    cache_sim = 0.0

    if not skip_web_search and web_search_enabled:
        state = _run_web_search_pipeline(state, standalone_query, action)
        cache_hit = state.get("search_cache_hit", False)
        cache_sim = state.get("search_cache_similarity", 0.0)

    # ══════════════════════════════════════════════════════════
    # NHÁNH C: WEB SEARCH TẮT → Chỉ dùng rag_context
    # Vẫn chạy Synthesizer + Sanitizer để format chuẩn
    # ══════════════════════════════════════════════════════════
    if not web_search_enabled:
        logger.info(
            "Web Search OFF -> Synthesizer+Sanitizer se dung RAG context only (%d chars)",
            len(rag_context)
        )

    # ══════════════════════════════════════════════════════════
    # Bước 4: GỘP NGỮ CẢNH (nếu có web results)
    # ══════════════════════════════════════════════════════════
    web_results = state.get("web_search_results") or ""
    web_citations = state.get("web_search_citations") or []

    if web_results:
        merged = _merge_context(
            rag_context=rag_context,
            web_results=web_results,
            web_citations=web_citations,
        )
        state["final_response"] = merged if merged else rag_context
    else:
        # Không có web results (web tắt hoặc web lỗi) → dùng rag_context
        state["final_response"] = rag_context

    # ══════════════════════════════════════════════════════════
    # Bước 5: SYNTHESIZER + SANITIZER (LUÔN CHẠY)
    # Input: rag_context + web_search_results (có thể None)
    # ══════════════════════════════════════════════════════════
    logger.info("Buoc 5: Synthesizer + Sanitizer...")
    state = _run_synthesizer_sanitizer_loop(state)

    # ══════════════════════════════════════════════════════════
    # Kết thúc pipeline
    # ══════════════════════════════════════════════════════════
    elapsed = time.time() - pipeline_start
    logger.info(
        "PROCEED RAG SEARCH v3 - Hoan tat (%.2fs) | web=%s | cache=%s (%.4f) | "
        "web_citations=%d | final=%d chars | source=%s",
        elapsed,
        "ON" if web_search_enabled else "OFF",
        'HIT' if cache_hit else 'MISS',
        cache_sim,
        len(state.get('web_search_citations') or []),
        len(state.get('final_response') or ''),
        state.get('response_source', 'unknown'),
    )

    return state

