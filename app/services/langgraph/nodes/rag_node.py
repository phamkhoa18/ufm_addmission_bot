"""
RAG Node — Truy xuất Context từ VectorDB nội bộ + LLM Context Curator.

Vị trí trong Graph:
  [embedding_node] → [rag_node] → [intent_node] hoặc [proceed_rag_search]

Nhiệm vụ:
  1. Nhận query_embeddings[0] (vector của standalone_query) từ Embedding Node.
  2. Gọi Hybrid Retriever: Vector Search + BM25 → RRF → Top 6 Parent.
  3. ★ CONTEXT CURATOR (MỚI): Gemini 2.5 Flash đọc standalone_query + chunks,
     lọc/giữ lại CHỈ thông tin liên quan, loại bỏ noise.
  4. Ghi kết quả đã curate vào state["rag_context"].

Nếu Database chưa sẵn sàng → rag_context = "" (fallback an toàn).

Model: Gemini 2.5 Flash (context curator) — ~$0.0001/query
Latency: ~100-300ms (DB) + ~500-1500ms (LLM curator)
"""

import time
from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _curate_context(standalone_query: str, raw_context: str) -> str:
    """
    ★ CONTEXT CURATOR — Gemini 2.5 Flash lọc ngữ cảnh.
    
    Input: standalone_query + raw_context (chuỗi text thô từ DB)
    Output: curated_context (chỉ giữ info liên quan) hoặc "" (không liên quan)
    
    Nếu LLM lỗi/timeout → trả về raw_context nguyên bản (fail-safe).
    """
    from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
    from app.core.prompts import prompt_manager

    config = query_flow_config.context_evaluator  # Dùng chung config context_evaluator (Gemini 2.5 Flash)

    try:
        sys_prompt = prompt_manager.get_system("context_curator")
        user_content = prompt_manager.render_user(
            "context_curator",
            standalone_query=standalone_query,
            rag_context=raw_context,
        )

        t0 = time.time()
        curated = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=config,
            node_key="context_evaluator",
        )
        elapsed = time.time() - t0

        if curated and curated.strip():
            # Nếu LLM trả "KHÔNG CÓ THÔNG TIN LIÊN QUAN" → context rỗng
            curated_clean = curated.strip()
            if curated_clean.upper() in ("KHÔNG CÓ THÔNG TIN LIÊN QUAN", "NONE", "N/A", "KHÔNG CÓ"):
                logger.info("Context Curator [%.3fs]: DB không liên quan → context rỗng", elapsed)
                return ""
            logger.info("Context Curator [%.3fs]: Curate OK, %d→%d chars", elapsed, len(raw_context), len(curated_clean))
            return curated_clean
        else:
            logger.warning("Context Curator [%.3fs]: LLM trả rỗng → dùng raw context", elapsed)
            return raw_context

    except Exception as e:
        logger.error("Context Curator lỗi: %s → dùng raw context", e)
        return raw_context


def rag_node(state: GraphState) -> GraphState:
    """
    🗄️ RAG NODE — Truy xuất ngữ cảnh từ VectorDB + LLM Context Curator.

    Input:
      - state["standalone_query"]:  Câu hỏi đã reformulate (từ Context Node)
      - state["query_embeddings"]:  [0] = vector 1024D của standalone_query

    Output:
      - state["rag_context"]:       Chuỗi text ĐÃ CURATE (chỉ info liên quan)
      - state["retrieved_chunks"]:  Danh sách RRF ranked chunks (debug/monitoring)

    Logic:
      1. Lấy vector embedding từ state
      2. Gọi hybrid_retrieve() → Vector + BM25 + RRF → 5 Parents
      3. ★ Context Curator (Gemini 2.5 Flash) lọc giữ info liên quan
      4. Ghi curated context vào state
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    query_embeddings = state.get("query_embeddings", [])
    start_time = time.time()

    # ── Kiểm tra: Có vector embedding nào không? ──
    if not query_embeddings:
        elapsed = time.time() - start_time
        logger.warning("RAG Node [%.3fs] Khong co query_embeddings -> bo qua RAG", elapsed)
        return {
            **state,
            "rag_context": "",
            "retrieved_chunks": [],
        }

    # ── Lấy vector chính (standalone_query) ──
    primary_embedding = query_embeddings[0]

    # ── Gọi Hybrid Retriever ──
    try:
        from app.services.retriever_service import hybrid_retrieve

        # Lấy metadata filter từ Intent Node (nếu có)
        program_level = state.get("program_level_filter")
        program_name = state.get("program_name_filter")

        result = hybrid_retrieve(
            query_text=standalone_query,
            query_embedding=primary_embedding,
            program_level=program_level,
            program_name=program_name,
            query_embeddings=query_embeddings,
        )

        raw_context = result["rag_context"]
        retrieved_chunks = result["retrieved_chunks"]
        top1_cosine = result.get("top1_cosine_score", 0.0)
        elapsed_db = time.time() - start_time

        logger.info(
            "RAG Node [%.3fs] Hybrid OK: vec=%d, bm25=%d, parents=%d, ctx=%d chars, top1=%.4f",
            elapsed_db, result['vector_count'], result['bm25_count'],
            len(result['parent_ids']), len(raw_context), top1_cosine
        )

        # ════════════════════════════════════════════════════════
        # ★ CONTEXT CURATOR — Chỉ curate khi context QUÁ LỚN (>8000 chars)
        # Với context nhỏ/vừa từ VectorDB nội bộ → dùng raw context trực tiếp
        # để tránh LLM Curator lọc mất thông tin hữu ích.
        # ════════════════════════════════════════════════════════
        _CURATOR_THRESHOLD = 4000  # Chỉ curate khi context vượt ngưỡng này

        if raw_context and len(raw_context) > _CURATOR_THRESHOLD:
            logger.info("RAG Node: Context lớn (%d chars > %d) → Curator lọc bớt", len(raw_context), _CURATOR_THRESHOLD)
            curated_context = _curate_context(standalone_query, raw_context)
        elif raw_context:
            logger.info("RAG Node: Context vừa đủ (%d chars) → dùng raw context trực tiếp (bypass Curator)", len(raw_context))
            curated_context = raw_context
        else:
            curated_context = ""
            logger.info("RAG Node: Context DB rỗng → bỏ qua Curator")

        elapsed_total = time.time() - start_time
        logger.info(
            "RAG Node [%.3fs total] Curated context: %d chars (raw=%d chars)",
            elapsed_total, len(curated_context), len(raw_context),
        )

        return {
            **state,
            "rag_context": curated_context,
            "retrieved_chunks": retrieved_chunks,
        }

    except ImportError as e:
        elapsed = time.time() - start_time
        logger.warning("RAG Node [%.3fs] psycopg2 chua cai: %s", elapsed, e)
        return {
            **state,
            "rag_context": "",
            "retrieved_chunks": [],
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("RAG Node [%.3fs] Loi truy xuat DB: %s", elapsed, e, exc_info=True)
        return {
            **state,
            "rag_context": "",
            "retrieved_chunks": [],
        }


