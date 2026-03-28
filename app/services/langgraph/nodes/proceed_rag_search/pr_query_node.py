"""
PR & UFM Query Node — Sinh truy vấn cho Web Search tùy theo Intent.

Vị trí:
  [Intent] → [pr_query_node] → [web_search_node]

Logic Phân Luồng:
  - Nếu intent_action == "PROCEED_RAG_UFM_SEARCH":
    Dùng ufm_query config → Sinh tối đa 2 câu truy vấn hành chính.
    Ghi vào: state["ufm_search_queries"] (list)
    
  - Nếu intent_action == "PROCEED_RAG_PR_SEARCH":
    Dùng pr_query config → Sinh 1 câu truy vấn khoe thành tích.
    Ghi vào: state["pr_search_query"] (str)
"""

import time
from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _generate_ufm_queries(standalone_query: str) -> list:
    """Gọi LLM sinh 1-2 câu truy cập subdomain UFM."""
    config = query_flow_config.ufm_query
    if not config.enabled:
        return []

    try:
        raw_output = _call_gemini_api_with_fallback(
            system_prompt=prompt_manager.get_system("ufm_query_node"),
            user_content=prompt_manager.render_user("ufm_query_node", standalone_query=standalone_query),
            config_section=config,
            node_key="ufm_query",
        )
        
        # Tách dòng, loại bỏ dòng rỗng và ký tự thừa
        queries = []
        for line in raw_output.split('\n'):
            q = line.strip().strip('-').strip('*').strip('"').strip("'").strip()
            # Bỏ các line đánh số (VD: "1. Điểm chuẩn...")
            import re
            q = re.sub(r'^\d+\.\s*', '', q)
            if q:
                queries.append(q)
        
        return queries[:2]  # Lấy tối đa 2 câu
    except Exception as e:
        logger.error("Loi sinh UFM Query: %s", e, exc_info=True)
        return []


def _generate_pr_query(standalone_query: str) -> str:
    """Gọi LLM sinh 1 câu truy vấn PR."""
    config = query_flow_config.pr_query
    if not config.enabled:
        return None

    try:
        raw_output = _call_gemini_api_with_fallback(
            system_prompt=prompt_manager.get_system("pr_query_node"),
            user_content=prompt_manager.render_user("pr_query_node", standalone_query=standalone_query),
            config_section=config,
            node_key="pr_query",
        )
        q = raw_output.strip().strip('"').strip("'").strip()
        import re
        q = re.sub(r'^\d+\.\s*', '', q)
        return q if q else None
    except Exception as e:
        logger.error("Loi sinh PR Query: %s", e, exc_info=True)
        return None


def pr_query_node(state: GraphState) -> GraphState:
    """
    🎯 PR & UFM QUERY NODE — Sinh truy vấn Web Search.

    Đầu ra State:
      - Nếu UFM Search: state["ufm_search_queries"] = [...]
      - Nếu PR Search:  state["pr_search_query"] = "..."
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    action = state.get("intent_action", "")
    start_time = time.time()

    logger.info("Query Gen Node - Nhanh: %s", action)

    # ── NHÁNH 1: UFM INFO SEARCH ──
    if action == "PROCEED_RAG_UFM_SEARCH":
        queries = _generate_ufm_queries(standalone_query)
        elapsed = time.time() - start_time
        logger.info("Query Gen [%.3fs] Sinh %d UFM Queries", elapsed, len(queries))
        return {
            **state,
            "ufm_search_queries": queries,
            "pr_search_query": None,
            "next_node": "web_search",
        }

    # ── NHÁNH 2: PR SEARCH (HOẶC MẶC ĐỊNH) ──
    else:
        pr_q = _generate_pr_query(standalone_query)
        elapsed = time.time() - start_time
        logger.info("Query Gen [%.3fs] Sinh PR Query: '%s'", elapsed, pr_q[:80])
        return {
            **state,
            "ufm_search_queries": None,
            "pr_search_query": pr_q,
            "next_node": "web_search",
        }
