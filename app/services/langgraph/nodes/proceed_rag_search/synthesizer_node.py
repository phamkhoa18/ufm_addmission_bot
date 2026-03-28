"""
Synthesizer Node — Tổng hợp câu trả lời theo tiêu chuẩn User-centric và Tôn trọng.

Vị trí:
  [web_search_node] → [synthesizer_node] → [sanitizer_node]

Nhiệm vụ:
  - Nếu nhánh UFM_SEARCH: Dùng Info Synthesizer (CẤM PR, CẤM VĂN VẺ BOT, đi thẳng vấn đề).
  - Nếu nhánh PR_SEARCH: Dùng PR Synthesizer (Lồng ghép uy tín nhưng KHÔNG DÌM trường bạn).
"""

import time
from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _build_synthesis_prompt(
    standalone_query: str,
    rag_context: str,
    web_results: str,
    web_citations: list,
    critique: str = None,
) -> str:
    """Gộp Data rành mạch."""
    sections = [f"## CÂU HỎI CỦA NGƯỜI DÙNG:\n{standalone_query}"]

    sections.append(
        f"## DỮ LIỆU NỘI BỘ (từ cơ sở dữ liệu UFM):\n{rag_context}" if rag_context 
        else "## DỮ LIỆU NỘI BỘ:\n(Không có)"
    )

    if web_results:
        sections.append(f"## THÔNG TIN TỪ WEB (đã xác minh nguồn):\n{web_results}")
        if web_citations:
            url_list = "\n".join(f"  - [{c['text']}]({c['url']})" for c in web_citations)
            sections.append(f"## DANH SÁCH NGUỒN HỢP LỆ (chỉ được trích dẫn từ đây):\n{url_list}")
    else:
        sections.append("## THÔNG TIN TỪ WEB:\n(Không có)")

    if critique:
        sections.append(
            f"## ⚠️ LỖI CẦN SỬA (từ lần kiểm duyệt trước):\n{critique}\n"
            f"BẠN PHẢI SỬA BẢN NHÁP CŨ ĐỂ KHÔNG MẮC LẠI CÁC LỖI TRÊN."
        )

    return "\n\n".join(sections)


def synthesizer_node(state: GraphState) -> GraphState:
    """
    ✍️ SYNTHESIZER NODE (Chọn Config tùy Intent).
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    action = state.get("intent_action", "")
    rag_context = state.get("rag_context")
    web_results = state.get("web_search_results")
    web_citations = state.get("web_search_citations") or []
    critique = state.get("sanitizer_critique")
    
    # Rẽ domain dựa trên action
    if action == "PROCEED_RAG_UFM_SEARCH":
        config = query_flow_config.info_synthesizer
        prompt_domain = "info_synthesizer"
    else:
        config = query_flow_config.pr_synthesizer
        prompt_domain = "pr_synthesizer"

    start_time = time.time()

    # Render user_prompt từ Prompt Hub (thay thế _build_synthesis_prompt cũ)
    user_content = prompt_manager.render_user(
        prompt_domain,
        standalone_query=standalone_query,
        rag_context=rag_context or "",
        web_results=web_results or "",
        web_citations=web_citations,
        critique=critique,
    )

    try:
        draft = _call_gemini_api_with_fallback(
            system_prompt=prompt_manager.get_system(prompt_domain),
            user_content=user_content,
            config_section=config,
            node_key=prompt_domain,
        )

        elapsed = time.time() - start_time
        loop_info = f" (sửa lần {state.get('sanitizer_loop_count', 0)})" if critique else ""
        logger.info("Synthesizer [%.3fs] Draft%s: %d ky tu", elapsed, loop_info, len(draft))

        return {
            **state,
            "synthesized_draft": draft,
            "next_node": "sanitizer",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Synthesizer [%.3fs] Loi: %s", elapsed, e, exc_info=True)
        fallback = rag_context or "Xin lỗi, hiện tại mình chưa thể tổng hợp thông tin."
        return {
            **state,
            "synthesized_draft": fallback,
            "next_node": "sanitizer",
        }
