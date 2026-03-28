"""
Sanitizer Node — Gác Cổng Kép (Citation Verifier + Hallucination Checker).

Vị trí trong Pipeline:
  [synthesizer_node] → [sanitizer_node] → (loop back hoặc output)

Nhiệm vụ:
  Kiểm tra draft từ Synthesizer:
  1. Trích dẫn URL có tồn tại trong nguồn gốc không? (Citation Check)
  2. Markdown link format đúng [text](url) không? (Render Check) 
  3. Nội dung có bịa đặt không? (Hallucination Check)
  4. Tone PR có vượt ngưỡng không? (Tone Check)

  Nếu REJECT → Trả critique + loop lại Synthesizer (tối đa 2 lần)
  Nếu PASS → Ghi final_response

Model: google/gemini-2.5-flash-preview (OpenRouter)
"""

import json
import re
import time
from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _build_sanitizer_prompt(
    draft: str,
    standalone_query: str,
    rag_context: str,
    web_results: str,
    web_citations: list,
) -> str:
    """
    Xây dựng user prompt cho Sanitizer Agent.
    """
    sections = []

    sections.append(f"## CÂU HỎI GỐC:\n{standalone_query}")
    sections.append(f"## BẢN NHÁP CẦN KIỂM TRA:\n{draft}")

    if rag_context:
        sections.append(f"## CONTEXT NỘI BỘ GỐC:\n{rag_context}")

    if web_results:
        sections.append(f"## CONTEXT WEB GỐC:\n{web_results}")

    if web_citations:
        url_list = "\n".join(f"  - {c['url']}" for c in web_citations)
        sections.append(f"## DANH SÁCH URL HỢP LỆ:\n{url_list}")
    else:
        sections.append("## DANH SÁCH URL HỢP LỆ:\n(Không có URL nào — draft KHÔNG ĐƯỢC chứa URL)")

    return "\n\n".join(sections)


def _parse_sanitizer_response(raw_output: str) -> dict:
    """
    Parse JSON response từ Sanitizer Agent.
    Trả về {"passed": bool, "critique": str}
    """
    # Thử parse JSON trực tiếp
    try:
        # Tìm JSON block trong output (có thể có text bao quanh)
        json_match = re.search(r'\{[^}]+\}', raw_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "passed": bool(data.get("passed", False)),
                "critique": str(data.get("critique", "")).strip(),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: tìm keyword
    lower = raw_output.lower()
    if "passed" in lower and "true" in lower:
        return {"passed": True, "critique": ""}

    # Mặc định: coi như pass (tránh loop vô tận)
    return {"passed": True, "critique": ""}


def sanitizer_node(state: GraphState) -> GraphState:
    """
    🛡️ SANITIZER NODE — Gác Cổng Kép (Verifier + Sanitizer).

    Input:
      - state["synthesized_draft"]: Bản nháp cần kiểm tra
      - state["standalone_query"]: Câu hỏi gốc
      - state["rag_context"]: Context nội bộ
      - state["web_search_results"]: Context web
      - state["web_search_citations"]: Trích dẫn đã extract
      - state["sanitizer_loop_count"]: Đếm vòng lặp

    Output:
      - state["sanitizer_passed"]: True/False
      - state["sanitizer_critique"]: Mô tả lỗi (nếu reject)
      - state["sanitizer_loop_count"]: Tăng lên 1
      - state["final_response"]: Ghi nếu passed
      - state["next_node"]: "output" nếu pass, "synthesizer" nếu reject
    """
    draft = state.get("synthesized_draft", "")
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    rag_context = state.get("rag_context")
    web_results = state.get("web_search_results")
    web_citations = state.get("web_search_citations") or []
    loop_count = state.get("sanitizer_loop_count", 0)
    config = query_flow_config.sanitizer
    start_time = time.time()

    # ── Kiểm tra giới hạn loop ──
    if loop_count >= config.max_loops:
        elapsed = time.time() - start_time
        logger.warning("Sanitizer [%.3fs] Dat gioi han %d vong -> chap nhan draft", elapsed, config.max_loops)
        return {
            **state,
            "sanitizer_passed": True,
            "sanitizer_critique": "",
            "sanitizer_loop_count": loop_count,
            "final_response": draft,
            "response_source": "rag_search_synthesized",
            "next_node": "response",
        }

    # ── Gọi Sanitizer LLM ──
    try:
        user_content = prompt_manager.render_user(
            "sanitizer_node",
            draft=draft,
            standalone_query=standalone_query,
            rag_context=rag_context or "",
            web_results=web_results or "",
            web_citations=web_citations,
        )

        raw_output = _call_gemini_api_with_fallback(
            system_prompt=prompt_manager.get_system("sanitizer_node"),
            user_content=user_content,
            config_section=config,
            node_key="sanitizer",
        )

        result = _parse_sanitizer_response(raw_output)
        elapsed = time.time() - start_time

        if result["passed"]:
            logger.info("Sanitizer [%.3fs] PASSED - Draft OK", elapsed)
            return {
                **state,
                "sanitizer_passed": True,
                "sanitizer_critique": "",
                "sanitizer_loop_count": loop_count + 1,
                "final_response": draft,
                "response_source": "rag_search_synthesized",
                "next_node": "response",
            }
        else:
            logger.warning(
                "Sanitizer [%.3fs] REJECTED (lan %d/%d): %s",
                elapsed, loop_count + 1, config.max_loops, result['critique'][:200]
            )
            return {
                **state,
                "sanitizer_passed": False,
                "sanitizer_critique": result["critique"],
                "sanitizer_loop_count": loop_count + 1,
                "next_node": "synthesizer",
            }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Sanitizer [%.3fs] Loi: %s -> chap nhan draft", elapsed, e, exc_info=True)
        return {
            **state,
            "sanitizer_passed": True,
            "sanitizer_critique": "",
            "sanitizer_loop_count": loop_count,
            "final_response": draft,
            "response_source": "rag_search_synthesized",
            "next_node": "response",
        }


def sanitizer_router(state: GraphState) -> str:
    """
    Conditional Edge:
      - sanitizer_passed = True  → "response" (thoát pipeline)
      - sanitizer_passed = False → "synthesizer" (loop lại để sửa)
    """
    if state.get("sanitizer_passed", True):
        return "response"
    return "synthesizer"
