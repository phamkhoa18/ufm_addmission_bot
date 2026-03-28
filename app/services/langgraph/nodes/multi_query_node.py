"""
Multi-Query Node — Sinh biến thể câu hỏi để tăng Recall cho Vector Search.

Vị trí trong Graph:
  [contextual_guard_node] → [multi_query_node] → [embedding_node]
  → CHỈ CHẠY KHI standalone_query ĐÃ QUA Contextual-Guard (SAFE)

Nhiệm vụ:
  Nhận standalone_query → Gọi Gemini Flash Lite → Sinh 3 biến thể.
  Mỗi biến thể dùng từ vựng khác nhau nhưng cùng nghĩa.

Model: google/gemini-3.1-flash-lite-preview (OpenRouter)
Chi phí: ~$0.000002/query
Latency: ~200ms

Ví dụ:
  standalone_query: "Học phí ngành Marketing là bao nhiêu?"
  → multi_queries: [
      "Chi phí đào tạo ngành Marketing tại UFM năm 2026",
      "Mức học phí chương trình cử nhân Marketing UFM",
      "Ngành Marketing UFM thu bao nhiêu tiền một kỳ học"
    ]
"""

import re
import time
from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_variants(raw_output: str) -> list:
    """
    Parse output dạng "1. ...\n2. ...\n3. ..." thành list[str].
    Xử lý nhiều format có thể (có hoặc không có số thứ tự).
    """
    lines = raw_output.strip().split("\n")
    variants = []
    for line in lines:
        # Xóa số thứ tự đầu dòng: "1. ", "2) ", "- ", "• "
        cleaned = re.sub(r"^\s*[\d]+[\.\)]\s*", "", line.strip())
        cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 5:  # Bỏ qua dòng quá ngắn
            variants.append(cleaned)
    return variants


def multi_query_node(state: GraphState) -> GraphState:
    """
    🔀 MULTI-QUERY NODE — Sinh 3 biến thể câu hỏi.

    Input:
      - state["standalone_query"]: Câu hỏi đã reformulate (từ Context Node)

    Output:
      - state["multi_queries"]: List 3 biến thể câu hỏi

    Logic:
      1. Nếu multi_query bị tắt → multi_queries = [] → chỉ dùng standalone_query
      2. Gọi Gemini Flash Lite sinh biến thể
      3. Parse output thành list
      4. Nếu API lỗi → multi_queries = [] (fallback, chỉ dùng standalone gốc)
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    config = query_flow_config.multi_query
    start_time = time.time()

    # ── Trường hợp 1: Multi-Query bị tắt ──
    if not config.enabled:
        return {
            **state,
            "multi_queries": [],
        }

    # ── Trường hợp 2: Gọi Gemini sinh biến thể ──
    try:
        # System prompt đã chứa {{ num_variants }} → render bằng Jinja2
        system_prompt = prompt_manager.render_user(
            "multi_query_node",
            num_variants=config.num_variants,
        )
        # Nhưng vì system_prompt nằm ở key 'system_prompt', ta dùng get_system
        # rồi replace thủ công cho đơn giản
        sys_prompt_raw = prompt_manager.get_system("multi_query_node")
        sys_prompt = sys_prompt_raw.replace(
            "{{ num_variants }}", str(config.num_variants)
        )

        user_content = prompt_manager.render_user(
            "multi_query_node",
            standalone_query=standalone_query,
        )

        raw_output = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=config,
            node_key="multi_query",
        )

        variants = _parse_variants(raw_output)

        # Giới hạn số biến thể đúng theo config
        variants = variants[:config.num_variants]

        elapsed = time.time() - start_time
        logger.info("Multi-Query [%.3fs] Sinh %d bien the", elapsed, len(variants))

        return {
            **state,
            "multi_queries": variants,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Multi-Query [%.3fs] Loi: %s", elapsed, e, exc_info=True)
        return {
            **state,
            "multi_queries": [],
        }


def multi_query_router(state: GraphState) -> str:
    """Conditional Edge: multi_query → embedding (luôn luôn)."""
    return "embedding"
