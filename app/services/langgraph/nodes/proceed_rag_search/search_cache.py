"""
Semantic Search Cache — Cache kết quả Web Search dựa trên ngữ nghĩa.

Mục đích:
  Tránh gọi lại gpt-4o-mini-search-preview cho các câu hỏi có ý nghĩa tương tự.
  Tiết kiệm ~$0.002/lần gọi + 10-15s latency mỗi lần cache hit.

Cơ chế:
  1. Nhúng query thành vector 1024D (BGE-M3, ~$0.00001)
  2. So sánh cosine similarity với tất cả entries trong cache
  3. Nếu ≥ threshold (0.9) và cùng nhánh intent → CACHE HIT
  4. Nếu < threshold → CACHE MISS → gọi API → lưu kết quả mới

Lưu trữ:
  In-memory dict (reset khi restart server)
  TTL: 24 giờ tự hết hạn
  Max: 200 entries (FIFO khi đầy)
"""

import time
import math
from typing import Optional, Tuple
from app.core.config import query_flow_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
from app.services.langgraph.nodes.embedding_node import _embed_batch


# ══════════════════════════════════════════════════════════
# GLOBAL CACHE STORE (In-Memory, sống cùng process)
# ══════════════════════════════════════════════════════════
_search_cache: list = []
# Mỗi entry: {
#   "query_text": str,
#   "query_vector": list[float],
#   "intent_action": str,
#   "web_results": str,
#   "web_citations": list,
#   "timestamp": float,
# }


def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Tính Cosine Similarity giữa 2 vectors.
    Dùng math thuần Python — không cần numpy.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cleanup_expired():
    """Xóa các entries hết hạn TTL."""
    global _search_cache
    config = query_flow_config.search_cache
    ttl_seconds = config.ttl_hours * 3600
    now = time.time()
    _search_cache = [
        entry for entry in _search_cache
        if (now - entry["timestamp"]) < ttl_seconds
    ]


def _embed_query(query_text: str) -> Optional[list]:
    """Nhúng 1 câu query thành vector 1024D (dùng BGE-M3)."""
    try:
        config = query_flow_config.embedding
        api_key = query_flow_config.api_keys.get_key(config.provider)
        base_url = query_flow_config.api_keys.get_base_url(config.provider)

        vectors = _embed_batch(
            texts=[query_text],
            api_key=api_key,
            base_url=base_url,
            model=config.model,
            dimensions=config.dimensions,
            max_retries=1,
            timeout=3,               # Cache là bổ trợ, không được tốn quá 3s
        )
        return vectors[0] if vectors else None
    except Exception as e:
        logger.warning("Search Cache - Loi embedding (khong anh huong): %s", e)
        return None


def cache_lookup(
    query_text: str, 
    intent_action: str
) -> Tuple[bool, float, Optional[str], Optional[list], Optional[list]]:
    """
    🔍 TRA CỨU CACHE — So sánh ngữ nghĩa với entries đã lưu.

    Returns: (cache_hit, similarity, web_results, web_citations, query_vector)
      - cache_hit=True:  Trả kết quả từ cache
      - cache_hit=False: Cần gọi Web Search API
      - query_vector:    Vector của query (để tái sử dụng khi lưu cache)
    """
    config = query_flow_config.search_cache

    if not config.enabled:
        return False, 0.0, None, None, None

    # Bước 1: Dọn entries hết hạn
    _cleanup_expired()

    if not _search_cache:
        # Cache rỗng → không có gì để so sánh → MISS ngay
        # Không gọi embedding ở đây (cache_save sẽ tự embed khi cần)
        logger.info("Search Cache - CACHE MISS (cache rong)")
        return False, 0.0, None, None, None

    # Bước 2: Nhúng query hiện tại (CHỈ khi có entries để so sánh)
    query_vector = _embed_query(query_text)
    if query_vector is None:
        return False, 0.0, None, None, None

    # Bước 3: Tìm entry có cosine cao nhất (cùng nhánh intent)
    best_sim = 0.0
    best_entry = None

    for entry in _search_cache:
        # Chỉ match cùng nhánh (UFM_SEARCH với UFM_SEARCH, PR với PR)
        if entry["intent_action"] != intent_action:
            continue

        sim = _cosine_similarity(query_vector, entry["query_vector"])
        if sim > best_sim:
            best_sim = sim
            best_entry = entry

    # Bước 4: Kiểm tra ngưỡng
    if best_sim >= config.similarity_threshold and best_entry:
        logger.info("Search Cache - CACHE HIT (cosine=%.4f >= %.2f)", best_sim, config.similarity_threshold)
        return True, best_sim, best_entry["web_results"], best_entry["web_citations"], query_vector
    else:
        if best_entry:
            logger.info("Search Cache - CACHE MISS (cosine=%.4f < %.2f)", best_sim, config.similarity_threshold)
        else:
            logger.info("Search Cache - CACHE MISS (khong co entry cung nhanh)")
        return False, best_sim, None, None, query_vector


def cache_save(
    query_text: str,
    intent_action: str,
    web_results: str,
    web_citations: list,
    query_vector: Optional[list] = None,
) -> None:
    """
    💾 LƯU VÀO CACHE — Ghi kết quả web search mới.
    Nhận `query_vector` từ bước lookup để tránh nhúng lại.
    """
    global _search_cache
    config = query_flow_config.search_cache

    if not config.enabled:
        return

    if not web_results:
        return  # Không cache kết quả null (fallback)

    # Nếu truyền sẵn vector từ bước lookup thì dùng luôn, không thì nhúng lại
    if query_vector is None:
        query_vector = _embed_query(query_text)
        
    if query_vector is None:
        return

    # Kiểm tra dung lượng
    if len(_search_cache) >= config.max_entries:
        # FIFO: xóa entry cũ nhất
        _search_cache.pop(0)
        logger.info("Search Cache - Cache day (%d), xoa entry cu nhat", config.max_entries)

    # Lưu entry mới
    _search_cache.append({
        "query_text": query_text,
        "query_vector": query_vector,
        "intent_action": intent_action,
        "web_results": web_results,
        "web_citations": web_citations,
        "timestamp": time.time(),
    })

    logger.info("Search Cache - Da luu cache (tong: %d entries)", len(_search_cache))


def cache_stats() -> dict:
    """Trả về thống kê cache hiện tại (dùng cho debug/monitoring)."""
    _cleanup_expired()
    return {
        "total_entries": len(_search_cache),
        "max_entries": query_flow_config.search_cache.max_entries,
        "ttl_hours": query_flow_config.search_cache.ttl_hours,
        "threshold": query_flow_config.search_cache.similarity_threshold,
    }
