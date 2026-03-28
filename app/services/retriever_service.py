"""
Retriever Service v2 — Hybrid Search (Vector + BM25 + RRF) + Parent Retrieval.

Cải tiến so với v1:
  ✅ ThreadedConnectionPool thay single connection
  ✅ Dynamic WHERE Builder (gộp SQL duplicate)
  ✅ DictCursor thay row[0], row[1]...
  ✅ Metadata Filter: program_level + program_name
  ✅ Multi-Query Score Boost (4 embeddings)
  ✅ Parallel Vector + BM25 (ThreadPoolExecutor)
  ✅ BM25 dùng stored tsvector column (GIN index)
  ✅ Observability metrics dict
"""

import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.config.retriever import RetrieverConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# DATABASE CONNECTION POOL (ThreadedConnectionPool)
# ══════════════════════════════════════════════════════════
_pool = None


def _get_pool(cfg: RetrieverConfig):
    """
    Lấy connection pool (thread-safe). Tạo mới nếu chưa có.
    """
    global _pool

    if _pool is not None:
        return _pool

    try:
        import psycopg2
        from psycopg2.pool import ThreadedConnectionPool
    except ImportError:
        raise ImportError(
            "psycopg2 chưa được cài. Chạy: pip install psycopg2-binary"
        )

    db = cfg.db
    _pool = ThreadedConnectionPool(
        minconn=db.pool_min,
        maxconn=db.pool_max,
        host=db.host,
        port=db.port,
        dbname=db.dbname,
        user=db.user,
        password=db.password,
        connect_timeout=db.connect_timeout,
        options=f"-c statement_timeout={db.query_timeout * 1000}",
    )
    logger.info("Retriever - Connection Pool: %s:%s/%s (min=%d, max=%d)",
                db.host, db.port, db.dbname, db.pool_min, db.pool_max)
    return _pool


def _get_connection(cfg: RetrieverConfig):
    """Lấy 1 connection từ pool."""
    pool = _get_pool(cfg)
    conn = pool.getconn()
    conn.autocommit = True
    return conn


def _put_connection(conn):
    """Trả connection về pool."""
    if _pool is not None and conn is not None:
        try:
            _pool.putconn(conn)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
# DYNAMIC WHERE BUILDER (Gộp tất cả filter logic)
# ══════════════════════════════════════════════════════════
def _build_filters(program_level: str = None, program_name: str = None):
    """
    Xây mệnh đề WHERE động cho SQL queries.
    Tránh duplicate code giữa vector / bm25.

    Returns:
        (where_clause: str, params: list)
    """
    clauses = ["is_active = TRUE"]
    params = []

    if program_level:
        clauses.append("(program_level = %s OR program_level IS NULL)")
        params.append(program_level)

    if program_name:
        clauses.append("(program_name ILIKE %s OR program_name IS NULL)")
        params.append(f"%{program_name}%")

    return " AND ".join(clauses), params


# ══════════════════════════════════════════════════════════
# VECTOR SEARCH (pgvector Cosine Similarity — HNSW Index)
# ══════════════════════════════════════════════════════════
def search_vector(
    query_embedding: list,
    cfg: RetrieverConfig,
    program_level: str = None,
    program_name: str = None,
) -> list:
    """
    Tìm Top K chunks có cosine similarity cao nhất với query vector.
    Hỗ trợ metadata filter: program_level + program_name.
    """
    vs_cfg = cfg.vector_search
    if not vs_cfg.enabled:
        return []

    conn = _get_connection(cfg)
    try:
        from psycopg2.extras import RealDictCursor

        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        where_clause, where_params = _build_filters(program_level, program_name)

        sql = f"""
            SELECT
                chunk_id,
                chunk_level,
                parent_id,
                section_path,
                program_name,
                1 - (embedding <=> %s::vector) AS cosine_score,
                LEFT(content, 200) AS content_preview
            FROM knowledge_chunks
            WHERE {where_clause}
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """

        params = [vec_str] + where_params + [vec_str, vs_cfg.top_k]

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row["chunk_id"],
                "chunk_level": row["chunk_level"],
                "parent_id": row["parent_id"],
                "section_path": row["section_path"],
                "program_name": row["program_name"],
                "score": float(row["cosine_score"]),
                "content_preview": row["content_preview"],
                "source": "vector",
            })

        # Log cảnh báo nếu top 1 dưới ngưỡng (chỉ warning, KHÔNG chặn data)
        if results and results[0]["score"] < vs_cfg.similarity_threshold:
            logger.warning(
                "Retriever - Top1 cosine=%.4f < threshold %.2f",
                results[0]['score'], vs_cfg.similarity_threshold
            )

        return results
    finally:
        _put_connection(conn)


# ══════════════════════════════════════════════════════════
# BM25 FULL-TEXT SEARCH (Postgres tsvector — stored column)
# ══════════════════════════════════════════════════════════
def search_bm25(
    query_text: str,
    cfg: RetrieverConfig,
    program_level: str = None,
    program_name: str = None,
) -> list:
    """
    Tìm Top K chunks bằng BM25 (stored tsvector + ts_rank_cd).
    Dùng cột content_tsvector (đã lưu sẵn) nếu có, fallback sang runtime.
    """
    bm_cfg = cfg.bm25_search
    if not bm_cfg.enabled:
        return []

    conn = _get_connection(cfg)
    try:
        from psycopg2.extras import RealDictCursor

        ts_cfg = bm_cfg.ts_config
        where_clause, where_params = _build_filters(program_level, program_name)

        # Ưu tiên dùng stored tsvector column (nếu đã tạo migration)
        # Fallback về runtime tsvector nếu chưa migrate
        tsvector_col = getattr(bm_cfg, 'use_stored_tsvector', False)

        if tsvector_col:
            # Dùng stored column — GIN index hoạt động
            tsvec_expr = "content_tsvector"
        else:
            # Fallback — tính runtime (chậm hơn)
            tsvec_expr = f"to_tsvector('{ts_cfg}', content)"

        sql = f"""
            SELECT
                chunk_id,
                chunk_level,
                parent_id,
                section_path,
                program_name,
                ts_rank_cd(
                    {tsvec_expr},
                    plainto_tsquery('{ts_cfg}', %s)
                ) AS bm25_score,
                LEFT(content, 200) AS content_preview
            FROM knowledge_chunks
            WHERE {where_clause}
              AND {tsvec_expr} @@ plainto_tsquery('{ts_cfg}', %s)
            ORDER BY bm25_score DESC
            LIMIT %s;
        """

        params = [query_text] + where_params + [query_text, bm_cfg.top_k]

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row["chunk_id"],
                "chunk_level": row["chunk_level"],
                "parent_id": row["parent_id"],
                "section_path": row["section_path"],
                "program_name": row["program_name"],
                "score": float(row["bm25_score"]),
                "content_preview": row["content_preview"],
                "source": "bm25",
            })

        return results
    finally:
        _put_connection(conn)


# ══════════════════════════════════════════════════════════
# RRF — Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════
def rrf_merge(
    vector_results: list,
    bm25_results: list,
    k: int = 60,
) -> list:
    """
    Gộp 2 danh sách kết quả bằng Reciprocal Rank Fusion.

    Công thức: RRF_score(d) = Σ 1 / (k + rank_i(d))
      - k = 60 (giá trị chuẩn từ paper gốc Cormack et al. 2009)
      - rank_i = ranking của document d trong danh sách i (1-indexed)
    """
    rrf_scores: dict = {}     # chunk_id → total RRF score
    chunk_data: dict = {}     # chunk_id → metadata dict

    # ── Tính RRF cho Vector results ──
    for rank, item in enumerate(vector_results, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
        chunk_data[cid] = item

    # ── Tính RRF cho BM25 results ──
    for rank, item in enumerate(bm25_results, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
        # Nếu chunk đã có từ vector, giữ bản từ vector (có cosine score)
        if cid not in chunk_data:
            chunk_data[cid] = item

    # ── Sắp xếp theo RRF score giảm dần ──
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for cid in sorted_ids:
        entry = chunk_data[cid].copy()
        entry["rrf_score"] = rrf_scores[cid]
        merged.append(entry)

    return merged


# ══════════════════════════════════════════════════════════
# MULTI-QUERY VECTOR BOOST — Tận dụng 4 embeddings
# ══════════════════════════════════════════════════════════
def search_vector_multi_query(
    query_embeddings: list,
    cfg: RetrieverConfig,
    program_level: str = None,
    program_name: str = None,
) -> list:
    """
    Chạy Vector Search với tất cả embeddings (standalone + multi-query).
    Deduplicate và giữ score cao nhất cho mỗi chunk.
    """
    best_scores: dict = {}   # chunk_id → best score
    best_data: dict = {}     # chunk_id → metadata dict

    for emb in query_embeddings:
        results = search_vector(emb, cfg, program_level, program_name)
        for item in results:
            cid = item["chunk_id"]
            if cid not in best_scores or item["score"] > best_scores[cid]:
                best_scores[cid] = item["score"]
                best_data[cid] = item

    # Sắp xếp theo cosine score giảm dần, lấy top_k
    sorted_ids = sorted(best_scores.keys(), key=lambda x: best_scores[x], reverse=True)
    top_k = cfg.vector_search.top_k

    return [best_data[cid] for cid in sorted_ids[:top_k]]


# ══════════════════════════════════════════════════════════
# PARENT EXTRACTION — Lấy top 5 Parent IDs duy nhất
# ══════════════════════════════════════════════════════════
def extract_unique_parent_ids(
    ranked_chunks: list,
    top_parents: int = 5,
) -> list:
    """
    Duyệt danh sách RRF đã sắp xếp → Trích ra top N parent_ids duy nhất.

    Chiến lược ưu tiên:
      - Nếu chunk là child → Lấy parent_id của nó.
      - Nếu chunk là parent → Lấy chính chunk_id.
      - Nếu chunk là standard (không cha-con) → Lấy chính chunk_id.
      - Nếu parent_id đã xuất hiện trước đó → BỎ QUA (chống trùng).
      - Dừng khi đủ top_parents IDs.
    """
    seen_parents = set()
    parent_ids = []

    for chunk in ranked_chunks:
        level = chunk.get("chunk_level", "standard")
        chunk_id = chunk["chunk_id"]
        parent_id = chunk.get("parent_id")

        # Xác định ID đại diện
        if level == "child" and parent_id:
            representative_id = parent_id
        else:
            representative_id = chunk_id

        if representative_id in seen_parents:
            continue

        seen_parents.add(representative_id)
        parent_ids.append(representative_id)

        if len(parent_ids) >= top_parents:
            break

    return parent_ids


# ══════════════════════════════════════════════════════════
# FETCH PARENT CONTENT — Lấy nội dung đầy đủ từ DB
# ══════════════════════════════════════════════════════════
def fetch_parent_contents(
    parent_ids: list,
    cfg: RetrieverConfig,
) -> list:
    """
    Query DB lấy nội dung đầy đủ của Parent Chunks.
    Giữ nguyên thứ tự ưu tiên từ danh sách RRF.
    Dùng DictCursor để truy cập theo tên cột (an toàn hơn index).
    """
    if not parent_ids:
        return []

    conn = _get_connection(cfg)
    try:
        from psycopg2.extras import RealDictCursor

        sql = """
            SELECT
                chunk_id,
                content,
                source,
                section_path,
                section_name,
                program_name,
                program_level,
                ma_nganh,
                academic_year,
                chunk_level,
                char_count,
                extra
            FROM knowledge_chunks
            WHERE chunk_id = ANY(%s)
              AND is_active = TRUE
        """

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (parent_ids,))
            rows = cur.fetchall()

        # Map kết quả theo chunk_id để giữ thứ tự ưu tiên
        content_map = {row["chunk_id"]: dict(row) for row in rows}

        # Trả về đúng thứ tự ưu tiên RRF
        return [content_map[pid] for pid in parent_ids if pid in content_map]
    finally:
        _put_connection(conn)


# ══════════════════════════════════════════════════════════
# FORMAT CONTEXT — Đóng gói context cho LLM chính
# ══════════════════════════════════════════════════════════
def format_rag_context(
    parent_docs: list,
    cfg: RetrieverConfig,
) -> str:
    """
    Gộp nội dung parent docs thành 1 chuỗi text có trích dẫn cấu trúc.
    Cắt ngắn nếu tổng vượt max_parent_chars.
    """
    pr_cfg = cfg.parent_retrieval
    parts = []
    total_chars = 0

    for i, doc in enumerate(parent_docs, start=1):
        content = doc["content"]

        # Kiểm tra tổng ký tự không vượt giới hạn
        if total_chars + len(content) > pr_cfg.max_parent_chars:
            remaining = pr_cfg.max_parent_chars - total_chars
            if remaining > 100:
                content = content[:remaining] + "..."
            else:
                break

        # Gắn metadata header trích dẫn cấu trúc (Tên văn bản + Mục + Ngành + URL)
        if pr_cfg.include_metadata:
            citation_parts = []
            extra = doc.get("extra") or {}

            # Tên văn bản chính thức (từ extra.title hoặc source filename)
            doc_title = extra.get("title") or ""
            source = doc.get("source") or ""
            source_url = extra.get("url") or ""

            # Tạo text citation, nếu có url thì chuyển thành the link
            doc_name = doc_title or source or f"Tài liệu {i}"
            if source_url:
                citation_parts.append(f"Nguồn: [{doc_name}]({source_url})")
            else:
                citation_parts.append(f"Nguồn: {doc_name}")

            # Số công văn (từ extra.doc_number)
            doc_number = extra.get("doc_number") or ""
            if doc_number:
                citation_parts.append(f"Số: {doc_number}")

            section_path = doc.get("section_path") or ""
            if section_path:
                formatted_path = section_path.replace(" > ", " > ")
                citation_parts.append(formatted_path)

            section_name = doc.get("section_name") or ""
            if section_name:
                citation_parts.append(f"Mục: {section_name}")

            program_name = doc.get("program_name") or ""
            if program_name:
                citation_parts.append(f"Ngành: {program_name}")

            ma_nganh = doc.get("ma_nganh") or ""
            if ma_nganh:
                citation_parts.append(f"Mã ngành: {ma_nganh}")

            academic_year = doc.get("academic_year") or ""
            if academic_year:
                citation_parts.append(f"Năm: {academic_year}")

            if citation_parts:
                citation = " | ".join(citation_parts)
                parts.append(f"[{citation}]\n{content}")
            else:
                parts.append(f"[Nguồn {i}]\n{content}")
        else:
            parts.append(f"[Nguồn {i}]\n{content}")

        total_chars += len(content)

    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════
# HÀM CHÍNH: HYBRID RETRIEVAL PIPELINE v2
# ══════════════════════════════════════════════════════════
def hybrid_retrieve(
    query_text: str,
    query_embedding: list,
    cfg: RetrieverConfig = None,
    program_level: str = None,
    program_name: str = None,
    query_embeddings: list = None,
) -> dict:
    """
    🔍 Pipeline Hybrid Search v2 — tối ưu tốc độ + chính xác.

    Args:
        query_text: Câu hỏi standalone (dùng cho BM25).
        query_embedding: Vector 1024D chính (dùng cho Cosine Search).
        cfg: Config object (mặc định tạo mới từ YAML).
        program_level: Filter bậc đào tạo (thac_si/tien_si/dai_hoc).
        program_name: Filter tên ngành (ILIKE match).
        query_embeddings: Danh sách tất cả embeddings (multi-query boost).

    Returns dict:
        {
            "rag_context": str,
            "retrieved_chunks": list,
            "parent_ids": list,
            "parent_docs": list,
            "vector_count": int,
            "bm25_count": int,
            "top1_cosine_score": float,
            "elapsed": float,
            "metrics": dict,
        }
    """
    if cfg is None:
        cfg = RetrieverConfig()

    start_time = time.time()

    # ── Log metadata filter ──
    if program_level:
        logger.info("Retriever - program_level_filter='%s'", program_level)
    if program_name:
        logger.info("Retriever - program_name_filter='%s'", program_name)

    # ════════════════════════════════════════════════════════
    # Bước 1+2: Vector + BM25 SONG SONG (ThreadPoolExecutor)
    # ════════════════════════════════════════════════════════
    use_multi = (
        query_embeddings
        and len(query_embeddings) > 1
        and getattr(cfg.vector_search, 'use_multi_query', False)
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        # ── Vector Search ──
        if use_multi:
            logger.info("Retriever - Multi-Query Vector Search (%d embeddings, top %d)...",
                        len(query_embeddings), cfg.vector_search.top_k)
            vec_future = pool.submit(
                search_vector_multi_query,
                query_embeddings, cfg, program_level, program_name,
            )
        else:
            logger.info("Retriever - Vector Search (top %d)...", cfg.vector_search.top_k)
            vec_future = pool.submit(
                search_vector,
                query_embedding, cfg, program_level, program_name,
            )

        # ── BM25 Search ──
        logger.info("Retriever - BM25 Search (top %d)...", cfg.bm25_search.top_k)
        bm25_future = pool.submit(
            search_bm25,
            query_text, cfg, program_level, program_name,
        )

        # ── Đợi kết quả ──
        vector_results = vec_future.result()
        bm25_results = bm25_future.result()

    logger.info("Retriever - Vector tim duoc: %d chunks", len(vector_results))
    logger.info("Retriever - BM25 tim duoc: %d chunks", len(bm25_results))

    if vector_results:
        top_score = vector_results[0]["score"]
        logger.debug("  Top1 Cosine: %.4f", top_score)

    # ── Bước 3: RRF Merge ──
    logger.info("Retriever - RRF Merge (k=%d)...", cfg.rrf.k)
    merged = rrf_merge(vector_results, bm25_results, k=cfg.rrf.k)
    logger.info("Retriever - Tong unique chunks sau RRF: %d", len(merged))

    # ── Bước 4: Extract Parent IDs ──
    top_n = cfg.parent_retrieval.top_parents
    logger.info("Retriever - Trich xuat Top %d Parent IDs...", top_n)
    parent_ids = extract_unique_parent_ids(merged, top_parents=top_n)
    logger.debug("Retriever - Parent IDs: %s", parent_ids)

    # ── Bước 5: Fetch Parent Contents ──
    parent_docs = fetch_parent_contents(parent_ids, cfg)
    logger.info("Retriever - Tai noi dung: %d parent docs", len(parent_docs))

    # ── Bước 6: Format Context ──
    rag_context = format_rag_context(parent_docs, cfg)

    elapsed = time.time() - start_time
    top1_cosine = vector_results[0]["score"] if vector_results else 0.0
    logger.info(
        "Retriever - Hybrid Search hoan tat (%.3fs) - %d ky tu, top1=%.4f",
        elapsed, len(rag_context), top1_cosine
    )

    # ════════════════════════════════════════════════════════
    # Observability Metrics
    # ════════════════════════════════════════════════════════
    metrics = {
        "vector_count": len(vector_results),
        "bm25_count": len(bm25_results),
        "rrf_unique": len(merged),
        "top1_cosine": top1_cosine,
        "parent_docs": len(parent_docs),
        "context_chars": len(rag_context),
        "latency_ms": round(elapsed * 1000, 1),
        "program_level": program_level or "all",
        "program_name": program_name or "all",
        "multi_query": use_multi,
    }

    return {
        "rag_context": rag_context,
        "retrieved_chunks": merged,
        "parent_ids": parent_ids,
        "parent_docs": parent_docs,
        "vector_count": len(vector_results),
        "bm25_count": len(bm25_results),
        "top1_cosine_score": top1_cosine,
        "elapsed": elapsed,
        "metrics": metrics,
    }
