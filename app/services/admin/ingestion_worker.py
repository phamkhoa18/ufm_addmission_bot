"""
Ingestion Worker — Background pipeline nạp file Markdown vào VectorDB.

Luồng: Header Normalize → Dedup Check → Hierarchical Chunk → Semantic Chunk → Embedding → DB Insert.
Tái sử dụng hoàn toàn HierarchicalChunker + SemanticChunkerBGE + PgVectorDB hiện có.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from app.core.config.admin_config import admin_cfg
from app.services.admin.header_normalizer import normalize_header
from app.services.admin.dedup_service import DedupService, compute_file_hash
from app.services.admin.task_store import TaskInfo, TaskStatus
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Ensure import path cho chunk_Process ──
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════
# DATABASE CONNECTION (tái sử dụng pattern từ retriever)
# ══════════════════════════════════════════════════════════
def _get_db_connection():
    """Tạo kết nối psycopg2 tới PostgreSQL."""
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "ufm_admission_db"),
        user=os.getenv("POSTGRES_USER", "ufm_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "ufm_secure_password_2026"),
    )


# ══════════════════════════════════════════════════════════
# EMBEDDING HELPER (tái sử dụng logic từ ingest_markdown)
# ══════════════════════════════════════════════════════════
_ing_cfg = admin_cfg.ingestion


def _generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Gọi OpenRouter Embedding API cho batch texts."""
    from chunk_Process.chunk_algorithms.utils import estimate_tokens

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    all_embeddings: list[list[float]] = []
    current_batch: list[str] = []
    current_tokens = 0
    max_tokens_per_batch = 6500

    def send_batch(batch: list[str]) -> list[list[float]]:
        url = f"{base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": _ing_cfg.embedding_model,
            "input": batch,
            "dimensions": _ing_cfg.embedding_dimensions,
        }

        for attempt in range(1, 4):
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                raw_data = sorted(result["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in raw_data]
            except urllib.error.HTTPError as e:
                if e.code in {429, 500, 502, 503} and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8")[:300]
                except Exception:
                    pass
                raise RuntimeError(f"Embedding API Error {e.code}: {error_body}")
            except (urllib.error.URLError, OSError) as e:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Network Error: {e}")
        return []

    for text in texts:
        tokens = estimate_tokens(text)
        if (current_tokens + tokens > max_tokens_per_batch
                or len(current_batch) >= _ing_cfg.embedding_batch_size) and current_batch:
            all_embeddings.extend(send_batch(current_batch))
            current_batch, current_tokens = [], 0
            time.sleep(0.5)

        current_batch.append(text)
        current_tokens += tokens

    if current_batch:
        all_embeddings.extend(send_batch(current_batch))

    return all_embeddings


# ══════════════════════════════════════════════════════════
# MAIN WORKER
# ══════════════════════════════════════════════════════════
def process_ingestion(
    file_name: str,
    file_content: str,
    task: TaskInfo,
    override_level: str = None,
    override_program: str = None,
    override_year: str = None,
    override_url: str = None,
) -> None:
    """
    Background worker — Xử lý 1 file Markdown từ đầu đến cuối.

    Args:
        file_name: Tên file gốc.
        file_content: Nội dung Markdown đầy đủ.
        task: TaskInfo object để cập nhật trạng thái.
        override_level: Bậc đào tạo do Admin chỉ định.
        override_program: Ngành do Admin chỉ định.
        override_year: Năm học do Admin chỉ định.
        override_url: Link gốc của tài liệu (tùy chọn).
    """
    conn = None
    start_time = time.time()

    try:
        # ── Bước 1: Normalize Header ──
        task.update(TaskStatus.VALIDATING, "Đang chuẩn hóa header...")
        meta, body = normalize_header(
            file_content, file_name,
            override_level=override_level,
            override_program=override_program,
            override_year=override_year,
        )
        logger.info("Worker - [%s] Header normalized: program='%s' level='%s'",
                     file_name, meta.get("program_name"), meta.get("program_level"))

        # ── Bước 2: Dedup Check ──
        file_hash = compute_file_hash(body)
        conn = _get_db_connection()
        conn.autocommit = False
        dedup = DedupService(conn)
        check = dedup.check_duplicate(file_hash, file_name)

        if check["action"] == "skip":
            task.update(TaskStatus.SKIPPED, check["reason"])
            logger.info("Worker - [%s] SKIPPED: %s", file_name, check["reason"])
            return

        if check["action"] == "update":
            logger.info("Worker - [%s] Updating: soft-delete old chunks", file_name)
            dedup.soft_delete_old_chunks(file_name)
            dedup.remove_old_log(file_name)

        # ── Bước 3: Chunking ──
        task.update(TaskStatus.CHUNKING, "Đang chia chunks (Hierarchical + Semantic)...")

        from chunk_Process.chunk_algorithms.hierarchical_chunker import HierarchicalChunker

        chunker = HierarchicalChunker()

        # Build metadata_extra từ header đã normalize
        chunk_meta = {}
        if meta.get("program_name"):
            chunk_meta["program_name"] = meta["program_name"]
        if meta.get("program_level"):
            chunk_meta["program_level"] = meta["program_level"]
        if meta.get("academic_year"):
            chunk_meta["academic_year"] = meta["academic_year"]
        if override_url:
            chunk_meta["url"] = override_url

        # Tạo lại nội dung với header chuẩn để chunker parse
        normalized_content = body
        if meta.get("program_name") or meta.get("program_level"):
            header_lines = ["---"]
            if meta.get("program_name"):
                header_lines.append(f'program_name: "{meta["program_name"]}"')
            if meta.get("program_level"):
                header_lines.append(f'program_level: "{meta["program_level"]}"')
            if meta.get("academic_year"):
                header_lines.append(f'academic_year: "{meta["academic_year"]}"')
            if override_url:
                header_lines.append(f'url: "{override_url}"')
            header_lines.append("---\n")
            normalized_content = "\n".join(header_lines) + body

        chunks = chunker.chunk(text=normalized_content, source=file_name, metadata_extra=chunk_meta)

        if not chunks:
            task.update(TaskStatus.FAILED, "Chunker trả về 0 chunks — file có thể quá ngắn hoặc sai format")
            dedup.record_ingestion(file_hash, file_name, "error", 0)
            return

        logger.info("Worker - [%s] Chunked: %d chunks (parent+child)", file_name, len(chunks))

        # ── Bước 4: Embedding ──
        task.update(TaskStatus.EMBEDDING, f"Đang embedding {len(chunks)} chunks...")

        # Chỉ embed child chunks (parent lấy embedding đại diện sau)
        child_texts = [c.content for c in chunks if c.metadata.chunk_level == "child"]
        parent_texts = [c.content for c in chunks if c.metadata.chunk_level == "parent"]
        all_texts = [c.content for c in chunks]

        embeddings = _generate_embeddings_batch(all_texts)

        logger.info("Worker - [%s] Embedded: %d vectors", file_name, len(embeddings))

        # ── Bước 5: Insert DB (batch) ──
        task.update(TaskStatus.INSERTING, f"Đang ghi {len(chunks)} chunks vào DB...")

        from ingestion.ingest_markdown import PgVectorDB

        db = PgVectorDB(config={
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "dbname": os.getenv("POSTGRES_DB", "ufm_admission_db"),
            "user": os.getenv("POSTGRES_USER", "ufm_admin"),
            "password": os.getenv("POSTGRES_PASSWORD", "ufm_secure_password_2026"),
        })
        db.conn = conn  # Tái sử dụng connection hiện có

        inserted = db.insert_chunks_batch(chunks, embeddings)

        # ── Bước 6: Ghi log ──
        dedup.record_ingestion(file_hash, file_name, "completed", inserted)

        elapsed = time.time() - start_time
        task.update(
            TaskStatus.COMPLETED,
            f"Hoàn tất! {inserted}/{len(chunks)} chunks đã nạp trong {elapsed:.1f}s",
            chunks_count=inserted,
        )
        logger.info(
            "Worker - [%s] COMPLETED: %d chunks inserted in %.1fs",
            file_name, inserted, elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Lỗi sau {elapsed:.1f}s: {str(e)}"
        task.update(TaskStatus.FAILED, error_msg, error=str(e))
        logger.error("Worker - [%s] FAILED: %s", file_name, e, exc_info=True)

        # Log lỗi vào DB nếu có connection
        if conn:
            try:
                dedup = DedupService(conn)
                dedup.record_ingestion(
                    compute_file_hash(file_content), file_name, "error", 0
                )
            except Exception:
                pass

    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
