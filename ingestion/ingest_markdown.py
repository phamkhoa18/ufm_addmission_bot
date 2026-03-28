"""
Ingestion Pipeline — Hierarchical + Semantic Chunking → pgvector.

Script này thực hiện toàn bộ pipeline:
  1. Quét thư mục data/unstructured/markdown/ → đọc file .md
  2. HierarchicalChunker tách Parent Chunks theo heading
  3. SemanticChunkerBGE tạo Child Chunks + Embedding (BGE-M3 via OpenRouter)
  4. Insert vào PostgreSQL/pgvector với đầy đủ metadata

Sử dụng:
  # Dry-run (chỉ preview, không insert DB)
  python ingestion/ingest_markdown.py --dry-run

  # Chạy thật — insert vào database
  python ingestion/ingest_markdown.py

  # Chỉ xử lý 1 file cụ thể
  python ingestion/ingest_markdown.py --file data/unstructured/markdown/thongtinchung/phuluc1.md

  # Rebuild (xóa hết data cũ trước)
  python ingestion/ingest_markdown.py --rebuild

  # Fallback mode (không gọi Embedding API)
  python ingestion/ingest_markdown.py --fallback
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ── Đảm bảo import path ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from chunk_Process.chunk_algorithms.semantic_chunker import SemanticChunkerBGE
from chunk_Process.chunk_algorithms.hierarchical_chunker import HierarchicalChunker
from models.chunk import ProcessedChunk

# ================================================================
# CẤU HÌNH
# ================================================================
# Thư mục chứa Markdown files (CHỈ thông tin tuyển sinh, KHÔNG mẫu đơn)
# Mẫu đơn (maudon/) được xử lý bởi FormAgent, không đưa vào VectorDB
MARKDOWN_DIRS = [
    PROJECT_ROOT / "data" / "unstructured" / "markdown" / "thongtinchung",
]

# Database
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "ufm_admission_db"),
    "user": os.getenv("POSTGRES_USER", "ufm_admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "ufm_secure_password_2026"),
}

# Embedding
EMBEDDING_MODEL = "baai/bge-m3"
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_BATCH_SIZE = 30        # Số chunks/batch gửi API
EMBEDDING_MAX_TOKENS = 6500      # Max tokens/API call

# Output
DRY_RUN_OUTPUT = PROJECT_ROOT / "data" / "unstructured" / "processed" / "chunks_preview.json"


# ================================================================
# DATABASE HELPER — psycopg2 (sync, đơn giản)
# ================================================================
class PgVectorDB:
    """Kết nối PostgreSQL + pgvector, insert/query chunks."""

    def __init__(self, config: dict):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError(
                "psycopg2 chưa được cài. Chạy: pip install psycopg2-binary"
            )
        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        self.config = config
        self.conn = None

    def connect(self):
        """Mở kết nối tới PostgreSQL."""
        self.conn = self._psycopg2.connect(**self.config)
        self.conn.autocommit = False
        print(f"  ✅ Kết nối DB thành công: {self.config['host']}:{self.config['port']}/{self.config['dbname']}")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def clear_all_chunks(self):
        """Xóa toàn bộ knowledge_chunks (dùng khi --rebuild)."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_chunks;")
        self.conn.commit()
        print("  🗑️  Đã xóa toàn bộ knowledge_chunks")

    def load_existing_hashes(self) -> set[str]:
        """
        Load TẤT CẢ content_hash hiện có vào RAM (1 query duy nhất).

        Trả về set các key dạng "<hash>|<source>|<version>".
        Dùng để filter trước khi gọi Embedding API → tiết kiệm tiền.

        Performance:
          - 10,000 chunks → ~50ms, ~500KB RAM
          - 100,000 chunks → ~200ms, ~5MB RAM (vẫn OK)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT content_hash, source, version FROM knowledge_chunks "
                "WHERE content_hash IS NOT NULL"
            )
            rows = cur.fetchall()

        hash_set = set()
        for row in rows:
            # Key format: "<hash>|<source>|<version>"
            hash_set.add(f"{row[0]}|{row[1]}|{row[2]}")

        return hash_set

    def insert_chunks_batch(
        self,
        chunks: list[ProcessedChunk],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """
        Batch INSERT chunks vào knowledge_chunks.

        Args:
            chunks: Danh sách ProcessedChunk
            embeddings: List vectors tương ứng (None nếu fallback)

        Returns:
            Số lượng chunks đã insert (bỏ qua duplicate)
        """
        if not chunks:
            return 0

        inserted = 0

        insert_sql = """
            INSERT INTO knowledge_chunks (
                chunk_id, source, section_path, section_name,
                program_name, program_level, ma_nganh,
                chunk_index, total_chunks_in_section,
                content, char_count, embedding,
                chunk_level, parent_id, children_ids, overlap_tokens,
                academic_year, valid_from, valid_until,
                is_active, version, replaced_by,
                content_hash, token_count, extra
            ) VALUES (
                %(chunk_id)s, %(source)s, %(section_path)s, %(section_name)s,
                %(program_name)s, %(program_level)s, %(ma_nganh)s,
                %(chunk_index)s, %(total_chunks_in_section)s,
                %(content)s, %(char_count)s, %(embedding)s,
                %(chunk_level)s, %(parent_id)s, %(children_ids)s, %(overlap_tokens)s,
                %(academic_year)s, %(valid_from)s, %(valid_until)s,
                %(is_active)s, %(version)s, %(replaced_by)s,
                %(content_hash)s, %(token_count)s, %(extra)s
            )
            ON CONFLICT (content_hash, source, version)
            WHERE content_hash IS NOT NULL
            DO NOTHING
        """

        with self.conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                meta = chunk.metadata

                # Chuẩn bị embedding vector string cho pgvector
                emb_str = None
                if embeddings and i < len(embeddings) and embeddings[i]:
                    emb_str = "[" + ",".join(str(v) for v in embeddings[i]) + "]"

                # Chuẩn bị params
                params = {
                    "chunk_id": meta.chunk_id,
                    "source": meta.source,
                    "section_path": meta.section_path,
                    "section_name": meta.section_name,
                    "program_name": meta.program_name,
                    "program_level": meta.program_level,
                    "ma_nganh": meta.ma_nganh,
                    "chunk_index": meta.chunk_index,
                    "total_chunks_in_section": meta.total_chunks_in_section,
                    "content": chunk.content,
                    "char_count": chunk.char_count,
                    "embedding": emb_str,
                    "chunk_level": meta.chunk_level,
                    "parent_id": meta.parent_id,
                    "children_ids": meta.children_ids if meta.children_ids else None,
                    "overlap_tokens": meta.overlap_tokens,
                    "academic_year": meta.academic_year,
                    "valid_from": meta.valid_from,
                    "valid_until": meta.valid_until,
                    "is_active": meta.is_active,
                    "version": meta.version,
                    "replaced_by": meta.replaced_by,
                    "content_hash": meta.content_hash,
                    "token_count": meta.token_count,
                    "extra": json.dumps(meta.extra, ensure_ascii=False) if meta.extra else "{}",
                }

                try:
                    # SAVEPOINT: Chi rollback 1 chunk loi, giu nguyen cac chunk khac
                    cur.execute("SAVEPOINT chunk_insert")
                    cur.execute(insert_sql, params)
                    cur.execute("RELEASE SAVEPOINT chunk_insert")
                    inserted += 1
                except Exception as e:
                    # Rollback CHI chunk nay, KHONG rollback toan bo batch
                    cur.execute("ROLLBACK TO SAVEPOINT chunk_insert")
                    print(f"    ⚠️  Lỗi insert chunk {meta.chunk_id[:8]}...: {e}")
                    continue

        self.conn.commit()
        return inserted


# ================================================================
# EMBEDDING HELPER — Gọi OpenRouter cho BGE-M3
# ================================================================
def generate_embeddings_batch(
    texts: list[str],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
    max_retries: int = 3,
) -> list[list[float]]:
    """
    Sinh embedding cho danh sách texts qua OpenRouter.

    Tự động chia nhỏ batch nếu vượt token limit.
    Retry với exponential backoff khi gặp lỗi tạm thời.
    """
    from chunk_Process.chunk_algorithms.utils import estimate_tokens

    all_embeddings: list[list[float]] = []
    current_batch: list[str] = []
    current_tokens = 0

    def send_batch(batch: list[str]) -> list[list[float]]:
        url = f"{base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UFM-Ingestion/1.0",
        }
        data = {
            "model": model,
            "input": batch,
            "dimensions": dimensions,
        }

        for attempt in range(1, max_retries + 1):
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
                if e.code in {429, 500, 502, 503} and attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"    ⏳ API {e.code}, retry {attempt}/{max_retries} sau {wait}s...")
                    time.sleep(wait)
                    continue
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8")[:300]
                except Exception:
                    pass
                raise RuntimeError(f"Embedding API Error {e.code}: {error_body}") from e

            except (urllib.error.URLError, OSError) as e:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"    ⏳ Network error, retry {attempt}/{max_retries} sau {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Network Error: {e}") from e

        return []  # Không bao giờ tới đây

    # Chia texts thành batches theo token limit
    for text in texts:
        tokens = estimate_tokens(text)

        if (current_tokens + tokens > EMBEDDING_MAX_TOKENS
                or len(current_batch) >= EMBEDDING_BATCH_SIZE) and current_batch:
            embeddings = send_batch(current_batch)
            all_embeddings.extend(embeddings)
            current_batch = []
            current_tokens = 0

            # Rate limit courtesy: nghỉ 500ms giữa các batch tránh Rate Limit (HTTP 429)
            time.sleep(0.5)

        current_batch.append(text)
        current_tokens += tokens

    if current_batch:
        embeddings = send_batch(current_batch)
        all_embeddings.extend(embeddings)

    return all_embeddings


# ================================================================
# LUỒNG CHÍNH
# ================================================================
def collect_markdown_files(
    dirs: list[Path] = None,
    single_file: str = None,
) -> list[Path]:
    """Thu thập tất cả file .md cần xử lý."""
    if single_file:
        p = Path(single_file)
        if not p.exists():
            raise FileNotFoundError(f"File không tồn tại: {single_file}")
        return [p]

    files = []
    for d in (dirs or MARKDOWN_DIRS):
        if d.exists():
            files.extend(sorted(d.rglob("*.md")))

    # Bỏ file temp
    files = [f for f in files if not f.name.startswith("~$")]
    return files


def run_ingestion(
    dry_run: bool = False,
    rebuild: bool = False,
    single_file: str = None,
    use_fallback: bool = False,
):
    """
    Pipeline chính: File .md → Hierarchical+Semantic Chunking → Embedding → pgvector.
    """
    start_time = time.time()

    print("=" * 70)
    print("  UFM ADMISSION BOT — Ingestion Pipeline")
    print(f"  Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'DRY-RUN' if dry_run else 'PRODUCTION'}")
    if use_fallback:
        print("  ⚠️  FALLBACK MODE — Không gọi Embedding API")
    print("=" * 70)

    # 1. Thu thập files
    files = collect_markdown_files(single_file=single_file)
    print(f"\n📂 Tìm thấy {len(files)} file Markdown:")
    for f in files:
        print(f"   • {f.name} ({f.stat().st_size:,} bytes)")

    if not files:
        print("❌ Không tìm thấy file nào để xử lý!")
        return

    # 2. Khởi tạo Chunkers
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    hierarchical = HierarchicalChunker()
    semantic = SemanticChunkerBGE(api_key=api_key, base_url=base_url)

    print(f"\n🔧 Chunker Config:")
    print(f"   Hierarchical: max_parent={hierarchical.cfg['max_parent_tokens']} tokens, merge_threshold={hierarchical.cfg['merge_threshold_tokens']} tokens")
    print(f"   Semantic:     similarity_threshold={semantic.cfg['similarity_threshold']}, overlap={semantic.cfg['overlap_tokens']} tokens")
    print(f"   Semantic:     base_block={semantic.cfg['base_block_tokens']} tokens, max_chunk={semantic.cfg['max_chunk_tokens']} tokens")
    print(f"   Embedding:    {semantic.cfg['model']} ({semantic.cfg['dimensions']}D)")

    # 3. Chunking — Xử lý từng file
    all_chunks: list[ProcessedChunk] = []
    file_stats = []

    for file_path in files:
        print(f"\n{'─' * 50}")
        print(f"📄 Xử lý: {file_path.name}")

        try:
            chunks = hierarchical.chunk_file(
                filepath=str(file_path),
                semantic_chunker=semantic,
                use_fallback=use_fallback,
            )

            parents = [c for c in chunks if c.metadata.chunk_level == "parent"]
            children = [c for c in chunks if c.metadata.chunk_level == "child"]

            print(f"   ✅ {len(chunks)} chunks tổng ({len(parents)} parents, {len(children)} children)")

            # Thống kê token
            total_tokens = sum(c.metadata.token_count for c in chunks)
            avg_tokens = total_tokens // max(1, len(chunks))
            print(f"   📊 Tổng ~{total_tokens:,} tokens, trung bình ~{avg_tokens}/chunk")

            all_chunks.extend(chunks)
            file_stats.append({
                "file": file_path.name,
                "parents": len(parents),
                "children": len(children),
                "total_tokens": total_tokens,
            })

        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'═' * 70}")
    print(f"📊 TỔNG KẾT CHUNKING:")
    print(f"   Files xử lý:    {len(file_stats)}/{len(files)}")
    print(f"   Tổng chunks:    {len(all_chunks)}")

    total_parents = sum(1 for c in all_chunks if c.metadata.chunk_level == "parent")
    total_children = sum(1 for c in all_chunks if c.metadata.chunk_level == "child")
    print(f"   Parents:        {total_parents}")
    print(f"   Children:       {total_children}")

    if semantic.stats["total_api_calls"] > 0:
        print(f"\n   🌐 Embedding API Stats:")
        print(f"      API calls:     {semantic.stats['total_api_calls']}")
        print(f"      Tokens sent:   {semantic.stats['total_tokens_sent']:,}")
        print(f"      Time:          {semantic.stats['total_time_embedding']:.1f}s")

    # 4. Dry-run: Lưu preview ra file JSON
    if dry_run:
        DRY_RUN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        preview = []
        for c in all_chunks:
            preview.append({
                "chunk_id": c.metadata.chunk_id[:12] + "...",
                "chunk_level": c.metadata.chunk_level,
                "parent_id": (c.metadata.parent_id[:12] + "...") if c.metadata.parent_id else None,
                "children_count": len(c.metadata.children_ids),
                "section_path": c.metadata.section_path,
                "section_name": c.metadata.section_name,
                "source": c.metadata.source,
                "program_level": c.metadata.program_level,
                "token_count": c.metadata.token_count,
                "char_count": c.char_count,
                "overlap_tokens": c.metadata.overlap_tokens,
                "content_preview": c.content[:300] + "..." if len(c.content) > 300 else c.content,
            })

        with open(DRY_RUN_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n📄 Dry-run output saved: {DRY_RUN_OUTPUT}")
        elapsed = time.time() - start_time
        print(f"⏱️  Thời gian: {elapsed:.1f}s")
        return

    # ================================================================
    # 5. DEDUP — Load hash cache từ DB, lọc chunks mới
    # ================================================================
    print(f"\n🔍 Kiểm tra trùng lặp với Database...")
    db = PgVectorDB(DB_CONFIG)
    inserted = 0
    try:
        db.connect()

        if rebuild:
            db.clear_all_chunks()
            existing_hashes: set[str] = set()
            print("   🗑️  Rebuild mode → bỏ qua dedup")
        else:
            # Load toàn bộ hash từ DB vào RAM (1 query duy nhất)
            existing_hashes = db.load_existing_hashes()
            print(f"   📦 Hash cache loaded: {len(existing_hashes):,} chunks đang có trong DB")

        # Lọc ra CHỈ chunks mới (chưa có hash trong DB)
        new_chunks: list[ProcessedChunk] = []
        skipped_chunks: list[ProcessedChunk] = []

        for chunk in all_chunks:
            meta = chunk.metadata
            dedup_key = f"{meta.content_hash}|{meta.source}|{meta.version}"

            if meta.content_hash and dedup_key in existing_hashes:
                skipped_chunks.append(chunk)
            else:
                new_chunks.append(chunk)

        print(f"   ✅ {len(new_chunks)} chunks MỚI cần xử lý")
        if skipped_chunks:
            print(f"   ⏭️  {len(skipped_chunks)} chunks đã tồn tại → BỎ QUA (tiết kiệm Embedding API)")

        # Nếu không có chunks mới → kết thúc sớm
        if not new_chunks:
            print(f"\n{'═' * 70}")
            print(f"✅ KHÔNG CÓ CHUNKS MỚI — Database đã lên mới nhất!")
            elapsed = time.time() - start_time
            print(f"   Thời gian: {elapsed:.1f}s")
            print(f"{'═' * 70}")
            db.close()
            return

        # ================================================================
        # 6. Sinh Embedding CHỈ cho chunks MỚI (tiết kiệm API calls)
        # ================================================================
        new_parents = sum(1 for c in new_chunks if c.metadata.chunk_level == "parent")
        new_children = sum(1 for c in new_chunks if c.metadata.chunk_level == "child")
        print(f"\n🧠 Sinh Embedding cho {len(new_chunks)} chunks MỚI ({new_parents} parents, {new_children} children)...")

        embeddings: list[list[float]] | None = None
        if not use_fallback:
            try:
                texts = [c.content for c in new_chunks]
                embed_start = time.time()
                embeddings = generate_embeddings_batch(
                    texts=texts,
                    api_key=api_key,
                    base_url=base_url,
                )
                embed_time = time.time() - embed_start
                print(f"   ✅ Embedding hoàn tất: {len(embeddings)} vectors, {embed_time:.1f}s")
            except Exception as e:
                print(f"   ❌ Embedding thất bại: {e}")
                print(f"   ⚠️  Chuyển sang insert KHÔNG có embedding (sẽ cần reprocess sau)")
                embeddings = None

        # ================================================================
        # 7. Insert CHỈ chunks MỚI vào PostgreSQL/pgvector
        # ================================================================
        print(f"\n💾 Insert {len(new_chunks)} chunks mới vào PostgreSQL...")
        inserted = db.insert_chunks_batch(new_chunks, embeddings)
        print(f"   ✅ Đã insert {inserted}/{len(new_chunks)} chunks vào knowledge_chunks")

    except Exception as e:
        print(f"   ❌ Database error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

    # 8. Tổng kết
    elapsed = time.time() - start_time
    print(f"\n{'═' * 70}")
    print(f"✅ INGESTION HOÀN TẤT")
    print(f"   Thời gian tổng:    {elapsed:.1f}s")
    print(f"   Files:             {len(file_stats)}")
    print(f"   Chunks tổng:       {len(all_chunks)}")
    print(f"   Chunks mới insert: {inserted}")
    print(f"   Chunks bỏ qua:    {len(skipped_chunks) if 'skipped_chunks' in locals() else 0} (đã có trong DB)")
    print(f"{'═' * 70}")


# ================================================================
# CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="UFM Admission Bot — Markdown Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python ingestion/ingest_markdown.py --dry-run
  python ingestion/ingest_markdown.py
  python ingestion/ingest_markdown.py --file data/unstructured/markdown/thongtinchung/phuluc1.md
  python ingestion/ingest_markdown.py --rebuild
  python ingestion/ingest_markdown.py --fallback
        """,
    )
    parser.add_argument("--dry-run", action="store_true", help="Chỉ preview chunks ra JSON, không insert DB")
    parser.add_argument("--rebuild", action="store_true", help="Xóa hết data cũ rồi insert mới")
    parser.add_argument("--file", type=str, help="Xử lý 1 file .md cụ thể")
    parser.add_argument("--fallback", action="store_true", help="Không gọi Embedding API (cắt tĩnh)")

    args = parser.parse_args()

    run_ingestion(
        dry_run=args.dry_run,
        rebuild=args.rebuild,
        single_file=args.file,
        use_fallback=args.fallback,
    )


if __name__ == "__main__":
    main()
