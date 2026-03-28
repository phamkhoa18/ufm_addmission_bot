"""
Ingestion cho Structured Text — Chương trình Thạc sĩ & Tiến sĩ.

Đặc thù dữ liệu:
  - Các file .txt với cấu trúc CỐ ĐỊNH: Tiêu đề → Giới thiệu → ĐIỀU KIỆN → CƠ HỘI → ƯU ĐÃI
  - KHÔNG dùng HierarchicalChunker (vì không có Markdown heading level)
  - Thay vào đó: Regex-based Section Splitter + Context Injection

Chiến lược Context Injection:
  Mỗi chunk (Parent & Child) đều được BƠM tên chương trình ở đầu:
    "[Chương trình: Thạc sĩ Kinh doanh quốc tế]"
  → Embeddings hiểu NGAY chunk đang thuộc ngành nào, tránh nhầm lẫn giữa
    "Điều kiện xét tuyển" của KDQT vs TC-NH vs Marketing...

Sử dụng:
  python ingestion/ingest_structured.py --dry-run
  python ingestion/ingest_structured.py
  python ingestion/ingest_structured.py --file "data/structured/processed/Ctrinh Thac Si/ThS KDQT.txt"
  python ingestion/ingest_structured.py --rebuild
  python ingestion/ingest_structured.py --fallback
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from models.chunk import ProcessedChunk, ChunkMetadata

# ================================================================
# CẤU HÌNH
# ================================================================
STRUCTURED_DIRS = [
    PROJECT_ROOT / "data" / "structured" / "processed" / "Ctrinh Thac Si",
    PROJECT_ROOT / "data" / "structured" / "processed" / "Ctrinh Tien Si",
]

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "ufm_admission_db"),
    "user": os.getenv("POSTGRES_USER", "ufm_admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "ufm_secure_password_2026"),
}

EMBEDDING_MODEL = "baai/bge-m3"
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_BATCH_SIZE = 30
EMBEDDING_MAX_TOKENS = 6500

DRY_RUN_OUTPUT = PROJECT_ROOT / "data" / "structured" / "processed" / "chunks_preview.json"

# Mapping folder → program_level
LEVEL_MAP = {
    "Ctrinh Thac Si": "thac_si",
    "Ctrinh Tien Si": "tien_si",
}

# ================================================================
# SECTION HEADERS — Regex patterns cho tất cả dạng heading có thể gặp
# ================================================================
# Các dạng: **ĐIỀU KIỆN XÉT TUYỂN**, ĐIỀU KIỆN XÉT TUYỂN, CHÍNH SÁCH ƯU ĐÃI...
# Lưu ý: \r? xử lý cả CRLF (Windows) và LF (Unix)
SECTION_PATTERN = re.compile(
    r"^(?:\*\*)?("
    r"CHƯƠNG TRÌNH ĐÀO TẠO[^*\r\n\.,\"]*"   # Cấm dấu chấm, phẩy, ngoặc kép để tránh match nhầm vào đoạn văn Intro
    r"|GIỚI THIỆU CHƯƠNG TRÌNH[^*\r\n\.,\"]*"
    r"|ĐIỀU KIỆN XÉT TUYỂN"
    r"|CƠ HỘI NGHỀ NGHIỆP"
    r"|CHÍNH SÁCH ƯU ĐÃI?\s*HỌC PHÍ"
    r"|MỤC TIÊU ĐÀO TẠO"
    r"|CHUẨN ĐẦU RA"
    r"|PHƯƠNG THỨC TUYỂN SINH"
    r")(?:\*\*)?\s*\r?$",
    re.MULTILINE | re.IGNORECASE,
)


# ================================================================
# SECTION SPLITTER — Tách file thành sections
# ================================================================
def split_sections(text: str) -> list[dict]:
    """
    Tách file .txt thành danh sách sections dựa trên heading pattern.

    Returns:
        List[dict] với keys: 'heading', 'body', 'is_intro'

    Ví dụ output cho NCS QLKT.txt:
      [
        { "heading": "CHƯƠNG TRÌNH ĐÀO TẠO TIẾN SĨ NGÀNH QUẢN LÝ KINH TẾ",
          "body": "Chương trình đào tạo...", "is_intro": True },
        { "heading": "ĐIỀU KIỆN XÉT TUYỂN",
          "body": "Ứng viên dự tuyển...", "is_intro": False },
        ...
      ]
    """
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        # Không tìm thấy heading → toàn bộ file là 1 section
        return [{"heading": "(Toàn văn)", "body": text.strip(), "is_intro": False}]

    sections = []

    for i, match in enumerate(matches):
        heading = match.group(1).strip().strip("*")

        # Lấy body = text từ cuối heading đến heading tiếp theo
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Section đầu tiên chứa tên chương trình → đánh dấu is_intro
        is_intro = (i == 0 and (
            "CHƯƠNG TRÌNH ĐÀO TẠO" in heading.upper()
            or "GIỚI THIỆU CHƯƠNG TRÌNH" in heading.upper()
        ))

        if body:
            sections.append({
                "heading": heading,
                "body": body,
                "is_intro": is_intro,
            })

    return sections


def extract_program_name(heading: str) -> str:
    """
    Trích xuất tên ngành từ heading.

    "CHƯƠNG TRÌNH ĐÀO TẠO TIẾN SĨ NGÀNH QUẢN LÝ KINH TẾ"
      → "Quản lý kinh tế"
    "GIỚI THIỆU CHƯƠNG TRÌNH NGÀNH TÀI CHÍNH – NGÂN HÀNG"
      → "Tài chính – Ngân hàng"
    """
    # Pattern: "NGÀNH <tên>" ở cuối. [^(]+ giúp bỏ qua các phần trong ngoặc như "(Định hướng ứng dụng)"
    m = re.search(r"NGÀNH\s+([^(]+)", heading, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().strip("*").strip()
        # Title case chuẩn tiếng Việt
        return raw.title() if raw == raw.upper() else raw

    # Fallback: Sau "CHƯƠNG TRÌNH" cắt bỏ "ĐÀO TẠO TIẾN SĨ/THẠC SĨ"
    cleaned = re.sub(
        r"(CHƯƠNG TRÌNH|ĐÀO TẠO|TIẾN SĨ|THẠC SĨ|GIỚI THIỆU)\s*",
        "", heading, flags=re.IGNORECASE,
    ).strip()
    return cleaned.title() if cleaned else heading


def detect_program_level(file_path: Path) -> str:
    """Phát hiện trình độ từ đường dẫn file."""
    for folder_name, level in LEVEL_MAP.items():
        if folder_name in str(file_path):
            return level
    return "unknown"


def lookup_ma_nganh(program_level: str, program_name: str) -> Optional[str]:
    """
    Tra mã ngành từ admissions_mapping.json.

    Tự động chuẩn hóa:
      - Em dash (–) → Hyphen (-)
      - Khoảng trắng thừa
      - Lowercase toàn bộ
    """
    json_path = PROJECT_ROOT / "app" / "core" / "config" / "yaml" / "admissions_mapping.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        ma_nganh_map = data.get("ma_nganh", {})

        # Chuẩn hóa: em dash → hyphen, strip, lowercase
        normalized_name = program_name.lower().strip()
        normalized_name = normalized_name.replace("\u2013", "-")  # – → -
        normalized_name = normalized_name.replace("\u2014", "-")  # — → -
        normalized_name = re.sub(r"\s+", " ", normalized_name)   # multi-space → single

        key = f"{program_level}|{normalized_name}"
        return ma_nganh_map.get(key)
    except Exception:
        return None


def lookup_viet_tat(program_name: str) -> Optional[str]:
    """
    Tra viết tắt của ngành từ admissions_mapping.json.

    Ví dụ:
      "Kinh Doanh Quốc Tế"  → "KDQT"
      "Quản Trị Kinh Doanh" → "QTKD"
      "Tài Chính – Ngân Hàng" → "TC-NH, TCNH"
    """
    json_path = PROJECT_ROOT / "app" / "core" / "config" / "yaml" / "admissions_mapping.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        viet_tat_map = data.get("viet_tat", {})

        # Chuẩn hóa tên ngành giống lookup_ma_nganh
        normalized = program_name.lower().strip()
        normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
        normalized = re.sub(r"\s+", " ", normalized)

        return viet_tat_map.get(normalized)
    except Exception:
        return None


# ================================================================
# CHUNKING: TẠO PARENT + CHILD CHUNKS VỚI CONTEXT INJECTION
# ================================================================
def chunk_structured_file(
    file_path: Path,
    use_fallback: bool = False,
) -> list[ProcessedChunk]:
    """
    Tách 1 file .txt thành Parent + Child chunks.

    Chiến lược:
      1. Split file thành sections (heading-based)
      2. Section đầu (intro + giới thiệu): → 1 PARENT chunk
      3. Mỗi section khác: → 1 PARENT chunk
      4. Mỗi PARENT: nếu > 300 tokens → tách thành CHILDREN (paragraph-based)
      5. MỌI chunk đều được bơm context prefix: "[Chương trình: Thạc sĩ XXX]"

    Returns:
        List[ProcessedChunk] — parents + children
    """
    text = file_path.read_text(encoding="utf-8-sig").strip()
    # Chuẩn hóa CRLF → LF (Windows → Unix) để regex $ hoạt động chính xác
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return []

    source = file_path.name
    program_level = detect_program_level(file_path)

    # Tách sections
    sections = split_sections(text)
    if not sections:
        return []

    # Trích program_name từ heading đầu tiên
    program_name = extract_program_name(sections[0]["heading"])
    ma_nganh = lookup_ma_nganh(program_level, program_name)
    viet_tat = lookup_viet_tat(program_name)

    # Context prefix sẽ gắn vào ĐẦU mọi chunk
    # Bao gồm: TÊN CHƯƠNG TRÌNH + VIẾT TẮT + MÃ NGÀNH
    # Ví dụ: [Chương trình: Thạc sĩ Kinh Doanh Quốc Tế (KDQT) | Mã ngành: 8340120]
    level_label = {
        "thac_si": "Thạc sĩ",
        "tien_si": "Tiến sĩ",
        "dai_hoc": "Đại học",
    }.get(program_level, program_level)

    # Build context prefix từng phần
    name_part = f"{level_label} {program_name}"
    if viet_tat:
        name_part += f" ({viet_tat})"

    if ma_nganh:
        context_prefix = f"[Chương trình: {name_part} | Mã ngành: {ma_nganh}]"
    else:
        context_prefix = f"[Chương trình: {name_part}]"

    all_chunks: list[ProcessedChunk] = []

    for sec_idx, section in enumerate(sections):
        heading = section["heading"]
        body = section["body"]

        # ── Section path để trace ──
        section_path = f"{level_label} {program_name} > {heading}"

        # ════════════════════════════════════════
        # TẠO PARENT CHUNK (toàn bộ section)
        # ════════════════════════════════════════
        parent_id = str(uuid.uuid4())

        # Bơm context: "[Chương trình: Thạc sĩ KDQT]\nĐIỀU KIỆN XÉT TUYỂN\n..."
        if section["is_intro"]:
            parent_content = f"{context_prefix}\n{heading}\n\n{body}"
        else:
            parent_content = f"{context_prefix}\n{heading}\n\n{body}"

        parent_hash = hashlib.sha256(parent_content.encode("utf-8")).hexdigest()

        parent_meta = ChunkMetadata(
            chunk_id=parent_id,
            source=source,
            section_path=section_path,
            section_name=heading,
            program_name=program_name,
            program_level=program_level,
            ma_nganh=ma_nganh,
            chunk_index=sec_idx + 1,
            total_chunks_in_section=len(sections),
            chunk_level="parent",
            parent_id=None,
            children_ids=[],
            overlap_tokens=0,
            academic_year="2026",
            is_active=True,
            content_hash=parent_hash,
        )

        parent_chunk = ProcessedChunk(
            content=parent_content,
            metadata=parent_meta,
        )

        # ════════════════════════════════════════
        # TẠO CHILD CHUNKS (paragraph-level split)
        # ════════════════════════════════════════
        # Ước tính tokens
        est_tokens = len(body) // 3  # ~3 chars/token cho tiếng Việt

        children_chunks: list[ProcessedChunk] = []

        if est_tokens > 300:
            # Section lớn → tách thành children theo paragraph
            paragraphs = _split_paragraphs(body)
            child_ids = []

            for p_idx, para in enumerate(paragraphs):
                if not para.strip():
                    continue

                child_id = str(uuid.uuid4())
                child_ids.append(child_id)

                # ★ Context injection cho CHILD: luôn gắn tên chương trình + heading
                child_content = f"{context_prefix} — {heading}\n\n{para}"

                child_hash = hashlib.sha256(child_content.encode("utf-8")).hexdigest()

                child_meta = ChunkMetadata(
                    chunk_id=child_id,
                    source=source,
                    section_path=section_path,
                    section_name=heading,
                    program_name=program_name,
                    program_level=program_level,
                    ma_nganh=ma_nganh,
                    chunk_index=p_idx + 1,
                    total_chunks_in_section=len(paragraphs),
                    chunk_level="child",
                    parent_id=parent_id,
                    children_ids=[],
                    overlap_tokens=0,
                    academic_year="2026",
                    is_active=True,
                    content_hash=child_hash,
                )

                children_chunks.append(ProcessedChunk(
                    content=child_content,
                    metadata=child_meta,
                ))

            # Gắn children_ids vào parent
            parent_chunk.metadata.children_ids = child_ids
        else:
            # Section nhỏ (< 300 tokens) → parent tự đóng vai child luôn
            # Tạo 1 child duy nhất = bản copy nội dung parent
            child_id = str(uuid.uuid4())
            child_content = f"{context_prefix} — {heading}\n\n{body}"
            child_hash = hashlib.sha256(child_content.encode("utf-8")).hexdigest()

            child_meta = ChunkMetadata(
                chunk_id=child_id,
                source=source,
                section_path=section_path,
                section_name=heading,
                program_name=program_name,
                program_level=program_level,
                ma_nganh=ma_nganh,
                chunk_index=1,
                total_chunks_in_section=1,
                chunk_level="child",
                parent_id=parent_id,
                children_ids=[],
                overlap_tokens=0,
                academic_year="2026",
                is_active=True,
                content_hash=child_hash,
            )

            children_chunks.append(ProcessedChunk(
                content=child_content,
                metadata=child_meta,
            ))
            parent_chunk.metadata.children_ids = [child_id]

        all_chunks.append(parent_chunk)
        all_chunks.extend(children_chunks)

    return all_chunks


def _split_paragraphs(text: str) -> list[str]:
    """
    Tách body thành các paragraphs ngữ nghĩa.

    Rules:
      - Dòng trống = ranh giới paragraph
      - Bullet list (bắt đầu bằng * hoặc -) gộp với dòng giới thiệu trước nó
      - Sub-bullets (bắt đầu bằng +) gộp vào bullet cha
    """
    lines = text.split("\n")
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Dòng trống → flush paragraph hiện tại
            if current:
                paragraphs.append("\n".join(current))
                current = []
            continue

        # Bullet chính (* hoặc -) → nếu đang có paragraph text, gộp
        if stripped.startswith(("* ", "- ")):
            current.append(line)
            continue

        # Sub-bullet (+ ) → luôn gộp vào paragraph hiện tại
        if stripped.startswith("+ ") or stripped.startswith("  +"):
            current.append(line)
            continue

        # Text bình thường
        current.append(line)

    if current:
        paragraphs.append("\n".join(current))

    # Gộp paragraph quá nhỏ (< 50 chars) vào paragraph kế tiếp
    merged: list[str] = []
    for para in paragraphs:
        if merged and len(merged[-1]) < 50:
            merged[-1] = merged[-1] + "\n\n" + para
        else:
            merged.append(para)

    return merged


# ================================================================
# EMBEDDING (tái sử dụng từ ingest_markdown.py)
# ================================================================
def generate_embeddings_batch(
    texts: list[str],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
    max_retries: int = 3,
) -> list[list[float]]:
    """Sinh embedding qua OpenRouter. Batch + Rate Limit + Retry."""
    from chunk_Process.chunk_algorithms.utils import estimate_tokens

    all_embeddings: list[list[float]] = []
    current_batch: list[str] = []
    current_tokens = 0

    def send_batch(batch: list[str]) -> list[list[float]]:
        url = f"{base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UFM-Ingestion-Structured/1.0",
        }
        data = {"model": model, "input": batch, "dimensions": dimensions}

        for attempt in range(1, max_retries + 1):
            req = urllib.request.Request(
                url, data=json.dumps(data).encode("utf-8"),
                headers=headers, method="POST",
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
        return []

    for text in texts:
        tokens = estimate_tokens(text)
        if (current_tokens + tokens > EMBEDDING_MAX_TOKENS
                or len(current_batch) >= EMBEDDING_BATCH_SIZE) and current_batch:
            embeddings = send_batch(current_batch)
            all_embeddings.extend(embeddings)
            current_batch = []
            current_tokens = 0
            time.sleep(0.5)  # Rate limit delay

        current_batch.append(text)
        current_tokens += tokens

    if current_batch:
        all_embeddings.extend(send_batch(current_batch))

    return all_embeddings


# ================================================================
# DATABASE (import từ ingest_markdown)
# ================================================================
# Tránh duplicate code — import class PgVectorDB
from ingestion.ingest_markdown import PgVectorDB


# ================================================================
# PIPELINE CHÍNH
# ================================================================
def collect_files(single_file: str = None) -> list[Path]:
    if single_file:
        p = Path(single_file)
        if not p.exists():
            raise FileNotFoundError(f"File không tồn tại: {single_file}")
        return [p]
    files = []
    for d in STRUCTURED_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("*.txt")))
    return [f for f in files if not f.name.startswith("~$")]


def run_ingestion(
    dry_run: bool = False,
    rebuild: bool = False,
    single_file: str = None,
    use_fallback: bool = False,
):
    start_time = time.time()

    print("=" * 70)
    print("  UFM — Structured Text Ingestion (Chương trình Thạc sĩ / Tiến sĩ)")
    print(f"  Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'DRY-RUN' if dry_run else 'PRODUCTION'}")
    print("=" * 70)

    # 1. Collect files
    files = collect_files(single_file=single_file)
    print(f"\n📂 Tìm thấy {len(files)} file:")
    for f in files:
        level = detect_program_level(f)
        print(f"   • [{level}] {f.name} ({f.stat().st_size:,} bytes)")

    if not files:
        print("❌ Không có file nào!")
        return

    # 2. Chunk tất cả files
    all_chunks: list[ProcessedChunk] = []

    for file_path in files:
        print(f"\n{'─' * 50}")
        print(f"📄 {file_path.name}")

        try:
            chunks = chunk_structured_file(file_path, use_fallback=use_fallback)
            parents = [c for c in chunks if c.metadata.chunk_level == "parent"]
            children = [c for c in chunks if c.metadata.chunk_level == "child"]

            print(f"   ✅ {len(chunks)} chunks ({len(parents)} parents, {len(children)} children)")
            print(f"   📋 Ngành: {chunks[0].metadata.program_name if chunks else '?'}")
            print(f"   🔖 Mã ngành: {chunks[0].metadata.ma_nganh if chunks else '?'}")

            # Show context injection demo
            if children:
                demo = children[0].content[:120].replace("\n", " ↵ ")
                print(f"   👁  Context demo: \"{demo}...\"")

            all_chunks.extend(chunks)

        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()

    # 3. Tổng kết chunking
    total_parents = sum(1 for c in all_chunks if c.metadata.chunk_level == "parent")
    total_children = sum(1 for c in all_chunks if c.metadata.chunk_level == "child")

    print(f"\n{'═' * 70}")
    print(f"📊 TỔNG KẾT CHUNKING:")
    print(f"   Files:     {len(files)}")
    print(f"   Parents:   {total_parents}")
    print(f"   Children:  {total_children}")
    print(f"   Tổng:      {len(all_chunks)}")

    # 4. Dry-run
    if dry_run:
        DRY_RUN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        preview = []
        for c in all_chunks:
            preview.append({
                "chunk_id": c.metadata.chunk_id[:12] + "...",
                "chunk_level": c.metadata.chunk_level,
                "parent_id": (c.metadata.parent_id[:12] + "...") if c.metadata.parent_id else None,
                "children_count": len(c.metadata.children_ids),
                "section_name": c.metadata.section_name,
                "program_name": c.metadata.program_name,
                "program_level": c.metadata.program_level,
                "ma_nganh": c.metadata.ma_nganh,
                "source": c.metadata.source,
                "token_count": c.metadata.token_count,
                "content_preview": c.content[:300] + "..." if len(c.content) > 300 else c.content,
            })
        with open(DRY_RUN_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📄 Preview saved: {DRY_RUN_OUTPUT}")
        elapsed = time.time() - start_time
        print(f"⏱️  {elapsed:.1f}s")
        return

    # 5. Dedup → Embed → Insert
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    print(f"\n🔍 Kiểm tra trùng lặp...")
    db = PgVectorDB(DB_CONFIG)
    inserted = 0
    try:
        db.connect()

        if rebuild:
            # Chỉ xóa structured chunks, giữ nguyên markdown chunks
            with db.conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM knowledge_chunks WHERE source LIKE %s OR source LIKE %s",
                    ("ThS %.txt", "NCS %.txt"),
                )
            db.conn.commit()
            print("   🗑️  Đã xóa structured chunks cũ")
            existing_hashes: set[str] = set()
        else:
            existing_hashes = db.load_existing_hashes()
            print(f"   📦 Hash cache: {len(existing_hashes):,} chunks")

        # Lọc chunks mới
        new_chunks = []
        skipped = 0
        for chunk in all_chunks:
            meta = chunk.metadata
            key = f"{meta.content_hash}|{meta.source}|{meta.version}"
            if meta.content_hash and key in existing_hashes:
                skipped += 1
            else:
                new_chunks.append(chunk)

        print(f"   ✅ {len(new_chunks)} chunks MỚI")
        if skipped:
            print(f"   ⏭️  {skipped} chunks đã có → BỎ QUA")

        if not new_chunks:
            print(f"\n✅ Database đã lên mới nhất!")
            db.close()
            elapsed = time.time() - start_time
            print(f"⏱️  {elapsed:.1f}s")
            return

        # Embedding chỉ chunks mới
        if not use_fallback:
            print(f"\n🧠 Embedding {len(new_chunks)} chunks...")
            try:
                embeddings = generate_embeddings_batch(
                    texts=[c.content for c in new_chunks],
                    api_key=api_key,
                    base_url=base_url,
                )
                print(f"   ✅ {len(embeddings)} vectors")
            except Exception as e:
                print(f"   ❌ Embedding lỗi: {e}")
                embeddings = None
        else:
            embeddings = None

        # Insert
        print(f"\n💾 Insert {len(new_chunks)} chunks...")
        inserted = db.insert_chunks_batch(new_chunks, embeddings)
        print(f"   ✅ {inserted}/{len(new_chunks)} inserted")

    except Exception as e:
        print(f"   ❌ DB error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

    elapsed = time.time() - start_time
    print(f"\n{'═' * 70}")
    print(f"✅ HOÀN TẤT — {inserted} chunks, {elapsed:.1f}s")
    print(f"{'═' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="UFM — Structured Text Ingestion (Chương trình ĐT)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--file", type=str)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    run_ingestion(
        dry_run=args.dry_run,
        rebuild=args.rebuild,
        single_file=args.file,
        use_fallback=args.fallback,
    )


if __name__ == "__main__":
    main()
