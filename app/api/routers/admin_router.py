"""
Admin Router — FastAPI Endpoints cho Admin Ingestion API.

Endpoints:
  POST /api/v1/admin/login          → Lấy JWT token
  POST /api/v1/admin/ingest         → Upload .md files → Background Task
  GET  /api/v1/admin/tasks          → Liệt kê tất cả tasks
  GET  /api/v1/admin/tasks/{id}     → Poll trạng thái 1 task
  GET  /api/v1/admin/documents      → Liệt kê tất cả documents từ VectorDB
  GET  /api/v1/admin/documents/stats → Thống kê VectorDB
  DELETE /api/v1/admin/documents    → Soft-delete chunks theo file
"""

import os
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form,
    HTTPException, Request, UploadFile, status,
)
from fastapi.responses import JSONResponse

from app.core.config.admin_config import admin_cfg
from app.core.security import (
    admin_rate_limiter,
    create_access_token,
    get_current_admin,
)
from app.services.admin.task_store import task_store
from app.services.admin.ingestion_worker import process_ingestion
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Router ──
router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])

# Allowed extensions
_ALLOWED_EXT = set(admin_cfg.rate_limit.allowed_extensions)
_MAX_FILE_BYTES = admin_cfg.rate_limit.max_file_size_mb * 1024 * 1024


# ══════════════════════════════════════════════════════════
# DB HELPER — Lấy kết nối PostgreSQL
# ══════════════════════════════════════════════════════════
def _get_pg_connection():
    """Tạo kết nối psycopg2 tới PostgreSQL VectorDB."""
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "ufm_admission_db"),
        user=os.getenv("POSTGRES_USER", "ufm_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "ufm_secure_password_2026"),
    )


# ══════════════════════════════════════════════════════════
# LOGIN — Lấy JWT Token
# ══════════════════════════════════════════════════════════
@router.post("/login", summary="Đăng nhập Admin → JWT Token")
async def admin_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """
    Xác thực Admin và trả về JWT token.

    Body (form-data):
        - username: Tên đăng nhập
        - password: Mật khẩu
    """
    # Rate limit
    client_ip = request.client.host if request.client else "unknown"
    admin_rate_limiter.check(client_ip)

    # Validate credentials
    cred = admin_cfg.credentials
    if username != cred.username or password != cred.password:
        logger.warning("Admin Login FAILED: username='%s' from %s", username, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tên đăng nhập hoặc mật khẩu.",
        )

    token = create_access_token(subject=username)
    logger.info("Admin Login OK: username='%s' from %s", username, client_ip)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_minutes": admin_cfg.jwt.access_token_expire_minutes,
    }


# ══════════════════════════════════════════════════════════
# INGEST — Upload files → Background Processing
# ══════════════════════════════════════════════════════════
@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Nạp file Markdown vào VectorDB",
)
async def ingest_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Danh sách file .md"),
    program_level: Optional[str] = Form(None, description="Bậc học: thac_si/tien_si/dai_hoc/chung"),
    program_name: Optional[str] = Form(None, description="Ngành: VD: Marketing"),
    academic_year: Optional[str] = Form(None, description="Năm học: VD: 2025-2026"),
    reference_url: Optional[str] = Form(None, description="Link tham khảo gốc"),
    admin: str = Depends(get_current_admin),
):
    """
    Upload 1 hoặc nhiều file Markdown → Nạp vào VectorDB.

    **Metadata fields (tùy chọn, fallback null nếu không điền):**
    - program_level: Bậc đào tạo (thac_si / tien_si / dai_hoc)
    - program_name: Tên ngành (VD: "Marketing", "QTKD")
    - academic_year: Năm học (VD: "2025-2026")

    Nếu Admin không điền → hệ thống sẽ cố trích xuất từ front-matter file.
    Nếu file cũng không có → giá trị = null (vẫn nạp bình thường).
    """
    # Rate limit
    admin_rate_limiter.check(admin)

    if not files:
        raise HTTPException(status_code=400, detail="Chưa có file nào được upload.")

    # ── Validate & Fallback null cho metadata ──
    # Empty string → None (front-end có thể gửi "" nếu user không điền)
    _VALID_LEVELS = {"thac_si", "tien_si", "dai_hoc", "chung"}
    clean_level = program_level.strip() if program_level else None
    if clean_level and clean_level not in _VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"program_level không hợp lệ: '{clean_level}'. "
                   f"Chỉ chấp nhận: {', '.join(_VALID_LEVELS)} hoặc để trống.",
        )
    clean_program = program_name.strip() if program_name else None
    clean_year = academic_year.strip() if academic_year else None
    clean_url = reference_url.strip() if reference_url else None

    # Log metadata admin gửi lên
    logger.info(
        "Ingest - Metadata override: level=%s, program=%s, year=%s, url=%s (by %s)",
        clean_level, clean_program, clean_year, clean_url, admin,
    )

    tasks_created = []

    for upload_file in files:
        filename = upload_file.filename or "unknown.md"

        # ── Validate extension ──
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _ALLOWED_EXT:
            tasks_created.append({
                "file_name": filename,
                "status": "rejected",
                "reason": f"Chỉ chấp nhận file {', '.join(_ALLOWED_EXT)}",
            })
            continue

        # ── Validate size ──
        content_bytes = await upload_file.read()
        if len(content_bytes) > _MAX_FILE_BYTES:
            tasks_created.append({
                "file_name": filename,
                "status": "rejected",
                "reason": f"File quá lớn ({len(content_bytes) / 1024 / 1024:.1f}MB > {admin_cfg.rate_limit.max_file_size_mb}MB)",
            })
            continue

        # ── Decode content ──
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            tasks_created.append({
                "file_name": filename,
                "status": "rejected",
                "reason": "File không phải UTF-8 encoding",
            })
            continue

        # ── Create task + schedule background worker ──
        task = task_store.create(file_name=filename)

        background_tasks.add_task(
            process_ingestion,
            file_name=filename,
            file_content=content,
            task=task,
            override_level=clean_level,
            override_program=clean_program,
            override_year=clean_year,
            override_url=clean_url,
        )

        tasks_created.append({
            "file_name": filename,
            "task_id": task.task_id,
            "status": "accepted",
        })
        logger.info("Ingest - Queued: '%s' → task_id=%s (by %s)", filename, task.task_id, admin)

    return {
        "total_files": len(files),
        "accepted": sum(1 for t in tasks_created if t.get("status") == "accepted"),
        "rejected": sum(1 for t in tasks_created if t.get("status") == "rejected"),
        "tasks": tasks_created,
    }


# ══════════════════════════════════════════════════════════
# COMPOSE — Soạn nội dung (HTML → Gemini → Markdown → Ingest)
# ══════════════════════════════════════════════════════════
from pydantic import BaseModel

class ComposeRequest(BaseModel):
    title: str
    html_content: str
    file_name: str = ""
    program_level: str = ""
    program_name: str = ""
    academic_year: str = ""
    reference_url: str = ""

def _html_to_markdown_via_gemini(title: str, html_content: str) -> str:
    """
    Gọi Gemini 2.5 Flash qua OpenRouter để chuyển HTML → Markdown chuẩn.
    Trả về chuỗi Markdown đã chuẩn hóa, sẵn sàng nạp vào VectorDB.
    """
    import requests as _req

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise ValueError("Thiếu OPENROUTER_API_KEY trong .env")

    system_prompt = (
        "Bạn là module chuyển đổi HTML sang Markdown cho hệ thống quản lý dữ liệu tuyển sinh Đại học Tài chính - Marketing (UFM).\n\n"
        "NHIỆM VỤ:\n"
        "Nhận tiêu đề và nội dung HTML từ trình soạn thảo, chuyển thành file Markdown chuẩn có cấu trúc rõ ràng.\n\n"
        "QUY TẮC:\n"
        "1. Bắt đầu bằng dòng `-start-` để đánh dấu bắt đầu nội dung.\n"
        "2. Tiêu đề chính dùng `# Tiêu đề`.\n"
        "3. Giữ nguyên TOÀN BỘ nội dung, số liệu, bảng, danh sách. KHÔNG tóm tắt, KHÔNG bỏ bớt.\n"
        "4. Chuyển bảng HTML sang bảng Markdown chuẩn | col1 | col2 |.\n"
        "5. Chuyển <ul>/<ol> sang danh sách Markdown (- hoặc 1.).\n"
        "6. Chuyển <strong> sang **bold**, <em> sang *italic*.\n"
        "7. Chuyển <a href> sang [text](url).\n"
        "8. Loại bỏ tất cả thẻ HTML, style, class, comment. CHỈ trả về Markdown thuần.\n"
        "9. Nếu có ảnh base64, BỎ ảnh đó (quá lớn cho VectorDB).\n"
        "10. KHÔNG bọc kết quả trong ```markdown```. Trả về Markdown thuần túy.\n"
    )

    user_content = f"TIÊU ĐỀ: {title}\n\nNỘI DUNG HTML:\n{html_content}"

    resp = _req.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 8000,
        },
        timeout=30,
    )

    if not resp.ok:
        error_body = resp.text[:500]
        logger.error("Gemini API error %d: %s", resp.status_code, error_body)
        resp.raise_for_status()

    data = resp.json()

    md = data["choices"][0]["message"]["content"].strip()
    # Remove markdown fences if the model wraps them
    if md.startswith("```"):
        lines = md.split("\n")
        lines = lines[1:]  # Remove first ```markdown
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        md = "\n".join(lines)

    return md


@router.post(
    "/compose",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Soạn nội dung → Gemini → Markdown → Ingest VectorDB",
)
async def compose_content(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ComposeRequest,
    admin: str = Depends(get_current_admin),
):
    """
    Nhận nội dung HTML từ trình soạn thảo, gọi Gemini chuyển sang Markdown,
    sau đó nạp vào VectorDB qua pipeline ingestion hiện có.

    Body (JSON):
      - title: Tiêu đề văn bản
      - html_content: Nội dung HTML từ rich text editor
      - file_name: (tùy chọn) Tên file .md (auto-generate nếu trống)
      - program_level: (tùy chọn) chung / thac_si / tien_si / dai_hoc
      - program_name: (tùy chọn) Tên ngành
      - academic_year: (tùy chọn) Năm học
      - reference_url: (tùy chọn) Link tham khảo gốc
    """
    admin_rate_limiter.check(admin)

    if not body.html_content or not body.html_content.strip():
        raise HTTPException(status_code=400, detail="Nội dung soạn thảo đang trống.")

    if not body.title or not body.title.strip():
        raise HTTPException(status_code=400, detail="Tiêu đề không được để trống.")

    # ── Gọi Gemini chuyển HTML → Markdown ──
    try:
        markdown_content = _html_to_markdown_via_gemini(body.title.strip(), body.html_content)
    except Exception as e:
        logger.error("Compose - Gemini conversion failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Lỗi chuyển đổi Markdown: {str(e)}")

    if not markdown_content or len(markdown_content.strip()) < 20:
        raise HTTPException(status_code=422, detail="Nội dung Markdown sau chuyển đổi quá ngắn hoặc lỗi.")

    # ── Tạo file_name nếu trống ──
    import re
    import unicodedata

    file_name = body.file_name.strip() if body.file_name else ""
    if not file_name:
        # Slugify title → file name
        slug = body.title.strip().lower()
        slug = unicodedata.normalize("NFD", slug)
        slug = re.sub(r"[\u0300-\u036f]", "", slug)  # Remove diacritics
        slug = re.sub(r"[đĐ]", "d", slug)
        slug = re.sub(r"[^a-z0-9]+", "_", slug)
        slug = slug.strip("_")[:60]
        file_name = f"{slug}.md"

    if not file_name.endswith(".md"):
        file_name += ".md"

    # ── Validate metadata ──
    _VALID_LEVELS = {"thac_si", "tien_si", "dai_hoc"}
    clean_level = body.program_level.strip() if body.program_level else None
    if clean_level and clean_level not in _VALID_LEVELS:
        clean_level = None
    clean_program = body.program_name.strip() if body.program_name else None
    clean_year = body.academic_year.strip() if body.academic_year else None

    logger.info(
        "Compose - title='%s' file='%s' level=%s program=%s year=%s md_len=%d (by %s)",
        body.title[:50], file_name, clean_level, clean_program, clean_year,
        len(markdown_content), admin,
    )

    # ── Tạo task + schedule background ingestion ──
    task = task_store.create(file_name=file_name)

    background_tasks.add_task(
        process_ingestion,
        file_name=file_name,
        file_content=markdown_content,
        task=task,
        override_level=clean_level,
        override_program=clean_program,
        override_year=clean_year,
    )

    return {
        "status": "accepted",
        "file_name": file_name,
        "task_id": task.task_id,
        "markdown_preview": markdown_content[:500] + ("..." if len(markdown_content) > 500 else ""),
        "markdown_length": len(markdown_content),
    }


# ══════════════════════════════════════════════════════════
# TASK STATUS — Poll trạng thái
# ══════════════════════════════════════════════════════════
@router.get("/tasks", summary="Liệt kê tất cả tasks")
async def list_tasks(admin: str = Depends(get_current_admin)):
    """Trả về danh sách tất cả ingestion tasks (mới nhất trước)."""
    return {"tasks": task_store.list_all()}


@router.get("/tasks/{task_id}", summary="Xem trạng thái 1 task")
async def get_task_status(
    task_id: str,
    admin: str = Depends(get_current_admin),
):
    """Poll trạng thái chi tiết của 1 ingestion task."""
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' không tồn tại.",
        )
    return task.to_dict()


@router.post("/tasks/{task_id}/cancel", summary="Hủy 1 task đang xử lý")
async def cancel_task(
    task_id: str,
    admin: str = Depends(get_current_admin),
):
    """Hủy task nếu đang ở trạng thái chờ/xử lý."""
    success = task_store.cancel(task_id)
    if not success:
        task = task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' không tồn tại.")
        return {
            "task_id": task_id,
            "cancelled": False,
            "reason": f"Task đã ở trạng thái '{task.status.value}', không thể hủy.",
        }
    logger.info("Admin CANCEL task '%s' (by %s)", task_id, admin)
    return {"task_id": task_id, "cancelled": True}

# DOCUMENTS — Liệt kê tất cả documents từ VectorDB
# ══════════════════════════════════════════════════════════
@router.get("/documents", summary="Liệt kê tất cả documents trong VectorDB")
async def list_documents(admin: str = Depends(get_current_admin)):
    """
    Truy vấn PostgreSQL VectorDB → trả danh sách documents kèm chunk counts.

    Response:
    ```json
    {
      "documents": [
        {
          "source": "phuluc1.md",
          "program_level": "thac_si",
          "program_name": "Marketing",
          "academic_year": "2025-2026",
          "chunk_count": 42,
          "total_chars": 15200,
          "is_active": true,
          "created_at": "2026-03-25T10:30:00",
          "updated_at": "2026-03-25T10:30:00"
        }
      ],
      "total_documents": 5
    }
    ```
    """
    import asyncio
    result = await asyncio.to_thread(_query_documents)
    return result


def _query_documents():
    """Sync function — Query PostgreSQL documents grouped by source file."""
    conn = None
    try:
        conn = _get_pg_connection()
        from psycopg2.extras import RealDictCursor

        sql = """
            SELECT
                source,
                program_level,
                program_name,
                academic_year,
                COUNT(*) AS chunk_count,
                SUM(char_count) AS total_chars,
                bool_and(is_active) AS is_active,
                MIN(created_at) AS created_at,
                MAX(updated_at) AS updated_at
            FROM knowledge_chunks
            WHERE is_active = TRUE
            GROUP BY source, program_level, program_name, academic_year
            ORDER BY MAX(updated_at) DESC;
        """

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        documents = []
        for row in rows:
            documents.append({
                "source": row["source"],
                "program_level": row["program_level"],
                "program_name": row["program_name"],
                "academic_year": row["academic_year"],
                "chunk_count": row["chunk_count"],
                "total_chars": int(row["total_chars"] or 0),
                "is_active": row["is_active"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            })

        return {
            "documents": documents,
            "total_documents": len(documents),
        }
    except Exception as e:
        logger.error("Admin - List documents error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi truy vấn VectorDB: {str(e)}",
        )
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════
# DOCUMENTS STATS — Thống kê VectorDB real-time
# ══════════════════════════════════════════════════════════
@router.get("/documents/stats", summary="Thống kê VectorDB chi tiết")
async def documents_stats(admin: str = Depends(get_current_admin)):
    """
    Thống kê real-time từ PostgreSQL VectorDB:
    - Tổng số documents (theo source file)
    - Tổng số chunks (active/inactive)
    - Phân bổ theo bậc đào tạo
    - Tổng ký tự, DB size

    Response mẫu:
    ```json
    {
      "total_documents": 17,
      "total_chunks": 226,
      "active_chunks": 220,
      "inactive_chunks": 6,
      "total_characters": 456000,
      "has_embeddings": 226,
      "by_level": {
        "thac_si": { "documents": 8, "chunks": 120 },
        "tien_si": { "documents": 2, "chunks": 30 },
        "dai_hoc": { "documents": 5, "chunks": 60 },
        "unknown": { "documents": 2, "chunks": 16 }
      },
      "db_status": "connected"
    }
    ```
    """
    import asyncio
    result = await asyncio.to_thread(_query_stats)
    return result


def _query_stats():
    """Sync function — Query detailed VectorDB statistics."""
    conn = None
    try:
        conn = _get_pg_connection()
        from psycopg2.extras import RealDictCursor

        stats = {"db_status": "connected"}

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Tổng quan
            cur.execute("""
                SELECT
                    COUNT(DISTINCT source) AS total_documents,
                    COUNT(*) AS total_chunks,
                    COUNT(*) FILTER (WHERE is_active = TRUE) AS active_chunks,
                    COUNT(*) FILTER (WHERE is_active = FALSE) AS inactive_chunks,
                    COALESCE(SUM(char_count), 0) AS total_characters,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS has_embeddings
                FROM knowledge_chunks;
            """)
            overview = cur.fetchone()
            stats.update({
                "total_documents": overview["total_documents"],
                "total_chunks": overview["total_chunks"],
                "active_chunks": overview["active_chunks"],
                "inactive_chunks": overview["inactive_chunks"],
                "total_characters": int(overview["total_characters"]),
                "has_embeddings": overview["has_embeddings"],
            })

            # 2. Phân bổ theo bậc đào tạo
            cur.execute("""
                SELECT
                    COALESCE(program_level, 'unknown') AS level,
                    COUNT(DISTINCT source) AS documents,
                    COUNT(*) AS chunks
                FROM knowledge_chunks
                WHERE is_active = TRUE
                GROUP BY program_level
                ORDER BY chunks DESC;
            """)
            by_level = {}
            for row in cur.fetchall():
                by_level[row["level"]] = {
                    "documents": row["documents"],
                    "chunks": row["chunks"],
                }
            stats["by_level"] = by_level

            # 3. Top 5 documents mới nhất
            cur.execute("""
                SELECT
                    source,
                    COUNT(*) AS chunks,
                    MAX(updated_at) AS last_updated
                FROM knowledge_chunks
                WHERE is_active = TRUE
                GROUP BY source
                ORDER BY MAX(updated_at) DESC
                LIMIT 5;
            """)
            recent = []
            for row in cur.fetchall():
                recent.append({
                    "source": row["source"],
                    "chunks": row["chunks"],
                    "last_updated": row["last_updated"].isoformat() if row["last_updated"] else None,
                })
            stats["recent_documents"] = recent

        return stats

    except Exception as e:
        logger.error("Admin - Stats query error: %s", e, exc_info=True)
        return {
            "db_status": "error",
            "error": str(e),
            "total_documents": 0,
            "total_chunks": 0,
            "active_chunks": 0,
            "inactive_chunks": 0,
            "total_characters": 0,
            "has_embeddings": 0,
            "by_level": {},
        }
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════
# DELETE — Soft-delete document
# ══════════════════════════════════════════════════════════
@router.delete("/documents", summary="Hard-delete chunks theo tên file")
async def delete_document(
    file_name: str,
    admin: str = Depends(get_current_admin),
):
    """
    Hard-delete tất cả chunks của 1 file (XÓA CHẾT).

    Query params:
        - file_name: Tên file cần xóa (VD: "phuluc1.md")
    """
    from app.services.admin.dedup_service import DedupService
    from app.services.admin.ingestion_worker import _get_db_connection

    conn = None
    try:
        conn = _get_db_connection()
        conn.autocommit = False
        dedup = DedupService(conn)
        count = dedup.soft_delete_old_chunks(file_name)
        dedup.remove_old_log(file_name)

        logger.info("Admin DELETE: '%s' → %d chunks hard-deleted (by %s)", file_name, count, admin)

        return {
            "file_name": file_name,
            "chunks_deleted": count,
            "message": f"Đã xóa vĩnh viễn {count} chunks.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa: {str(e)}",
        )
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════
# DOCUMENT DETAIL — Xem chi tiết chunks của 1 tài liệu
# ══════════════════════════════════════════════════════════
@router.get("/documents/detail", summary="Xem chi tiết chunks của 1 tài liệu")
async def document_detail(
    source: str,
    admin: str = Depends(get_current_admin),
):
    """
    Trả về danh sách tất cả chunks thuộc 1 source file.

    Query params:
        - source: Tên file nguồn (VD: "phuluc1.md")

    Response: Danh sách chunks kèm nội dung, metadata, section path.
    """
    import asyncio
    result = await asyncio.to_thread(_query_document_detail, source)
    return result


def _query_document_detail(source: str):
    """Sync function — Query chi tiết chunks của 1 document."""
    conn = None
    try:
        conn = _get_pg_connection()
        from psycopg2.extras import RealDictCursor

        sql = """
            SELECT
                chunk_id,
                chunk_level,
                parent_id,
                section_path,
                section_name,
                program_name,
                program_level,
                ma_nganh,
                academic_year,
                content,
                char_count,
                is_active,
                extra,
                created_at,
                updated_at
            FROM knowledge_chunks
            WHERE source = %s AND is_active = TRUE
            ORDER BY chunk_id ASC;
        """

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (source,))
            rows = cur.fetchall()

        chunks = []
        for row in rows:
            extra = row.get("extra") or {}
            chunks.append({
                "chunk_id": row["chunk_id"],
                "chunk_level": row["chunk_level"],
                "parent_id": row["parent_id"],
                "section_path": row["section_path"],
                "section_name": row["section_name"],
                "program_name": row["program_name"],
                "program_level": row["program_level"],
                "ma_nganh": row["ma_nganh"],
                "academic_year": row["academic_year"],
                "content": row["content"],
                "content_preview": (row["content"] or "")[:300],
                "char_count": row["char_count"],
                "is_active": row["is_active"],
                "extra": extra,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            })

        return {
            "source": source,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }
    except Exception as e:
        logger.error("Admin - Document detail error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi truy vấn chi tiết document: {str(e)}",
        )
    finally:
        if conn:
            conn.close()

