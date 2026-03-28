"""
Dedup Service — Chống trùng lặp dữ liệu khi nạp VectorDB.

Chiến lược:
  - SHA-256 hash toàn bộ nội dung file (sau khi normalize header).
  - Check hash trong bảng ingestion_logs.
  - Trùng 100% → skip.
  - Cùng tên nhưng khác nội dung → soft-delete cũ, insert mới.
"""

import hashlib
from datetime import datetime
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


def compute_file_hash(content: str) -> str:
    """Tính SHA-256 hash của nội dung file."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class DedupService:
    """
    Quản lý Deduplication qua bảng ingestion_logs.

    Bảng ingestion_logs:
        file_hash   TEXT PRIMARY KEY,
        file_name   TEXT NOT NULL,
        status      TEXT DEFAULT 'completed',
        chunks_count INT DEFAULT 0,
        created_at  TIMESTAMP DEFAULT NOW(),
        updated_at  TIMESTAMP DEFAULT NOW()
    """

    def __init__(self, conn):
        """
        Args:
            conn: psycopg2 connection object.
        """
        self.conn = conn

    def check_duplicate(self, file_hash: str, file_name: str) -> dict:
        """
        Kiểm tra file đã tồn tại trong DB chưa.

        Returns:
            {
                "action": "skip" | "update" | "insert",
                "reason": str,
                "existing_hash": str | None,
            }
        """
        with self.conn.cursor() as cur:
            # Check hash trùng
            cur.execute(
                "SELECT file_name, status, created_at FROM ingestion_logs "
                "WHERE file_hash = %s",
                (file_hash,),
            )
            row = cur.fetchone()

            if row:
                return {
                    "action": "skip",
                    "reason": f"File trùng 100% với '{row[0]}' (nạp lúc {row[2]})",
                    "existing_hash": file_hash,
                }

            # Check cùng tên nhưng nội dung khác
            cur.execute(
                "SELECT file_hash FROM ingestion_logs WHERE file_name = %s",
                (file_name,),
            )
            old_row = cur.fetchone()

            if old_row:
                return {
                    "action": "update",
                    "reason": f"File '{file_name}' đã tồn tại nhưng nội dung khác → sẽ cập nhật",
                    "existing_hash": old_row[0],
                }

            return {
                "action": "insert",
                "reason": "File mới, chưa từng nạp",
                "existing_hash": None,
            }

    def soft_delete_old_chunks(self, file_name: str) -> int:
        """
        Hard-delete tất cả chunks cũ của file. (Giữ tên hàm cũ để tương thích với router)

        Returns:
            Số chunks đã hard-delete.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM knowledge_chunks WHERE source = %s",
                (file_name,),
            )
            count = cur.rowcount

        self.conn.commit()
        logger.info("DedupService - Deleted %d chunks for '%s'", count, file_name)
        return count

    def record_ingestion(
        self,
        file_hash: str,
        file_name: str,
        status: str = "completed",
        chunks_count: int = 0,
    ) -> None:
        """Ghi log nạp file vào bảng ingestion_logs."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_logs (file_hash, file_name, status, chunks_count, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (file_hash) DO UPDATE SET
                    status = EXCLUDED.status,
                    chunks_count = EXCLUDED.chunks_count,
                    updated_at = EXCLUDED.updated_at
                """,
                (file_hash, file_name, status, chunks_count, datetime.utcnow()),
            )
        self.conn.commit()
        logger.info(
            "DedupService - Recorded: file='%s' hash='%s...' status='%s' chunks=%d",
            file_name, file_hash[:12], status, chunks_count,
        )

    def remove_old_log(self, file_name: str) -> None:
        """Xóa log cũ của file (trước khi insert bản mới)."""
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM ingestion_logs WHERE file_name = %s",
                (file_name,),
            )
        self.conn.commit()
