-- ============================================================
-- Migration 004: Bảng ingestion_logs — Theo dõi file đã nạp
-- Dùng cho Dedup Service (chống trùng lặp khi nạp VectorDB)
-- ============================================================

-- Bảng log tracking file đã nạp
CREATE TABLE IF NOT EXISTS ingestion_logs (
    file_hash       TEXT        PRIMARY KEY,
    file_name       TEXT        NOT NULL,
    status          TEXT        DEFAULT 'completed',    -- pending | processing | completed | error
    chunks_count    INTEGER     DEFAULT 0,
    error_message   TEXT,
    created_at      TIMESTAMP   DEFAULT NOW(),
    updated_at      TIMESTAMP   DEFAULT NOW()
);

-- Index tìm nhanh theo tên file
CREATE INDEX IF NOT EXISTS idx_ingestion_file_name
    ON ingestion_logs (file_name);

-- Comment
COMMENT ON TABLE ingestion_logs IS 'Theo dõi file đã nạp vào VectorDB, dùng SHA-256 hash chống trùng';
