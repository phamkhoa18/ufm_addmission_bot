-- ============================================================
-- MIGRATION: Retriever Performance Indexes
-- Chạy 1 lần trên Postgres để tối ưu tốc độ tìm kiếm
-- ============================================================

-- #1 BM25 GIN Index — tsvector stored column
-- Thay vì tính to_tsvector(content) mỗi query, lưu sẵn vào cột
ALTER TABLE knowledge_chunks
  ADD COLUMN IF NOT EXISTS content_tsvector tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_content_tsvector
  ON knowledge_chunks USING GIN(content_tsvector);

-- #2 Metadata Filter Index — program_level
-- Tối ưu WHERE program_level = 'thac_si' queries
CREATE INDEX IF NOT EXISTS idx_chunks_program_level
  ON knowledge_chunks(program_level)
  WHERE is_active = TRUE;

-- #3 Composite Index — program_level + is_active + embedding NOT NULL
-- Tối ưu vector search với metadata filter
CREATE INDEX IF NOT EXISTS idx_chunks_active_level
  ON knowledge_chunks(is_active, program_level)
  WHERE is_active = TRUE AND embedding IS NOT NULL;

-- #4 Index program_name cho filter ngành (tương lai)
CREATE INDEX IF NOT EXISTS idx_chunks_program_name
  ON knowledge_chunks(program_name)
  WHERE is_active = TRUE;

-- #5 Unaccent extension (cho BM25 tiếng Việt không dấu)
CREATE EXTENSION IF NOT EXISTS unaccent;
