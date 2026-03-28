-- ============================================================================
-- database/init.sql
-- Tự động chạy MỘT LẦN khi container PostgreSQL khởi tạo lần đầu.
-- Schema thiết kế theo models/chunk.py (ChunkMetadata + ProcessedChunk)
-- ============================================================================

-- ────────────────────────────────────────────────────────────
-- 1. BẬT EXTENSIONS CẦN THIẾT
-- ────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;         -- pgvector cho Vector Search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";    -- Hàm uuid_generate_v4()


-- ============================================================================
-- 2. BẢNG CHÍNH: knowledge_chunks
--    Map 1:1 với ChunkMetadata + ProcessedChunk (Pydantic model)
--
--    ┌─────────────────────────────────────────────────────────────────────┐
--    │  CHUNK PARENT (chunk_level = 'parent')                            │
--    │  id: aaaaa-1111                                                    │
--    │  section_path: "Tuyển sinh 2026 > Thạc sĩ > Điều kiện"           │
--    │  children_ids: ['bbbbb-2222', 'bbbbb-3333']                       │
--    │                                                                    │
--    │   ┌───────────────────────┐  ┌───────────────────────┐            │
--    │   │ CHILD bbbbb-2222      │  │ CHILD bbbbb-3333      │            │
--    │   │ parent_id: aaaaa-1111 │  │ parent_id: aaaaa-1111 │            │
--    │   │ embedding: [0.12,...] │  │ embedding: [0.34,...] │            │
--    │   │ (Vector Search trúng  │  │                       │            │
--    │   │  → Lôi Parent ra RAG)│  │                       │            │
--    │   └───────────────────────┘  └───────────────────────┘            │
--    └─────────────────────────────────────────────────────────────────────┘
-- ============================================================================
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    -- ════════════════════════════════════════════════════════
    -- CORE — Identity & Structure
    -- ════════════════════════════════════════════════════════
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id          VARCHAR(64) NOT NULL UNIQUE,    -- UUID từ Python (ChunkMetadata.chunk_id)
    source            VARCHAR(255) NOT NULL,          -- File nguồn: "ThS_KDQT.docx"
    section_path      TEXT,                           -- Breadcrumb: "Thạc sĩ KDQT > Xét tuyển"
    section_name      VARCHAR(500),                   -- Tên mục: "ĐIỀU KIỆN XÉT TUYỂN"
    program_name      VARCHAR(255),                   -- Ngành: "KINH DOANH QUỐC TẾ"
    program_level     VARCHAR(30),                    -- thac_si | tien_si | dai_hoc
    ma_nganh          VARCHAR(20),                    -- Mã ngành Bộ GD&ĐT: "8340120"
    chunk_index       INTEGER,                        -- Vị trí chunk trong section (1-indexed)
    total_chunks_in_section INTEGER,                  -- Tổng chunks trong section

    -- ════════════════════════════════════════════════════════
    -- CONTENT — Nội dung + Embedding
    -- ════════════════════════════════════════════════════════
    content           TEXT NOT NULL,                   -- Nội dung văn bản đầy đủ
    char_count        INTEGER DEFAULT 0,              -- Số ký tự (ProcessedChunk.char_count)
    embedding         VECTOR(1024),                   -- Vector BGE-M3 1024 chiều
    content_tsvector  TSVECTOR,                       -- Stored tsvector cho BM25 (GIN index)

    -- ════════════════════════════════════════════════════════
    -- HIERARCHY — Parent-Child Relationship
    -- ════════════════════════════════════════════════════════
    chunk_level       VARCHAR(10) NOT NULL DEFAULT 'standard'
                      CHECK (chunk_level IN ('parent', 'child', 'standard')),
    parent_id         VARCHAR(64),                    -- chunk_id của Parent (nếu là child)
    children_ids      TEXT[] DEFAULT '{}',             -- Array UUID con (nếu là parent)
    overlap_tokens    INTEGER DEFAULT 0,              -- Token overlap với chunk liền kề trước

    -- ════════════════════════════════════════════════════════
    -- LIFECYCLE — Validity & Version Control
    -- ════════════════════════════════════════════════════════
    academic_year     VARCHAR(20),                    -- "2025-2026" hoặc "2026"
    valid_from        TIMESTAMP WITH TIME ZONE,       -- Ngày bắt đầu hiệu lực
    valid_until       TIMESTAMP WITH TIME ZONE,       -- Ngày hết hạn
    is_active         BOOLEAN DEFAULT TRUE,           -- Soft-delete flag
    version           INTEGER DEFAULT 1,              -- Phiên bản nội dung
    replaced_by       VARCHAR(64),                    -- chunk_id của bản thay thế

    -- ════════════════════════════════════════════════════════
    -- RETRIEVAL — Search & Filter Enhancement
    -- ════════════════════════════════════════════════════════
    content_hash      VARCHAR(64),                    -- SHA256 hash (dedup detection)
    token_count       INTEGER DEFAULT 0,              -- Ước tính token count

    -- ════════════════════════════════════════════════════════
    -- EXTENSIBILITY — Future-proof
    -- ════════════════════════════════════════════════════════
    extra             JSONB DEFAULT '{}'::jsonb,       -- Dict mở rộng tùy ý

    -- ════════════════════════════════════════════════════════
    -- TIMESTAMPS
    -- ════════════════════════════════════════════════════════
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- ────────────────────────────────────────────────────────────
-- 3. INDEXES — Tối ưu cho từng loại truy vấn
-- ────────────────────────────────────────────────────────────

-- 3.1. HNSW Index cho Vector Search (Cosine Similarity)
--      Đây là index quan trọng nhất: tìm chunks giống nhất với câu hỏi user
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON knowledge_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 3.2. B-Tree Indexes cho filter truy vấn
--      Khi search, thường kèm filter: WHERE is_active = TRUE AND program_level = 'thac_si'
CREATE INDEX IF NOT EXISTS idx_chunks_active
    ON knowledge_chunks (is_active)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_chunks_program_level
    ON knowledge_chunks (program_level);

CREATE INDEX IF NOT EXISTS idx_chunks_ma_nganh
    ON knowledge_chunks (ma_nganh);

CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON knowledge_chunks (source);

CREATE INDEX IF NOT EXISTS idx_chunks_academic_year
    ON knowledge_chunks (academic_year);

-- 3.3. Index cho Parent-Child lookups
--      Khi search trúng Child → cần lôi ra Parent cực nhanh
CREATE INDEX IF NOT EXISTS idx_chunks_parent_id
    ON knowledge_chunks (parent_id)
    WHERE parent_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_level
    ON knowledge_chunks (chunk_level);

-- 3.4. Dedup Index — content_hash phải unique nếu cùng source + version
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_dedup
    ON knowledge_chunks (content_hash, source, version)
    WHERE content_hash IS NOT NULL;

-- 3.5. GIN Index trên JSONB extra (cho truy vấn linh hoạt)
CREATE INDEX IF NOT EXISTS idx_chunks_extra_gin
    ON knowledge_chunks USING gin (extra);

-- 3.6. GIN Index cho BM25 Full-Text Search (stored tsvector)
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsvector_gin
    ON knowledge_chunks USING gin (content_tsvector);


-- ────────────────────────────────────────────────────────────
-- 3.7. TRIGGER: Tự động tính content_tsvector khi INSERT/UPDATE
-- ────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_content_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector = to_tsvector('simple', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_content_tsvector
    BEFORE INSERT OR UPDATE OF content ON knowledge_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_content_tsvector();


-- ────────────────────────────────────────────────────────────
-- 4. TRIGGER: Tự động cập nhật updated_at
-- ────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_knowledge_chunks_updated
    BEFORE UPDATE ON knowledge_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();


-- ============================================================================
-- 5. BẢNG PHỤ: intent_examples (Layer 3.1: Vector Router)
-- ============================================================================
CREATE TABLE IF NOT EXISTS intent_examples (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    intent_name     VARCHAR(100) NOT NULL,
    example_text    TEXT NOT NULL,
    embedding       VECTOR(1024),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_intent_embedding_hnsw
    ON intent_examples
    USING hnsw (embedding vector_cosine_ops);


-- ────────────────────────────────────────────────────────────
-- 6. VIEW: Truy vấn nhanh Parent kèm danh sách Children
-- ────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_parent_children AS
SELECT
    p.chunk_id          AS parent_chunk_id,
    p.section_path      AS parent_section_path,
    p.section_name      AS parent_section_name,
    p.content           AS parent_content,
    p.program_level,
    p.ma_nganh,
    c.chunk_id          AS child_chunk_id,
    c.content           AS child_content,
    c.chunk_index       AS child_chunk_index,
    c.overlap_tokens    AS child_overlap_tokens,
    c.token_count       AS child_token_count
FROM knowledge_chunks p
JOIN knowledge_chunks c ON c.parent_id = p.chunk_id
WHERE p.chunk_level = 'parent'
  AND c.chunk_level = 'child'
  AND p.is_active = TRUE
  AND c.is_active = TRUE
ORDER BY p.chunk_id, c.chunk_index;


-- ────────────────────────────────────────────────────────────
-- 7. MOCK DATA CHO INTENT ROUTER (chưa có embedding)
-- ────────────────────────────────────────────────────────────
INSERT INTO intent_examples (intent_name, example_text) VALUES
('THONG_TIN_TUYEN_SINH', 'Cho em hỏi điểm chuẩn ngành Marketing năm 2024 là bao nhiêu ạ?'),
('THONG_TIN_TUYEN_SINH', 'Trường tuyển sinh khối nào? Có xét học bạ không?'),
('THONG_TIN_TUYEN_SINH', 'Điều kiện xét tuyển thạc sĩ ngành Kinh doanh quốc tế là gì?'),
('HOC_PHI_HOC_BONG', 'Học phí 1 kỳ của trường Tài chính Marketing là bao nhiêu?'),
('HOC_PHI_HOC_BONG', 'Trường có chính sách giảm học phí hay học bổng cho sinh viên nghèo không?'),
('DOI_SONG_SINH_VIEN', 'Ký túc xá của trường nằm ở đâu? Có điều hòa không ạ?'),
('THU_TUC_HANH_CHINH', 'Thủ tục làm hồ sơ nhập học cần mang theo những giấy tờ gì?'),

-- Intent: Yêu cầu mẫu đơn / biểu mẫu (→ FormAgent, KHÔNG qua VectorDB)
('TAO_MAU_DON', 'Cho em xin mẫu đơn đăng ký dự tuyển tiến sĩ'),
('TAO_MAU_DON', 'Gửi em giấy cam đoan học thạc sĩ với ạ'),
('TAO_MAU_DON', 'Tải đơn dự tuyển thạc sĩ ở đâu?'),
('TAO_MAU_DON', 'Trường có mẫu viết đề cương nghiên cứu không?'),
('TAO_MAU_DON', 'Cho em xin file word để điền thông tin xét tuyển'),
('TAO_MAU_DON', 'Em cần mẫu đơn đăng ký thi nghiên cứu sinh'),

-- Intent: Hỏi về chương trình đào tạo (→ VectorDB structured chunks)
('CHUONG_TRINH_DAO_TAO', 'Chương trình thạc sĩ Kinh doanh quốc tế học những gì?'),
('CHUONG_TRINH_DAO_TAO', 'Tiến sĩ Quản trị kinh doanh ra trường làm ở đâu?'),
('CHUONG_TRINH_DAO_TAO', 'Cơ hội nghề nghiệp sau khi tốt nghiệp thạc sĩ Tài chính Ngân hàng?'),
('CHUONG_TRINH_DAO_TAO', 'Thạc sĩ Marketing học mấy năm? Bao nhiêu tín chỉ?');


-- ════════════════════════════════════════════════════════════
-- ✅ INIT HOÀN TẤT
-- Schema sẵn sàng cho Semantic + Hierarchical Chunking
-- ════════════════════════════════════════════════════════════
