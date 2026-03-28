"""
Pydantic models for document chunks.

Thiết kế cho hệ thống RAG Tuyển sinh UFM:
  - ChunkMetadata: Metadata phong phú hỗ trợ Hierarchical Chunking (Parent-Child)
  - ProcessedChunk: Chunk đã xử lý, sẵn sàng Embedding
  - EmbeddingScore: Đánh giá chất lượng Embedding
"""
import uuid
import hashlib
from datetime import datetime
from typing import Optional, Literal, List
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Metadata for each chunk stored in VectorDB.

    Design principles:
      - Core fields: always present, used for search/filter
      - Hierarchy fields: parent-child relationships for Hierarchical Chunking
      - Lifecycle fields: validity control, version tracking
      - Retrieval fields: search enhancement (hash, token count)
      - Extensibility: extra dict for future use

    ┌──────────────────────────────────────────────────────────┐
    │ CORE — Identity & Structure                             │
    ├──────────────────────────────────────────────────────────┤
    │ chunk_id        UUID duy nhất (tạo từ Python)           │
    │ source          File nguồn (ThS KDQT.docx)              │
    │ section_path    Breadcrumb (Thạc sĩ KDQT > XÉT TUYỂN)   │
    │ section_name    Tên mục (ĐIỀU KIỆN XÉT TUYỂN)           │
    │ program_name    Tên ngành (KINH DOANH QUỐC TẾ)          │
    │ program_level   Trình độ (thac_si / tien_si / dai_hoc)  │
    │ ma_nganh        Mã ngành (8340120)                      │
    │ chunk_index     Vị trí chunk trong section (1, 2, ...)   │
    │ total_chunks    Tổng chunks trong section                │
    ├──────────────────────────────────────────────────────────┤
    │ HIERARCHY — Parent-Child (Hierarchical Chunking)        │
    ├──────────────────────────────────────────────────────────┤
    │ chunk_level     parent | child | standard               │
    │ parent_id       UUID của chunk cha (nếu là child)       │
    │ children_ids    Danh sách UUID các chunk con             │
    │ overlap_tokens  Số token overlap với chunk liền kề       │
    ├──────────────────────────────────────────────────────────┤
    │ LIFECYCLE — Validity & Version Control                  │
    ├──────────────────────────────────────────────────────────┤
    │ academic_year   Năm học áp dụng ("2025-2026")           │
    │ valid_from      Hiệu lực từ (date)                      │
    │ valid_until     Hết hạn (date) → auto-exclude outdated  │
    │ is_active       Còn hiệu lực? (True/False)              │
    │ version         Phiên bản nội dung (1, 2, 3...)         │
    │ replaced_by     ID chunk thay thế (nếu bị supersede)    │
    ├──────────────────────────────────────────────────────────┤
    │ RETRIEVAL — Search & Filter Enhancement                 │
    ├──────────────────────────────────────────────────────────┤
    │ content_hash    SHA256 hash (dedup detection)           │
    │ token_count     Số token ước tính (để filter/sort)      │
    ├──────────────────────────────────────────────────────────┤
    │ EXTENSIBILITY                                           │
    ├──────────────────────────────────────────────────────────┤
    │ extra           Dict mở rộng bất kỳ                     │
    └──────────────────────────────────────────────────────────┘
    """

    # ═══════════════════════════════════════════════════
    # CORE — Identity & Structure (always present)
    # ═══════════════════════════════════════════════════
    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID duy nhất, tạo từ Python để map Parent-Child trước khi insert DB",
    )
    source: str = Field(
        ..., description="Tên file nguồn, VD: ThS KDQT.docx"
    )
    section_path: Optional[str] = Field(
        default=None,
        description="Breadcrumb path, VD: Thạc sĩ KDQT > ĐIỀU KIỆN XÉT TUYỂN",
    )
    section_name: Optional[str] = Field(
        default=None,
        description="Tên section/mục, VD: ĐIỀU KIỆN XÉT TUYỂN",
    )
    program_name: Optional[str] = Field(
        default=None,
        description="Tên chương trình đào tạo, VD: KINH DOANH QUỐC TẾ",
    )
    program_level: Optional[str] = Field(
        default=None,
        description="Trình độ: thac_si | tien_si | dai_hoc",
    )
    ma_nganh: Optional[str] = Field(
        default=None,
        description="Mã ngành theo Bộ GD&ĐT, VD: 8340120",
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Vị trí chunk trong section (1-indexed)",
    )
    total_chunks_in_section: Optional[int] = Field(
        default=None,
        description="Tổng số chunks trong section",
    )

    # ═══════════════════════════════════════════════════
    # HIERARCHY — Parent-Child Relationship
    # Dùng cho Hierarchical Chunking:
    #   Parent (lớn, ~2000 tokens) → LLM đọc ngữ cảnh rộng
    #   Child  (nhỏ, ~300 tokens)  → Vector Search chính xác
    #   Khi Search trúng Child → Lôi Parent ra đưa cho RAG
    # ═══════════════════════════════════════════════════
    chunk_level: Literal["parent", "child", "standard"] = Field(
        default="standard",
        description=(
            "'parent': Chunk lớn chứa ngữ cảnh rộng (dùng để LLM đọc). "
            "'child': Chunk nhỏ tách từ Parent (dùng để Vector Search). "
            "'standard': Chunk bình thường (không phân cấp)."
        ),
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="UUID của chunk cha. Chỉ có giá trị khi chunk_level='child'",
    )
    children_ids: List[str] = Field(
        default_factory=list,
        description="Danh sách UUID của các chunk con. Chỉ có giá trị khi chunk_level='parent'",
    )
    overlap_tokens: int = Field(
        default=0,
        description=(
            "Số token overlap giữa chunk này và chunk liền kề trước đó. "
            "Mặc định 100 tokens cho Semantic Chunking. Giúp văn cảnh không bị đứt gãy."
        ),
    )

    # ═══════════════════════════════════════════════════
    # LIFECYCLE — Validity & Version Control
    # ═══════════════════════════════════════════════════
    academic_year: Optional[str] = Field(
        default=None,
        description="Năm học áp dụng, VD: 2025-2026. Dùng để filter khi data nhiều năm",
    )
    valid_from: Optional[datetime] = Field(
        default=None,
        description="Thời điểm chunk bắt đầu có hiệu lực",
    )
    valid_until: Optional[datetime] = Field(
        default=None,
        description="Thời điểm chunk hết hạn. Chatbot tự exclude chunk quá hạn khi truy vấn",
    )
    is_active: bool = Field(
        default=True,
        description="True = chunk còn hiệu lực. Set False để soft-delete thay vì xóa",
    )
    version: int = Field(
        default=1,
        description="Phiên bản nội dung. Tăng khi cập nhật thông tin tuyển sinh mới",
    )
    replaced_by: Optional[str] = Field(
        default=None,
        description="chunk_id của chunk thay thế. Khi update, chunk cũ trỏ tới chunk mới",
    )

    # ═══════════════════════════════════════════════════
    # RETRIEVAL — Search & Filter Enhancement
    # ═══════════════════════════════════════════════════
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA256 hash của content. Dùng detect trùng lặp khi re-import",
    )
    token_count: int = Field(
        default=0,
        description="Số token ước tính của chunk. Dùng để filter/sort khi retrieval",
    )

    # ═══════════════════════════════════════════════════
    # EXTENSIBILITY — Future-proof
    # ═══════════════════════════════════════════════════
    extra: dict = Field(
        default_factory=dict,
        description="Dict mở rộng cho fields chưa dự kiến. VD: {'url': '...', 'keywords': [...]}",
    )


class ProcessedChunk(BaseModel):
    """
    A processed text chunk ready for embedding.
    
    Tự động tính:
      - char_count: số ký tự
      - metadata.content_hash: SHA256 hash (chống trùng lặp)
      - metadata.token_count: ước tính số token (~3.5 chars/token cho tiếng Việt)
    """
    content: str
    metadata: ChunkMetadata
    char_count: int = 0

    def model_post_init(self, __context):
        """Auto-fill các trường tính toán sau khi khởi tạo."""
        # Tính char_count
        if not self.char_count:
            self.char_count = len(self.content)

        # Tự động tính content_hash nếu chưa có (SHA256)
        if not self.metadata.content_hash:
            self.metadata.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()

        # Ước tính token_count nếu chưa có
        # Tiếng Việt BPE tokenizer: ~3.5 chars/token (kém hiệu quả hơn tiếng Anh)
        if not self.metadata.token_count:
            self.metadata.token_count = max(1, len(self.content) // 4)


class EmbeddingScore(BaseModel):
    """Quality evaluation of embeddings."""
    avg_score: float
    min_score: float
    max_score: float
    std_dev: float
    distribution: dict = Field(default_factory=dict)
    total_chunks: int = 0
