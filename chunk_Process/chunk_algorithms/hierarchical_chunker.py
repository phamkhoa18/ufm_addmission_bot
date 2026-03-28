"""
Hierarchical Chunker — Chia văn bản Markdown theo cấu trúc phân cấp (Parent-Child).

Thiết kế phối hợp chặt chẽ với SemanticChunkerBGE:
  - Tầng 1 (Hierarchical): Đọc cấu trúc Markdown (# H1, ## H2, ### H3...)
                            → Tạo Parent Chunks (ngữ cảnh rộng ~2000 tokens)
  - Tầng 2 (Semantic):     Đưa nội dung Parent vào SemanticChunkerBGE
                            → Tạo Child Chunks (nhỏ gọn ~100-300 tokens)
                            → Gắn parent_id vào metadata_extra

Luồng xử lý:
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. ĐỌC FILE MARKDOWN                                        │
  │     → Parse heading structure (H1 → H2 → H3 → body)          │
  │                                                                │
  │  2. XÂY DỰNG CÂY PHÂN CẤP (Heading Tree)                    │
  │     → Mỗi section = { heading, level, content, children }    │
  │                                                                │
  │  3. TẠO PARENT CHUNKS (Tầng 1)                               │
  │     → Gộp nội dung theo section, gắn section_path breadcrumb │
  │     → chunk_level = "parent"                                  │
  │                                                                │
  │  4. TẠO CHILD CHUNKS (Tầng 2 — gọi SemanticChunkerBGE)      │
  │     → Đưa nội dung parent → semantic_chunker.chunk()          │
  │     → chunk_level = "child", parent_id = UUID của parent      │
  │     → Parent nhận children_ids                                │
  │                                                                │
  │  5. TRẢ VỀ List[ProcessedChunk]                               │
  │     → Gồm cả Parent và Child chunks, sẵn sàng insert DB     │
  └─────────────────────────────────────────────────────────────────┘

Kỹ thuật đặc biệt cho Markdown:
  - Xử lý front-matter YAML (nếu có)
  - Nhận diện bảng Markdown, code block, danh sách
  - Merge sections quá nhỏ vào section liền kề
  - Breadcrumb (section_path) tự động: H1 > H2 > H3
"""

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from models.chunk import ChunkMetadata, ProcessedChunk
from chunk_Process.chunk_algorithms.utils import (
    normalize_vietnamese,
    estimate_tokens,
    split_sentences_vietnamese,
    is_markdown_table,
    parse_document_header,
    lookup_ma_nganh,
)


# ================================================================
# CẤU HÌNH MẶC ĐỊNH
# ================================================================
DEFAULT_HIERARCHICAL_CONFIG = {
    "min_parent_tokens": 80,        # Parent chunk tối thiểu (dưới → gộp)
    "max_parent_tokens": 2000,      # Parent chunk tối đa (~6000 chars)
    "min_child_tokens": 70,         # Child chunk tối thiểu (đồng bộ semantic)
    "max_child_tokens": 800,        # Child chunk tối đa (đồng bộ semantic)
    "overlap_tokens": 120,          # Overlap giữa các child chunks liền kề
    "chars_per_token": 3.0,         # Hệ số ước tính token (conservative cho tiếng Việt)
    "merge_threshold_tokens": 30,   # Section < 30 tokens → gộp vào section kế
    "max_heading_depth": 4,         # Chỉ parse đến #### (H4)
    "context_prefix": True,         # Gắn tiền tố ngữ cảnh cho mỗi chunk
}


# ================================================================
# HÀM TIỆN ÍCH — (hàm chung đã chuyển sang utils.py)
# ================================================================
# Alias nội bộ để không phá vỡ các call hiện có trong class
_normalize_vietnamese = normalize_vietnamese
_split_sentences_vietnamese = split_sentences_vietnamese
_is_markdown_table = is_markdown_table


def _estimate_tokens(text: str, chars_per_token: float = 0) -> int:
    """Alias bảo tương thích, trực tiếp gọi estimate_tokens từ utils."""
    return estimate_tokens(text, chars_per_token)


def _content_hash(text: str) -> str:
    """SHA256 hash cho dedup detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_table_preserve_header(
    table_text: str, max_chars: int
) -> List[str]:
    """
    Cắt bảng Markdown quá dài thành nhiều phần,
    MỖI PHẦN giữ nguyên dòng Header + Separator.

    Chiến lược:
      - Dòng 1: Header row  (| STT | Môn học | ...)
      - Dòng 2: Separator   (| --- | ------- | ...)
      - Dòng 3+: Data rows  → chia theo nhóm, ghim header lại

    Ví dụ output (bảng 50 dòng, max_chars cho phép 20 dòng):
      Phần 1: Header + Sep + Row 1-18
      Phần 2: Header + Sep + Row 19-36
      Phần 3: Header + Sep + Row 37-50
    """
    lines = table_text.strip().split("\n")
    if len(lines) < 3:
        return [table_text]

    # Tìm header và separator
    header_line = None
    separator_line = None
    data_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and re.match(r'^\|[\s\-:|]+\|', stripped):
            separator_line = line
            # Header là dòng ngay trước separator
            if i > 0:
                header_line = lines[i - 1]
                data_start = i + 1
            else:
                data_start = i + 1
            break

    # Nếu không tìm thấy separator → trả nguyên
    if separator_line is None:
        return [table_text]

    # Prefix = header + separator (sẽ ghim vào đầu mỗi sub-chunk)
    header_block = ""
    if header_line:
        header_block = header_line + "\n" + separator_line
    else:
        header_block = separator_line
    header_len = len(header_block) + 2  # +2 cho \n\n

    data_rows = lines[data_start:]
    # Nội dung trước bảng (nếu có text ở trên header)
    pre_table = "\n".join(lines[:max(0, data_start - 2)]).strip()

    if not data_rows:
        return [table_text]

    # Chia data rows thành các nhóm vừa vặn max_chars
    available_chars = max_chars - header_len
    if available_chars < 200:
        available_chars = 200

    groups: List[List[str]] = []
    current_group: List[str] = []
    current_len = 0

    for row in data_rows:
        row_len = len(row) + 1  # +1 cho \n
        if current_len + row_len > available_chars and current_group:
            groups.append(current_group)
            current_group = []
            current_len = 0
        current_group.append(row)
        current_len += row_len

    if current_group:
        groups.append(current_group)

    # Ghép mỗi nhóm với header
    result = []
    for group in groups:
        chunk = header_block + "\n" + "\n".join(group)
        if pre_table:
            chunk = pre_table + "\n\n" + chunk
        result.append(chunk)

    return result


# ================================================================
# CẤU TRÚC DỮ LIỆU — Section Node trong cây phân cấp
# ================================================================
@dataclass
class MarkdownSection:
    """
    Một node trong cây phân cấp Markdown.

    Ví dụ với file:
      # Tuyển sinh 2025          → level=1, heading="Tuyển sinh 2025"
      ## Điều kiện xét tuyển     → level=2, heading="Điều kiện xét tuyển"
      ### Đối tượng              → level=3, heading="Đối tượng"
    """
    level: int                              # Heading level (1=H1, 2=H2, ...)
    heading: str                            # Nội dung heading (không có dấu #)
    content: str = ""                       # Nội dung body (dưới heading)
    children: List["MarkdownSection"] = field(default_factory=list)

    def full_content(self, include_children: bool = True) -> str:
        """
        Lấy toàn bộ nội dung gồm heading + body + children (đệ quy).
        Dùng để tạo Parent Chunk.
        """
        parts = []
        if self.heading:
            prefix = "#" * self.level + " "
            parts.append(prefix + self.heading)
        if self.content.strip():
            parts.append(self.content.strip())
        if include_children:
            for child in self.children:
                parts.append(child.full_content(include_children=True))
        return "\n\n".join(parts)

    def body_only(self) -> str:
        """
        Chỉ lấy nội dung body (không heading, không children).
        Dùng khi cần nội dung riêng của section này.
        """
        return self.content.strip()

    def flat_sections(self) -> List["MarkdownSection"]:
        """Trả về danh sách phẳng tất cả sections (DFS)."""
        result = [self]
        for child in self.children:
            result.extend(child.flat_sections())
        return result


# ================================================================
# LỚP CHÍNH: HierarchicalChunker
# ================================================================
class HierarchicalChunker:
    """
    Hierarchical Chunker cho file Markdown.

    Tạo Parent Chunks theo cấu trúc heading, sau đó (tùy chọn) gọi
    SemanticChunkerBGE để tạo Child Chunks bên trong mỗi Parent.

    Có 2 chế độ sử dụng:
      1. Chỉ tạo Parent Chunks:
         chunks = chunker.chunk(text, source="file.md")

      2. Tạo Parent + Child Chunks (phối hợp Semantic):
         chunks = chunker.chunk_with_semantic(
             text, source="file.md",
             semantic_chunker=my_semantic_chunker
         )

    Ví dụ:
        chunker = HierarchicalChunker()
        # Chế độ 1: Chỉ Hierarchical
        parents = chunker.chunk(text="...", source="tuyen_sinh.md")

        # Chế độ 2: Hierarchical + Semantic (Parent-Child)
        from chunk_Process.chunk_algorithms.semantic_chunker import SemanticChunkerBGE
        semantic = SemanticChunkerBGE(api_key="sk-or-v1-...")
        all_chunks = chunker.chunk_with_semantic(
            text="...", source="tuyen_sinh.md",
            semantic_chunker=semantic
        )
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Dict ghi đè cấu hình mặc định (xem DEFAULT_HIERARCHICAL_CONFIG)
        """
        self.cfg = {**DEFAULT_HIERARCHICAL_CONFIG, **(config or {})}

    # ================================================================
    # BƯỚC 1: PARSE MARKDOWN → CÂY PHÂN CẤP
    # ================================================================
    def _strip_frontmatter(self, text: str) -> str:
        """
        Loại bỏ YAML front-matter (nếu có).
        Front-matter nằm giữa 2 dấu --- ở đầu file.

        Ví dụ:
          ---
          title: Tuyển sinh 2025
          date: 2025-01-15
          ---
          # Nội dung chính...
        """
        pattern = r"^---\s*\n.*?\n---\s*\n"
        return re.sub(pattern, "", text, count=1, flags=re.DOTALL)

    def _parse_heading_line(self, line: str) -> Optional[Tuple[int, str, str]]:
        """
        Parse một dòng heading Markdown (bao gồm pseudo-heading bold).

        Nhận diện 2 dạng:
          1. Heading chuẩn:  # / ## / ### ... (ký tự # đầu dòng)
          2. Pseudo-heading: **X. Tiêu đề** hoặc **X. Tiêu đề:**
             (Dòng bắt đầu bằng bold text với số thứ tự → coi như heading)
             Có thể có text theo sau trên cùng một dòng.

        Returns:
            (level, heading_text, remaining_text) hoặc None nếu không phải heading.

        Ví dụ:
            "# Tuyển sinh"          → (1, "Tuyển sinh", "")
            "**4. Thời gian** là 2 năm" → (max_depth, "4. Thời gian", "là 2 năm")
        """
        stripped = line.strip()

        # ── Dạng 1: Heading Markdown chuẩn ──
        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            if level > self.cfg["max_heading_depth"]:
                return None
            return (level, heading_text, "")

        # ── Dạng 2: Pseudo-heading bằng Bold + Số thứ tự ──
        # Pattern: Bắt đầu bằng **X. Tiêu đề** (hoặc **X.Y. Tiêu đề:**), sau đó có thể có text
        bold_match = re.match(
            r"^\*\*((\d+[\.\)])+\s*[^(\*\*)]+?)\*\*(.*)$",
            stripped,
        )
        if bold_match:
            heading_text = bold_match.group(1).strip().rstrip(":")
            remaining_text = bold_match.group(3).strip()
            # Xác định level từ độ sâu số thứ tự:
            #   "4. Title"     → 1 cấp số → level 2 (cùng cấp ##)
            #   "3.1. Title"   → 2 cấp số → level 3 (cùng cấp ###)
            number_prefix = re.match(r"^([\d\.\)]+)", heading_text)
            if number_prefix:
                num_parts = [p for p in re.split(r'[\.\)]', number_prefix.group(1)) if p.strip()]
                pseudo_level = min(len(num_parts) + 1, self.cfg["max_heading_depth"])
            else:
                pseudo_level = 2
            return (pseudo_level, heading_text, remaining_text)

        return None

    def _is_inside_code_block(self, lines: List[str], current_idx: int) -> bool:
        """
        Kiểm tra xem dòng hiện tại có nằm trong code block (```) hay không.
        Heading trong code block sẽ được bỏ qua (không phải heading thật).
        """
        fence_count = 0
        for i in range(current_idx):
            stripped = lines[i].strip()
            if stripped.startswith("```"):
                fence_count += 1
        # Lẻ → đang ở trong code block
        return fence_count % 2 == 1

    def parse_markdown(self, text: str) -> List[MarkdownSection]:
        """
        Parse văn bản Markdown thành cây phân cấp MarkdownSection.

        Thuật toán:
          1. Loại bỏ front-matter YAML
          2. Duyệt từng dòng, nhận diện heading (bỏ qua heading trong code block)
          3. Gom nội dung body vào section tương ứng
          4. Xây dựng cây phân cấp bằng stack

        Returns:
            List[MarkdownSection] — Các section cấp cao nhất (roots)
        """
        text = _normalize_vietnamese(text)
        text = self._strip_frontmatter(text)
        lines = text.split("\n")

        # Parse tất cả sections (phẳng, chưa phân cấp)
        flat_sections: List[MarkdownSection] = []
        current_heading = None
        current_level = 0
        content_lines: List[str] = []
        preamble_lines: List[str] = []  # Nội dung trước heading đầu tiên

        for idx, line in enumerate(lines):
            # Bỏ qua heading trong code block
            if self._is_inside_code_block(lines, idx):
                content_lines.append(line)
                continue

            heading_info = self._parse_heading_line(line)

            if heading_info:
                level, heading_text, remaining = heading_info

                # Lưu section trước đó (nếu có)
                if current_heading is not None:
                    flat_sections.append(MarkdownSection(
                        level=current_level,
                        heading=current_heading,
                        content="\n".join(content_lines).strip(),
                    ))
                elif content_lines:
                    # Nội dung trước heading đầu tiên (preamble)
                    preamble_lines = content_lines.copy()

                # Bắt đầu section mới
                current_heading = heading_text
                current_level = level
                content_lines = []
                if remaining:
                    content_lines.append(remaining)
            else:
                content_lines.append(line)

        # Section / Nội dung cuối cùng
        if current_heading is not None:
            flat_sections.append(MarkdownSection(
                level=current_level,
                heading=current_heading,
                content="\n".join(content_lines).strip(),
            ))
        elif content_lines:
            preamble_lines = content_lines

        # Nếu có preamble (nội dung trước heading đầu tiên) → tạo section ảo
        if preamble_lines:
            preamble_content = "\n".join(preamble_lines).strip()
            if preamble_content:
                flat_sections.insert(0, MarkdownSection(
                    level=0,
                    heading="Giới thiệu",
                    content=preamble_content,
                ))

        # Xây dựng cây phân cấp từ flat list
        return self._build_tree(flat_sections)

    def _build_tree(self, flat_sections: List[MarkdownSection]) -> List[MarkdownSection]:
        """
        Xây dựng cây phân cấp từ danh sách phẳng các MarkdownSection.

        Sử dụng stack-based algorithm:
          - Duyệt từng section
          - Nếu level > stack top → section là con của top → push vào children
          - Nếu level <= stack top → pop stack cho đến khi tìm được cha phù hợp

        Ví dụ:
          H1: A, H2: B, H2: C, H3: D → A(children=[B, C(children=[D])])
        """
        if not flat_sections:
            return []

        roots: List[MarkdownSection] = []
        stack: List[MarkdownSection] = []

        for section in flat_sections:
            # Pop stack cho đến khi stack rỗng hoặc top có level < section.level
            while stack and stack[-1].level >= section.level:
                stack.pop()

            if stack:
                # Section là con của top
                stack[-1].children.append(section)
            else:
                # Section là root (không có cha)
                roots.append(section)

            stack.append(section)

        return roots

    # ================================================================
    # BƯỚC 2: TẠO PARENT CHUNKS TỪ CÂY PHÂN CẤP
    # ================================================================
    def _build_breadcrumb(self, ancestors: List[str]) -> str:
        """
        Tạo breadcrumb path từ danh sách tổ tiên.

        Ví dụ:
            ["Tuyển sinh 2025", "Thạc sĩ", "Điều kiện"] 
            → "Tuyển sinh 2025 > Thạc sĩ > Điều kiện"
        """
        return " > ".join(ancestors)

    def _build_context_prefix(
        self, breadcrumb: str, source: str, extra: Optional[dict] = None
    ) -> str:
        """Tạo prefix ngữ cảnh cho embedding quality."""
        if not self.cfg["context_prefix"]:
            return ""

        parts = []
        # Ưu tiên Header Context (thông báo/phụ lục) nếu có
        # header_context nằm trong extra["extra"] (Pydantic extra dict)
        header_ctx = None
        if extra:
            header_ctx = extra.get("header_context")  # direct (backward compat)
            if not header_ctx and isinstance(extra.get("extra"), dict):
                header_ctx = extra["extra"].get("header_context")
        if header_ctx:
            parts.append(header_ctx)
            
        parts.append(f"[Nguồn: {source}]")
        if breadcrumb:
            parts.append(f"[Mục: {breadcrumb}]")
        return "\n".join(parts) + "\n\n"

    def _collect_parent_chunks(
        self,
        sections: List[MarkdownSection],
        source: str,
        ancestors: Optional[List[str]] = None,
        metadata_extra: Optional[dict] = None,
    ) -> List[ProcessedChunk]:
        """
        Duyệt cây phân cấp (DFS) và tạo Parent Chunks.

        Chiến lược:
          - Mỗi section có nội dung đủ lớn → 1 Parent Chunk
          - Section quá nhỏ (< merge_threshold) → gộp vào section liền kề
          - Section quá lớn (> max_parent_tokens) → cắt tại ranh giới đoạn văn
          - Breadcrumb tự động từ cây tổ tiên

        Args:
            sections: List các MarkdownSection ở cùng cấp
            source: Tên file nguồn
            ancestors: Danh sách heading tổ tiên (cho breadcrumb)
            metadata_extra: Metadata bổ sung (program_name, academic_year...)

        Returns:
            List[ProcessedChunk] — Các Parent Chunks
        """
        if ancestors is None:
            ancestors = []

        extra = metadata_extra or {}
        chunks: List[ProcessedChunk] = []
        max_parent_chars = int(self.cfg["max_parent_tokens"] * self.cfg["chars_per_token"])
        min_parent_chars = int(self.cfg["min_parent_tokens"] * self.cfg["chars_per_token"])
        merge_threshold_chars = int(self.cfg["merge_threshold_tokens"] * self.cfg["chars_per_token"])

        # Pending small section — chờ gộp vào section tiếp theo
        pending_content = ""
        pending_headings: List[str] = []  # Danh sách heading đã gộp

        for section in sections:
            current_ancestors = ancestors + [section.heading] if section.heading else ancestors
            breadcrumb = self._build_breadcrumb(current_ancestors)
            section_name = section.heading or "Giới thiệu"

            # Nội dung riêng (body, không children)
            body = section.body_only()

            # Tổng nội dung (body + children recursive)
            full = section.full_content(include_children=True)
            full_chars = len(full)

            # ── TRƯỜNG HỢP 1: Section có children → Đệ quy vào children ──
            if section.children:
                # Nếu body riêng đủ lớn → tạo parent chunk cho body
                if body and len(body) >= min_parent_chars:
                    # Gộp pending vào trước nếu có
                    if pending_content:
                        body = pending_content + "\n\n" + body
                        pending_headings.append(section_name)
                        merged_name = " & ".join(pending_headings)
                        merged_breadcrumb = self._build_breadcrumb(
                            ancestors + [merged_name]
                        )
                        pending_content = ""
                        pending_headings = []
                    else:
                        merged_name = section_name
                        merged_breadcrumb = breadcrumb

                    parent_chunks = self._create_parent_chunk_with_split(
                        content=body,
                        breadcrumb=merged_breadcrumb,
                        section_name=merged_name,
                        source=source,
                        max_chars=max_parent_chars,
                        extra=extra,
                    )
                    chunks.extend(parent_chunks)
                elif body:
                    # Body quá nhỏ — chờ gộp, kèm heading
                    pending_content = (pending_content + "\n\n" + body).strip() if pending_content else body
                    if section.heading:
                        pending_headings.append(section.heading)

                # Đệ quy vào children
                child_chunks = self._collect_parent_chunks(
                    sections=section.children,
                    source=source,
                    ancestors=current_ancestors,
                    metadata_extra=metadata_extra,
                )
                chunks.extend(child_chunks)
                continue

            # ── TRƯỜNG HỢP 2: Section lá (không children) ──
            if not body:
                # Heading-only section → gộp heading vào pending
                if section.heading:
                    pending_content = (
                        (pending_content + "\n\n" + section.heading).strip()
                        if pending_content else section.heading
                    )
                    pending_headings.append(section.heading)
                continue

            # Section quá nhỏ → chờ gộp, LƯU HEADING
            if len(body) < merge_threshold_chars:
                pending_content = (pending_content + "\n\n" + body).strip() if pending_content else body
                if section.heading:
                    pending_headings.append(section.heading)
                continue

            # Section đủ lớn → tạo chunk
            # Gộp pending vào trước nếu có
            if pending_content:
                body = pending_content + "\n\n" + body
                pending_headings.append(section_name)
                merged_name = " & ".join(pending_headings)
                merged_breadcrumb = self._build_breadcrumb(
                    ancestors + [merged_name]
                )
                pending_content = ""
                pending_headings = []
            else:
                merged_name = section_name
                merged_breadcrumb = breadcrumb

            # Tạo parent chunk (cắt nếu quá lớn)
            parent_chunks = self._create_parent_chunk_with_split(
                content=body,
                breadcrumb=merged_breadcrumb,
                section_name=merged_name,
                source=source,
                max_chars=max_parent_chars,
                extra=extra,
            )
            chunks.extend(parent_chunks)

        # Flush nội dung pending cuối cùng
        if pending_content:
            merged_name = " & ".join(pending_headings) if pending_headings else "Phần bổ sung"
            # Gộp vào chunk cuối nếu vừa đủ
            if chunks and len(chunks[-1].content) + len(pending_content) <= max_parent_chars:
                prev = chunks[-1]
                merged = prev.content + "\n\n" + pending_content
                # Cập nhật section_name để phản ánh đã gộp
                old_name = prev.metadata.section_name or ""
                new_name = old_name + " & " + merged_name if old_name else merged_name
                prev.metadata.section_name = new_name
                prev.metadata.section_path = self._build_breadcrumb(
                    ancestors + [new_name]
                )
                chunks[-1] = ProcessedChunk(
                    content=merged,
                    metadata=prev.metadata,
                )
            else:
                merged_breadcrumb = self._build_breadcrumb(ancestors + [merged_name])
                parent_chunks = self._create_parent_chunk_with_split(
                    content=pending_content,
                    breadcrumb=merged_breadcrumb,
                    section_name=merged_name,
                    source=source,
                    max_chars=max_parent_chars,
                    extra=extra,
                )
                chunks.extend(parent_chunks)

        return chunks

    def _create_parent_chunk_with_split(
        self,
        content: str,
        breadcrumb: str,
        section_name: str,
        source: str,
        max_chars: int,
        extra: dict,
    ) -> List[ProcessedChunk]:
        """
        Tạo 1 hoặc nhiều Parent Chunks từ nội dung.

        Chiến lược cắt ưu tiên (Production-safe):
          1. Nếu nội dung vừa vặn max_chars → trả 1 chunk nguyên vẹn
          2. Tách nội dung thành các "paragraph blocks" bằng \n\n
          3. Nhận diện Markdown Tables → KHÔNG BAO GIỜ cắt ngang bảng
             - Bảng nhỏ: giữ nguyên khối (atomic block)
             - Bảng quá dài: cắt theo ROW, ghim lại Header mỗi phần
          4. Paragraph thường quá dài → cắt theo ranh giới câu
        """
        prefix = self._build_context_prefix(breadcrumb, source, extra)
        effective_max = max_chars - len(prefix)
        if effective_max < 500:
            effective_max = 500

        # ── Nội dung vừa đủ → 1 chunk ──
        if len(content) <= effective_max:
            full_content = prefix + content
            meta = ChunkMetadata(
                source=source,
                section_path=breadcrumb,
                section_name=section_name,
                chunk_level="parent",
                chunk_index=1,
                total_chunks_in_section=1,
                **extra,
            )
            return [ProcessedChunk(content=full_content, metadata=meta)]

        # ── Nội dung quá lớn → Tách thành paragraph blocks ──
        # Dùng regex giữ bảng Markdown nguyên khối (không split giữa các dòng |)
        raw_paragraphs = self._split_paragraphs_table_aware(content)

        groups: List[List[str]] = []
        current_group: List[str] = []
        current_len = 0

        for para in raw_paragraphs:
            para_len = len(para)

            # ── CAS 1: Paragraph là một BẢNG MARKDOWN ──
            if _is_markdown_table(para):
                # Bảng nhỏ → giữ nguyên khối (atomic)
                if para_len <= effective_max:
                    if current_len + para_len + 2 > effective_max and current_group:
                        groups.append(current_group)
                        current_group = []
                        current_len = 0
                    current_group.append(para)
                    current_len += para_len + 2
                else:
                    # Bảng quá lớn → cắt theo ROW, ghim Header
                    if current_group:
                        groups.append(current_group)
                        current_group = []
                        current_len = 0

                    table_parts = _split_table_preserve_header(para, effective_max)
                    for tp in table_parts:
                        groups.append([tp])
                continue

            # ── CAS 2: Paragraph thường nhưng quá dài → cắt theo câu ──
            if para_len > effective_max:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_len = 0

                sentences = _split_sentences_vietnamese(para)
                sent_group: List[str] = []
                sent_len = 0
                for sent in sentences:
                    if sent_len + len(sent) > effective_max and sent_group:
                        groups.append([" ".join(sent_group)])
                        sent_group = []
                        sent_len = 0
                    sent_group.append(sent)
                    sent_len += len(sent)
                if sent_group:
                    groups.append([" ".join(sent_group)])
                continue

            # ── CAS 3: Paragraph bình thường ──
            if current_len + para_len + 2 > effective_max and current_group:
                groups.append(current_group)
                current_group = []
                current_len = 0

            current_group.append(para)
            current_len += para_len + 2

        if current_group:
            groups.append(current_group)

        # ── Tạo ProcessedChunk objects ──
        total = len(groups)
        chunks: List[ProcessedChunk] = []

        for idx, group in enumerate(groups, 1):
            group_content = "\n\n".join(group)
            if total > 1:
                part_label = f"[Phần {idx}/{total}]\n"
                full_content = prefix + part_label + group_content
            else:
                full_content = prefix + group_content

            meta = ChunkMetadata(
                source=source,
                section_path=breadcrumb,
                section_name=section_name,
                chunk_level="parent",
                chunk_index=idx,
                total_chunks_in_section=total,
                **extra,
            )
            chunks.append(ProcessedChunk(content=full_content, metadata=meta))

        return chunks

    @staticmethod
    def _split_paragraphs_table_aware(content: str) -> List[str]:
        """
        Tách content thành danh sách paragraphs, nhưng GIỮ NGUYÊN bảng Markdown.

        Thuật toán:
          - Duyệt từng dòng
          - Khi gặp dòng bắt đầu bằng '|' → bắt đầu tích lũy vào table_block
          - Khi dòng không bắt đầu bằng '|' mà đang trong table_block → flush table
          - Dòng trống → phân cách paragraph thường

        Ví dụ input:
          'Nội dung mở đầu\n\n| A | B |\n| - | - |\n| 1 | 2 |\n\nKết luận'
        Output:
          ['Nội dung mở đầu', '| A | B |\n| - | - |\n| 1 | 2 |', 'Kết luận']
        """
        lines = content.split("\n")
        paragraphs: List[str] = []
        current_text_lines: List[str] = []
        table_lines: List[str] = []
        in_table = False

        def flush_text():
            nonlocal current_text_lines
            if current_text_lines:
                text = "\n".join(current_text_lines).strip()
                if text:
                    # Tách thêm theo \n\n cho paragraph thường
                    sub_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
                    paragraphs.extend(sub_paras)
                current_text_lines = []

        def flush_table():
            nonlocal table_lines, in_table
            if table_lines:
                table_text = "\n".join(table_lines).strip()
                if table_text:
                    paragraphs.append(table_text)
                table_lines = []
            in_table = False

        for line in lines:
            stripped = line.strip()
            is_table_line = stripped.startswith("|")

            if is_table_line:
                if not in_table:
                    # Bắt đầu bảng mới → flush text trước
                    flush_text()
                    in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    # Dòng trống ngay sau bảng → flush bảng
                    if not stripped:
                        flush_table()
                    else:
                        # Dòng không phải bảng nhưng đang trong bảng
                        # → flush bảng, bắt đầu text mới
                        flush_table()
                        current_text_lines.append(line)
                else:
                    current_text_lines.append(line)

        # Flush remaining
        flush_table()
        flush_text()

        return paragraphs

    # ================================================================
    # BƯỚC 3: API CÔNG KHAI — chunk() và chunk_with_semantic()
    # ================================================================
    def chunk(
        self,
        text: str,
        source: str,
        metadata_extra: Optional[dict] = None,
    ) -> List[ProcessedChunk]:
        """
        Tạo Parent Chunks từ file Markdown (Chỉ Tầng 1 — Hierarchical).

        Tự động phân tích Header trước `-start-` để trích xuất:
          - valid_from (Ngày hiệu lực)
          - program_level (thac_si / tien_si / dai_hoc)
          - academic_year (năm tuyển sinh)
          - header_context (ngữ cảnh gốc: tiêu đề + thông báo)

        Args:
            text: Nội dung Markdown thô (có thể chứa Header + `-start-`)
            source: Tên file nguồn (VD: "tuyen_sinh_2025.md")
            metadata_extra: Dict metadata bổ sung, ghi đè auto-parsed values
                (program_name, academic_year, ma_nganh, ...)

        Returns:
            List[ProcessedChunk] — Chỉ chứa Parent Chunks (chunk_level="parent")
        """
        if not text or not text.strip():
            return []

        # ── Auto-parse Header metadata ──
        parsed = parse_document_header(text)
        content = parsed["content"]  # Noi dung SAU -start-

        # Xay dung metadata tu dong tu header
        auto_meta = {}
        if parsed["valid_from"]:
            auto_meta["valid_from"] = parsed["valid_from"]
        if parsed["program_level"]:
            auto_meta["program_level"] = parsed["program_level"]
        if parsed["academic_year"]:
            auto_meta["academic_year"] = parsed["academic_year"]

        # Nap TAT CA fields moi tu YAML Frontmatter vao extra dict
        # (doc_type, doc_id, doc_number, keywords, source_url, ...)
        # Dat vao extra{} de map voi DB column `extra JSONB`
        header_extra = parsed.get("extra", {})
        if parsed.get("header_context"):
            header_extra["header_context"] = parsed["header_context"]
        if header_extra:
            auto_meta.setdefault("extra", {}).update(header_extra)

        # Tra cuu ma nganh neu co program_name
        extra = metadata_extra or {}
        program_name = extra.get("program_name")
        program_level = extra.get("program_level") or auto_meta.get("program_level")
        if program_name and program_level:
            ma_nganh = lookup_ma_nganh(program_level, program_name)
            if ma_nganh:
                auto_meta["ma_nganh"] = ma_nganh

        # Merge: auto_meta (nen) <- metadata_extra (uu tien user)
        # Deep-merge extra dict de khong mat header fields (doc_type, doc_id...)
        merged_extra = {**auto_meta, **extra}
        if "extra" in auto_meta and "extra" in extra:
            merged_extra["extra"] = {**auto_meta["extra"], **extra["extra"]}

        # Parse cấu trúc Markdown (chỉ phần content, không header)
        tree = self.parse_markdown(content)

        if not tree:
            return []

        # Tạo Parent Chunks từ cây phân cấp
        parent_chunks = self._collect_parent_chunks(
            sections=tree,
            source=source,
            metadata_extra=merged_extra,
        )

        # Đánh lại index tổng thể
        for i, chunk in enumerate(parent_chunks, 1):
            chunk.metadata.chunk_index = i
            chunk.metadata.total_chunks_in_section = len(parent_chunks)

        return parent_chunks

    def chunk_with_semantic(
        self,
        text: str,
        source: str,
        semantic_chunker,
        metadata_extra: Optional[dict] = None,
        use_fallback: bool = False,
    ) -> List[ProcessedChunk]:
        """
        Tạo Parent + Child Chunks (Hierarchical + Semantic — Parent-Child Retrieval).

        Đây là chế độ mạnh nhất, phối hợp cả 2 thuật toán:
          1. Hierarchical: Tạo Parent Chunks theo cấu trúc heading
          2. Semantic: Lấy nội dung mỗi Parent → SemanticChunkerBGE.chunk()
                       → Tạo Child Chunks gắn parent_id

        Khi Vector Search:
          - Search trên Child Chunks (nhỏ, semantic-dense → chính xác cao)
          - Truy ngược parent_id → lấy Parent Chunk (lớn, ngữ cảnh đầy đủ)
          - Đưa Parent content cho LLM → trả lời chất lượng cao

        Args:
            text: Nội dung Markdown thô
            source: Tên file nguồn
            semantic_chunker: Instance của SemanticChunkerBGE (đã có api_key)
            metadata_extra: Dict metadata bổ sung
            use_fallback: True → dùng chunk_fallback() thay vì chunk()
                          (khi API Embedding không khả dụng)

        Returns:
            List[ProcessedChunk] — Gồm cả Parent (chunk_level="parent")
                                    và Child (chunk_level="child") chunks

        Ví dụ sử dụng:
            from chunk_Process.chunk_algorithms.semantic_chunker import SemanticChunkerBGE
            from chunk_Process.chunk_algorithms.hierarchical_chunker import HierarchicalChunker

            semantic = SemanticChunkerBGE(api_key="sk-or-v1-...")
            hierarchical = HierarchicalChunker()

            with open("tuyen_sinh.md", "r", encoding="utf-8") as f:
                text = f.read()

            all_chunks = hierarchical.chunk_with_semantic(
                text=text,
                source="tuyen_sinh.md",
                semantic_chunker=semantic,
                metadata_extra={"academic_year": "2025-2026"}
            )

            parents = [c for c in all_chunks if c.metadata.chunk_level == "parent"]
            children = [c for c in all_chunks if c.metadata.chunk_level == "child"]
        """
        if not text or not text.strip():
            return []

        # ── Bước 1: Tạo Parent Chunks ──
        parent_chunks = self.chunk(
            text=text,
            source=source,
            metadata_extra=metadata_extra,
        )

        if not parent_chunks:
            return []

        all_chunks: List[ProcessedChunk] = []

        # ── Bước 2: Với mỗi Parent → tạo Children bằng Semantic Chunker ──
        for parent in parent_chunks:
            parent_id = parent.metadata.chunk_id

            # Lấy nội dung thuần (bỏ prefix nếu có)
            parent_content = parent.content

            # Tạo metadata_extra cho children, kế thừa toàn bộ từ parent.metadata
            child_extra = {
                "parent_id": parent_id,
                "chunk_level": "child",
                "section_path": parent.metadata.section_path,
                "section_name": parent.metadata.section_name,
                "program_level": parent.metadata.program_level,
                "valid_from": parent.metadata.valid_from,
                "academic_year": parent.metadata.academic_year,
                "ma_nganh": parent.metadata.ma_nganh,
                "program_name": parent.metadata.program_name,
                "extra": parent.metadata.extra.copy() if parent.metadata.extra else {}
            }

            # Gọi Semantic Chunker
            if use_fallback:
                child_chunks = semantic_chunker.chunk_fallback(
                    text=parent_content,
                    source=source,
                    metadata_extra=child_extra,
                )
            else:
                child_chunks = semantic_chunker.chunk(
                    text=parent_content,
                    source=source,
                    metadata_extra=child_extra,
                )

            # Gắn children_ids vào parent
            children_ids = [c.metadata.chunk_id for c in child_chunks]
            parent.metadata.children_ids = children_ids

            # Thêm parent vào danh sách
            all_chunks.append(parent)

            # Thêm tất cả children
            all_chunks.extend(child_chunks)

        return all_chunks

    # ================================================================
    # TIỆN ÍCH: Đọc file Markdown và chunk
    # ================================================================
    def chunk_file(
        self,
        filepath: str,
        metadata_extra: Optional[dict] = None,
        semantic_chunker=None,
        use_fallback: bool = False,
    ) -> List[ProcessedChunk]:
        """
        Đọc file Markdown từ disk và chunk.

        Tự động chọn chế độ:
          - Có semantic_chunker → chunk_with_semantic()
          - Không có → chunk() (chỉ Parent)

        Args:
            filepath: Đường dẫn file .md
            metadata_extra: Metadata bổ sung
            semantic_chunker: (Optional) SemanticChunkerBGE instance
            use_fallback: Dùng fallback chunking cho Semantic

        Returns:
            List[ProcessedChunk]
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File không tồn tại: {filepath}")

        if not path.suffix.lower() == ".md":
            raise ValueError(
                f"HierarchicalChunker chỉ hỗ trợ file Markdown (.md). "
                f"Nhận được: {path.suffix}"
            )

        text = path.read_text(encoding="utf-8")
        source = path.name

        if semantic_chunker:
            return self.chunk_with_semantic(
                text=text,
                source=source,
                semantic_chunker=semantic_chunker,
                metadata_extra=metadata_extra,
                use_fallback=use_fallback,
            )
        else:
            return self.chunk(
                text=text,
                source=source,
                metadata_extra=metadata_extra,
            )

    # ================================================================
    # THỐNG KÊ & DEBUG
    # ================================================================
    def get_tree_summary(self, text: str) -> str:
        """
        Trả về chuỗi mô tả cây cấu trúc Markdown (dạng tree view).
        Hữu ích cho debugging.

        Ví dụ output:
          📄 Tuyển sinh 2025
          ├── 📁 Thạc sĩ KDQT
          │   ├── 📝 Điều kiện xét tuyển (450 chars)
          │   ├── 📝 Hồ sơ cần nộp (380 chars)
          │   └── 📝 Lịch tuyển sinh (200 chars)
          └── 📁 Thạc sĩ Marketing
              └── 📝 Điều kiện xét tuyển (500 chars)
        """
        tree = self.parse_markdown(text)
        lines = []
        for section in tree:
            self._tree_to_string(section, lines, prefix="", is_last=True)
        return "\n".join(lines)

    def _tree_to_string(
        self,
        section: MarkdownSection,
        lines: List[str],
        prefix: str = "",
        is_last: bool = True,
    ):
        """Đệ quy xây dựng tree view string."""
        connector = "└── " if is_last else "├── "
        icon = "📁" if section.children else "📝"
        content_len = len(section.body_only())
        token_est = _estimate_tokens(section.body_only())

        label = f"{section.heading or '(no heading)'}"
        if content_len > 0:
            label += f" ({token_est} tokens, {content_len} chars)"

        lines.append(f"{prefix}{connector}{icon} {label}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(section.children):
            self._tree_to_string(
                child, lines,
                prefix=child_prefix,
                is_last=(i == len(section.children) - 1),
            )
