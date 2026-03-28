"""
Semantic Chunker — Embedding-based chunking with configurable provider.
Config: app/core/config/chunker_config.yaml
"""

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from models.chunk import ChunkMetadata, ProcessedChunk
from chunk_Process.chunk_algorithms.utils import (
    normalize_vietnamese,
    clean_whitespace,
    estimate_tokens,
    split_sentences_vietnamese,
    is_markdown_table,
    parse_document_header,
    lookup_ma_nganh,
    build_context_prefix,
)


# ================================================================
# YAML CONFIG LOADER
# ================================================================
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "app" / "core" / "config" / "yaml" / "chunker_config.yaml"

# Fallback nếu YAML không tồn tại hoặc thiếu key
_FALLBACK_CONFIG = {
    "provider": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "baai/bge-m3",
    "dimensions": 1024,
    "similarity_threshold": 0.57,
    "overlap_tokens": 120,
    "base_block_tokens": 100,
    "min_chunk_tokens": 70,
    "max_chunk_tokens": 800,
    "max_tokens_per_api_call": 6500,
    "api_batch_size": 40,
    "api_timeout": 45,
    "api_max_retries": 3,
    "api_retry_base_wait": 2,
}


def _load_chunker_config() -> dict:
    """Đọc semantic_chunker section từ chunker_config.yaml, merge với fallback."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        yaml_cfg = raw.get("semantic_chunker", {})
        return {**_FALLBACK_CONFIG, **yaml_cfg}
    return _FALLBACK_CONFIG.copy()


# Provider → Env var mapping
_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
}

# Alias nội bộ
_normalize_vietnamese = normalize_vietnamese
_clean_whitespace = clean_whitespace
_estimate_tokens = estimate_tokens
_split_sentences_vietnamese = split_sentences_vietnamese


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine Similarity giữa 2 vector."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ================================================================
# LỚP CHÍNH
# ================================================================
class SemanticChunkerBGE:
    """
    Semantic Chunker với config từ YAML.
    Provider/model được cấu hình trong chunker_config.yaml.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        config: Optional[dict] = None,
    ):
        # Load YAML config → merge user overrides
        yaml_cfg = _load_chunker_config()
        self.cfg = {**yaml_cfg, **(config or {})}

        # Resolve API key: param > env var > error
        provider = self.cfg.get("provider", "openrouter")
        env_key = _API_KEY_ENV.get(provider, "OPENROUTER_API_KEY")
        self.api_key = api_key or os.environ.get(env_key, "")

        # Resolve base_url: param > yaml > fallback
        self.base_url = (base_url or self.cfg.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/")

        # Runtime stats
        self.stats = {
            "total_api_calls": 0,
            "total_tokens_sent": 0,
            "total_time_embedding": 0.0,
        }

    # ================================================================
    # BƯỚC 1: TÁCH TEXT THÀNH BASE BLOCKS
    # ================================================================
    def _split_into_base_blocks(self, text: str) -> List[str]:
        """
        Tách văn bản thành các Base Blocks nhỏ (~100 tokens/block).

        Ưu tiên cắt theo ranh giới câu để giữ nguyên ý nghĩa.
        Nếu 1 câu quá dài → cắt theo ký tự.

        🛡️ TABLE GUARD: Nếu toàn bộ text là Markdown Table → không tách,
        trả về nguyên 1 block để bảo toàn cấu trúc bảng.
        """
        # ── TABLE GUARD: Bảo vệ bảng Markdown khỏi bị tách nát ──
        # Bảng không có dấu chấm câu cuối dòng → split_sentences sẽ thất bại
        # và hard-cut giữa chừng làm nát hàng/cột.
        if is_markdown_table(text):
            return [text.strip()] if text.strip() else []

        sentences = _split_sentences_vietnamese(text)
        if not sentences:
            return [text] if text.strip() else []

        target_chars = int(self.cfg["base_block_tokens"] * 3.0)  # ~300 chars/block (conservative cho mixed text)
        blocks = []
        current_block = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)

            # Câu quá dài → cắt nhỏ theo ký tự
            if sent_len > target_chars * 2:
                # Flush block hiện tại trước
                if current_block:
                    blocks.append(" ".join(current_block))
                    current_block = []
                    current_len = 0

                # Cắt câu dài thành nhiều phần
                for i in range(0, sent_len, target_chars):
                    part = sent[i:i + target_chars].strip()
                    if part:
                        blocks.append(part)
                continue

            # Block hiện tại đã đủ lớn → flush
            if current_len + sent_len > target_chars and current_block:
                blocks.append(" ".join(current_block))
                current_block = []
                current_len = 0

            current_block.append(sent)
            current_len += sent_len

        # Block cuối cùng
        if current_block:
            blocks.append(" ".join(current_block))

        return blocks

    # ================================================================
    # BƯỚC 2: GỌI API BGE-M3 ĐỂ SINH EMBEDDING
    # ================================================================
    def _call_embedding_api(self, texts: List[str]) -> List[np.ndarray]:
        """
        Gọi API OpenRouter Embedding cho BAAI/bge-m3.
        Tự động chia batch nếu tổng token vượt 8192.

        Returns:
            List các vector 1024 chiều (np.ndarray)
        """
        all_embeddings = []
        current_batch = []
        current_batch_tokens = 0
        max_tokens = self.cfg["max_tokens_per_api_call"]
        max_batch = self.cfg["api_batch_size"]

        for text in texts:
            text_tokens = _estimate_tokens(text)

            # Nếu thêm text này sẽ vượt limit → gửi batch hiện tại trước
            if (current_batch_tokens + text_tokens > max_tokens
                    or len(current_batch) >= max_batch) and current_batch:
                embeddings = self._send_embedding_batch(current_batch)
                all_embeddings.extend(embeddings)
                current_batch = []
                current_batch_tokens = 0
                
                # Delay tránh Rate Limit (RPS/RPM) của API provider
                import time
                time.sleep(0.5)

            # Text đơn lẻ vượt limit → cắt bớt (cực hiếm với base_block ~100 tokens)
            if text_tokens > max_tokens:
                safe_len = int(max_tokens * 3.0)  # Conservative: dùng 3.0 thay vì 3.5 cho mixed text
                text = text[:safe_len]
                text_tokens = _estimate_tokens(text)

            current_batch.append(text)
            current_batch_tokens += text_tokens

        # Gửi batch cuối cùng
        if current_batch:
            embeddings = self._send_embedding_batch(current_batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _send_embedding_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Gửi 1 batch texts tới API Embedding và trả về list vectors.

        Production-safe với Exponential Backoff Retry:
          - Retry tự động khi gặp HTTP 429 (Rate Limit), 500, 502, 503
          - Thời gian chờ tăng gấp đôi mỗi lần: 2s → 4s → 8s
          - Sau 3 lần thất bại → raise RuntimeError
          - Lỗi 4xx khác (400, 401, 403) → KHÔNG retry (lỗi logic)
        """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UFM-Admission-Bot/1.0",
        }
        data = {
            "model": self.cfg["model"],
            "input": texts,
            "dimensions": self.cfg["dimensions"],
        }

        max_retries = self.cfg["api_max_retries"]
        base_wait = self.cfg["api_retry_base_wait"]

        # Mã lỗi HTTP cho phép retry (lỗi tạm thời do server)
        RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

        last_error = None

        for attempt in range(1, max_retries + 1):
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            start_time = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.cfg["api_timeout"]) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                elapsed = time.time() - start_time

                # Cập nhật stats
                self.stats["total_api_calls"] += 1
                self.stats["total_tokens_sent"] += sum(_estimate_tokens(t) for t in texts)
                self.stats["total_time_embedding"] += elapsed

                if attempt > 1:
                    self.stats.setdefault("total_retries", 0)
                    self.stats["total_retries"] += attempt - 1

                # Parse embeddings (API trả về sorted by index)
                raw_data = sorted(result["data"], key=lambda x: x["index"])
                embeddings = [
                    np.array(item["embedding"], dtype=np.float32)
                    for item in raw_data
                ]
                return embeddings

            except urllib.error.HTTPError as e:
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8")[:500]
                except Exception:
                    error_body = "(không đọc được response body)"

                last_error = e

                # Chỉ retry với lỗi tạm thời (429, 500, 502, 503)
                if e.code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                    wait_time = base_wait * (2 ** (attempt - 1))  # 2s, 4s, 8s
                    time.sleep(wait_time)
                    continue

                # Lỗi không thể retry (400, 401, 403, 404) hoặc hết lượt retry
                raise RuntimeError(
                    f"Embedding API Error ({e.code}) sau {attempt} lần thử: "
                    f"{error_body}"
                ) from e

            except (urllib.error.URLError, OSError) as e:
                # Lỗi kết nối mạng (DNS, timeout, connection refused)
                last_error = e

                if attempt < max_retries:
                    wait_time = base_wait * (2 ** (attempt - 1))
                    time.sleep(wait_time)
                    continue

                raise RuntimeError(
                    f"Network Error sau {attempt} lần thử: {str(e)}"
                ) from e

        # Fallback cuối cùng (không bao giờ tới đây nếu logic đúng)
        raise RuntimeError(
            f"Embedding API thất bại sau {max_retries} lần retry. "
            f"Lỗi cuối: {str(last_error)}"
        )

    # ================================================================
    # BƯỚC 3: TÌM RANH GIỚI CẮT CHUNK (Similarity Drop)
    # ================================================================
    def _find_chunk_boundaries(
        self, embeddings: List[np.ndarray]
    ) -> List[Tuple[int, float]]:
        """
        Tìm các điểm cắt chunk dựa trên sự sụt giảm Cosine Similarity.

        Returns:
            List of (block_index, similarity_score) tại mỗi ranh giới.
            Block_index = vị trí bắt đầu chunk MỚI.
        """
        if len(embeddings) <= 1:
            return [(0, 1.0)]

        boundaries = [(0, 1.0)]  # Chunk đầu tiên luôn bắt đầu ở index 0
        threshold = self.cfg["similarity_threshold"]

        for i in range(1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < threshold:
                boundaries.append((i, sim))

        return boundaries

    # ================================================================
    # BƯỚC 4: GỘP BLOCKS THÀNH CHUNKS + OVERLAP
    # ================================================================
    def _merge_blocks_to_chunks(
        self,
        blocks: List[str],
        boundaries: List[Tuple[int, float]],
    ) -> List[dict]:
        """
        Gộp các Base Blocks thành Chunks dựa trên ranh giới đã tìm.
        Áp dụng thuật toán Ovelap Block nguyên vẹn để không bao giờ cắt gãy câu.
        """
        if not blocks:
            return []

        # Cấu hình gối đầu bằng số lượng Block 
        # Ví dụ: overlap 100 tokens, 1 block = 100 tokens -> cần overlap 1 block liên tục
        overlap_block_count = max(1, int(self.cfg["overlap_tokens"] / max(1, self.cfg["base_block_tokens"])))
        
        min_chunk_tokens = self.cfg["min_chunk_tokens"]
        max_chunk_tokens = self.cfg["max_chunk_tokens"]
        max_chunk_chars = int(max_chunk_tokens * 3.0)  # Conservative cho mixed Việt-Anh

        boundary_starts = [b[0] for b in boundaries]
        if 0 not in boundary_starts:
            boundary_starts.insert(0, 0)

        chunks = []
        
        for idx, start in enumerate(boundary_starts):
            end = boundary_starts[idx + 1] if idx + 1 < len(boundary_starts) else len(blocks)
            
            # LẤY NỘI DUNG CHÍNH CỦA CHUNK HIỆN TẠI
            current_chunk_blocks = blocks[start:end]
            
            # TẠO OVERLAP TỪ CHUNK TRƯỚC (NẾU CÓ)
            overlap_content = ""
            actual_overlap_tokens = 0
            
            if idx > 0:
                # Lấy các block cuối của chunk TRƯỚC ĐÓ để làm overlap
                prev_start = boundary_starts[idx-1]
                prev_end = start
                # Lấy tối đa 'overlap_block_count' blocks dư cuối của đoạn trước đưa vào
                overlap_source_blocks = blocks[max(prev_start, prev_end - overlap_block_count):prev_end]
                if overlap_source_blocks:
                    overlap_content = " ".join(overlap_source_blocks)
                    actual_overlap_tokens = _estimate_tokens(overlap_content)

            # Hợp nhất Overlap + Current Content
            content = (overlap_content + " " + " ".join(current_chunk_blocks)).strip()

            # Giới hạn tổng dung lượng (phòng hờ mốc chót an toàn)
            if len(content) > max_chunk_chars:
                content = content[:max_chunk_chars]

            # Kiểm tra gộp chunk nếu kích thước hiện tại quá nhỏ nhắn
            if (_estimate_tokens(content) < min_chunk_tokens 
                    and chunks 
                    and _estimate_tokens(chunks[-1]["content"]) + _estimate_tokens(content) <= max_chunk_tokens):
                chunks[-1]["content"] += " " + content
                chunks[-1]["block_range"] = (chunks[-1]["block_range"][0], end)
                continue

            chunks.append({
                "content": content,
                "overlap_tokens": actual_overlap_tokens,
                "block_range": (start, end),
            })

        return chunks

    # ================================================================
    # BƯỚC 5: LUỒNG TỔNG — chunk()
    # ================================================================
    def chunk(
        self,
        text: str,
        source: str,
        metadata_extra: Optional[dict] = None,
    ) -> List[ProcessedChunk]:
        """
        Luồng xử lý hoàn chỉnh Semantic Chunking.

        Tự động phân tích Header trước `-start-` để trích xuất:
          - valid_from, program_level, academic_year, header_context

        Args:
            text: Văn bản thô cần chunking (có thể chứa Header + `-start-`)
            source: Tên file nguồn (VD: "tuyensinh2025.docx")
            metadata_extra: Dict metadata bổ sung, ghi đè auto-parsed values

        Returns:
            List[ProcessedChunk] - Các chunk đã xử lý, sẵn sàng Embedding & Insert DB
        """
        # 0. Auto-parse Header metadata
        parsed = parse_document_header(text)
        content = parsed["content"]

        auto_meta = {}
        if parsed["valid_from"]:
            auto_meta["valid_from"] = parsed["valid_from"]
        if parsed["program_level"]:
            auto_meta["program_level"] = parsed["program_level"]
        if parsed["academic_year"]:
            auto_meta["academic_year"] = parsed["academic_year"]
        # Nap TAT CA fields tu YAML Frontmatter vao extra dict
        header_extra = parsed.get("extra", {})
        if parsed.get("header_context"):
            header_extra["header_context"] = parsed["header_context"]
        if header_extra:
            auto_meta.setdefault("extra", {}).update(header_extra)

        extra = metadata_extra or {}
        program_name = extra.get("program_name")
        program_level = extra.get("program_level") or auto_meta.get("program_level")
        if program_name and program_level:
            ma_nganh = lookup_ma_nganh(program_level, program_name)
            if ma_nganh:
                auto_meta["ma_nganh"] = ma_nganh

        merged_extra = {**auto_meta, **extra}
        if "extra" in auto_meta and "extra" in extra:
            merged_extra["extra"] = {**auto_meta["extra"], **extra["extra"]}

        # Prefix cho mỗi chunk con (để bơm context parent vào)
        section_path = extra.get("section_path", "")
        context_prefix = build_context_prefix(section_path, source, extra=merged_extra)

        # 1. Tiền xử lý (chỉ strip prefix gốc nếu parent vô tình truyền text đã gắn prefix xuống)
        if len(content) > len(context_prefix) and content.startswith(context_prefix):
            content = content[len(context_prefix):].strip()

        content = _normalize_vietnamese(content)
        content = _clean_whitespace(content)

        if not content.strip():
            return []

        # 2. Tách thành Base Blocks (~100 tokens/block)
        blocks = self._split_into_base_blocks(content)

        if not blocks:
            return []

        # 3. Sinh Embedding cho các blocks
        if len(blocks) == 1:
            meta = ChunkMetadata(source=source, chunk_index=1, total_chunks_in_section=1, **merged_extra)
            # Chèn prefix vào chunk duy nhất
            final_content = context_prefix + blocks[0] if not blocks[0].startswith(context_prefix) else blocks[0]
            return [ProcessedChunk(content=final_content, metadata=meta)]

        embeddings = self._call_embedding_api(blocks)

        # 4. Tìm ranh giới cắt chunk (Cosine Similarity < 60%)
        boundaries = self._find_chunk_boundaries(embeddings)

        # 5. Gộp blocks thành chunks + overlap 100 tokens
        raw_chunks = self._merge_blocks_to_chunks(blocks, boundaries)

        # 6. Tạo ProcessedChunk objects với metadata đầy đủ
        total = len(raw_chunks)
        processed_chunks = []

        for idx, raw in enumerate(raw_chunks, 1):
            meta = ChunkMetadata(
                source=source,
                chunk_index=idx,
                total_chunks_in_section=total,
                overlap_tokens=raw["overlap_tokens"],
                **merged_extra,
            )
            # Gắn lại prefix cho từng chunk mới tách
            chunk_content = raw["content"]
            if not chunk_content.startswith(context_prefix):
                chunk_content = context_prefix + chunk_content
            
            processed_chunks.append(
                ProcessedChunk(content=chunk_content, metadata=meta)
            )

        return processed_chunks

    # ================================================================
    # FALLBACK: Chunk không cần Embedding (Khi API lỗi)
    # ================================================================
    def chunk_fallback(
        self,
        text: str,
        source: str,
        metadata_extra: Optional[dict] = None,
    ) -> List[ProcessedChunk]:
        """
        Fallback chunking (không cần API Embedding).
        Cắt theo kích thước cố định + overlap câu.
        Dùng khi API BGE-M3 bị lỗi hoặc không có kết nối mạng.

        Tự động phân tích Header nếu có `-start-`.
        """
        # Auto-parse Header
        parsed = parse_document_header(text)
        content = parsed["content"]

        auto_meta = {}
        if parsed["valid_from"]:
            auto_meta["valid_from"] = parsed["valid_from"]
        if parsed["program_level"]:
            auto_meta["program_level"] = parsed["program_level"]
        if parsed["academic_year"]:
            auto_meta["academic_year"] = parsed["academic_year"]
        # Nap TAT CA fields tu YAML Frontmatter vao extra dict
        header_extra = parsed.get("extra", {})
        if parsed.get("header_context"):
            header_extra["header_context"] = parsed["header_context"]
        if header_extra:
            auto_meta.setdefault("extra", {}).update(header_extra)

        extra = metadata_extra or {}
        merged_extra = {**auto_meta, **extra}
        if "extra" in auto_meta and "extra" in extra:
            merged_extra["extra"] = {**auto_meta["extra"], **extra["extra"]}

        # Tạo context prefix
        section_path = extra.get("section_path", "")
        context_prefix = build_context_prefix(section_path, source, extra=merged_extra)

        if len(content) > len(context_prefix) and content.startswith(context_prefix):
            content = content[len(context_prefix):].strip()

        content = _normalize_vietnamese(content)
        content = _clean_whitespace(content)

        if not content.strip():
            return []

        sentences = _split_sentences_vietnamese(content)
        target_chars = int(self.cfg["max_chunk_tokens"] * 3.5)
        overlap_sents = 2  # Overlap 2 câu cuối

        chunks = []
        current = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > target_chars and current:
                chunk_content = " ".join(current)
                if not chunk_content.startswith(context_prefix):
                    chunk_content = context_prefix + chunk_content
                meta = ChunkMetadata(source=source, **merged_extra)
                chunks.append(ProcessedChunk(content=chunk_content, metadata=meta))

                # Overlap: giữ lại 2 câu cuối
                current = current[-overlap_sents:]
                current_len = sum(len(s) for s in current)

            current.append(sent)
            current_len += len(sent)

        # Chunk cuối cùng
        if current:
            chunk_content = " ".join(current)
            if not chunk_content.startswith(context_prefix):
                chunk_content = context_prefix + chunk_content
            meta = ChunkMetadata(source=source, **merged_extra)
            chunks.append(ProcessedChunk(content=chunk_content, metadata=meta))

        # Cập nhật chunk_index
        for i, c in enumerate(chunks, 1):
            c.metadata.chunk_index = i
            c.metadata.total_chunks_in_section = len(chunks)

        return chunks

    # ================================================================
    # THỐNG KÊ
    # ================================================================
    def get_stats(self) -> dict:
        """Trả về thống kê runtime (số API calls, tokens, thời gian)."""
        return {**self.stats}

    def reset_stats(self):
        """Reset thống kê."""
        self.stats = {
            "total_api_calls": 0,
            "total_tokens_sent": 0,
            "total_time_embedding": 0.0,
        }
