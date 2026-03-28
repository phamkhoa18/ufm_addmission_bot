"""
Export Chunks → JSON files để theo dõi dữ liệu trước/sau khi nạp VectorDB.

Xuất ra 2 file cho mỗi pipeline:
  1. chunks_raw_<type>.json    — Chunk gốc (content + metadata + parent-child)
  2. chunks_embedded_<type>.json — Chunk đã có embedding (metadata + vector preview)

Sử dụng:
  # Xuất cả 2 pipeline (markdown + structured)
  python ingestion/export_chunks.py

  # Chỉ xuất markdown
  python ingestion/export_chunks.py --type markdown

  # Chỉ xuất structured
  python ingestion/export_chunks.py --type structured

  # Xuất kèm embedding (tốn API calls)
  python ingestion/export_chunks.py --with-embedding

  # Chỉ xuất 1 file cụ thể
  python ingestion/export_chunks.py --file "data/structured/processed/Ctrinh Thac Si/ThS KDQT.txt"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Output directory
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

# ================================================================
# HELPER: Serialize chunk → dict (cho JSON)
# ================================================================
def chunk_to_dict(chunk, include_full_content: bool = True) -> dict:
    """Chuyển ProcessedChunk → dict dễ đọc."""
    meta = chunk.metadata
    d = {
        "chunk_id": meta.chunk_id,
        "chunk_level": meta.chunk_level,
        "source": meta.source,
        "section_path": meta.section_path,
        "section_name": meta.section_name,
        "program_name": meta.program_name,
        "program_level": meta.program_level,
        "ma_nganh": meta.ma_nganh,
        "chunk_index": meta.chunk_index,
        "total_chunks_in_section": meta.total_chunks_in_section,
        "parent_id": meta.parent_id,
        "children_ids": meta.children_ids,
        "children_count": len(meta.children_ids),
        "overlap_tokens": meta.overlap_tokens,
        "token_count": meta.token_count,
        "char_count": chunk.char_count,
        "content_hash": meta.content_hash,
        "academic_year": meta.academic_year,
        "valid_from": str(meta.valid_from) if meta.valid_from else None,
        "valid_until": str(meta.valid_until) if meta.valid_until else None,
        "is_active": meta.is_active,
        "version": meta.version,
        "extra": meta.extra,
    }

    if include_full_content:
        d["content"] = chunk.content
    else:
        d["content_preview"] = chunk.content[:400] + ("..." if len(chunk.content) > 400 else "")

    return d


def build_parent_child_tree(chunks: list) -> list[dict]:
    """
    Xây dựng cấu trúc cây Parent → Children cho dễ đọc.

    Output: List[dict] mỗi parent có key "children" chứa list children.
    """
    # Tách parents và children
    parents = [c for c in chunks if c.metadata.chunk_level == "parent"]
    children_map: dict[str, list] = {}
    for c in chunks:
        if c.metadata.chunk_level == "child" and c.metadata.parent_id:
            children_map.setdefault(c.metadata.parent_id, []).append(c)

    tree = []
    for parent in parents:
        parent_dict = chunk_to_dict(parent)
        parent_dict["children"] = []

        # Gắn children vào parent (sắp xếp theo chunk_index)
        kids = children_map.get(parent.metadata.chunk_id, [])
        kids.sort(key=lambda c: c.metadata.chunk_index or 0)
        for child in kids:
            parent_dict["children"].append(chunk_to_dict(child))

        tree.append(parent_dict)

    # Chunks không thuộc parent-child (orphans)
    orphan_ids = {p.metadata.chunk_id for p in parents}
    for c in chunks:
        if c.metadata.chunk_level not in ("parent", "child"):
            tree.append(chunk_to_dict(c))

    return tree


def save_json(data: any, filepath: Path, label: str):
    """Lưu data ra JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    size_kb = filepath.stat().st_size / 1024
    print(f"   💾 {label}: {filepath.name} ({size_kb:.1f} KB)")


# ================================================================
# EXPORT MARKDOWN CHUNKS
# ================================================================
def export_markdown(single_file: str = None, with_embedding: bool = False):
    """Export chunks từ pipeline markdown."""
    from ingestion.ingest_markdown import (
        collect_markdown_files,
        generate_embeddings_batch,
    )
    from chunk_Process.chunk_algorithms.semantic_chunker import SemanticChunkerBGE
    from chunk_Process.chunk_algorithms.hierarchical_chunker import HierarchicalChunker

    print("\n" + "=" * 60)
    print("📝 EXPORT: Markdown Pipeline (thông tin tuyển sinh chung)")
    print("=" * 60)

    files = collect_markdown_files(single_file=single_file)
    print(f"   📂 {len(files)} files")

    if not files:
        print("   ❌ Không có file!")
        return

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    hierarchical = HierarchicalChunker()
    semantic = SemanticChunkerBGE(api_key=api_key, base_url=base_url)

    all_chunks = []
    for fp in files:
        print(f"   📄 {fp.name}...")
        try:
            chunks = hierarchical.chunk_file(
                filepath=str(fp),
                semantic_chunker=semantic,
                use_fallback=(not with_embedding),  # fallback nếu không cần embedding
            )
            all_chunks.extend(chunks)
            parents = sum(1 for c in chunks if c.metadata.chunk_level == "parent")
            children = sum(1 for c in chunks if c.metadata.chunk_level == "child")
            print(f"      ✅ {len(chunks)} chunks ({parents}P + {children}C)")
        except Exception as e:
            print(f"      ❌ {e}")

    if not all_chunks:
        print("   ⚠️ Không có chunks nào!")
        return

    # ── File 1: Chunks gốc (RAW) ──
    tree = build_parent_child_tree(all_chunks)
    raw_file = EXPORT_DIR / "chunks_raw_markdown.json"
    save_json({
        "exported_at": datetime.now().isoformat(),
        "pipeline": "markdown",
        "total_chunks": len(all_chunks),
        "total_parents": sum(1 for c in all_chunks if c.metadata.chunk_level == "parent"),
        "total_children": sum(1 for c in all_chunks if c.metadata.chunk_level == "child"),
        "files_processed": [f.name for f in files],
        "chunks": tree,
    }, raw_file, "Chunks gốc (RAW)")

    # ── File 2: Chunks + Embedding ──
    if with_embedding:
        print(f"\n   🧠 Generating embeddings for {len(all_chunks)} chunks...")
        try:
            texts = [c.content for c in all_chunks]
            embeddings = generate_embeddings_batch(
                texts=texts, api_key=api_key, base_url=base_url,
            )
            print(f"      ✅ {len(embeddings)} vectors")

            # Build embedded export
            embedded_data = []
            for i, chunk in enumerate(all_chunks):
                d = chunk_to_dict(chunk, include_full_content=False)
                if i < len(embeddings):
                    d["embedding_dims"] = len(embeddings[i])
                    d["embedding_preview"] = embeddings[i][:8]  # Chỉ show 8 số đầu
                    d["embedding_norm"] = round(sum(v**2 for v in embeddings[i]) ** 0.5, 6)
                embedded_data.append(d)

            emb_file = EXPORT_DIR / "chunks_embedded_markdown.json"
            save_json({
                "exported_at": datetime.now().isoformat(),
                "pipeline": "markdown",
                "embedding_model": "baai/bge-m3",
                "embedding_dimensions": 1024,
                "total_chunks": len(all_chunks),
                "chunks": embedded_data,
            }, emb_file, "Chunks + Embedding")

        except Exception as e:
            print(f"      ❌ Embedding lỗi: {e}")
    else:
        print("   ℹ️  Bỏ qua embedding (dùng --with-embedding để bật)")

    print(f"   ✅ Markdown export hoàn tất!")


# ================================================================
# EXPORT STRUCTURED CHUNKS
# ================================================================
def export_structured(single_file: str = None, with_embedding: bool = False):
    """Export chunks từ pipeline structured (Thạc sĩ/Tiến sĩ)."""
    from ingestion.ingest_structured import (
        collect_files,
        chunk_structured_file,
        detect_program_level,
        generate_embeddings_batch,
    )

    print("\n" + "=" * 60)
    print("📊 EXPORT: Structured Pipeline (Chương trình Thạc sĩ/Tiến sĩ)")
    print("=" * 60)

    files = collect_files(single_file=single_file)
    print(f"   📂 {len(files)} files")

    if not files:
        print("   ❌ Không có file!")
        return

    all_chunks = []
    for fp in files:
        level = detect_program_level(fp)
        print(f"   📄 [{level}] {fp.name}...")
        try:
            chunks = chunk_structured_file(fp, use_fallback=(not with_embedding))
            all_chunks.extend(chunks)
            parents = sum(1 for c in chunks if c.metadata.chunk_level == "parent")
            children = sum(1 for c in chunks if c.metadata.chunk_level == "child")
            print(f"      ✅ {len(chunks)} chunks ({parents}P + {children}C)")

            # Show context injection
            if chunks:
                demo = chunks[0].content[:100].replace("\n", " ↵ ")
                print(f"      👁  \"{demo}...\"")
        except Exception as e:
            print(f"      ❌ {e}")

    if not all_chunks:
        print("   ⚠️ Không có chunks nào!")
        return

    # ── File 1: Chunks gốc (RAW) ──
    tree = build_parent_child_tree(all_chunks)
    raw_file = EXPORT_DIR / "chunks_raw_structured.json"
    save_json({
        "exported_at": datetime.now().isoformat(),
        "pipeline": "structured",
        "total_chunks": len(all_chunks),
        "total_parents": sum(1 for c in all_chunks if c.metadata.chunk_level == "parent"),
        "total_children": sum(1 for c in all_chunks if c.metadata.chunk_level == "child"),
        "files_processed": [f.name for f in files],
        "program_summary": _build_program_summary(all_chunks),
        "chunks": tree,
    }, raw_file, "Chunks gốc (RAW)")

    # ── File 2: Chunks + Embedding ──
    if with_embedding:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        print(f"\n   🧠 Generating embeddings for {len(all_chunks)} chunks...")
        try:
            texts = [c.content for c in all_chunks]
            embeddings = generate_embeddings_batch(
                texts=texts, api_key=api_key, base_url=base_url,
            )
            print(f"      ✅ {len(embeddings)} vectors")

            embedded_data = []
            for i, chunk in enumerate(all_chunks):
                d = chunk_to_dict(chunk, include_full_content=False)
                if i < len(embeddings):
                    d["embedding_dims"] = len(embeddings[i])
                    d["embedding_preview"] = embeddings[i][:8]
                    d["embedding_norm"] = round(sum(v**2 for v in embeddings[i]) ** 0.5, 6)
                embedded_data.append(d)

            emb_file = EXPORT_DIR / "chunks_embedded_structured.json"
            save_json({
                "exported_at": datetime.now().isoformat(),
                "pipeline": "structured",
                "embedding_model": "baai/bge-m3",
                "embedding_dimensions": 1024,
                "total_chunks": len(all_chunks),
                "program_summary": _build_program_summary(all_chunks),
                "chunks": embedded_data,
            }, emb_file, "Chunks + Embedding")

        except Exception as e:
            print(f"      ❌ Embedding lỗi: {e}")
    else:
        print("   ℹ️  Bỏ qua embedding (dùng --with-embedding để bật)")

    print(f"   ✅ Structured export hoàn tất!")


def _build_program_summary(chunks: list) -> list[dict]:
    """Tổng hợp thống kê theo từng chương trình đào tạo."""
    programs: dict[str, dict] = {}
    for c in chunks:
        key = f"{c.metadata.program_level}|{c.metadata.program_name}"
        if key not in programs:
            programs[key] = {
                "program_name": c.metadata.program_name,
                "program_level": c.metadata.program_level,
                "ma_nganh": c.metadata.ma_nganh,
                "source": c.metadata.source,
                "parents": 0,
                "children": 0,
                "sections": set(),
            }
        if c.metadata.chunk_level == "parent":
            programs[key]["parents"] += 1
        elif c.metadata.chunk_level == "child":
            programs[key]["children"] += 1
        if c.metadata.section_name:
            programs[key]["sections"].add(c.metadata.section_name)

    result = []
    for info in programs.values():
        info["sections"] = sorted(info["sections"])
        info["total_chunks"] = info["parents"] + info["children"]
        result.append(info)
    return result


# ================================================================
# CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Export chunks ra JSON để theo dõi trước khi nạp VectorDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python ingestion/export_chunks.py                          # Cả 2 pipeline, không embedding
  python ingestion/export_chunks.py --type markdown          # Chỉ markdown
  python ingestion/export_chunks.py --type structured        # Chỉ structured
  python ingestion/export_chunks.py --with-embedding         # Kèm embedding vectors
  python ingestion/export_chunks.py --file "ThS KDQT.txt"   # 1 file cụ thể
        """,
    )
    parser.add_argument(
        "--type", choices=["markdown", "structured", "all"], default="all",
        help="Pipeline muốn export (default: all)",
    )
    parser.add_argument(
        "--with-embedding", action="store_true",
        help="Xuất kèm embedding vectors (tốn API calls)",
    )
    parser.add_argument(
        "--file", type=str,
        help="File cụ thể muốn export",
    )
    args = parser.parse_args()

    start = time.time()
    print("╔" + "═" * 58 + "╗")
    print("║  UFM ADMISSION BOT — Chunk Export Tool                   ║")
    print(f"║  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                     ║")
    print("╚" + "═" * 58 + "╝")

    if args.type in ("markdown", "all"):
        export_markdown(
            single_file=args.file if args.file and args.file.endswith(".md") else None,
            with_embedding=args.with_embedding,
        )

    if args.type in ("structured", "all"):
        export_structured(
            single_file=args.file if args.file and args.file.endswith(".txt") else None,
            with_embedding=args.with_embedding,
        )

    elapsed = time.time() - start
    print(f"\n{'═' * 60}")
    print(f"✅ Export hoàn tất — {elapsed:.1f}s")
    print(f"📁 Output: {EXPORT_DIR}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
