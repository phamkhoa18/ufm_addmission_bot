import sys
import time
import os
# Đảm bảo đường dẫn module được load đúng
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from chunk_Process.chunk_algorithms.hierarchical_chunker import HierarchicalChunker
from chunk_Process.chunk_algorithms.semantic_chunker import SemanticChunkerBGE
from chunk_Process.chunk_algorithms.utils import LEVEL_DISPLAY

def main():
    target_file = r"E:\UFM - THƯƠNG MẠI\ufm_admission_bot\data\unstructured\markdown\thongtinchung\tuyển sinh trình độ tiến sĩ đợt 1 năm 2026.md"
    output_report = r"E:\UFM - THƯƠNG MẠI\ufm_admission_bot\test\test_results_5_parent_child.txt"

    print(f"🎯 Bắt đầu quá trình test Parent-Child Pipeline...")
    
    # 1. Khởi tạo Chunkers
    hierarchical = HierarchicalChunker()
    # Semantic Chunker tự đọc config/api_key từ YAML + env
    semantic = SemanticChunkerBGE()

    with open(target_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Sinh Parent Chunks từ Hierarchical
    print(f"⏳ Đang xử lý Hierarchical Chunking...")
    all_parents = hierarchical.chunk(
        text=raw_text,
        source=os.path.basename(target_file)
    )

    # 3. Chỉ lấy 5 Parent Chunks đầu tiên
    first_5_parents = all_parents[:5]

    with open(output_report, "w", encoding="utf-8") as out:
        out.write(f"=== BÁO CÁO PIPELINE HIERARCHICAL -> SEMANTIC (Top 5 Parents) ===\n")
        out.write(f"Tập tin: {os.path.basename(target_file)}\n")
        out.write("=" * 80 + "\n\n")

        total_children_created = 0

        # 4. Truyền 5 Parent Chunk này vào Semantic Chunker
        for p_idx, parent in enumerate(first_5_parents, 1):
            out.write("*" * 80 + "\n")
            out.write(f"📁 [PARENT CHUNK #{p_idx}] (Tổng: {parent.metadata.total_chunks_in_section})\n")
            out.write(f"   - Parent ID:     {parent.metadata.chunk_id}\n")
            out.write(f"   - Path:          {parent.metadata.section_path}\n")
            out.write(f"   - Level:         {parent.metadata.chunk_level.upper()}\n")
            out.write(f"   - Tokens Est:    ~{parent.metadata.token_count}\n")
            out.write(f"   - Program Level: {parent.metadata.program_level} ({LEVEL_DISPLAY.get(parent.metadata.program_level or '', 'N/A')})\n")
            out.write(f"   - Valid From:    {parent.metadata.valid_from}\n")
            out.write(f"   - Academic Year: {parent.metadata.academic_year}\n")
            # In nội dung Parent (rút gọn nếu quá dài)
            content_preview = parent.content[:200].replace('\n', ' ') + "..." if len(parent.content) > 200 else parent.content.replace('\n', ' ')
            out.write(f"   - Nội dung (Preview): {content_preview}\n\n")

            # ── CHUẨN BỊ METADATA CHO CHILD ──
            # Kế thừa dữ liệu từ parent
            child_extra = {
                "parent_id": parent.metadata.chunk_id,
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

            # Chạy Semantic Chunker (sử dụng fallback để mô phỏng thuật toán mà k gọi API)
            print(f"   → Semantic chunking cho Parent #{p_idx}...")
            child_chunks = semantic.chunk_fallback(
                text=parent.content,
                source=parent.metadata.source,
                metadata_extra=child_extra
            )

            parent.metadata.children_ids = [c.metadata.chunk_id for c in child_chunks]
            total_children_created += len(child_chunks)

            # In chi tiết từng Child
            for c_idx, child in enumerate(child_chunks, 1):
                out.write(f"      📍 [CHILD CHUNK #{c_idx} / {len(child_chunks)}]\n")
                out.write(f"         - Child ID:     {child.metadata.chunk_id}\n")
                out.write(f"         - Parent ID Ref:{child.metadata.parent_id}  <-- Trỏ về Parent!\n")
                out.write(f"         - Level:        {child.metadata.chunk_level}\n")
                out.write(f"         - Tokens:       ~{child.metadata.token_count}\n")
                # Indent nội dung con
                child_content = "\n            ".join(child.content.splitlines())
                out.write("         - " + "-"*40 + "\n")
                out.write(f"            {child_content}\n")
                out.write("         - " + "-"*40 + "\n\n")

            # ── TRÁNH OVER REQUEST (Rate Limit) ──
            if p_idx < len(first_5_parents):
                print(f"   (Chờ 2s để tránh rate limit...)")
                time.sleep(2)

        out.write("=" * 80 + "\n")
        out.write(f"🏁 Đã xử lý {len(first_5_parents)} Parent Chunks, sinh ra {total_children_created} Child Chunks.\n")

    print(f"\n🎉 Hoàn tất kiểm thử luồng Parent-to-Child! Báo cáo tại:")
    print(f"👉 {output_report}")

if __name__ == "__main__":
    main()
