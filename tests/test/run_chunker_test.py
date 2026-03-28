import os
import sys
import glob

# Thêm rễ dự án vào sys.path để Python có thể tìm thấy module chunk_Process
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from chunk_Process.chunk_algorithms.hierarchical_chunker import HierarchicalChunker
from chunk_Process.chunk_algorithms.utils import (
    parse_document_header,
    lookup_ma_nganh,
    LEVEL_DISPLAY,
)


def main():
    # 1. Khởi tạo CHỈ Hierarchical Chunker
    hierarchical = HierarchicalChunker()
    
    # 2. Cấu hình đường dẫn
    # TÙY CHỌN 1: Test cả thư mục
    target_dir = r"E:\UFM - THƯƠNG MẠI\ufm_admission_bot\data\unstructured\markdown\thongtinchung"
    
    # TÙY CHỌN 2: Test 1 file chỉ định (Ưu tiên file này trước)
    target_file = r"E:\UFM - THƯƠNG MẠI\ufm_admission_bot\data\unstructured\markdown\thongtinchung\tuyển sinh trình độ tiến sĩ đợt 1 năm 2026.md"
    # target_file = r"E:\UFM - THƯƠNG MẠI\ufm_admission_bot\data\unstructured\markdown\thongtinchung\phuluc2.md"
    # target_file = None  # Bỏ comment để quét cả thư mục

    output_report = r"E:\UFM - THƯƠNG MẠI\ufm_admission_bot\test\test_results_thacsi_hierarchical.txt"

    # Lấy danh sách file cần xử lý (Hỗ trợ cả .txt và .md)
    if target_file and os.path.exists(target_file):
        txt_files = [target_file]
        print(f"🎯 Đang xử lý FILE CHỈ ĐỊNH: {os.path.basename(target_file)}")
    elif os.path.exists(target_dir):
        txt_files = glob.glob(os.path.join(target_dir, "*.txt")) + glob.glob(os.path.join(target_dir, "*.md"))
        print(f"🔍 Tìm thấy {len(txt_files)} file trong thư mục. Đang xử lý...")
    else:
        print(f"❌ Không tìm thấy đường dẫn mục tiêu.")
        return

    if not txt_files:
        print(f"⚠️ Không có file .txt hoặc .md nào để xử lý.")
        return

    with open(output_report, "w", encoding="utf-8") as out:
        out.write(f"=== BÁO CÁO CHI TIẾT KẾT QUẢ HIERARCHICAL CHUNKING ===\n")
        source_info = target_file if target_file else target_dir
        out.write(f"Nguồn dữ liệu: {source_info}\n")
        out.write("=" * 80 + "\n\n")

        for file_path in txt_files:
            file_name = os.path.basename(file_path)
            
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            if not raw_text.strip():
                continue

            # --- Phân tách Header (chỉ để hiển thị trong báo cáo) ---
            # Hàm chunk() bên trong HierarchicalChunker sẽ tự động gọi
            # parse_document_header() nên không cần truyền metadata_extra thủ công nữa
            parsed = parse_document_header(raw_text)

            out.write(f"📄 TẬP TIN: {file_name}\n")
            
            # Hiển thị metadata đã parse được
            out.write(f"📅 Ngày hiệu lực: {parsed['valid_from'] or 'N/A'}\n")
            level_display = LEVEL_DISPLAY.get(parsed['program_level'] or '', 'N/A')
            out.write(f"🎓 Trình độ (auto): {level_display} ({parsed['program_level'] or 'N/A'})\n")
            out.write(f"📆 Năm tuyển sinh: {parsed['academic_year'] or 'N/A'}\n")
            if parsed["header_context"]:
                out.write(f"🏷️ Header Context:\n{parsed['header_context']}\n")
            out.write("-" * 80 + "\n")
            
            # --- In Cấu Trúc Cây Markdown ---
            tree_summary = hierarchical.get_tree_summary(parsed["content"])
            out.write("🌳 CẤU TRÚC LOGIC (TREE VIEW):\n")
            out.write(tree_summary + "\n\n")

            # --- Thực thi quá trình Chunking ---
            # Truyền raw_text VÀO → chunk() sẽ tự parse header + metadata
            chunks = hierarchical.chunk(
                text=raw_text,
                source=file_name,
                # metadata_extra chỉ cần truyền nếu muốn GHI ĐÈ giá trị auto-parsed
                # VD: metadata_extra={"program_name": "Tài chính – Ngân hàng"}
            )

            out.write(f"📊 THỐNG KÊ: Tổng cộng {len(chunks)} chunks được tạo.\n\n")

            # --- In chi tiết TỪNG Chunk ---
            for idx, chunk in enumerate(chunks, 1):
                m = chunk.metadata
                out.write(f"📍 [CHUNK #{idx}]\n")
                out.write(f"   - Level:         {m.chunk_level.upper()}\n")
                out.write(f"   - Index:         {m.chunk_index}/{m.total_chunks_in_section}\n")
                out.write(f"   - Path:          {m.section_path}\n")
                out.write(f"   - Tokens Est:    ~{m.token_count}\n")
                out.write(f"   - Source:        {m.source}\n")
                out.write(f"   - ID:            {m.chunk_id}\n")
                out.write(f"   - Program Level: {m.program_level or 'N/A'}\n")
                out.write(f"   - Valid From:    {m.valid_from or 'N/A'}\n")
                out.write(f"   - Academic Year: {m.academic_year or 'N/A'}\n")
                out.write(f"   - Ma Nganh:      {m.ma_nganh or 'N/A'}\n")
                
                out.write(f"\n   📝 [NỘI DUNG ĐẦY ĐỦ]:\n")
                out.write("   " + "-"*40 + "\n")
                content_indented = "\n   ".join(chunk.content.splitlines())
                out.write(f"   {content_indented}\n")
                out.write("   " + "-"*40 + "\n\n")
                
            out.write("=" * 80 + "\n\n")
            print(f"✅ Hoàn thành: {file_name}")

    print(f"\n🎉 Test thành công! Vui lòng mở file để xem chi tiết:")
    print(f"👉 {output_report}")

if __name__ == "__main__":
    main()
