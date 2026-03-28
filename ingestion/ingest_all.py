"""
Master Ingestion Script — Chạy cả 2 Pipeline Nạp VectorDB
1. Markdown Pipeline (Thông báo chung)
2. Structured Pipeline (Chương trình đào tạo)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.ingest_markdown import run_ingestion as run_markdown
from ingestion.ingest_structured import run_ingestion as run_structured


def main():
    parser = argparse.ArgumentParser(description="Chạy nạp toàn bộ dữ liệu vào VectorDB (Full Pipeline)")
    parser.add_argument("--rebuild", action="store_true", help="Xóa sạch dữ liệu cũ trong DB trước khi nạp lại")
    parser.add_argument("--fallback", action="store_true", help="Chạy ở chế độ Fallback (cắt chay, không gọi API Embedding)")
    args = parser.parse_args()

    start_time = time.time()
    print("╔" + "═" * 60 + "╗")
    print("║  🚀 UFM ADMISSION BOT — MASTER INGESTION PIPELINE        ║")
    print(f"║  Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                     ║")
    print("╚" + "═" * 60 + "╝\n")

    # ==========================================
    # 1. RUN MARKDOWN PIPELINE
    # ==========================================
    print("Mời bạn theo dõi quá trình nạp dữ liệu — TIẾN TRÌNH 1/2")
    try:
        # Gọi thẳng hàm chính của ingest_markdown
        run_markdown(
            rebuild=args.rebuild,
            dry_run=False,
            use_fallback=args.fallback
        )
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng ở Markdown Pipeline: {e}")
        sys.exit(1)

    print("\n" + "─" * 62 + "\n")

    # ==========================================
    # 2. RUN STRUCTURED PIPELINE
    # ==========================================
    print("Mời bạn theo dõi quá trình nạp dữ liệu — TIẾN TRÌNH 2/2")
    try:
        # Gọi thẳng hàm chính của ingest_structured
        run_structured(
            rebuild=args.rebuild,
            dry_run=False,
            use_fallback=args.fallback
        )
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng ở Structured Pipeline: {e}")
        sys.exit(1)

    # ==========================================
    # SUMMARY
    # ==========================================
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print("╔" + "═" * 60 + "╗")
    print("║  🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH NẠP VECTORDB!             ║")
    print(f"║  Tổng thời gian   : {minutes} phút {seconds} giây                        ║")
    print("║  Status           : Database đã được Update 100%         ║")
    print("╚" + "═" * 60 + "╝")


if __name__ == "__main__":
    main()
