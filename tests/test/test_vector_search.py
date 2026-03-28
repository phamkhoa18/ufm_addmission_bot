"""
Script test chức năng Vector Search (Tìm kiếm ngữ nghĩa) bằng pgvector.
Quy trình:
1. Nhập câu hỏi (Query) từ người dùng.
2. Gọi API OpenRouter sinh vector (1024 chiều) bằng model BAAI/BGE-M3.
3. Tính Cosine Similarity bằng HNSW Index trong DB.
4. Trả về TOP 10 kết quả (Chunks) giống nhất.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import psycopg2

# ── Cấu hình ──
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "ufm_admission_db"),
    "user": os.getenv("POSTGRES_USER", "ufm_admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "ufm_secure_password_2026"),
}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
EMBEDDING_MODEL = "baai/bge-m3"
DIMENSIONS = 1024

def get_embedding(text: str) -> list[float]:
    """Gọi API Embedding (OpenRouter) cho câu hỏi đầu vào."""
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": EMBEDDING_MODEL,
        "input": text,
        "dimensions": DIMENSIONS,
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result["data"][0]["embedding"]

def search_top_k(query: str, k: int = 10):
    """Quét dữ liệu dưới Database để trích xuất k chunks sát nghĩa nhất."""
    print("\n" + "=" * 80)
    print(f"🔍 TÌM KIẾM NGỮ NGHĨA - Model: {EMBEDDING_MODEL} (Top {k})")
    print(f"🗣️  Câu hỏi: '{query}'")
    print("=" * 80)
    
    # 1. Sinh Vecto câu hỏi
    print("1. Đang gọi API lấy Vector...")
    try:
        query_vector = get_embedding(query)
        print("   ✅ Lấy Vector thành công (1024 chiều)")
    except Exception as e:
        print(f"   ❌ Lỗi gọi API: {e}")
        return

    vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

    # 2. Quét Vector Database
    print("2. Đang quét kho dữ liệu Vector (pgvector)...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            # <=> là toán tử Cosine Distance trong pgvector
            # Cosine Similarity = 1 - Distance
            sql = """
                SELECT 
                    chunk_id, 
                    chunk_level, 
                    program_name, 
                    section_name, 
                    content,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_chunks
                WHERE is_active = true
                ORDER BY embedding <=> %s::vector  -- Sắp xếp ưu tiên Distance càng nhỏ càng tốt
                LIMIT %s;
            """
            cur.execute(sql, (vector_str, vector_str, k))
            rows = cur.fetchall()
            
            print(f"   ✅ Đã tìm thấy kết quả! (Dữ liệu xếp từ chuẩn xác Nhất -> Kém dần)\n")
            
            # --- IN RA MÀN HÌNH MỘT CÁCH ĐẸP MẮT ---
            for index, row in enumerate(rows, 1):
                chunk_id, chunk_level, program_name, section_name, content, similarity = row
                
                print("-" * 80)
                print(f"🏆 RANK #{index} | Độ chuẩn xác (Cosine): {similarity:.4f}")
                print(f"📦 Loại Chunk      : {chunk_level.upper()} (ID: {chunk_id[:8]}...)")
                print(f"📌 Chuyên ngành    : {program_name if program_name else 'Thông báo chung UFM'}")
                print(f"🔖 Tiêu đề nội dung: {section_name}")
                
                # Cắt gọn nội dung
                content_preview = content.replace('\n', ' ↵ ')
                if len(content_preview) > 200:
                    content_preview = content_preview[:200] + "..."
                
                print(f"📝 Trích đoạn (Preview):\n  \"{content_preview}\"")
                
    except Exception as e:
        print(f"   ❌ Lỗi truy vấn Database: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    print("\n🚀 CHÀO MỪNG BẠN ĐẾN VỚI BỘ CÔNG CỤ TEST VECTOR SEARCH UFM! 🚀")
    print("Hãy đặt một câu hỏi kỳ quặc hay sai chính tả để thử sức bật của hệ thống.")
    
    while True:
        try:
            user_input = input("\n👉 Nhập câu hỏi vào đây (hoặc gõ 'q' để thoát): ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("👋 Đã thoát phiên test. Hẹn gặp lại!")
                break
            if user_input.strip():
                search_top_k(user_input.strip(), k=10)
        except KeyboardInterrupt:
            print("\n👋 Đã thoát phiên test.")
            break
