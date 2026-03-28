import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.proceed_rag_search.graph import proceed_rag_search_pipeline
from app.services.intent_service import classify_intent
from app.services.langgraph.nodes.proceed_rag_search.search_cache import _search_cache

class Logger(object):
    """Ghi log đồng thời ra Terminal và File TXT"""
    def __init__(self, filename="rag_test_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    print("="*80)
    print("  🧪 KIỂM THỬ LUỒNG RAG SEARCH PIPELINE (NHẬP TAY)")
    print("="*80)
    
    # 1. Nhập Query
    query = input("\n[1] Nhập câu hỏi của bạn: ").strip()
    if not query:
        print("Câu hỏi trống. Thoát chương trình.")
        return

    # 2. Phân loại Intent
    print("\n[2] Đang phân loại Intent qua AI...")
    intent_res = classify_intent(query)
    intent_action = intent_res["intent_action"]
    print(f" 👉 Intent phân loại: {intent_res['intent']} -> Action: {intent_action}")
    
    # 3. Ép luồng nếu không phải RAG_SEARCH
    if intent_action not in ["PROCEED_RAG_UFM_SEARCH", "PROCEED_RAG_PR_SEARCH"]:
        override = input(f" ⚠️ Luồng hệ thống đề xuất là '{intent_action}'. Bạn có muốn ép chạy web search không? (y/n): ").strip().lower()
        if override == 'y':
            print("   Chọn nhánh Web Search:")
            print("   1: PROCEED_RAG_UFM_SEARCH (Tìm hành chính, cấm PR)")
            print("   2: PROCEED_RAG_PR_SEARCH (Tìm thành tích, báo chí)")
            choice = input("   => Lựa chọn (1 hoặc 2): ").strip()
            if choice == "1":
                intent_action = "PROCEED_RAG_UFM_SEARCH"
            else:
                intent_action = "PROCEED_RAG_PR_SEARCH"
        else:
            print("Thoát.")
            return

    # 4. Nhập RAG Context
    rag_context = input("\n[3] Nhập RAG Context (từ VectorDB nội bộ) - Nhấn Enter để bỏ qua/mock: ").strip()
    if not rag_context:
        rag_context = "(Context mock nội bộ tự động: UFM tuyển sinh 2024 có nhiều ưu đãi. Cơ sở Q7 rất rộng.)"
        print(f" -> Dùng context mock: {rag_context}")

    # Xóa cache cũ để test minh bạch
    _search_cache.clear()

    # Bắt đầu ghi log file
    log_filename = "rag_test_log.txt"
    sys.stdout = Logger(log_filename)

    print("\n" + "★"*80)
    print(f"🚀 BẮT ĐẦU CHẠY PIPELINE VÀO LÚC: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("★"*80)
    print(f" 👉 Câu hỏi gốc: {query}")
    print(f" 👉 Nhánh Action: {intent_action}")
    print(f" 👉 RAG Context truyền vào: {rag_context}")
    print("-" * 80)

    # State khởi tạo
    state: GraphState = {
        "session_id": "test-manual-session",
        "user_query": query,
        "standalone_query": query,
        "intent_action": intent_action,
        "rag_context": rag_context,
    }

    start_t = time.time()
    try:
        # Gọi Graph
        final_state = proceed_rag_search_pipeline(state)
        elapsed = time.time() - start_t

        print("\n" + "="*80)
        print(f"🏁 TỔNG KẾT SAU KHI QUA MỌI NODE ({elapsed:.2f}s)")
        print("="*80)
        
        print("\n[🎯 NGỮ CẢNH CUỐI CÙNG ĐƯỢC GỘP CHO LLM CHÍNH]")
        print("-" * 80)
        final_resp = final_state.get("final_response", "")
        if final_resp:
            print(final_resp)
        else:
            print("(Không có dữ liệu trả về)")
        print("-" * 80)
        
        print("\n[📊 THỐNG KÊ STATE TRẢ VỀ]")
        print(f" 1. Nguồn kết luận (Source): {final_state.get('response_source')}")
        citations = final_state.get('web_search_citations') or []
        print(f" 2. Trích dẫn URL (Citations): {len(citations)} trích dẫn")
        for i, c in enumerate(citations):
            print(f"    - [{i+1}] {c['text']}: {c['url']}")
            
        print(f" 3. Truy vấn sinh LLM phụ (UFM queries): {final_state.get('ufm_search_queries')}")
        print(f" 4. Truy vấn sinh PR (PR query)        : {final_state.get('pr_search_query')}")
        print(f" 5. Lưu cache? (Cache Hit)           : {final_state.get('search_cache_hit')}")

    except Exception as e:
        print(f"\n❌ LỖI NGHIÊM TRỌNG TRONG PIPELINE: {e}")
    finally:
        # Trả lại sys.stdout gốc
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal
        print(f"\n✅ Đã hoàn tất! Toàn bộ quá trình chạy node và câu trả lời đã được lưu tại: {log_filename}")


if __name__ == "__main__":
    main()
