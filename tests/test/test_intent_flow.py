"""
Test script cho luồng Phân loại Ý định (Intent Node).
Kiểm tra:
  1. Câu mở đầu siêu ngắn (CHAO_HOI) -> Nhận GREET template ngay
  2. Câu hỏi trực diện (THONG_TIN_TUYEN_SINH) -> Đi tới RAG Node
  3. Câu hỏi form mẫu (TAO_MAU_DON) -> Đi tới Form Node
  4. Câu hỏi lạc đề (CAU_HOI_LAC_DE) -> Bị khoá ở BLOCK_FALLBACK
"""

import os
import sys
import time
from typing import TypedDict

# Import môi trường
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.intent_node import intent_node, intent_router


def print_banner(title: str):
    print(f"\n{'='*80}")
    print(f"  ✨ {title.upper()}")
    print(f"{'='*80}")


def run_intent_test(scenario_name: str, standalone_query: str):
    print_banner(scenario_name)
    print(f"   [Input] standalone_query = \"{standalone_query}\"")
    
    # Khởi tạo giả lập State từ các node trước truyền sang
    initial_state: GraphState = {
        "session_id": "test_intent",
        "chat_history": [],
        "user_query": standalone_query,
        "standalone_query": standalone_query, # Đã qua Context Node
    }
    
    start_time = time.time()
    
    # Chạy Intent Node
    out_state = intent_node(initial_state)
    
    # Định tuyến (như LangGraph Edge)
    out_state["next_node"] = intent_router(out_state)
    
    elapsed = time.time() - start_time
    status = "✅ PASS" if out_state["next_node"] != "response" or out_state.get("final_response") else "🔴 FAIL"
    
    print(f"\n   ┌──────────────────────────────────────────────────")
    print(f"   │ 📍 INTENT NODE → {status} (Tổng thời gian: {elapsed:.3f}s)")
    print(f"   ├──────────────────────────────────────────────────")
    print(f"   │  intent         : {out_state.get('intent')}")
    print(f"   │  intent_action  : {out_state.get('intent_action')}")
    print(f"   │  next_node      : {out_state.get('next_node')}  <-- Hướng tiếp theo")
    
    if out_state.get("intent_summary"):
        print(f"   │  summary        : {out_state['intent_summary'][:60]}")
        
    if out_state.get("final_response"):
        print(f"   │  final_response : {out_state['final_response'][:80]}...")
        print(f"   │  source         : {out_state.get('response_source')}")
    print(f"   └──────────────────────────────────────────────────\n")


if __name__ == "__main__":
    
    # SCENARIO 1: Chào hỏi siêu ngắn (< 5 ký tự)
    # Kỳ vọng: Bắt hàm CHAO_HOI ngay, KHÔNG GỌI Qwen, lấy GREET template → next_node = response
    run_intent_test(
        "TEST 1: CÂU CHÀO NGẮN (< 5 ký tự)",
        "Alo "
    )
    time.sleep(1)
    
    # SCENARIO 2: Tìm kiếm thông tin cốt lõi
    # Kỳ vọng: Qwen phân loại THONG_TIN_TUYEN_SINH → Action = PROCEED_RAG → next_node = rag
    run_intent_test(
        "TEST 2: HỎI TRỰC DIỆN TUYỂN SINH RAG",
        "Chỉ tiêu xét tuyển học bạ vào ngành Tài chính năm nay thế nào?"
    )
    time.sleep(1)

    # SCENARIO 3: Xin mẫu đơn
    # Kỳ vọng: Qwen phân loại TAO_MAU_DON → Action = PROCEED_FORM → next_node = form
    run_intent_test(
        "TEST 3: YÊU CẦU MẪU ĐƠN (FORM AGENT)",
        "Gửi cho em file word giấy cam đoan nhập học với ạ"
    )
    time.sleep(1)

    # SCENARIO 4: Lạc đề, nằm ngoài phạm vi
    # Kỳ vọng: Qwen phân loại CAU_HOI_LAC_DE → Action = BLOCK_FALLBACK → next_node = response (Kèm tin nhắn cự tuyệt)
    run_intent_test(
        "TEST 4: HỎI LẠC ĐỀ (BLOCK FALLBACK)",
        "Tại sao trời lại mưa thế giới này thật kì diệu"
    )
    time.sleep(1)
