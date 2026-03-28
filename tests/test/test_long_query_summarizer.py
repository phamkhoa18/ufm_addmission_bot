# test/test_long_query_summarizer.py
# ============================================================
# SCRIPT KIỂM TRA LUỒNG TÓM TẮT QUERY DÀI
# Test 3 trường hợp:
#   1. Query ngắn (< 1999 chars) → đi qua bình thường
#   2. Query dài (1999-2000 chars) → được LLM tóm tắt
#   3. Query quá dài (> 2000 chars) → bị chặn DoS
# ============================================================
# Chạy: python test/test_long_query_summarizer.py

import sys
import time
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from app.services.langgraph.nodes.fast_scan_node import fast_scan_node
from app.core.config import query_flow_config


def build_test_state(query: str) -> dict:
    """Tạo state mẫu cho test."""
    return {
        "user_query": query,
        "chat_history": [],
        "original_query": "",
        "query_was_summarized": False,
        "normalized_query": "",
        "standalone_query": "",
        "fast_scan_passed": None,
        "fast_scan_blocked_layer": None,
        "fast_scan_message": "",
        "next_node": "",
        "final_response": "",
        "response_source": "",
    }


def print_result(test_num: int, label: str, query_len: int, state: dict, elapsed: float):
    """In kết quả test đẹp."""
    passed = state.get("fast_scan_passed")
    summarized = state.get("query_was_summarized", False)
    original = state.get("original_query", "")
    user_query = state.get("user_query", "")

    print(f"\n{'━'*70}")
    print(f"  TEST #{test_num}: {label}")
    print(f"  Input: {query_len} ký tự")
    print(f"{'━'*70}")

    if not passed:
        print(f"  ❌ CHẶN tại Layer {state.get('fast_scan_blocked_layer')}")
        print(f"     Message: {state.get('fast_scan_message', '')}")
        print(f"     Response: {state.get('final_response', '')[:120]}")
    else:
        print(f"  ✅ PASS")
        if summarized:
            print(f"     🔄 Query ĐÃ TÓM TẮT:")
            print(f"        Gốc ({len(original)} chars):")
            print(f"        \"{(original[:150] + '...') if len(original) > 150 else original}\"")
            print(f"        Tóm tắt ({len(user_query)} chars):")
            print(f"        👉 \"{user_query}\"")
        else:
            print(f"     📝 Query bình thường (không cần tóm tắt)")

    print(f"     ⏱️  Thời gian: {elapsed:.3f}s")
    print(f"     Message: {state.get('fast_scan_message', '')}")
    print(f"  {'═'*56}")


def main():
    print()
    print("╔" + "═"*70 + "╗")
    print("║   UFM ADMISSION BOT — LONG QUERY SUMMARIZER TESTER              ║")
    print("║   Test luồng: Short → Summarize → DoS Block                     ║")
    print("╚" + "═"*70 + "╝")
    print()
    print(f"  Config:")
    print(f"    max_input_chars:     {query_flow_config.input_validation.max_input_chars}")
    print(f"    summarize_threshold: {query_flow_config.input_validation.summarize_threshold}")
    print(f"    summarizer model:    {query_flow_config.long_query_summarizer.provider}/{query_flow_config.long_query_summarizer.model}")
    print(f"    summarizer tokens:   {query_flow_config.long_query_summarizer.max_tokens}")

    # ══════════════════════════════════════════════
    # TEST 1: Query ngắn → Đi qua bình thường
    # ══════════════════════════════════════════════
    short_query = "Học phí ngành Công nghệ thông tin năm 2026 là bao nhiêu?"
    state1 = build_test_state(short_query)
    t1 = time.time()
    result1 = fast_scan_node(state1)
    e1 = time.time() - t1
    print_result(1, "Query NGẮN (< 1999 chars) → Pass bình thường", len(short_query), result1, e1)

    # ══════════════════════════════════════════════
    # TEST 2: Query dài tự nhiên → Được LLM tóm tắt
    # ══════════════════════════════════════════════
    long_query = (
        "Em chào anh chị ạ, em là học sinh lớp 12 ở Bình Dương. "
        "Em rất muốn hỏi về ngành Công nghệ thông tin ở UFM, "
        "em nghe nói là ngành này học phí cao lắm, mà em thì nhà không có điều kiện cho lắm. "
        "Em muốn biết là có học bổng gì không, rồi còn điều kiện xét tuyển năm 2026 là như thế nào? "
        "Em thi tổ hợp A01, em được 25 điểm. Ngoài ra em cũng muốn biết thêm về chương trình đào tạo, "
        "có được thực tập ở công ty nào không, rồi ký túc xá có không, phí bao nhiêu? "
        "Em cũng nghe nói UFM có liên kết quốc tế gì đó, em muốn biết thêm. "
        "À em cũng muốn hỏi là nếu em đăng ký ngành Marketing thì điều kiện có khác không? "
        "Vì em cũng thích Marketing lắm, nhưng mà không biết ngành nào phù hợp hơn. "
        "Rồi còn vấn đề nữa là em muốn biết UFM có câu lạc bộ sinh viên gì không, "
        "em rất thích hoạt động ngoại khóa. Cảm ơn anh chị nhiều ạ! "
        "Em mong anh chị giúp em giải đáp hết tất cả các thắc mắc trên ạ. "
        "Em đã tìm hiểu trên website nhưng thông tin hơi nhiều quá, em không biết đâu là đúng. "
        "Em cũng hỏi mấy anh chị đi trước nhưng mỗi người nói một kiểu. "
        "Nên em muốn hỏi trực tiếp chatbot xem thông tin chính xác như thế nào ạ. "
        "À em quên nữa, em còn muốn hỏi về việc chuyển ngành nữa, nếu em vào ngành CNTT rồi mà "
        "muốn chuyển sang Marketing thì có được không, quy trình như thế nào? "
        "Rồi học phí có thay đổi không nếu em chuyển ngành? "
        "Em cũng nghe nói là UFM có vị trí rất đẹp ở quận 7, "
        "em muốn biết thêm về cơ sở vật chất, phòng lab, thư viện có tốt không ạ? "
        "Em rất mong nhận được phản hồi sớm ạ, em cần biết thông tin để quyết định trước ngày 30/6! "
    )
    # Pad thêm cho đủ >= 1999 chars nếu cần
    while len(long_query) < 1999:
        long_query += " Em mong nhận được tư vấn chi tiết ạ!"
    long_query = long_query[:2000]  # Đảm bảo trong khoảng [1999, 2000]

    state2 = build_test_state(long_query)
    t2 = time.time()
    result2 = fast_scan_node(state2)
    e2 = time.time() - t2
    print_result(2, "Query DÀI (1999-2000 chars) → LLM tóm tắt", len(long_query), result2, e2)

    # ══════════════════════════════════════════════
    # TEST 3: Query quá dài → Chặn DoS
    # ══════════════════════════════════════════════
    dos_query = "A" * 2000
    state3 = build_test_state(dos_query)
    t3 = time.time()
    result3 = fast_scan_node(state3)
    e3 = time.time() - t3
    print_result(3, "Query QUÁ DÀI (> 2000 chars) → Chặn DoS", len(dos_query), result3, e3)

    # ══════════════════════════════════════════════
    # TỔNG KẾT
    # ══════════════════════════════════════════════
    print(f"\n{'╔' + '═'*70 + '╗'}")
    passed_count = sum([
        result1.get("fast_scan_passed", False),
        result2.get("fast_scan_passed", False),
        not result3.get("fast_scan_passed", True),  # Test 3 PHẢI bị chặn
    ])
    was_summarized = result2.get("query_was_summarized", False)
    status = "✅ ALL PASSED" if passed_count == 3 and was_summarized else "⚠️ CÓ LỖI"
    print(f"║  🏁 KẾT QUẢ: {status:<56s} ║")
    print(f"║  Test 1 (short):     {'✅ PASS' if result1.get('fast_scan_passed') else '❌ FAIL':<48s} ║")
    print(f"║  Test 2 (summarize): {'✅ SUMMARIZED' if was_summarized else '❌ NOT SUMMARIZED':<48s} ║")
    print(f"║  Test 3 (DoS block): {'✅ BLOCKED' if not result3.get('fast_scan_passed') else '❌ NOT BLOCKED':<48s} ║")
    print(f"{'╚' + '═'*70 + '╝'}")


if __name__ == "__main__":
    main()
