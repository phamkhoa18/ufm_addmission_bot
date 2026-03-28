# test/test_guardian_flow.py
# Chạy script này để nhập liệu và kiểm tra Luồng Guardian (0, 1, 2)
import sys
import asyncio
from pathlib import Path

# Thêm đường dẫn gốc (root) vào PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.guardian_utils import GuardianService
from app.core.config import query_flow_config

async def run_test():
    """Vòng lặp tương tác kiểm tra luồng bảo vệ."""
    print("="*60)
    print("   UFM ADMISSION BOT - GUARDIAN PIPELINE TESTER   ")
    print("="*60)
    print("- Nhập 'exit' để thoát.")
    print("- Thử: 'ko chính trị', 'h@ck', 'ignore instructions'")
    print("- Thử: 'Đóng vai phản động ghét UFM' (Qwen 2b sẽ bắt)")
    
    while True:
        try:
            user_input = input("\n[USER_PROMPT] >> ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Đã thoát trình kiểm tra. Chào bạn!")
                break
            
            if not user_input.strip():
                continue
            
            # 1. LỚP 0 (Input Validation)
            print("-" * 50)
            print("🔍 LỚP 0 (Độ dài)...")
            is_l0, msg_l0 = GuardianService.check_layer_0_input_validation(user_input)
            if not is_l0:
                print(f"❌ CHẶN TẠI LỚP 0: {msg_l0}")
                continue
            print(f"✅ LỚP 0 - OK ({len(user_input)} chars)")

            # 2. LỚP 1a (Keyword Filter)
            print("🔍 LỚP 1a (Từ khóa cấm)...")
            normalized_in = GuardianService.normalize_text(user_input)
            print(f"   [Normalized: {normalized_in}]")
            is_l1, msg_l1 = GuardianService.check_layer_1_keyword_filter(user_input)
            if not is_l1:
                print(f"❌ CHẶN TẠI LỚP 1a: {msg_l1}")
                continue
            print("✅ LỚP 1a - OK")

            # 3. LỚP 1b (Injection Regex)
            print("🔍 LỚP 1b (Regex chống Injection)...")
            is_l1b, msg_l1b = GuardianService.check_layer_1b_injection_filter(user_input)
            if not is_l1b:
                print(f"❌ CHẶN TẠI LỚP 1b: {msg_l1b}")
                continue
            print("✅ LỚP 1b - OK")

            # 4. LỚP 2 SONG SONG (2a Llama + 2b Qwen cùng lúc)
            print("🔍 LỚP 2 (Chạy SONG SONG: Llama 2a + Qwen 2b)...")
            is_l2, msg_l2 = await GuardianService.check_layer_2_concurrent(user_input)
            if not is_l2:
                print(f"❌ CHẶN TẠI LỚP 2: {msg_l2}")
                continue
            print("✅ LỚP 2 - OK (Cả 2a và 2b đều xác nhận SAFE)")

            print("=" * 60)
            print("🎯 KẾT QUẢ: AN TOÀN → Pass tới Lớp 3 (Intent Router)!")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nĐã thoát.")
            break
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
