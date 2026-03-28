# test/testtk.py
# ============================================================
# SCRIPT KIỂM TRA SỐ LƯỢNG TOKEN CHO TẤT CẢ CÁC MODEL BẰNG TIKTOKEN
# KHÔNG gọi API, chỉ dùng thư viện cục bộ
# ============================================================
import sys
from pathlib import Path
import tiktoken

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import query_flow_config

def get_encoding_for_model(model_name: str):
    """
    Cố gắng lấy encoding của tiktoken cho model.
    Nếu tiktoken không hỗ trợ trực tiếp model này (ví dụ model của Google/Llama/Qwen),
    sẽ fallback về 'cl100k_base' (encoding mặc định phổ biến của OpenAI cho GPT-3.5/4/4o).
    """
    # Xử lý một số tên model đặc thù sang chuẩn OpenAI để tiktoken nhận diện
    mapping = {
        "gpt-4o-mini-search-preview": "gpt-4o",
        "gpt-4o-mini": "gpt-4o",
        "gpt-4o": "gpt-4o",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base"
    }
    
    # Thử mapping trước
    for key, enc_name in mapping.items():
        if key in model_name.lower():
            try:
                if enc_name in ["gpt-4o", "cl100k_base"]:
                    return tiktoken.encoding_for_model(enc_name)
                return tiktoken.get_encoding(enc_name)
            except KeyError:
                pass
                
    try:
        # Nhờ tiktoken đoán bừa name
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Các model OpenSource / Google đa phần không có trong tiktoken
        # Dùng cl100k_base làm xấp xỉ tương đối (Geminiflash, Llama3, Qwen cũng dùng BPE tương tự)
        return tiktoken.get_encoding("cl100k_base")

def test_tokens():
    print("="*80)
    print("   UFM ADMISSION BOT — TOKEN COUNTER (LOCAL TIKTOKEN)   ")
    print("="*80)
    
    text = "Trong thời đại công nghệ phát triển mạnh mẽ như hiện nay, trí tuệ nhân tạo đang dần trở thành một phần không thể thiếu trong cuộc sống của con người, từ những ứng dụng nhỏ như gợi ý nội dung trên mạng xã hội cho đến các համակարգ lớn phục vụ doanh nghiệp và giáo dục. Việc tiếp cận và ứng dụng AI không còn là điều xa vời mà đang dần trở nên phổ biến, đặc biệt đối với sinh viên và những người làm trong lĩnh vực công nghệ thông tin. Tuy nhiên, để tận dụng được sức mạnh của AI một cách hiệu quả, người dùng không chỉ cần hiểu về cách sử dụng mà còn cần nắm được nguyên lý hoạt động cơ bản, cách dữ liệu được xử lý, cũng như những giới hạn của hệ thống. Một trong những yếu tố quan trọng khi làm việc với các mô hình ngôn ngữ lớn là việc quản lý token – đơn vị cơ bản mà mô hình sử dụng để hiểu và sinh văn bản. Việc tối ưu số lượng token không chỉ giúp giảm chi phí mà còn cải thiện tốc độ xử lý và độ chính xác của kết quả. Bên cạnh đó, trong các hệ thống như chatbot tuyển sinh hay hệ thống hỏi đáp thông minh, việc chia nhỏ dữ liệu thành các đoạn phù hợp (chunking) đóng vai trò then chốt trong việc nâng cao chất lượng truy xuất thông tin. Nếu dữ liệu được chia quá nhỏ, ngữ cảnh có thể bị mất; ngược lại, nếu quá lớn, hệ thống có thể bị quá tải hoặc trả về kết quả không chính xác. Do đó, việc cân bằng giữa kích thước và nội dung của mỗi đoạn là một bài toán cần được nghiên cứu kỹ lưỡng. Ngoài ra, tiếng Việt cũng mang những đặc thù riêng khiến việc xử lý ngôn ngữ trở nên phức tạp hơn so với tiếng Anh, chẳng hạn như dấu thanh, từ ghép và cấu trúc câu linh hoạt. Điều này đòi hỏi các mô hình phải được huấn luyện hoặc tinh chỉnh phù hợp để đạt hiệu quả tối ưu. Không chỉ dừng lại ở kỹ thuật, việc xây dựng một hệ thống AI còn cần sự kết hợp giữa tư duy sản phẩm, trải nghiệm người dùng và khả năng mở rộng trong tương lai. Một hệ thống tốt không chỉ trả lời đúng mà còn phải trả lời nhanh, dễ hiểu và mang lại giá trị thực sự cho người sử dụng. Trong bối cảnh cạnh tranh ngày càng cao, những ai biết tận dụng AI một cách thông minh sẽ có lợi thế rất lớn, سواء trong học tập, công việc hay kinh doanh. Chính vì vậy, việc đầu tư thời gian để tìm hiểu, thử nghiệm và tối ưu các hệ thống AI ngay từ bây giờ là một quyết định hoàn toàn đúng đắn, giúp bạn không chỉ bắt kịp xu hướng mà còn có thể dẫn đầu trong lĩnh vực của mình trong tương lai.Trong thời đại công nghệ phát triển mạnh mẽ như hiện nay, trí tuệ nhân tạo đang dần trở thành một phần không thể thiếu trong cuộc sống của con người, từ những ứng dụng nhỏ như gợi ý nội dung trên mạng xã hội cho đến các համակարգ lớn phục vụ doanh nghiệp và giáo dục. Việc tiếp cận và ứng dụng AI không còn là điều xa vời mà đang dần trở nên phổ biến, đặc biệt đối với sinh viên và những người làm trong lĩnh vực công nghệ thông tin. Tuy nhiên, để tận dụng được sức mạnh của AI một cách hiệu quả, người dùng không chỉ cần hiểu về cách sử dụng mà còn cần nắm được nguyên lý hoạt động cơ bản, cách dữ liệu được xử lý, cũng như những giới hạn của hệ thống. Một trong những yếu tố quan trọng khi làm việc với các mô hình ngôn ngữ lớn là việc quản lý token – đơn vị cơ bản mà mô hình sử dụng để hiểu và sinh văn bản. Việc tối ưu số lượng token không chỉ giúp giảm chi phí mà còn cải thiện tốc độ xử lý và độ chính xác của kết quả. Bên cạnh đó, trong các hệ thống như chatbot tuyển sinh hay hệ thống hỏi đáp thông minh, việc chia nhỏ dữ liệu thành các đoạn phù hợp (chunking) đóng vai trò then chốt trong việc nâng cao chất lượng truy xuất thông tin. Nếu dữ liệu được chia quá nhỏ, ngữ cảnh có thể bị mất; ngược lại, nếu quá lớn, hệ thống có thể bị quá tải hoặc trả về kết quả không chính xác. Do đó, việc cân bằng giữa kích thước và nội dung của mỗi đoạn là một bài toán cần được nghiên cứu kỹ lưỡng. Ngoài ra, tiếng Việt cũng mang những đặc thù riêng khiến việc xử lý ngôn ngữ trở nên phức tạp hơn so với tiếng Anh, chẳng hạn như dấu thanh, từ ghép và cấu trúc câu linh hoạt. Điều này đòi hỏi các mô hình phải được huấn luyện hoặc tinh chỉnh phù hợp để đạt hiệu quả tối ưu. Không chỉ dừng lại ở kỹ thuật, việc xây dựng một hệ thống AI còn cần sự kết hợp giữa tư duy sản phẩm, trải nghiệm người dùng và khả năng mở rộng trong tương lai. Một hệ thống tốt không chỉ trả lời đúng mà còn phải trả lời nhanh, dễ hiểu và mang lại giá trị thực sự cho người sử dụng. Trong bối cảnh cạnh tranh ngày càng cao, những ai biết tận dụng AI một cách thông minh sẽ có lợi thế rất lớn, سواء trong học tập, công việc hay kinh doanh. Chính vì vậy, việc đầu tư thời gian để tìm hiểu, thử nghiệm và tối ưu các hệ thống AI ngay từ bây giờ là một quyết định hoàn toàn đúng đắn, giúp bạn không chỉ bắt kịp xu hướng mà còn có thể dẫn đầu trong lĩnh vực của mình trong tương lai."
    print(f"Bản text mẫu ({len(text)} ký tự): '{text}'\n")
    
    # Thu thập tất cả cấu hình model đang dùng trong hệ thống
    models_to_test = {
        "Long Query Summarizer": query_flow_config.long_query_summarizer.model,
        "Fast Guard (Layer 2a)": query_flow_config.prompt_guard_fast.model,
        "Deep Guard (Layer 2b)": query_flow_config.prompt_guard_deep.model,
        "Intent Router": query_flow_config.semantic_router.model,
        "Reformulator (Context)": query_flow_config.query_reformulation.model,
        "Multi-Query": query_flow_config.multi_query.model,
        "Main Bot (Response)": query_flow_config.main_bot.model,
    }
    
    # Tùy config mà PR search / UFM search có model độc lập không
    if hasattr(query_flow_config, "pr_query") and hasattr(query_flow_config.pr_query, "model"):
        models_to_test["PR Query Node"] = query_flow_config.pr_query.model
        
    if hasattr(query_flow_config, "main_bot") and hasattr(query_flow_config.main_bot, "model"):
         models_to_test["Web Search Synthesizer"] = query_flow_config.main_bot.model # Thường lấy chung main_bot
    
    # Loại bỏ duplicate model name để in cho gọn nếu muốn, nhưng in theo role sẽ rõ hơn
    
    print(f"{'VAI TRÒ (NODE)':<30} | {'MODEL TỪ CONFIG':<40} | {'TOKEN COUNT'}")
    print("-" * 85)
    
    for role, model_name in models_to_test.items():
        if not model_name: 
            continue
            
        enc = get_encoding_for_model(model_name)
        tokens = enc.encode(text)
        token_count = len(tokens)
        
        # Xóa prefix tên họ (vd "google/gemini..." -> "gemini...") cho gọn bảng
        short_model = model_name
        if "/" in short_model:
            short_model = short_model.split("/")[-1]
            
        print(f"{role:<30} | {short_model:<40} | {token_count} tokens")

    print("\n" + "="*80)
    print("* Lưu ý: Tiktoken hỗ trợ chuẩn nhất cho dòng OpenAI (GPT-4o, GPT-3.5).")
    print("  Với Llama, Qwen, Gemini, mã script đang dùng cl100k_base làm chuẩn đếm xấp xỉ tương đương.")
    print("="*80)

if __name__ == "__main__":
    test_tokens()