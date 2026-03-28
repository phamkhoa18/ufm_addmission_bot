"""
Script kiểm tra toàn bộ Fallback Models trên OpenRouter.
Gửi request siêu nhỏ (max 5 tokens) để check xem model nào còn sống, 
model nào báo lỗi (402, 503, 404, Timeout).
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import query_flow_config

def check_chat_model(model_name: str, api_key: str, base_url: str):
    """
    Test các model sinh text/chat.
    """
    req_data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
        "temperature": 0.0
    }
    return _send_request(f"{base_url}/chat/completions", req_data, api_key)

def check_embedding_model(model_name: str, api_key: str, base_url: str):
    """
    Test các model nhóm embedding.
    """
    req_data = {
        "model": model_name,
        "input": "test request for pinging embedding model"
    }
    return _send_request(f"{base_url}/embeddings", req_data, api_key)

def _send_request(url: str, req_data: dict, api_key: str):
    data_bytes = json.dumps(req_data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
        },
        method="POST"
    )
    
    start_t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
            elapsed = time.time() - start_t
            return True, f"Online ({elapsed:.2f}s)"
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_t
        error_body = e.read().decode("utf-8", errors="ignore")
        try:
            err_json = json.loads(error_body)
            err_msg = err_json.get("error", {}).get("message", "Unknown API error")
        except:
            err_msg = error_body[:100]
            
        # Format lại lỗi cho dễ nhìn
        if e.code == 402:
            clean_msg = "Lỗi 402: Hết credits OpenRouter"
        elif e.code == 404:
            clean_msg = "Lỗi 404: Model không tồn tại/bị xóa"
        elif e.code == 502 or e.code == 503:
            clean_msg = f"Lỗi {e.code}: Model đang sập/bảo trì"
        else:
            clean_msg = f"Lỗi {e.code}: {err_msg[:50]}..."
            
        return False, f"{clean_msg} ({elapsed:.2f}s)"
    except Exception as e:
        elapsed = time.time() - start_t
        return False, f"Lỗi mạng: {str(e)[:40]} ({elapsed:.2f}s)"

def main():
    print("\n" + "★"*95)
    print("  🚀 KIỂM TRA TÌNH TRẠNG SỨC KHỎE CỦA CÁC FALLBACK MODEL TRÊN OPENROUTER")
    print("★"*95 + "\n")

    fb = query_flow_config.fallback_models
    api_key = query_flow_config.api_keys.openrouter_api_key
    base_url = query_flow_config.api_keys.openrouter_base_url

    if not api_key:
        print("❌ KHÔNG TÌM THẤY OPENROUTER_API_KEY TRONG .env!")
        return

    groups = [
        ("light", fb.light_models),
        ("medium", fb.medium_models),
        ("search", fb.search_models),
        ("embedding", fb.embedding_models),
        ("guard", fb.guard_models),
        ("main_bot", fb.main_bot_models)
    ]

    total_models = 0
    passed_models = 0

    for group_name, cfg in groups:
        print(f"━" * 95)
        print(f" 📦 NHÓM: {group_name.upper()}")
        print(f"━" * 95)

        all_entries = [cfg.primary] + cfg.fallbacks
        
        for idx, entry in enumerate(all_entries):
            total_models += 1
            label = "PRIMARY" if idx == 0 else f"BACKUP {idx}"

            # Lấy API key + base_url theo provider của từng model
            entry_api_key = query_flow_config.api_keys.get_key(entry.provider)
            entry_base_url = query_flow_config.api_keys.get_base_url(entry.provider)

            if not entry_api_key:
                print(f" ⚠️ [{label.center(9)}] {entry.provider}/{entry.model} -> Không có API key cho provider '{entry.provider}'")
                continue
            
            display = f"{entry.provider}/{entry.model}"
            print(f" ⏳ Đang test {label.ljust(9)}: {display} ...", end="", flush=True)
            
            if group_name == "embedding":
                ok, msg = check_embedding_model(entry.model, entry_api_key, entry_base_url)
            else:
                ok, msg = check_chat_model(entry.model, entry_api_key, entry_base_url)
                
            if ok:
                passed_models += 1
                
            icon = "✅" if ok else "❌"
            print(f"\r {icon} [{label.center(9)}] {display.ljust(45)} -> {msg.ljust(40)}")
            
            time.sleep(0.5)
            
    print("\n" + "="*95)
    print(f"  🎯 TỔNG KẾT: {passed_models}/{total_models} models đang hoạt động bình thường")
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
