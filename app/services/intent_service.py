"""
Intent Service — Phân loại ý định bằng LLM (Primary + Backup Retry).

Luồng xử lý:
  1. Edge case pre-check (miễn phí, ~0ms):
     - Câu < min_query_length ký tự → CHAO_HOI
  2. Primary LLM — Qwen (~400ms, ~$0.00003):
     - Gọi API, nhận JSON {"intent": "...", "summary": "..."}
     - Validate JSON + validate intent name
  3. Nếu JSON KHÔNG HỢP LỆ → Retry Backup LLM (~200ms, ~$0.000002):
     - Gọi Gemini Flash Lite với cùng system_prompt
     - Parse lại + validate
  4. Nếu Backup vẫn lỗi → KHONG_XAC_DINH (fallback cứng)

Tại sao cần Backup Model?
  - Qwen đôi khi trả: markdown block (```json...```), text thừa, JSON sai format
  - Gemini Flash Lite tuân thủ JSON format tốt hơn trong edge cases
  - Backup chỉ tốn ~$0.000002 thêm khi gặp lỗi — cực rẻ
"""

import json
import re
import time
import urllib.request
import urllib.error
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager


# ================================================================
# JSON VALIDATOR + CLEANER
# ================================================================
def _extract_json(raw: str) -> dict | None:
    """
    Cố gắng trích xuất JSON hợp lệ từ text thô.
    Xử lý các edge case phổ biến từ LLM output:
      - Text thừa trước/sau JSON object
      - Markdown code block: ```json ... ```
      - Single quotes thay vì double quotes
      - JSON bị truncate
    """
    if not raw or not raw.strip():
        return None

    # Bước 1: Xoá markdown code block nếu có
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Bước 2: Tìm JSON object đầu tiên trong text (bắt đầu = {, kết thúc = })
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)

    # Bước 3: Parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Bước 4: Thử sửa single quotes → double quotes
        try:
            fixed = json_str.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _validate_parsed(parsed: dict, allowed_intents: set, fallback: str) -> tuple:
    """
    Validate kết quả đã parse từ LLM.
    Trả về (intent, summary) đã qua kiểm tra.
    """
    intent  = str(parsed.get("intent",  "")).strip()
    summary = str(parsed.get("summary", "")).strip()

    if not intent or intent not in allowed_intents:
        return fallback, summary or ""

    return intent, summary


# ================================================================
# GỌI API LLM (Dùng chung cho cả Primary và Backup)
# ================================================================
def _call_llm(
    standalone_query: str,
    system_prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    """
    Gọi 1 lần API chat completion.
    Trả về raw content string. Raise exception nếu lỗi network/HTTP.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "UFM-Admission-Bot/1.0",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": standalone_query},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"]


# ================================================================
# TẦNG PHÂN LOẠI — Primary (Qwen) + Backup (Gemini Flash Lite)
# ================================================================
def classify_by_llm(standalone_query: str) -> tuple:
    """
    Phân loại intent theo chiến lược Primary → Backup:

      Lần 1: Primary model (Qwen) → Validate JSON
              ✅ OK  → Trả kết quả ngay
              ❌ Lỗi JSON → Ghi log + chuyển sang Backup

      Lần 2: Backup model (Gemini Flash Lite) → Validate JSON
              ✅ OK  → Trả kết quả
              ❌ Lỗi → Trả fallback KHONG_XAC_DINH

    Returns: (intent_name: str, summary: str)
    """
    semantic_cfg = query_flow_config.semantic_router
    validator_cfg = query_flow_config.intent_validator
    allowed_intents = set(semantic_cfg.allowed_intents)
    fallback_intent = validator_cfg.fallback_intent

    # Đọc primary + fallbacks trực tiếp từ models_config.yaml["semantic_router"]
    from app.core.config import models_yaml_data
    node_cfg = models_yaml_data.get("semantic_router", {})
    fb_settings = query_flow_config.fallback_models.settings

    # Xây model chain: [primary, fallback1, fallback2, ...]
    model_chain = []
    if node_cfg.get("model"):
        model_chain.append({
            "provider": node_cfg.get("provider", "openrouter"),
            "model": node_cfg["model"],
        })
    for fb in (node_cfg.get("fallbacks", []) or []):
        model_chain.append({
            "provider": fb.get("provider", "openrouter"),
            "model": fb.get("model", ""),
        })

    if not model_chain:
        print(f"   [IntentService] ⚠️ Không có model nào trong semantic_router → {fallback_intent}")
        return fallback_intent, standalone_query

    # ── DUYỆT QUA TỪNG MODEL (Primary → Fallback 1 → Fallback 2) ──
    for i, entry in enumerate(model_chain):
        label = "Primary" if i == 0 else f"Fallback #{i}"
        
        api_key = query_flow_config.api_keys.get_key(entry["provider"])
        base_url = query_flow_config.api_keys.get_base_url(entry["provider"])

        if not api_key:
            print(f"   [IntentService] ⚠️ Chưa có API key cho {entry['provider']} → Bỏ qua {label}")
            continue

        start_t = time.time()
        try:
            # Ưu tiên lấy max_tokens/temperature từ config gốc
            # hoặc ghi đè cho phù hợp với intent classification
            raw = _call_llm(
                standalone_query=standalone_query,
                system_prompt=prompt_manager.get_system("intent_classification"),
                api_key=api_key,
                base_url=base_url,
                model=entry["model"],
                temperature=query_flow_config.semantic_router.temperature,
                max_tokens=query_flow_config.semantic_router.max_tokens,
                timeout=query_flow_config.semantic_router.timeout_seconds,
            )
            elapsed = time.time() - start_t
            parsed = _extract_json(raw)

            if parsed is not None:
                intent, summary = _validate_parsed(parsed, allowed_intents, fallback_intent)
                print(
                    f"   [IntentService] ✅ {label} ({entry['provider']}/{entry['model']}) "
                    f"— {elapsed:.3f}s "
                    f"→ intent='{intent}'"
                )
                return intent, summary
            else:
                print(
                    f"   [IntentService] ⚠️ {label} JSON lỗi định dạng "
                    f"({elapsed:.3f}s) — raw: {raw[:80]!r}"
                )
                # Tiếp tục vòng lặp thử model tiếp theo

        except urllib.error.HTTPError as e:
            elapsed = time.time() - start_t
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                error_body = ""
            print(f"   [IntentService] ⚠️ {label} HTTP {e.code} ({elapsed:.3f}s) → {error_body[:50]}")
        except Exception as e:
            elapsed = time.time() - start_t
            print(f"   [IntentService] ⚠️ {label} Error ({elapsed:.3f}s): {type(e).__name__}: {str(e)[:50]}")

        # Đợi chút trước khi chuyển sang model dự phòng tiếp theo
        if i < len(model_chain) - 1:
            time.sleep(fb_settings.retry_delay_ms / 1000)

    # Nếu tất cả các model đều tạch (timeout hoặc JSON lỗi hết)
    print(f"   [IntentService] ❌ Toàn bộ model đều thất bại → Fallback cứng ({fallback_intent})")
    return fallback_intent, standalone_query


# ================================================================
# HÀM CHÍNH: CLASSIFY INTENT
# ================================================================
def classify_intent(standalone_query: str) -> dict:
    """
    Phân loại intent của câu hỏi.

    Args:
        standalone_query: Câu hỏi đã reformulate từ Context Node.

    Returns dict:
        {
            "intent":         "HOC_PHI_HOC_BONG",
            "intent_summary": "học phí ngành marketing",
            "intent_action":  "PROCEED_RAG",
        }
    """
    action_cfg = query_flow_config.intent_actions
    query = standalone_query.strip()

    # ── Edge case: Câu quá ngắn → Coi là mở đầu hội thoại ──
    min_len = query_flow_config.intent_threshold.min_query_length
    if len(query) < min_len:
        intent = "CHAO_HOI"
        print(f"   [IntentService] Câu ngắn ({len(query)} ký tự < {min_len}) → {intent}")
        return {
            "intent": intent,
            "intent_summary": query,
            "intent_action": action_cfg.get_action(intent),
        }

    # ── Phân loại bằng LLM (Primary → Backup nếu JSON lỗi) ──
    start = time.time()
    intent, summary = classify_by_llm(query)
    elapsed = time.time() - start

    print(f"   [IntentService — tổng {elapsed:.3f}s] intent='{intent}' | summary='{summary[:60]}'")

    return {
        "intent": intent,
        "intent_summary": summary,
        "intent_action": action_cfg.get_action(intent),
    }
