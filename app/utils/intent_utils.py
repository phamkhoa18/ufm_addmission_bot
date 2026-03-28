# app/utils/intent_utils.py
# Chứa logic xử lý thực tế cho lớp PHÂN LOẠI Ý ĐỊNH (Layer 3)
# Layer 3.2: LLM Semantic Router (Qwen) → JSON {"intent": "...", "summary": "..."}
# Layer 3 Validator: Kiểm tra intent hợp lệ, fallback nếu sai

import json
import time
import urllib.request
import urllib.error
from typing import Tuple, Dict, Optional
from app.core.config import query_flow_config


class IntentService:
    """Dịch vụ phân loại ý định câu hỏi người dùng."""

    # ================================================================
    # HÀM GỌI API CHUNG (Tái sử dụng từ GuardianService pattern)
    # ================================================================
    @staticmethod
    def _call_llm_api(
        provider: str,
        model: str,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 100,
        response_format: str = None
    ) -> str:
        """Gọi API LLM (Groq/OpenRouter). Hỗ trợ response_format JSON."""
        api_key = query_flow_config.api_keys.get_key(provider)
        base_url = query_flow_config.api_keys.get_base_url(provider)

        if not api_key:
            raise ValueError(f"Chưa cấu hình API Key cho provider '{provider}'")

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UFM-Admission-Bot/1.0"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if response_format:
            data["response_format"] = {"type": response_format}

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()

    # ================================================================
    # LỚP 3.2: LLM Semantic Router
    # ================================================================
    @staticmethod
    def classify_intent(text: str) -> Dict:
        """
        Phân loại ý định bằng LLM Semantic Router.
        Trả về dict: {"intent": "...", "summary": "...", "raw": "..."}
        """
        config = query_flow_config.semantic_router
        start_time = time.time()

        try:
            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": text}
            ]
            output = IntentService._call_llm_api(
                provider=config.provider,
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=150,
                response_format=config.response_format
            )
            elapsed = time.time() - start_time

            # Parse JSON response
            try:
                result = json.loads(output)
                intent = result.get("intent", "").upper().strip()
                summary = result.get("summary", "")
            except json.JSONDecodeError:
                # Fallback: Nếu LLM không trả JSON hợp lệ
                intent = output.upper().strip()
                summary = ""

            return {
                "intent": intent,
                "summary": summary,
                "raw": output,
                "elapsed_s": round(elapsed, 2),
                "error": None
            }

        except ValueError as e:
            return {"intent": "", "summary": "", "raw": "", "elapsed_s": 0, "error": str(e)}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")[:300]
            return {"intent": "", "summary": "", "raw": "", "elapsed_s": 0, "error": f"API Error: {error_body}"}
        except Exception as e:
            return {"intent": "", "summary": "", "raw": "", "elapsed_s": 0, "error": str(e)}

    # ================================================================
    # LỚP 3 - VALIDATOR: Kiểm tra intent hợp lệ
    # ================================================================
    @staticmethod
    def validate_intent(intent: str) -> Tuple[bool, str]:
        """
        Kiểm tra intent có nằm trong danh sách allowed_intents không.
        Trả về (is_valid, validated_intent).
        """
        config_sr = query_flow_config.semantic_router
        config_val = query_flow_config.intent_validator

        if not config_val.enabled:
            return True, intent

        if intent in config_sr.allowed_intents:
            return True, intent

        # Thử fuzzy match (cắt khoảng trắng, viết hoa)
        normalized = intent.upper().replace(" ", "_").replace("-", "_")
        if normalized in config_sr.allowed_intents:
            return True, normalized

        # Không khớp → fallback
        return False, config_val.fallback_intent

    # ================================================================
    # LỚP 3 - FALLBACK ROUTER: Xử lý intent Nhóm 4
    # ================================================================
    @staticmethod
    def get_fallback_response(intent: str) -> Optional[str]:
        """
        Nếu intent thuộc Nhóm 4 (Bảo vệ), trả về fallback message.
        Trả về None nếu intent không thuộc nhóm fallback.
        """
        config = query_flow_config.semantic_router
        return config.fallbacks.get(intent, None)

    # ================================================================
    # LUỒNG TỔNG: classify_and_route (Kết hợp tất cả)
    # ================================================================
    @classmethod
    def classify_and_route(cls, text: str) -> Dict:
        """
        Luồng xử lý hoàn chỉnh Layer 3:
        1. LLM phân loại intent
        2. Validator kiểm tra hợp lệ
        3. Fallback Router xử lý Nhóm 4

        Trả về dict hoàn chỉnh.
        """
        # 3.2: LLM Semantic Router
        llm_result = cls.classify_intent(text)

        if llm_result["error"]:
            return {
                "intent": query_flow_config.intent_validator.fallback_intent,
                "summary": "",
                "validated": False,
                "fallback_msg": None,
                "action": "FALLBACK_ERROR",
                "llm_raw": llm_result["raw"],
                "elapsed_s": llm_result["elapsed_s"],
                "error": llm_result["error"]
            }

        # 3 Validator: Kiểm tra intent hợp lệ
        is_valid, validated_intent = cls.validate_intent(llm_result["intent"])

        # Kiểm tra Fallback Nhóm 4
        fallback_msg = cls.get_fallback_response(validated_intent)

        # Xác định action
        if fallback_msg:
            action = "BLOCK_FALLBACK"       # Nhóm 4: Chặn + Trả fallback
        elif validated_intent == "TAO_MAU_DON":
            action = "FORM_AGENT"           # Nhóm 1.5: Route sang FormAgent (không qua RAG)
        elif validated_intent == "CHAO_HOI":
            action = "GREETING"             # Nhóm 5: Chào hỏi
        elif validated_intent == "KHONG_XAC_DINH":
            action = "UNKNOWN"              # Nhóm 5: Không xác định
        elif not is_valid:
            action = "VALIDATOR_FALLBACK"   # LLM trả intent sai
        else:
            action = "PROCEED_RAG"          # Nhóm 1-3: Tiến tới RAG

        return {
            "intent": validated_intent,
            "summary": llm_result["summary"],
            "validated": is_valid,
            "fallback_msg": fallback_msg,
            "action": action,
            "llm_raw": llm_result["raw"],
            "elapsed_s": llm_result["elapsed_s"],
            "error": None
        }
