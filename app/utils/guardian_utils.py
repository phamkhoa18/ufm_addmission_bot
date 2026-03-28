# app/utils/guardian_utils.py
# Chứa logic xử lý thực tế cho các lớp bảo vệ (0, 1, 2)
# Fix: Async concurrent cho 2a & 2b, JSON parsing cho 2b

import re
import json
import asyncio
import urllib.request
import urllib.error
from typing import Tuple
from app.core.config import query_flow_config
from app.services.langgraph.nodes.context_node import _call_gemini_api
from app.core.prompts import prompt_manager


class GuardianService:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Chuẩn hóa văn bản: Chuyển chữ thường, thay thế Teencode/Leetspeak."""
        text = text.lower()
        
        teencode_map = query_flow_config.keyword_filter.teencode_map
        sorted_keys = sorted(teencode_map.keys(), key=len, reverse=True)
        
        for key in sorted_keys:
            text = re.sub(rf'\b{re.escape(key)}\b', teencode_map[key], text)
            
        return text

    @staticmethod
    def check_layer_0_input_validation(text: str) -> Tuple[bool, str]:
        """LỚP 0: Kiểm tra độ dài ký tự (hard limit chống DoS)."""
        config = query_flow_config.input_validation
        if len(text) > config.max_input_chars:
            return False, config.fallback_too_long
        return True, ""

    @staticmethod
    def check_layer_1_keyword_filter(text: str) -> Tuple[bool, str]:
        """LỚP 1a: Kiểm tra từ khóa cấm (nội dung nhạy cảm) bằng Regex."""
        config = query_flow_config.keyword_filter
        normalized_text = GuardianService.normalize_text(text)
        
        for pattern in config.banned_regex_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return False, config.fallback_message
        return True, ""

    @staticmethod
    def check_layer_1b_injection_filter(text: str) -> Tuple[bool, str]:
        """LỚP 1b: Regex chống Injection/Jailbreak sơ đẳng (0ms)."""
        config = query_flow_config.keyword_filter
        normalized_text = GuardianService.normalize_text(text)
        
        for pattern in config.injection_regex_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return False, config.fallback_injection
        return True, ""

    @staticmethod
    def _call_llm_api(
        config_section, 
        system_prompt: str,
        user_content: str,
    ) -> str:
        """Gọi API LLM theo đúng cấu hình chuyên biệt của từng Layer, KHÔNG trộn fallback chung."""
        return _call_gemini_api(
            system_prompt=system_prompt,
            user_content=user_content,
            config_section=config_section
        )

    # ================================================================
    # LỚP 2a: Llama 86M - Score-based (Quét nhanh)
    # Groq text classification: CHỈ chấp nhận 1 user message duy nhất
    # ================================================================
    @staticmethod
    def check_layer_2a_prompt_guard_fast(text: str) -> Tuple[bool, str]:
        """LỚP 2a: Llama 86M quét nhanh bằng điểm số (Groq text classification)."""
        config = query_flow_config.prompt_guard_fast
        
        try:
            api_key = query_flow_config.api_keys.get_key(config.provider)
            base_url = query_flow_config.api_keys.get_base_url(config.provider)

            if not api_key:
                return True, "Bỏ qua 2a (chưa cấu hình Groq API Key)"

            url = f"{base_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            }
            # Groq text classification: CHỈ 1 user message, KHÔNG có system
            data = {
                "model": config.model,
                "messages": [
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
                "max_tokens": config.max_tokens_per_chunk,
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                output = result["choices"][0]["message"]["content"].strip()

            print(f"   [Debug 2a] Llama Guard Score: '{output}'")
            
            try:
                score = float(output)
            except ValueError:
                if "unsafe" in output.lower():
                    return False, config.fallback_unsafe
                return True, ""
            
            if score >= config.score_threshold:
                return False, f"{config.fallback_unsafe} (Score: {score:.2%})"
            return True, ""
            
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            return True, f"Bỏ qua 2a (API Error: {error_body[:200]})"
        except Exception as e:
            return True, f"Bỏ qua 2a (Lỗi: {str(e)})"

    # ================================================================
    # LỚP 2b: Qwen 7B - Vietnamese JSON check (Quét sâu)
    # ================================================================
    @staticmethod
    def check_layer_2b_prompt_guard_deep(text: str) -> Tuple[bool, str]:
        """LỚP 2b: Qwen 7B quét sâu bằng tiếng Việt. Trả về JSON."""
        config = query_flow_config.prompt_guard_deep
        
        try:
            class _TempConfig:
                pass
            tc = _TempConfig()
            tc.provider = config.provider
            tc.model = config.model
            tc.temperature = config.temperature
            tc.max_tokens = getattr(config, 'max_tokens', 50)
            tc.timeout_seconds = getattr(config, 'timeout_seconds', 5)

            output = GuardianService._call_llm_api(
                config_section=tc,
                system_prompt=prompt_manager.get_system("prompt_guard_deep"),
                user_content=text,
            )
            print(f"   [Debug 2b] Qwen Guard trả về: '{output}'")
            
            # Parse JSON: {"status": "SAFE"} hoặc {"status": "UNSAFE"}
            try:
                result = json.loads(output)
                status = result.get("status", "").upper()
            except json.JSONDecodeError:
                # Fallback: nếu Qwen không trả JSON, kiểm tra text thô
                status = output.upper()
            
            if "UNSAFE" in status:
                return False, config.fallback_unsafe
            return True, ""
            
        except ValueError as e:
            return True, f"Bỏ qua 2b ({str(e)})"
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            return True, f"Bỏ qua 2b (API Error: {error_body[:200]})"
        except Exception as e:
            return True, f"Bỏ qua 2b (Lỗi: {str(e)})"

    # ================================================================
    # LỚP 2 TỔNG: Chạy SONG SONG 2a & 2b (Concurrent)
    # Ai báo UNSAFE trước thì CHẶN ngay, hủy thằng còn lại
    # ================================================================
    @staticmethod
    async def _run_2a_async(text: str) -> Tuple[bool, str]:
        """Wrapper async cho 2a (chạy trong thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, GuardianService.check_layer_2a_prompt_guard_fast, text
        )

    @staticmethod
    async def _run_2b_async(text: str) -> Tuple[bool, str]:
        """Wrapper async cho 2b (chạy trong thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, GuardianService.check_layer_2b_prompt_guard_deep, text
        )

    @classmethod
    async def check_layer_2_concurrent(cls, text: str) -> Tuple[bool, str]:
        """Chạy 2a và 2b SONG SONG. Ai UNSAFE trước thì chặn ngay, hủy task còn lại."""
        tasks = [
            asyncio.create_task(cls._run_2a_async(text)),
            asyncio.create_task(cls._run_2b_async(text)),
        ]
        
        # as_completed: task nào xong trước xử lý trước
        for future in asyncio.as_completed(tasks):
            try:
                is_safe, msg = await future
                if not is_safe:
                    # Chặn ngay lập tức và hủy task còn lại
                    for t in tasks:
                        t.cancel()
                    return False, msg
            except Exception as e:
                print(f"   ⚠️ Lớp Guard lỗi: {e}")
        
        return True, ""

    # ================================================================
    # LUỒNG TỔNG: validate_query (Sync wrapper cho test script)
    # ================================================================
    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, str, int]:
        """Chạy toàn bộ luồng Guardian (sync wrapper)."""
        # LỚP 0
        is_l0_ok, msg_l0 = cls.check_layer_0_input_validation(query)
        if not is_l0_ok:
            return False, msg_l0, 0

        # LỚP 1a: Từ khóa cấm (nội dung)
        is_l1_ok, msg_l1 = cls.check_layer_1_keyword_filter(query)
        if not is_l1_ok:
            return False, msg_l1, 1

        # LỚP 1b: Regex chống Injection (0ms)
        is_l1b_ok, msg_l1b = cls.check_layer_1b_injection_filter(query)
        if not is_l1b_ok:
            return False, msg_l1b, 1

        # LỚP 2: Chạy SONG SONG 2a + 2b
        is_l2_ok, msg_l2 = asyncio.run(cls.check_layer_2_concurrent(query))
        if not is_l2_ok:
            return False, msg_l2, 2

        return True, "SAFE", 2
