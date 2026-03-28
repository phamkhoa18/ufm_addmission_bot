"""
Context Node — Cối Xay Ngữ Cảnh (Query Reformulation).

Vị trí trong Graph:
  [fast_scan_node] → [context_node] → [contextual_guard_node] → ...

Nhiệm vụ:
  1. Đọc chat_history (10 lượt gần nhất) + user_query
  2. Gọi Gemini Flash Lite để "dịch" câu hỏi lửng lơ
     thành câu hỏi độc lập có đầy đủ ngữ cảnh (standalone_query).
  3. Nếu không có lịch sử → skip reformulation (tiết kiệm 1 API call).

Model: google/gemini-3.1-flash-lite-preview (OpenRouter)
Chi phí: ~$0.000002/query (gần như miễn phí)
Latency: ~200ms

Ví dụ:
  chat_history: [User: "Học phí QTKD?", Bot: "600.000đ/tín chỉ"]
  user_query: "Thế còn ngành Marketing?"
  → standalone_query: "Học phí ngành Marketing là bao nhiêu?"
"""

import json
import time
import urllib.request
import urllib.error
from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _summarize_message(content: str) -> str:
    """
    Tóm tắt message dài bằng Gemini Flash Lite.
    Nếu lỗi → fallback cắt cứng.
    """
    config = query_flow_config.memory.auto_summarize
    try:
        summary = _call_gemini_api_with_fallback(
            system_prompt=prompt_manager.get_system("auto_summarize"),
            user_content=content,
            config_section=config,
        )
        return summary[:config.target_length]
    except Exception:
        return content[:config.target_length] + "..."


def _build_history_prompt(chat_history: list, max_turns: int) -> str:
    """
    Xây dựng chuỗi lịch sử hội thoại.
    Message quá dài sẽ được tóm tắt tự động (nếu bật).
    """
    if not chat_history:
        return ""

    max_messages = max_turns * 2
    recent = chat_history[-max_messages:]

    summarize_cfg = query_flow_config.memory.auto_summarize
    max_len = query_flow_config.memory.max_tokens_per_message

    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Tóm tắt tự động nếu message quá dài
        if summarize_cfg.enabled and len(content) > summarize_cfg.trigger_length:
            content = _summarize_message(content)
        elif len(content) > max_len:
            content = content[:max_len] + "..."

        if role == "user":
            lines.append(f"Người dùng: {content}")
        else:
            lines.append(f"Bot: {content}")

    return "\n".join(lines)


def _call_gemini_api(
    system_prompt: str,
    user_content: str,
    config_section,
) -> str:
    """
    Gọi API LLM (OpenRouter) cho Gemini Flash Lite.
    Dùng chung cho cả Reformulation và Multi-Query.
    """
    api_key = query_flow_config.api_keys.get_key(config_section.provider)
    base_url = query_flow_config.api_keys.get_base_url(config_section.provider)

    if not api_key:
        raise ValueError(
            f"Chưa cấu hình API Key cho provider '{config_section.provider}'"
        )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "UFM-Admission-Bot/1.0",
        "HTTP-Referer": "https://ufm.edu.vn",
    }
    data = {
        "model": config_section.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": config_section.temperature,
        "max_tokens": config_section.max_tokens,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config_section.timeout_seconds) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()


def _stream_gemini_api(
    system_prompt: str,
    user_content: str,
    config_section,
):
    """
    Gọi API LLM (OpenRouter) với stream=true.
    Yield từng token (chunk text) theo kiểu ChatGPT streaming.

    Yields:
        str — Từng đoạn text nhỏ (token/chunk)
    """
    api_key = query_flow_config.api_keys.get_key(config_section.provider)
    base_url = query_flow_config.api_keys.get_base_url(config_section.provider)

    if not api_key:
        raise ValueError(
            f"Chưa cấu hình API Key cho provider '{config_section.provider}'"
        )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "UFM-Admission-Bot/1.0",
        "HTTP-Referer": "https://ufm.edu.vn",
    }
    data = {
        "model": config_section.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": config_section.temperature,
        "max_tokens": config_section.max_tokens,
        "stream": True,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=config_section.timeout_seconds) as resp:
        buffer = ""
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue


def _stream_gemini_api_with_fallback(
    system_prompt: str,
    user_content: str,
    config_section,
    node_key: str = "",
):
    """
    Stream LLM với fallback tự động.
    Thử primary model trước, nếu lỗi thử fallback.

    Yields:
        str — Từng token text
    """
    fb_settings = query_flow_config.fallback_models.settings

    # ── Bước 1: Thử primary model ──
    try:
        yield from _stream_gemini_api(system_prompt, user_content, config_section)
        return
    except Exception as primary_error:
        if fb_settings.log_fallback:
            logger.warning(
                "Stream Primary FAIL (%s/%s): %s",
                getattr(config_section, 'provider', '?'),
                getattr(config_section, 'model', '?'),
                primary_error,
            )
        if not node_key:
            raise
        last_error = primary_error

    # ── Bước 2: Fallback models ──
    from app.core.config import models_yaml_data
    node_cfg = models_yaml_data.get(node_key, {})
    fallbacks_raw = node_cfg.get("fallbacks", []) or []

    if not fallbacks_raw:
        raise last_error

    max_retries = fb_settings.max_retries

    for i, fb_entry in enumerate(fallbacks_raw[:max_retries]):
        try:
            class _TempConfig:
                pass
            temp = _TempConfig()
            temp.provider = fb_entry.get("provider", "openrouter")
            temp.model = fb_entry.get("model", "")
            temp.temperature = getattr(config_section, 'temperature', 0.0)
            temp.max_tokens = getattr(config_section, 'max_tokens', 500)
            temp.timeout_seconds = getattr(config_section, 'timeout_seconds', 15)

            delay = fb_settings.retry_delay_ms / 1000
            time.sleep(delay)

            yield from _stream_gemini_api(system_prompt, user_content, temp)

            if fb_settings.log_fallback:
                logger.info(
                    "Stream Fallback #%d OK: %s/%s (node_key='%s')",
                    i + 1, temp.provider, temp.model, node_key,
                )
            return

        except Exception as e:
            last_error = e
            if fb_settings.log_fallback:
                logger.warning(
                    "Stream Fallback #%d FAIL (%s/%s): %s",
                    i + 1,
                    fb_entry.get("provider", "?"),
                    fb_entry.get("model", "?"),
                    e,
                )

    raise last_error


def _call_gemini_api_with_fallback(
    system_prompt: str,
    user_content: str,
    config_section,
    node_key: str = "",
) -> str:
    """
    Gọi API LLM với cơ chế fallback tự động.

    Luồng:
      1. Thử primary model từ config_section (provider, model, temperature...)
      2. Nếu lỗi → đọc fallbacks[] từ models_config.yaml[node_key]
      3. Thử lần lượt từng fallback model (giữ nguyên temperature/max_tokens)

    Args:
        system_prompt: System prompt cho LLM
        user_content: User prompt đã render
        config_section: Pydantic config chứa provider, model, temperature, max_tokens, timeout_seconds
        node_key: Key trong models_config.yaml để đọc fallbacks[] (vd: "sanitizer", "main_bot")
                  Nếu rỗng → không có fallback, chỉ thử primary.
    """
    fb_settings = query_flow_config.fallback_models.settings

    # ── Bước 1: Thử primary model (từ config_section) ──
    try:
        return _call_gemini_api(system_prompt, user_content, config_section)
    except Exception as primary_error:
        if fb_settings.log_fallback:
            logger.warning(
                "Primary FAIL (%s/%s): %s",
                getattr(config_section, 'provider', '?'),
                getattr(config_section, 'model', '?'),
                primary_error,
            )

        # Nếu không có node_key → không có fallback → raise ngay
        if not node_key:
            raise

        last_error = primary_error

    # ── Bước 2: Đọc fallbacks[] từ models_config.yaml ──
    from app.core.config import models_yaml_data
    node_cfg = models_yaml_data.get(node_key, {})
    fallbacks_raw = node_cfg.get("fallbacks", []) or []

    if not fallbacks_raw:
        raise last_error

    max_retries = fb_settings.max_retries

    for i, fb_entry in enumerate(fallbacks_raw[:max_retries]):
        try:
            class _TempConfig:
                pass
            temp = _TempConfig()
            temp.provider = fb_entry.get("provider", "openrouter")
            temp.model = fb_entry.get("model", "")
            temp.temperature = getattr(config_section, 'temperature', 0.0)
            temp.max_tokens = getattr(config_section, 'max_tokens', 500)
            temp.timeout_seconds = getattr(config_section, 'timeout_seconds', 15)

            delay = fb_settings.retry_delay_ms / 1000
            time.sleep(delay)

            result = _call_gemini_api(system_prompt, user_content, temp)

            if fb_settings.log_fallback:
                logger.info(
                    "Fallback #%d OK: %s/%s (node_key='%s')",
                    i + 1, temp.provider, temp.model, node_key,
                )

            return result

        except Exception as e:
            last_error = e
            if fb_settings.log_fallback:
                logger.warning(
                    "Fallback #%d FAIL (%s/%s): %s",
                    i + 1,
                    fb_entry.get("provider", "?"),
                    fb_entry.get("model", "?"),
                    e,
                )

    raise last_error


def _reformulate_query(user_query: str, chat_history: list) -> str:
    """
    Gọi Gemini Flash Lite để dịch câu hỏi lửng lơ thành câu hỏi độc lập.

    Input:
      user_query: "Thế còn ngành Marketing?"
      chat_history: [{"role": "user", "content": "Học phí QTKD?"}, ...]

    Output:
      "Học phí ngành Marketing là bao nhiêu?"
    """
    config = query_flow_config.query_reformulation

    # Xây prompt lịch sử
    max_turns = query_flow_config.memory.max_history_turns
    history_text = _build_history_prompt(chat_history, max_turns)

    # Render user_prompt từ Prompt Hub
    user_content = prompt_manager.render_user(
        "context_node",
        chat_history_text=history_text,
        user_query=user_query,
    )

    return _call_gemini_api(
        system_prompt=prompt_manager.get_system("context_node"),
        user_content=user_content,
        config_section=config,
    )


# ================================================================
# CONTEXT NODE — Hàm chính cho Graph
# ================================================================
def context_node(state: GraphState) -> GraphState:
    """
    🔄 CONTEXT NODE — Cối Xay Ngữ Cảnh.

    Input:
      - state["user_query"]: Câu hỏi thô (đã qua Fast-Scan)
      - state["chat_history"]: Lịch sử hội thoại (10 lượt gần nhất)

    Output:
      - state["standalone_query"]: Câu hỏi đã reformulate (hoặc giữ nguyên)

    Logic:
      1. Nếu chat_history RỖNG + skip_if_no_history = true
         → standalone_query = user_query (BỎ QUA API call, tiết kiệm tiền)
      2. Nếu có lịch sử
         → Gọi Gemini Flash Lite dịch câu hỏi
      3. Nếu API lỗi
         → Fallback: standalone_query = user_query (fail-safe)
    """
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    config = query_flow_config.query_reformulation
    start_time = time.time()

    # Pre-build history text for downstream nodes
    max_turns = query_flow_config.memory.max_history_turns
    chat_history_text = _build_history_prompt(chat_history, max_turns)

    # ── Trường hợp 1: Reformulation bị tắt ──
    if not config.enabled:
        return {
            **state,
            "standalone_query": user_query,
            "chat_history_text": chat_history_text,
        }

    # ── Trường hợp 2: Không có lịch sử → Skip (tiết kiệm $) ──
    if config.skip_if_no_history and (not chat_history or len(chat_history) == 0):
        elapsed = time.time() - start_time
        logger.info("Context Node [%.3fs] Khong co history -> giu nguyen user_query", elapsed)
        return {
            **state,
            "standalone_query": user_query,
            "chat_history_text": chat_history_text,
        }

    # ── Trường hợp 3: Có lịch sử → Gọi Gemini để reformulate ──
    try:
        standalone = _reformulate_query(user_query, chat_history)
        elapsed = time.time() - start_time
        logger.info(
            "Context Node [%.3fs] Reformulated: '%s' -> '%s'",
            elapsed, user_query[:50], standalone[:80]
        )
        return {
            **state,
            "standalone_query": standalone,
            "chat_history_text": chat_history_text,
        }
    except urllib.error.URLError as e:
        # Timeout hoặc lỗi mạng → Fallback giữ nguyên
        elapsed = time.time() - start_time
        logger.error("Context Node [%.3fs] API timeout/error: %s", elapsed, e, exc_info=True)
        return {
            **state,
            "standalone_query": user_query,
            "chat_history_text": chat_history_text,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Context Node [%.3fs] Unexpected error: %s", elapsed, e, exc_info=True)
        return {
            **state,
            "standalone_query": user_query,
            "chat_history_text": chat_history_text,
        }

