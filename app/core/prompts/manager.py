# app/core/prompts/manager.py
# PromptManager — Engine render Prompt bằng Jinja2
# Nạp 1 lần khi app khởi động, render ~0ms mỗi lần gọi

import os
import yaml
from pathlib import Path
from jinja2 import Environment, BaseLoader


class PromptManager:
    """
    Quản lý tập trung toàn bộ Prompt (System + User).

    Cách dùng:
        from app.core.prompts import prompt_manager

        # 1) Lấy system prompt (tĩnh, không có biến)
        sys = prompt_manager.get_system("sanitizer_node")

        # 2) Render user prompt (động, truyền context từ GraphState)
        usr = prompt_manager.render_user("sanitizer_node",
            standalone_query="Học phí ngành Marketing?",
            draft="Học phí khoảng 25 triệu...",
            rag_context="...",
            web_citations=[{"text": "UFM", "url": "https://..."}]
        )
    """

    def __init__(self, yaml_path: str = None):
        if yaml_path is None:
            # Ưu tiên file hợp nhất mới: config/yaml/prompts_config.yaml
            _new_path = Path(__file__).parent.parent / "config" / "yaml" / "prompts_config.yaml"
            if _new_path.exists():
                yaml_path = str(_new_path)
            else:
                # Backward compatible: dùng file cũ prompts.yaml
                yaml_path = str(Path(__file__).parent / "prompts.yaml")

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"[PromptManager] Không tìm thấy file prompt: {yaml_path}"
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}

        # Jinja2 Environment không gắn filesystem loader (vì template nằm trong YAML)
        self._env = Environment(
            loader=BaseLoader(),
            # Giữ whitespace gốc — quan trọng cho prompt formatting
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Pre-compile tất cả user_prompt templates (tối ưu tốc độ runtime)
        self._compiled = {}
        for domain, prompts in self._raw.items():
            if isinstance(prompts, dict) and "user_prompt" in prompts:
                try:
                    self._compiled[domain] = self._env.from_string(
                        prompts["user_prompt"]
                    )
                except Exception as e:
                    print(f"[PromptManager] WARN Lo compile template '{domain}': {e}")

        print(
            f"[PromptManager] OK Nap {len(self._raw)} domain, "
            f"compile {len(self._compiled)} user_prompt templates"
        )

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_system(self, domain: str) -> str:
        """
        Lấy system_prompt tĩnh cho 1 domain.

        Args:
            domain: Tên domain trong prompts.yaml
                    (vd: "sanitizer_node", "context_node", "intent_classification")

        Returns:
            System prompt string đã strip whitespace.
            Trả về "" nếu domain không tồn tại.
        """
        section = self._raw.get(domain)
        if not section or not isinstance(section, dict):
            return ""
        return str(section.get("system_prompt", "")).strip()

    def render_user(self, domain: str, **kwargs) -> str:
        """
        Render user_prompt bằng Jinja2, truyền context qua kwargs.

        Args:
            domain: Tên domain trong prompts.yaml
            **kwargs: Biến context để chèn vào template
                      (standalone_query, draft, rag_context, web_citations, ...)

        Returns:
            User prompt string đã render và strip.
            Nếu domain không có user_prompt → trả về "" rỗng.
        """
        template = self._compiled.get(domain)
        if template is None:
            return ""

        # Render, cho phép biến không tồn tại → trả về "" thay vì crash
        try:
            return template.render(**kwargs).strip()
        except Exception as e:
            print(f"[PromptManager] WARN Loi render '{domain}': {e}")
            # Fallback: trả raw template không render
            raw = self._raw.get(domain, {}).get("user_prompt", "")
            return str(raw).strip()

    def get_fallback(self, key: str) -> str:
        """
        Lấy fallback message tĩnh (không cần render).

        Args:
            key: Tên key trong section 'fallback_messages'
                 (vd: "too_long", "injection", "cau_hoi_lac_de")

        Returns:
            Fallback message string.
        """
        section = self._raw.get("fallback_messages", {})
        return str(section.get(key, "")).strip()

    def list_domains(self) -> list:
        """Liệt kê tất cả domain đã nạp (debug/introspection)."""
        return list(self._raw.keys())


# ─────────────────────────────────────────────────────────
# SINGLETON — Import 1 lần, dùng mọi nơi
# ─────────────────────────────────────────────────────────
prompt_manager = PromptManager()
