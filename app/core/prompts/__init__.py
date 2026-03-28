# app/core/prompts/__init__.py
# Prompt Hub — Trung tâm quản lý toàn bộ System + User Prompt
# Import singleton để sử dụng ở mọi Node

from app.core.prompts.manager import prompt_manager

__all__ = ["prompt_manager"]
