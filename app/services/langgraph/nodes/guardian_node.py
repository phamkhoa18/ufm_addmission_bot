"""
[DEPRECATED] Guardian Node cũ — Đã được tách thành 2 Node riêng biệt:
  1. fast_scan_node.py      → Chốt 1 (Regex, trước Context Node)
  2. contextual_guard_node.py → Chốt 2 (LLM, sau Context Node)

Xem file mới tại:
  app/services/langgraph/nodes/fast_scan_node.py
  app/services/langgraph/nodes/contextual_guard_node.py
"""

# File này được giữ lại để tránh import error từ code cũ.
# Sẽ xóa hoàn toàn khi migration xong.

from app.services.langgraph.nodes.fast_scan_node import fast_scan_node, fast_scan_router
from app.services.langgraph.nodes.contextual_guard_node import contextual_guard_node, contextual_guard_router

# Backward-compatible alias
guardian_node = fast_scan_node
guardian_router = fast_scan_router
