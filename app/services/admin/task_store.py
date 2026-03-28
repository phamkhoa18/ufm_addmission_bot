"""
Task Store — Theo dõi trạng thái Background Tasks (in-memory).

Admin POST file → nhận task_id → GET /tasks/{task_id} để poll trạng thái.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INSERTING = "inserting"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskInfo:
    """Thông tin 1 task ngập."""

    __slots__ = (
        "task_id", "file_name", "status", "message",
        "chunks_count", "created_at", "updated_at", "error",
    )

    def __init__(self, file_name: str):
        self.task_id = str(uuid.uuid4())
        self.file_name = file_name
        self.status = TaskStatus.PENDING
        self.message = "Đang chờ xử lý..."
        self.chunks_count = 0
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.error: Optional[str] = None

    def update(
        self,
        status: TaskStatus,
        message: str = "",
        chunks_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        self.status = status
        if message:
            self.message = message
        if chunks_count > 0:
            self.chunks_count = chunks_count
        if error:
            self.error = error
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "file_name": self.file_name,
            "status": self.status.value,
            "message": self.message,
            "chunks_count": self.chunks_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error": self.error,
        }


class TaskStore:
    """In-memory store cho task tracking. Thread-safe đủ cho single-instance."""

    def __init__(self, max_history: int = 100):
        self._tasks: dict[str, TaskInfo] = {}
        self._max_history = max_history

    def create(self, file_name: str) -> TaskInfo:
        task = TaskInfo(file_name=file_name)
        self._tasks[task.task_id] = task

        # Cleanup old tasks
        if len(self._tasks) > self._max_history:
            oldest_key = min(self._tasks, key=lambda k: self._tasks[k].created_at)
            del self._tasks[oldest_key]

        return task

    def get(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    def list_all(self) -> list[dict]:
        return [t.to_dict() for t in sorted(
            self._tasks.values(), key=lambda t: t.created_at, reverse=True
        )]

    def cancel(self, task_id: str) -> bool:
        """Cancel a task if it's still in a cancellable state."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED):
            return False
        task.update(TaskStatus.CANCELLED, "Đã hủy bởi Admin.")
        return True


# Singleton
task_store = TaskStore()
