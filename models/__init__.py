"""Pydantic models for chat messages."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' | 'assistant' | 'system'")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatContext(BaseModel):
    """Context assembled for the synthesizer."""
    chunks: list[dict] = Field(default_factory=list)
    web_results: list[dict] = Field(default_factory=list)
    retrieval_score: float = 0.0
    source_type: str = "rag"  # "rag" | "web" | "hybrid"
