"""
Stream Utils — Tien ich stream van ban theo cau cho trai nghiem nguoi dung.

Cach dung:
  from app.utils.stream_utils import stream_response

  # Trong FastAPI endpoint:
  async def chat_endpoint():
      final_text = state["final_response"]
      return StreamingResponse(
          stream_response(final_text),
          media_type="text/event-stream"
      )

Chien luoc cat cau:
  Tach van ban tai cac dau: . , ! ? ; - :
  Moi phan duoc yield kem ky tu phan cach, tao hieu ung "go chu" tu nhien.
  Delay giua cac phan co the tuy chinh qua tham so delay_seconds.
"""

import re
import json
import asyncio
from typing import Generator, AsyncGenerator


# Regex tach van ban theo cac dau cau
# Giu lai dau cau lam phan cuoi cua moi doan
_SENTENCE_SPLIT_PATTERN = re.compile(
    r"([^,.!?;\-:\n]+[,.!?;\-:\n])"
)


def split_into_chunks(text: str) -> list:
    """
    Tach van ban thanh cac doan nho theo dau cau.

    Input:  "Chao ban! Minh la UFM Bot. Ban can ho tro gi?"
    Output: ["Chao ban!", " Minh la UFM Bot.", " Ban can ho tro gi?"]

    Xu ly edge case:
      - Giu nguyen khoang trang dau dong (tranh mat indent)
      - Doan cuoi khong co dau cau van duoc bat
      - Van ban rong tra list rong
    """
    if not text or not text.strip():
        return []

    # Tach theo pattern
    chunks = _SENTENCE_SPLIT_PATTERN.findall(text)

    # Bat phan du cuoi (text sau dau cau cuoi cung)
    matched_length = sum(len(c) for c in chunks)
    if matched_length < len(text):
        remainder = text[matched_length:]
        if remainder.strip():
            chunks.append(remainder)

    # Neu regex khong match gi (van ban khong co dau cau)
    if not chunks:
        chunks = [text]

    return chunks


def stream_response_sync(text: str, delay_seconds: float = 0.03) -> Generator:
    """
    Generator dong bo — stream van ban theo tung cum cau.

    Yield format SSE (Server-Sent Events):
      data: {"chunk": "Chao ban!", "done": false}
      data: {"chunk": " Minh la Bot.", "done": false}
      data: {"chunk": "", "done": true}

    Args:
      text: Van ban can stream
      delay_seconds: Thoi gian nghi giua cac chunk (mac dinh 30ms)
    """
    import time

    chunks = split_into_chunks(text)

    for chunk in chunks:
        payload = json.dumps(
            {"chunk": chunk, "done": False},
            ensure_ascii=False,
        )
        yield f"data: {payload}\n\n"
        time.sleep(delay_seconds)

    # Tin hieu ket thuc
    done_payload = json.dumps({"chunk": "", "done": True})
    yield f"data: {done_payload}\n\n"


async def stream_response_async(
    text: str,
    delay_seconds: float = 0.03,
) -> AsyncGenerator:
    """
    Async Generator — stream van ban cho FastAPI StreamingResponse.

    Su dung:
      from starlette.responses import StreamingResponse

      return StreamingResponse(
          stream_response_async(final_text),
          media_type="text/event-stream",
      )

    Args:
      text: Van ban can stream
      delay_seconds: Thoi gian nghi giua cac chunk (mac dinh 30ms)
    """
    chunks = split_into_chunks(text)

    for chunk in chunks:
        payload = json.dumps(
            {"chunk": chunk, "done": False},
            ensure_ascii=False,
        )
        yield f"data: {payload}\n\n"
        await asyncio.sleep(delay_seconds)

    # Tin hieu ket thuc
    done_payload = json.dumps({"chunk": "", "done": True})
    yield f"data: {done_payload}\n\n"


def format_sse_complete(text: str) -> str:
    """
    Tra ve toan bo van ban trong 1 SSE event duy nhat (khong stream).
    Dung cho bypass responses (GREET, CLARIFY, BLOCK) de khoi stream.
    """
    payload = json.dumps(
        {"chunk": text, "done": True},
        ensure_ascii=False,
    )
    return f"data: {payload}\n\n"
