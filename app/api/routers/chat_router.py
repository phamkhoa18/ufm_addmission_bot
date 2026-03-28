"""
Chat Router — FastAPI Endpoint công khai cho Chat API.

Bảo mật:
  - Origin/Referer check: Chỉ cho phép từ domain đã đăng ký
  - Rate Limit per IP: 8 msg/phút
  - Không cần đăng nhập (public chatbot)

Endpoints:
  POST /api/v1/chat/message  → Gửi tin nhắn → LangGraph Pipeline
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.core.config.chat_config import chat_cfg
from app.core.security import chat_rate_limiter, verify_origin
from app.services.chat_workflow import run_chat_pipeline, stream_chat_pipeline, stream_chat_pipeline_tokens_v2
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Router ──
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


# ══════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' hoặc 'assistant'")
    content: str = Field(..., description="Nội dung tin nhắn")


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=chat_cfg.history.max_query_length,
        description="Câu hỏi của người dùng (tối đa 2000 ký tự)",
    )
    chat_history: Optional[list[ChatMessage]] = Field(
        default=[],
        description="Lịch sử chat (tối đa 20 tin gần nhất)",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="ID phiên chat (tùy chọn, dùng cho tracking)",
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Câu trả lời của Bot")
    source: str = Field("", description="Nguồn gốc: rag_direct, form_template, care_agent...")
    intent: str = Field("", description="Intent phân loại: HOC_PHI_HOC_BONG, CHAO_HOI...")
    intent_action: str = Field("", description="Action routing: PROCEED_RAG, GREET, BLOCK_FALLBACK...")
    blocked: bool = Field(False, description="True nếu tin nhắn bị Guardian chặn")
    blocked_reason: str = Field("", description="Lý do bị chặn (rỗng nếu không bị chặn)")
    elapsed_seconds: float = Field(0.0, description="Thời gian xử lý (giây)")


# ══════════════════════════════════════════════════════════
# MESSAGE — Public Endpoint (Domain-Locked)
# ══════════════════════════════════════════════════════════
@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Gửi tin nhắn Chat (public, domain-locked)",
)
async def send_message(body: ChatRequest, request: Request):
    """
    Gửi tin nhắn và nhận câu trả lời từ Bot.

    **Bảo mật (không cần đăng nhập):**
    - ✅ Chỉ chấp nhận request từ domain đã đăng ký (Origin check)
    - ✅ Rate Limit: 8 tin nhắn/phút/IP
    - ✅ Tin nhắn spam/injection → bị chặn bởi Guardian Nodes bên trong

    **Body (JSON):**
    ```json
    {
      "query": "Học phí ngành Marketing là bao nhiêu?",
      "chat_history": [
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn!"}
      ],
      "session_id": "abc123"
    }
    ```
    """
    # ── Lớp 1: Origin Check (chặn domain lạ) ──
    client_ip = verify_origin(request)

    # ── Lớp 2: Rate Limit per IP ──
    chat_rate_limiter.check(client_ip)

    # ── Validate & Trim history ──
    max_hist = chat_cfg.history.max_history_messages
    history = []
    if body.chat_history:
        recent = body.chat_history[-max_hist:]
        history = [{"role": m.role, "content": m.content} for m in recent]

    logger.info(
        "Chat - Message from %s | session=%s | query='%s'",
        client_ip, body.session_id or "none", body.query[:60],
    )

    # ── Lớp 3: Chạy Pipeline (bên trong có Fast Scan + Guard) ──
    # NOTE: run_chat_pipeline là sync function (gọi HTTP APIs, DB queries, tốn 5-30s).
    # Phải dùng asyncio.to_thread() để offload sang threadpool,
    # tránh block event loop của FastAPI.
    try:
        result = await asyncio.to_thread(
            run_chat_pipeline,
            query=body.query,
            chat_history=history,
            session_id=body.session_id or client_ip,
        )
    except Exception as e:
        logger.error("Chat - Pipeline crashed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hệ thống gặp lỗi khi xử lý. Vui lòng thử lại.",
        )

    return ChatResponse(
        response=result["response"],
        source=result.get("source", ""),
        intent=result.get("intent", ""),
        intent_action=result.get("intent_action", ""),
        blocked=result.get("blocked", False),
        blocked_reason=result.get("blocked_reason", ""),
        elapsed_seconds=result.get("elapsed_seconds", 0.0),
    )


# ══════════════════════════════════════════════════════════
# STREAM — SSE Endpoint (LangGraph .stream() chuẩn thư viện)
# ══════════════════════════════════════════════════════════
@router.post(
    "/stream",
    summary="Stream tin nhắn Chat qua SSE (LangGraph node-by-node)",
)
async def stream_message(body: ChatRequest, request: Request):
    """
    Stream câu trả lời qua **Server-Sent Events (SSE)**.

    Frontend nhận từng event khi mỗi LangGraph node hoàn thành:
    ```
    data: {"type":"node","node":"fast_scan","label":"🛡️ Kiểm tra nhanh"}
    data: {"type":"node","node":"intent","label":"🧭 Phân loại ý định"}
    data: {"type":"result","response":"Chào bạn!","source":"greet_template",...}
    ```
    """
    import json
    import queue
    from starlette.responses import StreamingResponse

    # ── Lớp 1-2: Bảo mật (giống endpoint /message) ──
    client_ip = verify_origin(request)
    chat_rate_limiter.check(client_ip)

    max_hist = chat_cfg.history.max_history_messages
    history = []
    if body.chat_history:
        recent = body.chat_history[-max_hist:]
        history = [{"role": m.role, "content": m.content} for m in recent]

    logger.info(
        "Chat Stream - %s | session=%s | query='%s'",
        client_ip, body.session_id or "none", body.query[:60],
    )

    # ── Async SSE Generator ──
    # stream_chat_pipeline() là sync generator (vì LangGraph .stream() là sync).
    # Dùng thread + queue để bridge sang async cho FastAPI.
    event_queue = queue.Queue()

    def _run_stream():
        """Chạy pipeline sync trong background thread, đẩy events vào queue."""
        try:
            for event in stream_chat_pipeline(
                query=body.query,
                chat_history=history,
                session_id=body.session_id or client_ip,
            ):
                event_queue.put(event)
        except Exception as e:
            logger.error("Stream thread crashed: %s", e, exc_info=True)
            event_queue.put({
                "type": "error",
                "message": "Hệ thống gặp lỗi khi xử lý.",
            })
        finally:
            event_queue.put(None)  # Sentinel: kết thúc stream

    async def _sse_generator():
        """Async generator yield SSE events từ queue."""
        # Start sync pipeline in background thread
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_stream)

        while True:
            # Poll queue (non-blocking) + yield control
            event = await asyncio.to_thread(event_queue.get)
            if event is None:
                break  # Stream kết thúc
            line = json.dumps(event, ensure_ascii=False)
            yield f"data: {line}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx: tắt buffer
        },
    )


# ══════════════════════════════════════════════════════════
# STREAM TOKENS — ChatGPT-style Token-by-Token SSE
# ══════════════════════════════════════════════════════════
@router.post(
    "/stream-tokens",
    summary="Stream tin nhắn Chat qua SSE — token-by-token (ChatGPT-style)",
)
async def stream_tokens(body: ChatRequest, request: Request):
    """
    Stream câu trả lời token-by-token qua **Server-Sent Events (SSE)**.

    Events:
    ```
    data: {"type":"node","node":"fast_scan","label":"🛡️ Kiểm tra nhanh"}
    data: {"type":"node","node":"response","label":"💬 Sinh câu trả lời"}
    data: {"type":"token","content":"Xin "}
    data: {"type":"token","content":"chào "}
    data: {"type":"token","content":"bạn!"}
    data: {"type":"done","source":"rag_db_only_generated",...}
    ```

    Frontend nên:
      - `type: node` → Hiển thị trạng thái pipeline
      - `type: token` → Append text vào bubble (ChatGPT-style)
      - `type: done` → Kết thúc streaming, lưu metadata
      - `type: result` → Nhận response đầy đủ (dùng cho bypass: chào hỏi, block)
      - `type: error` → Hiển thị lỗi
    """
    import json
    import queue
    from starlette.responses import StreamingResponse

    # ── Bảo mật ──
    client_ip = verify_origin(request)
    chat_rate_limiter.check(client_ip)

    max_hist = chat_cfg.history.max_history_messages
    history = []
    if body.chat_history:
        recent = body.chat_history[-max_hist:]
        history = [{"role": m.role, "content": m.content} for m in recent]

    logger.info(
        "Chat Stream Tokens - %s | session=%s | query='%s'",
        client_ip, body.session_id or "none", body.query[:60],
    )

    # ── Async SSE Generator (bridge sync→async via queue) ──
    event_queue = queue.Queue()

    def _run_stream():
        try:
            for event in stream_chat_pipeline_tokens_v2(
                query=body.query,
                chat_history=history,
                session_id=body.session_id or client_ip,
            ):
                event_queue.put(event)
        except Exception as e:
            logger.error("Stream tokens thread crashed: %s", e, exc_info=True)
            event_queue.put({
                "type": "error",
                "message": "Hệ thống gặp lỗi khi xử lý.",
            })
        finally:
            event_queue.put(None)  # Sentinel

    async def _sse_generator():
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_stream)

        while True:
            event = await asyncio.to_thread(event_queue.get)
            if event is None:
                break
            line = json.dumps(event, ensure_ascii=False)
            yield f"data: {line}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
