"""
LangGraph State v3 — Schema chung cho toàn bộ Agent Graph.
Thiết kế cho bài toán: Chatbot Tư vấn Tuyển sinh Đại học (Hội thoại đa lượt)

 LIFECYCLE v3 (Guardian 2 Chốt):

  User Query
    │
    ▼
  [Fast-Scan Node] ── Chặn thô trên user_query (Regex, ~0ms, $0)
    │                  Layer 0a: Độ dài > 2000 ký tự → CHẶN (DoS)
    │                  Layer 0b: Độ dài >= 1999 → LLM tóm tắt
    │                            (google/gemini-2.5-flash-lite ~$0.000005)
    │                  Layer 1a: Từ cấm nhạy cảm (bạo lực, ma tuý)
    │                  Layer 1b: Injection regex (ignore instructions, sudo)
    │                  → BLOCKED? → END (trả fallback, KHÔNG tốn tiền Gemini)
    ▼
  [Context Node] ── Reformulate (Gemini Flash Lite — siêu rẻ)
    │                user_query + chat_history → standalone_query
    ▼
  [Contextual-Guard Node] ── Chặn tinh trên standalone_query (LLM, ~500ms)
    │                         Layer 2a: Llama 86M Score (Groq)
    │                         Layer 2b: Qwen 7B Vietnamese SAFE/UNSAFE
    │                         → BLOCKED? → END
    ▼
  [Multi-Query Node] ── Sinh 3 biến thể (Gemini Flash Lite)
    │                    Chỉ chạy khi standalone_query đã SAFE
    ▼
  [Embedding Node] ── Nhúng batch [standalone + 3 biến thể] (BGE-M3)
    │
    ▼
  [Cache Node] ── Kiểm tra Semantic Cache (Cosine ≥ 0.9 → trả ngay)
    │
    ▼
  [Intent Node] ── Phân loại ý định trên standalone_query (Qwen)
    │
    ▼
  [Agent Node] ── RAG / Form / PR / Care
    │
    ├─ (Nếu PROCEED_RAG_SEARCH):
    │   [PR Query Gen] → sinh query PR riêng (KHÔNG vào embedding)
    │   [Web Search Agent] → gpt-4o-mini-search-preview (fallback=null)
    │   [Synthesizer] → tổng hợp RAG context + web context
    │   [Sanitizer] → kiểm tra trích dẫn, tone PR, loop nếu lỗi
    │
    ▼
  [Response Node] ── Ghi cache + Cập nhật lịch sử
    │
    ▼
  Response → User
"""

from typing import TypedDict, Optional


class GraphState(TypedDict, total=False):
    """
    State Schema v3 — Guardian 2 Chốt + Multi-Query + Conversational Memory

    OWNER MAP:
    ┌──────────────────────────┬──────────────────────────────────────────────┐
    │ Field                    │ Written by                                  │
    ├──────────────────────────┼──────────────────────────────────────────────┤
    │ session_id               │ API Layer (FastAPI endpoint)                │
    │ user_query               │ API Layer                                   │
    │ chat_history             │ API Layer (load) + Response Node (append)   │
    │ normalized_query         │ Fast-Scan Node                              │
    │ fast_scan_*              │ Fast-Scan Node                              │
    │ standalone_query         │ Context Node                                │
    │ contextual_guard_*       │ Contextual-Guard Node                       │
    │ multi_queries            │ Multi-Query Node                            │
    │ query_embeddings         │ Embedding Node                              │
    │ cache_*                  │ Cache Node                                  │
    │ intent, intent_*         │ Intent Router Node                          │
    │ retrieved_chunks         │ RAG Agent Node                              │
    │ rag_context              │ RAG Agent Node                              │
    │ final_response           │ Response Node / Guardian / Cache            │
    │ response_source          │ Response Node / Guardian / Cache            │
    │ next_node                │ Mọi Node (quyết định routing)              │
    └──────────────────────────┴──────────────────────────────────────────────┘
    """

    # ════════════════════════════════════════════════════════
    # 1. PHIÊN LÀM VIỆC (Session)
    # Ghi bởi: API Layer
    # ════════════════════════════════════════════════════════
    session_id: str                    # UUID phiên chat

    # ════════════════════════════════════════════════════════
    # 2. BỘ NHỚ HỘI THOẠI (Conversational Memory)
    # Sliding Window: 10 lượt gần nhất (config trong YAML)
    # Ghi bởi: API Layer (load) + Response Node (append)
    # ════════════════════════════════════════════════════════
    chat_history: list                 # 10 cặp (user + assistant) gần nhất
                                       # Format: [{"role": "user", "content": "..."}, ...]

    chat_history_text: str             # Chuỗi history đã format (người dùng: ..., bot: ...)
                                       # Ghi bởi: Context Node
                                       # Mục đích: Truyền cho Response Node làm ngữ cảnh để LLM không mất trí nhớ

    # ════════════════════════════════════════════════════════
    # 3. INPUT — Câu hỏi qua 4 giai đoạn biến đổi
    # ════════════════════════════════════════════════════════

    # 3a. Câu hỏi thô (API Layer ghi)
    user_query: str                    # VD: "Thế còn ngành Marketing?"

    # 3a.1 Metadata tóm tắt query dài (Fast-Scan Node ghi)
    original_query: str                # Lưu query gốc nếu bị summarize (để logging/debug)
                                       # = "" nếu không bị summarize
    query_was_summarized: bool         # True nếu user_query đã qua LLM summarizer
                                       # (query gốc >= 1999 chars → đã nén lại)

    # 3b. Chuẩn hóa teencode (Fast-Scan Node ghi)
    normalized_query: str              # VD: "thế còn ngành marketing?"
                                       # → user_query sau khi lowercase + thay teencode
                                       # → Dùng cho regex filter Layer 1

    # 3c. Câu hỏi độc lập (Context Node ghi)
    standalone_query: str              # VD: "Học phí ngành Marketing là bao nhiêu?"
                                       # → Gemini Flash Lite dịch từ user_query + chat_history
                                       # ⚡ TẤT CẢ NODE TỪ ĐÂY TRỞ ĐI CHỈ DÙNG standalone_query

    # 3d. Các biến thể mở rộng (Multi-Query Node ghi)
    multi_queries: list                # VD: ["Chi phí đào tạo Marketing UFM", ...]
                                       # → Gemini Flash Lite sinh thêm 3 biến thể
                                       # → CHỈ SINH KHI standalone_query ĐÃ QUA Contextual-Guard

    # ════════════════════════════════════════════════════════
    # 4. CHỐT 1: FAST-SCAN NODE (Chặn thô — trên user_query)
    # Regex 0ms, $0. Chặn trước khi tốn tiền gọi Gemini.
    # Ghi bởi: Fast-Scan Node
    # ════════════════════════════════════════════════════════
    fast_scan_passed: bool             # True = Câu hỏi sạch, cho qua Context Node
    fast_scan_blocked_layer: Optional[int]   # 0 = quá dài, 1 = từ cấm/injection
    fast_scan_message: str             # Log hoặc fallback message

    # ════════════════════════════════════════════════════════
    # 5. CHỐT 2: CONTEXTUAL-GUARD NODE (Chặn tinh — trên standalone_query)
    # LLM-based, ~500ms. Phát hiện jailbreak tinh vi sau khi có ngữ cảnh.
    # Ghi bởi: Contextual-Guard Node
    # ════════════════════════════════════════════════════════
    contextual_guard_passed: bool      # True = standalone_query an toàn
    contextual_guard_blocked_layer: Optional[str]    # "2a" (Llama) hoặc "2b" (Qwen) hoặc None
    contextual_guard_message: str      # Log hoặc fallback message

    # ════════════════════════════════════════════════════════
    # 6. EMBEDDING NODE — Vector đa truy vấn
    # Ghi bởi: Embedding Node
    # ════════════════════════════════════════════════════════
    query_embeddings: list             # [0] = embedding(standalone_query)
                                       # [1..3] = embedding(multi_queries)

    # ════════════════════════════════════════════════════════
    # 7. SEMANTIC CACHE NODE
    # Ghi bởi: Cache Node
    # ════════════════════════════════════════════════════════
    cache_hit: bool
    cache_similarity: float
    cached_response: Optional[str]

    # ════════════════════════════════════════════════════════
    # 8. INTENT ROUTER NODE — Phân loại trên standalone_query
    # Ghi bởi: Intent Router Node
    # ════════════════════════════════════════════════════════
    intent: str                        # "THONG_TIN_TUYEN_SINH", "TAO_MAU_DON"...
    intent_summary: str                # Tóm tắt từ Qwen
    intent_action: str                 # "PROCEED_RAG", "PROCEED_FORM", "BLOCK_FALLBACK"...
    program_level_filter: Optional[str]  # "thac_si" | "tien_si" | "dai_hoc" | None
                                         # Metadata filter cho RAG SQL (trích từ query)
    program_name_filter: Optional[str]   # Tên ngành (ILIKE match) | None

    # ════════════════════════════════════════════════════════
    # 9. RAG NODE + CONTEXT CURATOR
    # Ghi bởi: RAG Agent Node + Context Curator (Gemini 2.5 Flash)
    # ════════════════════════════════════════════════════════
    retrieved_chunks: Optional[list]
    rag_context: Optional[str]                 # Context ĐÃ CURATE (chỉ info liên quan)

    # ════════════════════════════════════════════════════════
    # 9b. PROCEED RAG SEARCH (Web-Augmented RAG Pipeline)
    # Ghi bởi: PR Query Node, Web Search Node, Synthesizer, Sanitizer
    # ════════════════════════════════════════════════════════
    ufm_search_queries: Optional[list]      # Danh sách tối đa 2 câu tìm kiếm sub-domain UFM
                                            # (Dành cho PROCEED_RAG_UFM_SEARCH)

    pr_search_query: Optional[str]          # Query PR riêng
                                            # (Dành cho PROCEED_RAG_PR_SEARCH)

    web_search_results: Optional[list]      # Kết quả từ gpt-4o-mini-search-preview
                                            # [{"title": "...", "url": "...", "snippet": "..."}]
                                            # = None nếu node search fallback

    web_search_citations: Optional[list]    # Trích dẫn chuẩn markdown đã verify
                                            # ["[Tên bài](https://...)", ...]

    synthesized_draft: Optional[str]        # Bản nháp câu trả lời từ Synthesizer
    sanitizer_critique: Optional[str]       # Phản hồi lỗi từ Sanitizer (nếu có)
    sanitizer_passed: Optional[bool]        # True = draft OK, False = cần sửa
    sanitizer_loop_count: Optional[int]     # Đếm số vòng lặp Sanitizer (max 2)

    search_cache_hit: Optional[bool]        # True = lấy từ cache, False = gọi API mới
    search_cache_similarity: Optional[float]  # Điểm cosine similarity với entry cache

    # ════════════════════════════════════════════════════════
    # 10. RESPONSE
    # Ghi bởi: Response Node / Fast-Scan / Contextual-Guard / Cache
    # ════════════════════════════════════════════════════════
    final_response: str
    response_source: str               # "fast_scan", "contextual_guard", "cache",
                                       # "rag", "form", "pr", "care", "greeting"
    form_history_summary: str           # Ghost History: câu tóm tắt cho chat_history
                                       # Frontend lưu câu này thay vì toàn bộ nội dung đơn
                                       # Chỉ có giá trị khi intent_action = "PROCEED_FORM"

    # ════════════════════════════════════════════════════════
    # 11. ROUTING
    # Ghi bởi: Mọi Node
    # ════════════════════════════════════════════════════════
    next_node: str                     # "context", "contextual_guard", "multi_query",
                                       # "embedding", "cache", "intent", "rag",
                                       # "form", "pr", "care", "response", "end"
