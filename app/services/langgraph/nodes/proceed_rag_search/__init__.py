# app/services/langgraph/nodes/proceed_rag_search/__init__.py
"""
Proceed RAG Search — Web-Augmented RAG Pipeline.

Luồng xử lý cho các intent cần ngữ cảnh phong phú + PR thành tựu UFM:
  THONG_TIN_TUYEN_SINH, CHUONG_TRINH_DAO_TAO, THANH_TICH_UFM, DOI_SONG_SINH_VIEN

Pipeline:
  [PR Query Gen] → [Web Search] → [Synthesizer] → [Sanitizer] → Output
  
  - PR Query: Sinh 1 query riêng cho web search (KHÔNG vào embedding)
  - Web Search: gpt-4o-mini-search-preview (fallback = null)
  - Synthesizer: Tổng hợp RAG nội bộ + web context
  - Sanitizer: Kiểm tra trích dẫn, hallucination, tone PR
"""
