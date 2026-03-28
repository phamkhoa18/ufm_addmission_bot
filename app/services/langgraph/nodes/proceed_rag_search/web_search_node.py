"""
Web Search Node — Gemini 2.5 Flash + Google Search Tool (Native API).

Vị trí:
  [pr_query_node] → [web_search_node] → [synthesizer_node]

Nhiệm vụ:
  Gọi Gemini 2.5 Flash với google_search tool tìm thông tin từ web.
  - Nhánh UFM Search: Ưu tiên domain ufm.edu.vn và sub-domains
  - Nhánh PR Search: Ưu tiên báo lớn (thanhnien, vnexpress, tuoitre)

CƠ CHẾ:
  1. Gọi Google Gemini API native với tool google_search
  2. Gemini tự quyết định khi nào cần search và search gì
  3. Citations được trích từ groundingMetadata (chính xác, có verify)
  4. Nếu Google API lỗi → Fallback sang OpenRouter search models

Model: gemini-2.5-flash (Google native API + google_search tool)
Fallback: OpenRouter (gpt-4o-search-preview, perplexity/sonar)
"""

import json
import re
import time
import urllib.request
import urllib.error
from datetime import datetime
from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# DOMAIN MAPPING: Chọn domains phù hợp nhất theo ngữ cảnh
# ══════════════════════════════════════════════════════════

_UFM_DOMAIN_MAP = {
    # Tuyển sinh / Điểm chuẩn / Xét tuyển
    "tuyển sinh":      "tuyensinh.ufm.edu.vn",
    "điểm chuẩn":      "tuyensinh.ufm.edu.vn",
    "xét tuyển":        "tuyensinh.ufm.edu.vn",
    "chỉ tiêu":        "tuyensinh.ufm.edu.vn",
    "nguyện vọng":      "tuyensinh.ufm.edu.vn",
    "hồ sơ":           "tuyensinh.ufm.edu.vn",
    # Đào tạo / Học phí / Chương trình
    "đào tạo":         "pdt.ufm.edu.vn",
    "chương trình":    "pdt.ufm.edu.vn",
    "học phí":         "pdt.ufm.edu.vn",
    "tín chỉ":         "pdt.ufm.edu.vn",
    "lịch học":        "pdt.ufm.edu.vn",
    "thời khóa biểu":  "uis.ufm.edu.vn",
    # Nhập học
    "nhập học":        "nhaphoc.ufm.edu.vn",
    "thủ tục":         "nhaphoc.ufm.edu.vn",
    # Sinh viên
    "học bổng":        "ctsv.ufm.edu.vn",
    "rèn luyện":       "ctsv.ufm.edu.vn",
    "ký túc xá":       "ktx.ufm.edu.vn",
}

def _select_ufm_domains(query: str, max_domains: int = 3) -> list:
    """Chọn tối đa 3 domains UFM phù hợp nhất dựa trên từ khóa."""
    query_lower = query.lower()
    matched = set()

    for keyword, domain in _UFM_DOMAIN_MAP.items():
        if keyword in query_lower:
            matched.add(domain)

    matched.add("ufm.edu.vn")

    if len(matched) >= max_domains:
        return list(matched)[:max_domains]

    # Lấy thêm domain từ config nếu chưa đủ
    default_domains = query_flow_config.web_search.ufm_domains
    for d in default_domains:
        if d not in matched:
            matched.add(d)
        if len(matched) >= max_domains:
            break

    return list(matched)[:max_domains]


def _has_year(text: str) -> bool:
    """Kiểm tra câu query đã chứa năm (2020-2030) chưa."""
    return bool(re.search(r'20[2-3]\d', text))


def _inject_year_anchor(query: str) -> str:
    """Bơm neo thời gian vào query nếu chưa có năm."""
    if _has_year(query):
        return query
    current_year = datetime.now().year
    prev_year = current_year - 1
    return f"{query} năm {prev_year} {current_year}"


def _build_search_query(
    standalone_query: str,
    action: str,
    ufm_queries: list,
    pr_query: str,
) -> str:
    """Xây dựng user prompt cho Gemini + Google Search tool vắt chặt domain."""

    if action == "PROCEED_RAG_UFM_SEARCH":
        domains = _select_ufm_domains(standalone_query)
        site_filters = " OR ".join([f"site:{d}" for d in domains])
        
        search_terms = standalone_query
        if ufm_queries:
            search_terms = " | ".join(ufm_queries[:2])

        search_terms = _inject_year_anchor(search_terms)

        return (
            f"Bạn phải tìm kiếm bằng cú pháp sau để giới hạn đúng domain:\n"
            f"\"{search_terms} {site_filters}\"\n\n"
            f"CHỈ TRÍCH XUẤT THÔNG TIN TỪ CÁC TÊN MIỀN NÀY: {', '.join(domains)}.\n"
            f"TUYỆT ĐỐI KHÔNG lấy dữ liệu từ các trang báo ngoài hoặc nguồn khác.\n"
            f"Trả lời bằng tiếng Việt, ngắn gọn, có dẫn nguồn chính xác."
        )

    else:  # PROCEED_RAG_PR_SEARCH
        pr_domains = query_flow_config.web_search.pr_domains[:5] # limit to top 5
        site_filters = " OR ".join([f"site:{d}" for d in pr_domains])
        
        search_terms = pr_query or standalone_query
        search_terms = _inject_year_anchor(search_terms)

        return (
            f"Bạn phải tìm bài báo về Trường ĐH Tài chính - Marketing (UFM) bằng cú pháp sau:\n"
            f"\"{search_terms} {site_filters}\"\n\n"
            f"CHỈ ĐƯỢC PHÉP TRÍCH BÁO TỪ BÁO CHÍNH THỐNG: {', '.join(pr_domains)}.\n"
            f"TUYỆT ĐỐI KHÔNG dùng Wikipedia, Facebook hay nền tảng mạng xã hội.\n"
            f"Trả lời bằng tiếng Việt, trích dẫn rõ tên bài báo và link."
        )


def _extract_citations_from_text(text: str) -> list:
    """Fallback: Trích xuất [text](url) từ text nếu không có groundingMetadata."""
    pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    matches = re.findall(pattern, text)
    citations = []
    seen_urls = set()
    for text_part, url in matches:
        if url not in seen_urls:
            citations.append({"text": text_part.strip(), "url": url.strip()})
            seen_urls.add(url)
    return citations


# ══════════════════════════════════════════════════════════
# GOOGLE GEMINI NATIVE API — với google_search tool
# ══════════════════════════════════════════════════════════

def _call_gemini_native_with_search(
    system_prompt: str,
    user_content: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    allowed_domains: list = None,
) -> tuple:
    """
    Gọi Google Gemini API native với google_search tool.

    Returns: (raw_text, citations)
      - raw_text: Nội dung phản hồi từ Gemini
      - citations: List[{text, url}] từ groundingMetadata
    """
    api_key = query_flow_config.api_keys.get_key("google")
    base_url = query_flow_config.api_keys.get_base_url("google")

    if not api_key:
        raise ValueError("Chưa cấu hình GOOGLE_API_KEY trong .env")

    url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"

    body = {
        "contents": [
            {"role": "user", "parts": [{"text": user_content}]}
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    headers = {
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    # ── Parse text từ candidates ──
    raw_text = ""
    candidates = result.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        raw_text = "".join(p.get("text", "") for p in parts if "text" in p)

    # ── Parse citations từ groundingMetadata ──
    citations = []
    if candidates:
        grounding = candidates[0].get("groundingMetadata", {})
        grounding_chunks = grounding.get("groundingChunks", [])
        seen_urls = set()

        for chunk in grounding_chunks:
            web = chunk.get("web", {})
            chunk_url = web.get("uri", "")
            chunk_title = web.get("title", "").strip()
            if not chunk_url or chunk_url in seen_urls:
                continue

            # Google trả redirect URLs tạm thời → follow để lấy URL thật
            real_url = _resolve_google_redirect(chunk_url)
            if real_url and real_url not in seen_urls:
                citations.append({
                    "text": chunk_title or real_url,
                    "url": real_url,
                })
                seen_urls.add(real_url)

        # Fallback: extract URLs từ searchEntryPoint HTML
        if not citations:
            entry_html = grounding.get("searchEntryPoint", {}).get("renderedContent", "")
            if entry_html:
                html_citations = _extract_urls_from_html(entry_html)
                for c in html_citations:
                    if c["url"] not in seen_urls:
                        citations.append(c)
                        seen_urls.add(c["url"])

    # Regex fallback nếu vẫn chưa có
    if not citations:
        citations = _extract_citations_from_text(raw_text)

    # ── Validate URLs: loại bỏ link chết ──
    citations = _validate_citations(citations, allowed_domains=allowed_domains)

    return raw_text, citations


# ══════════════════════════════════════════════════════════
# GOOGLE REDIRECT RESOLVER — Follow redirect lấy URL thật
# ══════════════════════════════════════════════════════════

_GOOGLE_REDIRECT_HOSTS = (
    "vertexaisearch.cloud.google.com",
    "www.google.com/url",
)


def _resolve_google_redirect(url: str, timeout: int = 4) -> str | None:
    """
    Follow Google grounding redirect URL → trả về URL đích thực.
    Trả None nếu:
      - URL hết hạn (404)
      - Không redirect được
      - URL vẫn là Google redirect sau khi follow
    """
    # URL bình thường (không phải redirect) → giữ nguyên
    is_redirect = any(host in url for host in _GOOGLE_REDIRECT_HOSTS)
    if not is_redirect:
        return url

    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UFM-Bot/1.0")

        # Chặn auto-follow để lấy Location header
        class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None  # Không follow, trả Location thay thế

        opener = urllib.request.build_opener(NoRedirectHandler)
        try:
            opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                final_url = e.headers.get("Location", "")
                if final_url and "vertexaisearch.cloud.google.com" not in final_url:
                    logger.debug("Resolved Google redirect → %s", final_url[:120])
                    return final_url
            # 404 = redirect hết hạn
            logger.debug("Google redirect expired (HTTP %d): %s", e.code, url[:80])
            return None

    except Exception as e:
        logger.debug("Cannot resolve Google redirect: %s → %s", url[:80], e)

    return None


def _extract_urls_from_html(html: str) -> list:
    """
    Extract real URLs từ searchEntryPoint.renderedContent HTML.
    Tìm tất cả href="https://..." trong HTML.
    """
    urls = re.findall(r'href=["\']?(https?://[^\s"\'<>]+)', html)
    seen = set()
    results = []
    for u in urls:
        # Bỏ Google internal URLs
        if "google.com" in u or "googleapis.com" in u:
            continue
        if u not in seen:
            # Dùng domain làm title
            try:
                from urllib.parse import urlparse
                domain = urlparse(u).netloc
            except Exception:
                domain = u
            results.append({"text": domain, "url": u})
            seen.add(u)
    return results


def _validate_citations(citations: list, allowed_domains: list = None, timeout: int = 3) -> list:
    """
    Ping HEAD request song song tất cả URLs.
    Chỉ giữ lại citations có URL trả HTTP 200-399.
    Loại bỏ cứng mọi Google redirect URLs còn sót.
    Loại bỏ cứng mọi URL không nằm trong allowed_domains (nếu có).
    """
    if not citations:
        return []

    # Bước 1: Loại bỏ cứng Google redirect URLs và URL ngoài biên
    filtered = []
    
    # Chuẩn bị danh sách domain cho phép
    safe_domains = [d.lower() for d in allowed_domains] if allowed_domains else []

    for c in citations:
        url = c.get("url", "")
        if not url:
            continue
            
        if any(host in url for host in _GOOGLE_REDIRECT_HOSTS):
            logger.warning("Web Search - Loại Google redirect: %s", url[:80])
            continue
            
        # Hard-filter domain
        if safe_domains:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.lower()
                # Chấp nhận subdomain (vd: pdt.ufm.edu.vn endswith ufm.edu.vn)
                is_allowed = any(domain == d or domain.endswith(f".{d}") for d in safe_domains)
                if not is_allowed:
                    logger.warning("Web Search - Loai URL NGOAI LUONG: %s", url[:80])
                    continue
            except Exception:
                pass

        filtered.append(c)

    if not filtered:
        return []

    # Bước 2: Ping song song
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _ping(citation: dict) -> dict | None:
        url = citation.get("url", "")
        if not url:
            return None
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UFM-Bot/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status < 400:
                    return citation
        except urllib.error.HTTPError as e:
            # Một số server trả 405 cho HEAD nhưng GET OK
            if e.code == 405:
                return citation
        except Exception:
            pass
        logger.warning("Web Search - Link chết, loại bỏ: %s", url)
        return None

    valid = []
    with ThreadPoolExecutor(max_workers=min(len(filtered), 5)) as pool:
        futures = {pool.submit(_ping, c): c for c in filtered}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                valid.append(result)

    # Giữ nguyên thứ tự gốc
    url_set = {c["url"] for c in valid}
    ordered = [c for c in filtered if c["url"] in url_set]

    logger.info(
        "Web Search - URL validate: %d/%d links sống",
        len(ordered), len(citations),
    )
    return ordered


# ══════════════════════════════════════════════════════════
# WEB SEARCH NODE — Hàm chính cho Graph
# ══════════════════════════════════════════════════════════

def web_search_node(state: GraphState) -> GraphState:
    """
    🌐 WEB SEARCH NODE — Gemini 2.5 Flash + Google Search Tool.

    Luồng:
      1. Gọi Google Gemini native API với google_search tool (PRIMARY)
      2. Nếu lỗi → Fallback sang OpenRouter search models
      3. Trích citations từ groundingMetadata (hoặc regex fallback)
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    action = state.get("intent_action", "")
    ufm_queries = state.get("ufm_search_queries") or []
    pr_query = state.get("pr_search_query")

    config = query_flow_config.web_search
    start_time = time.time()

    if not config.enabled:
        return {
            **state,
            "web_search_results": None,
            "web_search_citations": None,
            "next_node": "synthesizer",
        }

    # ── Xây dựng search prompt ──
    search_prompt = _build_search_query(
        standalone_query=standalone_query,
        action=action,
        ufm_queries=ufm_queries,
        pr_query=pr_query,
    )

    system_prompt = prompt_manager.get_system("web_search_node")
    logger.info("Web Search - Prompt: '%s'", search_prompt[:120])

    # ── Chuẩn bị allowed_domains để validate ──
    if action == "PROCEED_RAG_UFM_SEARCH":
        allowed_domains = query_flow_config.web_search.ufm_domains
    else:
        allowed_domains = query_flow_config.web_search.pr_domains[:5]

    # ══════════════════════════════════════════════════════
    # BƯỚC 1: Thử Google Gemini Native + google_search tool
    # ══════════════════════════════════════════════════════
    try:
        raw_result, citations = _call_gemini_native_with_search(
            system_prompt=system_prompt,
            user_content=search_prompt,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
            allowed_domains=allowed_domains,
        )

        elapsed = time.time() - start_time
        logger.info(
            "Web Search [%.3fs] GOOGLE OK (%s) %d citations, %d chars",
            elapsed, config.model, len(citations), len(raw_result),
        )

        return {
            **state,
            "web_search_results": raw_result,
            "web_search_citations": citations,
            "next_node": "synthesizer",
        }

    except Exception as google_err:
        elapsed_google = time.time() - start_time
        logger.warning(
            "Web Search [%.3fs] GOOGLE FAIL (%s): %s → Thu fallback OpenRouter",
            elapsed_google, config.model, google_err,
        )

    # ══════════════════════════════════════════════════════
    # BƯỚC 2: Fallback → OpenRouter search models
    # ══════════════════════════════════════════════════════
    try:
        raw_result = _call_gemini_api_with_fallback(
            system_prompt=system_prompt,
            user_content=search_prompt,
            config_section=config,
            node_key="web_search",
        )

        citations = _extract_citations_from_text(raw_result)
        citations = _validate_citations(citations, allowed_domains=allowed_domains)
        elapsed = time.time() - start_time
        logger.info(
            "Web Search [%.3fs] FALLBACK OK, %d citations, %d chars",
            elapsed, len(citations), len(raw_result),
        )

        return {
            **state,
            "web_search_results": raw_result,
            "web_search_citations": citations,
            "next_node": "synthesizer",
        }

    except Exception as fallback_err:
        elapsed = time.time() - start_time
        logger.error(
            "Web Search [%.3fs] ALL FAILED: Google=%s | Fallback=%s",
            elapsed, google_err, fallback_err, exc_info=True,
        )
        return {
            **state,
            "web_search_results": None,
            "web_search_citations": None,
            "next_node": "synthesizer",
        }
