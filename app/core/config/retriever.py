# app/core/config/retriever.py
# Pydantic config cho Hybrid Retriever (Vector + BM25 + RRF)
# Đọc từ: retriever_config.yaml

import os
from pydantic import BaseModel, Field
from typing import Optional
from app.core.config import _load_yaml


_rt_data = _load_yaml("retriever_config.yaml")
_retriever = _rt_data.get("retriever", {})


# ── Vector Search ──
_vs = _retriever.get("vector_search", {})

class VectorSearchConfig(BaseModel):
    enabled: bool = _vs.get("enabled", True)
    top_k: int = _vs.get("top_k", 20)
    similarity_threshold: float = Field(
        default=_vs.get("similarity_threshold", 0.85),
        ge=0.0, le=1.0
    )
    use_multi_query: bool = _vs.get("use_multi_query", False)


# ── BM25 Search ──
_bm = _retriever.get("bm25_search", {})

class BM25SearchConfig(BaseModel):
    enabled: bool = _bm.get("enabled", True)
    top_k: int = _bm.get("top_k", 20)
    ts_config: str = _bm.get("ts_config", "simple")
    use_stored_tsvector: bool = _bm.get("use_stored_tsvector", False)


# ── RRF ──
_rrf = _retriever.get("rrf", {})

class RRFConfig(BaseModel):
    k: int = _rrf.get("k", 60)


# ── Parent Retrieval ──
_pr = _retriever.get("parent_retrieval", {})

class ParentRetrievalConfig(BaseModel):
    top_parents: int = _pr.get("top_parents", 5)
    max_parent_chars: int = _pr.get("max_parent_chars", 8000)
    include_metadata: bool = _pr.get("include_metadata", True)


# ── DB Connection ──
_db = _retriever.get("db", {})

class RetrieverDBConfig(BaseModel):
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname: str = os.getenv("POSTGRES_DB", "ufm_admission_db")
    user: str = os.getenv("POSTGRES_USER", "ufm_admin")
    password: str = os.getenv("POSTGRES_PASSWORD", "ufm_secure_password_2026")
    pool_min: int = _db.get("pool_min", 1)
    pool_max: int = _db.get("pool_max", 5)
    connect_timeout: int = _db.get("connect_timeout", 5)
    query_timeout: int = _db.get("query_timeout", 10)


# ── Filters ──
_fl = _retriever.get("filters", {})

class RetrieverFilterConfig(BaseModel):
    only_active: bool = _fl.get("only_active", True)
    academic_year: Optional[str] = _fl.get("academic_year", None)




# ── Metadata Filter ──
_mf = _retriever.get("metadata_filter", {})

class MetadataFilterConfig(BaseModel):
    filter_program_level: bool = _mf.get("filter_program_level", True)
    filter_program_name: bool = _mf.get("filter_program_name", False)


# ── Config tổng Retriever ──
class RetrieverConfig(BaseModel):
    vector_search: VectorSearchConfig = VectorSearchConfig()
    bm25_search: BM25SearchConfig = BM25SearchConfig()
    rrf: RRFConfig = RRFConfig()
    parent_retrieval: ParentRetrievalConfig = ParentRetrievalConfig()
    metadata_filter: MetadataFilterConfig = MetadataFilterConfig()
    db: RetrieverDBConfig = RetrieverDBConfig()
    filters: RetrieverFilterConfig = RetrieverFilterConfig()
