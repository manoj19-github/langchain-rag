"""
config.py — Stage 0: Global Configuration
==========================================
All settings live here. Change these values to tune the pipeline.
Never hardcode secrets — use environment variables for API keys.

Usage:
    from multimodal.config import cfg
    print(cfg.LLM_MODEL)
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------
    # LLM SETTINGS
    # ------------------------------------------------------------------
    LLM_PROVIDER: str    = os.getenv("LLM_PROVIDER", "ollama")   # "ollama" | "openai" | "anthropic"
    LLM_MODEL: str       = os.getenv("LLM_MODEL",    "qwen3:4b-instruct")
    LLM_TEMPERATURE: float = 0.0   # 0 = deterministic (best for fact answers)

    # ------------------------------------------------------------------
    # VISION MODEL (for describing charts / images)
    # ------------------------------------------------------------------
    VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "ollama")
    VISION_MODEL: str    = os.getenv("VISION_MODEL",    "llava:13b")

    # ------------------------------------------------------------------
    # HYDE (Hypothetical Document Embedding)
    # ------------------------------------------------------------------
    HYDE_MODEL: str       = os.getenv("HYDE_MODEL", "qwen3:4b-instruct")
    HYDE_TEMPERATURE: float = 0.4   # Slightly creative for better hypothetical docs
    HYDE_ENABLED: bool    = True

    # ------------------------------------------------------------------
    # EMBEDDING MODEL (text → vectors)
    # ------------------------------------------------------------------
    EMBEDDING_MODEL: str  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")   # "cuda" if GPU

    # ------------------------------------------------------------------
    # RERANKER (CrossEncoder)
    # ------------------------------------------------------------------
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # RETRIEVAL SETTINGS
    # ------------------------------------------------------------------
    TOP_K_DENSE: int      = 15   # Chunks from vector search
    TOP_K_BM25: int       = 15   # Chunks from keyword search
    TOP_K_TABLE: int      = 10   # Chunks from table-specific search
    TOP_K_CHART: int      = 8    # Chunks from chart search
    TOP_K_VISUAL: int     = 8    # Chunks from image search
    TOP_K_STRUCTURED: int = 5    # Chunks from exact numeric lookup
    MMR_FETCH_K: int      = 40   # Candidate pool before MMR diversity filter
    MMR_LAMBDA: float     = 0.6  # 0=max diversity, 1=max relevance
    RERANK_TOP_K: int     = 7    # Final chunks sent to LLM
    RERANK_BATCH_SIZE: int= 16

    # ------------------------------------------------------------------
    # RRF FUSION WEIGHTS (must sum to ≤ 1.0)
    # ------------------------------------------------------------------
    W_DENSE: float      = 0.35   # Vector search weight
    W_BM25: float       = 0.25   # Keyword search weight
    W_TABLE: float      = 0.20   # Table-specific weight
    W_CHART: float      = 0.10   # Chart weight
    W_VISUAL: float     = 0.07   # Image weight
    W_STRUCTURED: float = 0.03   # Exact numeric weight
    RRF_K: int          = 60     # RRF smoothing constant

    # ------------------------------------------------------------------
    # CHUNKING SETTINGS
    # ------------------------------------------------------------------
    TEXT_CHUNK_SIZE: int    = 600   # Max characters per text chunk
    TEXT_CHUNK_OVERLAP: int = 80    # Overlap between consecutive chunks
    MIN_CHUNK_CHARS: int    = 30    # Ignore chunks shorter than this
    MAX_TABLE_ROWS_PER_CHUNK: int = 60

    # ------------------------------------------------------------------
    # IMAGE EXTRACTION
    # ------------------------------------------------------------------
    EXTRACT_IMAGES: bool    = True
    MIN_IMAGE_WIDTH: int    = 80    # Ignore tiny images (icons, bullets)
    MIN_IMAGE_HEIGHT: int   = 80
    MAX_IMAGES_PER_PAGE: int= 10

    # ------------------------------------------------------------------
    # OCR (Tesseract — for scanned pages)
    # ------------------------------------------------------------------
    OCR_ENABLED: bool = True
    OCR_LANG: str     = "eng"
    OCR_DPI: int      = 300    # Higher = better OCR, slower

    # ------------------------------------------------------------------
    # CACHE (speeds up repeated runs dramatically)
    # ------------------------------------------------------------------
    CACHE_ENABLED: bool = True
    CACHE_DIR: str      = ".rag_cache"

    # ------------------------------------------------------------------
    # OLLAMA SERVER
    # ------------------------------------------------------------------
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ------------------------------------------------------------------
    # API KEYS (read from environment — never hardcode)
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str    = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------
    DEBUG: bool = True   # Set False in production to silence stage banners


# ── Singleton instance used throughout the package ──
cfg = Config()
