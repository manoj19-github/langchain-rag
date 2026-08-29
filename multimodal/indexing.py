"""
indexing.py — Stage 5: Indexing & Storage
==========================================
Builds two search indexes from the chunks:

  📦 ChromaDB  — Dense vector database (semantic similarity search)
  📦 BM25      — Sparse keyword index (exact term matching)

Why two indexes?
  ChromaDB: Finds "annual income" when you search "revenue" (semantic)
  BM25:     Finds "FY2025-26" exactly when you type "FY2025-26" (keyword)
  Both together = better recall than either alone.

Cache strategy:
  - BM25 index saved to .rag_cache/bm25_<hash>.pkl
  - ChromaDB persisted to .rag_cache/chroma_<hash>/
  - Loading from cache is ~50x faster than rebuilding

Inputs:  List[Chunk], pdf_hash (str)
Outputs: (vstore, bm25_retriever)  — passed directly to retrieval.py

Usage:
    from multimodal.indexing import build_index
    vstore, bm25 = build_index(chunks, pdf_hash="abc123def456")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from .config import cfg
from .embeddings import get_embedding_model
from .models import Chunk

logger = logging.getLogger("multimodal.indexing")


# ══════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════════

def _cache_path(key: str) -> Path:
    p = Path(cfg.CACHE_DIR)
    p.mkdir(exist_ok=True)
    return p / f"{key}.pkl"


def _cache_load(key: str):
    if not cfg.CACHE_ENABLED:
        return None
    cp = _cache_path(key)
    if cp.exists():
        try:
            with open(cp, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _cache_save(key: str, obj) -> None:
    if not cfg.CACHE_ENABLED:
        return
    try:
        with open(_cache_path(key), "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# CONVERT CHUNKS → LANGCHAIN DOCUMENTS
# ══════════════════════════════════════════════════════════════════

def _to_lc_docs(chunks: List[Chunk]):
    """
    Convert our Chunk objects to LangChain Document objects.

    LangChain's Document has two fields:
      page_content: str         — the text to embed
      metadata: dict            — anything you want to filter/retrieve later

    The metadata lets us recover the original Chunk after retrieval.
    """
    from langchain_core.documents import Document as LCDoc

    lc_docs = []
    for c in chunks:
        if len(c.text.strip()) < cfg.MIN_CHUNK_CHARS:
            continue    # Skip garbage chunks
        lc_docs.append(LCDoc(
            page_content = c.text,
            metadata = {
                "chunk_id":     c.chunk_id,
                "element_id":   c.element_id,
                "doc_name":     c.doc_name,
                "page_number":  c.page_number,
                "element_type": c.element_type.value,
                "strategy":     c.strategy.value,
                "has_image":    c.image_b64 is not None,
                "has_table":    c.table_data is not None,
                "parent_heading": c.parent_heading or "",
            },
        ))
    return lc_docs


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — build_index()
# ══════════════════════════════════════════════════════════════════

def build_index(chunks: List[Chunk], pdf_hash: str):
    """
    Stage 5 entry point.

    Builds and returns two search indexes:
      1. ChromaDB vector store (semantic search via cosine similarity)
      2. BM25 sparse retriever (keyword search)

    Both are either loaded from cache or built fresh.

    Args:
        chunks:   Output of Stage 3 (chunking.py)
        pdf_hash: Short SHA256 of the PDF file (for cache naming)

    Returns:
        (vstore, bm25_retriever) — tuple passed to retrieval.py
    """
    logger.info(f"[Stage 5] Building indexes for {len(chunks)} chunks")

    lc_docs = _to_lc_docs(chunks)
    logger.info(f"  {len(lc_docs)} valid documents for indexing")

    # ── BM25 (Sparse keyword index) ────────────────────────────
    bm25_key = f"bm25_{pdf_hash}"
    bm25_retriever = _cache_load(bm25_key)

    if bm25_retriever is not None:
        logger.info("  BM25 loaded from cache ✓")
    else:
        logger.info("  Building BM25 index...")
        from langchain_community.retrievers import BM25Retriever
        bm25_retriever = BM25Retriever.from_documents(lc_docs)
        bm25_retriever.k = cfg.TOP_K_BM25
        _cache_save(bm25_key, bm25_retriever)
        logger.info("  BM25 built and cached ✓")

    # ── ChromaDB (Dense vector index) ──────────────────────────
    # ChromaDB persists to disk — we always rebuild it here
    # (ChromaDB handles its own persistence internally)
    chroma_dir = f"{cfg.CACHE_DIR}/chroma_{pdf_hash}"
    logger.info("  Building ChromaDB vector index...")

    from langchain_chroma import Chroma
    vstore = Chroma.from_documents(
        documents        = lc_docs,
        embedding        = get_embedding_model(),
        collection_name  = f"mmrag_{pdf_hash}",
        persist_directory= chroma_dir,
    )
    logger.info(f"  ChromaDB built: {len(lc_docs)} vectors ✓")

    return vstore, bm25_retriever


# ══════════════════════════════════════════════════════════════════
# UTILITY — build chunks_by_id lookup
# ══════════════════════════════════════════════════════════════════

def build_chunk_lookup(chunks: List[Chunk]) -> Dict[str, Chunk]:
    """
    Creates a dict: chunk_id → Chunk for fast lookups.

    After retrieval, LangChain returns Documents with metadata["chunk_id"].
    We use this dict to recover the full Chunk object from that ID.

    Args:
        chunks: All chunks from chunking.py

    Returns:
        Dict mapping chunk_id to Chunk
    """
    return {c.chunk_id: c for c in chunks}
