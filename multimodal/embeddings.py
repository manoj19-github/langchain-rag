"""
embeddings.py — Stage 4: Embedding Generation
==============================================
Converts text chunks into dense vectors (numbers) so they can be
stored in a vector database and searched by semantic similarity.

Think of embeddings as coordinates in a 384-dimensional space where:
  "revenue growth" ≈ "annual income increase" (close together)
  "revenue growth" ≠ "employee headcount" (far apart)

Models used:
  📄 Text   → BAAI/bge-small-en-v1.5 (fast, high quality, free)
  🖼️  Image  → CLIP (if PIL available) — NOT used by default
               (We use vision text description instead)

Inputs:  List[Chunk]
Outputs: The same List[Chunk] — but embeddings are stored separately
         in the LangChain embedding wrapper (used by indexing.py)

Usage:
    from multimodal.embeddings import get_embedding_model
    embedder = get_embedding_model()
    vectors = embedder.embed_documents(["Revenue was 482.6 Cr", ...])
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .config import cfg

logger = logging.getLogger("multimodal.embeddings")

# ── Singleton cache — load model only once ──
_embeddings_instance = None


def get_embedding_model():
    """
    Returns a LangChain-compatible embedding model (singleton).

    The model is downloaded once (from HuggingFace Hub) and cached locally.
    After first load, subsequent calls return the already-loaded model
    without any download or re-initialization overhead.

    Model: BAAI/bge-small-en-v1.5
      - 384 dimensions (compact but powerful)
      - Licensed for commercial use
      - Runs on CPU without GPU
      - ~130 MB download

    Usage:
        embedder = get_embedding_model()
        vectors = embedder.embed_documents(["text1", "text2"])
        # vectors = [[0.12, -0.34, ...], [0.56, 0.78, ...]]
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info(f"[Stage 4] Loading embedding model: {cfg.EMBEDDING_MODEL}")
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name   = cfg.EMBEDDING_MODEL,
            model_kwargs = {"device": cfg.EMBEDDING_DEVICE},
            # cache_folder avoids re-downloading on each run
        )
        logger.info(f"  Embedding model loaded (device={cfg.EMBEDDING_DEVICE})")
    return _embeddings_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Utility: embed a list of strings and return their vectors.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors, one per input string.

    Example:
        vectors = embed_texts(["Revenue 482.6 Cr", "Net profit 61.8 Cr"])
        print(len(vectors[0]))   # → 384
    """
    embedder = get_embedding_model()
    return embedder.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.
    (Some models use a different query prefix than document prefix.)

    Args:
        query: The user's question string.

    Returns:
        A single float vector (384 dimensions).
    """
    embedder = get_embedding_model()
    return embedder.embed_query(query)
