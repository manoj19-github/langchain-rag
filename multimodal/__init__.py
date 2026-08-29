"""
__init__.py — Multimodal RAG Package
=====================================
Public API: import only what you need.

Usage examples:

  # Full pipeline in one call
  from multimodal.pipeline import run
  result = run("report.pdf", "What was revenue?")

  # Individual stages
  from multimodal.ingestion import ingest
  from multimodal.preprocessing import preprocess
  from multimodal.chunking import chunk_documents
  from multimodal.query_enhancement import enhance_query
  from multimodal.retrieval import hybrid_retrieve

  # Config
  from multimodal.config import cfg
  cfg.LLM_MODEL = "llama3:8b"
"""

from .config       import cfg
from .pipeline     import run
from .models       import (
    FinalAnswer, Chunk, ProcessedDocument, RawDocument,
    EnhancedQuery, ContextWindow, EvidenceBlock,
    ElementType, ChunkStrategy,
)

__version__ = "1.0.0"
__all__ = [
    "run", "cfg",
    "FinalAnswer", "Chunk", "ProcessedDocument", "RawDocument",
    "EnhancedQuery", "ContextWindow", "EvidenceBlock",
    "ElementType", "ChunkStrategy",
]
