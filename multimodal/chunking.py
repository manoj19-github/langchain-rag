"""
chunking.py — Stage 3: Intelligent Modal-Specific Chunking
===========================================================
Splits ProcessedDocuments into Chunks that can be embedded and searched.

Different content types use different chunking strategies:
  📄 TEXT / HEADING / PARAGRAPH → Semantic chunking (fixed-size + overlap)
  📊 TABLE                      → Whole-table chunk + one chunk per row
  🖼️  CHART / IMAGE / DIAGRAM   → One chunk per visual (vision description as text)
  📋 OCR_TEXT                   → Same as semantic text chunking

Why different strategies?
  - A table row "Revenue | 482.6 | 406.5" must stay together — don't split mid-row
  - A chart description is already short — no splitting needed
  - Long paragraphs need overlap so context isn't lost at boundaries

Inputs:  List[ProcessedDocument]
Outputs: List[Chunk]

Usage:
    from multimodal.chunking import chunk_documents
    chunks = chunk_documents(processed_docs)
    print(f"Created {len(chunks)} chunks")
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .config import cfg
from .models import Chunk, ChunkStrategy, ElementType, ProcessedDocument

logger = logging.getLogger("multimodal.chunking")


# ══════════════════════════════════════════════════════════════════
# HELPER: Semantic text chunking with overlap
# ══════════════════════════════════════════════════════════════════

def _split_text(
    text: str,
    chunk_size: int  = cfg.TEXT_CHUNK_SIZE,
    overlap: int     = cfg.TEXT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a long string into overlapping chunks.

    Visual example (chunk_size=20, overlap=5):
        "The quick brown fox jumped over the lazy dog"
         ├────── Chunk 1 ───────┤
                           ├────── Chunk 2 ───────┤
                                              ├── Chunk 3

    The overlap preserves context at boundaries.
    We prefer to break at paragraph → sentence → word boundaries (in that order).
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at a natural boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = text.rfind(sep, start + overlap, end)
                if idx > start:
                    end = idx + len(sep)
                    break

        part = text[start:end].strip()
        if len(part) >= cfg.MIN_CHUNK_CHARS:
            chunks.append(part)

        start = end - overlap if end < len(text) else len(text)

    return chunks


# ══════════════════════════════════════════════════════════════════
# CHUNKING STRATEGIES
# ══════════════════════════════════════════════════════════════════

def _chunk_text_element(doc: ProcessedDocument, parent_heading: str) -> List[Chunk]:
    """
    Strategy: SEMANTIC chunking.
    Used for: TEXT, HEADING, PARAGRAPH, OCR_TEXT, CAPTION.
    """
    chunks = []
    for ci, part in enumerate(_split_text(doc.content)):
        cid = f"{doc.element_id}_chunk_{ci}"
        chunks.append(Chunk(
            chunk_id      = cid,
            element_id    = doc.element_id,
            doc_name      = doc.doc_name,
            page_number   = doc.page_number,
            element_type  = doc.element_type,
            strategy      = ChunkStrategy.SEMANTIC,
            text          = part,
            parent_heading= parent_heading,
            metadata      = {
                "doc_name":     doc.doc_name,
                "page_number":  doc.page_number,
                "element_type": doc.element_type.value,
                "language":     doc.language,
                "confidence":   doc.confidence,
                "entities":     doc.named_entities,
            },
        ))
    return chunks


def _chunk_table(doc: ProcessedDocument, parent_heading: str) -> List[Chunk]:
    """
    Strategy: TABLE_WHOLE + TABLE_ROW chunking.
    Used for: TABLE elements.

    Creates:
      1. One "whole table" chunk — for broad semantic search
      2. One chunk per row — for precise row-level retrieval

    Example whole-table text:
        "Metric FY2025-26 FY2024-25 Change | Revenue 482.6 406.5 +18.7%"

    Example row chunk text:
        "n Crore: Revenue | FY2025-26: 482.6 | FY2024-25: 406.5 | Change: +18.7%"
    """
    if not doc.table_data:
        return _chunk_text_element(doc, parent_heading)

    td = doc.table_data
    chunks = []

    # Build structured values for exact numeric lookup
    structured = {
        "headers":      td.headers,
        "row_count":    len(td.rows),
        "numeric_cells": td.extract_numbers()[:50],
        "currency":     td.currency,
        "units":        td.units,
        "title":        td.title,
    }

    # ── Chunk 1: Whole table ────────────────────────────────────
    chunks.append(Chunk(
        chunk_id       = f"{doc.element_id}_whole",
        element_id     = doc.element_id,
        doc_name       = doc.doc_name,
        page_number    = doc.page_number,
        element_type   = ElementType.TABLE,
        strategy       = ChunkStrategy.TABLE_WHOLE,
        text           = td.to_search_text(),
        table_data     = td,
        structured_vals= structured,
        parent_heading = parent_heading,
        metadata       = {
            "doc_name":     doc.doc_name,
            "page_number":  doc.page_number,
            "element_type": "TABLE",
            "markdown":     td.to_markdown(),   # For LLM context window
            "has_table":    True,
        },
    ))

    # ── Chunks 2..N: One per row ────────────────────────────────
    for ri, row in enumerate(td.rows[:cfg.MAX_TABLE_ROWS_PER_CHUNK]):
        # Format: "Header1: Value1 | Header2: Value2 | ..."
        row_text = " | ".join(
            f"{h}: {v}" for h, v in zip(td.headers, row)
        )
        chunks.append(Chunk(
            chunk_id       = f"{doc.element_id}_row_{ri}",
            element_id     = doc.element_id,
            doc_name       = doc.doc_name,
            page_number    = doc.page_number,
            element_type   = ElementType.TABLE,
            strategy       = ChunkStrategy.TABLE_ROW,
            text           = row_text,
            table_data     = td,
            structured_vals= {"row_index": ri, "row_label": row[0] if row else ""},
            parent_heading = parent_heading,
            metadata       = {
                "doc_name":     doc.doc_name,
                "page_number":  doc.page_number,
                "element_type": "TABLE",
                "is_row_chunk": True,
                "row_index":    ri,
            },
        ))

    return chunks


def _chunk_visual(doc: ProcessedDocument, parent_heading: str) -> List[Chunk]:
    """
    Strategy: IMAGE_PATCH chunking.
    Used for: CHART, IMAGE, DIAGRAM.

    The "text" of this chunk is the AI-generated vision description.
    The original image bytes are kept for passing to a vision LLM at answer time.
    """
    desc = doc.content   # Already replaced with vision description in preprocessing
    if not desc or len(desc) < cfg.MIN_CHUNK_CHARS:
        desc = f"[{doc.element_type.value} on page {doc.page_number}]"

    return [Chunk(
        chunk_id       = f"{doc.element_id}_visual",
        element_id     = doc.element_id,
        doc_name       = doc.doc_name,
        page_number    = doc.page_number,
        element_type   = doc.element_type,
        strategy       = ChunkStrategy.IMAGE_PATCH,
        text           = desc,
        image_bytes    = doc.image_bytes,
        image_b64      = doc.image_b64,
        parent_heading = parent_heading,
        metadata       = {
            "doc_name":     doc.doc_name,
            "page_number":  doc.page_number,
            "element_type": doc.element_type.value,
            "has_image":    True,
            "caption":      doc.metadata.get("caption", ""),
        },
    )]


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — chunk_documents()
# ══════════════════════════════════════════════════════════════════

def chunk_documents(processed_docs: List[ProcessedDocument]) -> List[Chunk]:
    """
    Stage 3 entry point.

    Routes each ProcessedDocument to the correct chunking strategy:
      - TEXT/HEADING/PARAGRAPH/OCR_TEXT → _chunk_text_element()
      - TABLE                           → _chunk_table()
      - CHART/IMAGE/DIAGRAM             → _chunk_visual()

    Also tracks the current section heading per page so each chunk
    knows which section it belongs to (parent_heading).

    Args:
        processed_docs: Output of Stage 2 (preprocessing.py)

    Returns:
        List[Chunk] — the fundamental units for embedding & indexing
    """
    logger.info(f"[Stage 3] Chunking {len(processed_docs)} processed documents")

    chunks: List[Chunk] = []
    # Tracks the most recent heading seen on each page
    heading_by_page: Dict[int, str] = {}

    text_types   = {ElementType.TEXT, ElementType.HEADING, ElementType.PARAGRAPH,
                    ElementType.CAPTION, ElementType.OCR_TEXT}
    visual_types = {ElementType.CHART, ElementType.IMAGE, ElementType.DIAGRAM}

    for doc in processed_docs:
        # Skip flagged duplicates
        if doc.is_duplicate:
            continue

        # Track current heading context
        if doc.element_type == ElementType.HEADING:
            heading_by_page[doc.page_number] = doc.content
        parent_heading = heading_by_page.get(doc.page_number, "")

        # Route to correct strategy
        if doc.element_type in text_types:
            new_chunks = _chunk_text_element(doc, parent_heading)
        elif doc.element_type == ElementType.TABLE:
            new_chunks = _chunk_table(doc, parent_heading)
        elif doc.element_type in visual_types:
            new_chunks = _chunk_visual(doc, parent_heading)
        else:
            new_chunks = _chunk_text_element(doc, parent_heading)

        chunks.extend(new_chunks)

    # Summary
    by_strategy: Dict[str, int] = {}
    for c in chunks:
        by_strategy[c.strategy.value] = by_strategy.get(c.strategy.value, 0) + 1

    logger.info(f"  Total chunks: {len(chunks)}")
    logger.info(f"  By strategy: {by_strategy}")
    return chunks
