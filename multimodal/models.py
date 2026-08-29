"""
models.py — Stage 0: Data Models
=================================
All dataclasses and enums used across the pipeline.
Every stage imports from this file — no circular imports.

Types flow through the pipeline like this:
  RawDocument → ProcessedDocument → Chunk → ScoredChunk → EvidenceBlock → FinalAnswer
"""

from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════

class ElementType(str, Enum):
    """Type of content element extracted from a document."""
    TEXT      = "TEXT"       # Regular body text
    HEADING   = "HEADING"    # Section title (large/bold font)
    PARAGRAPH = "PARAGRAPH"  # Long descriptive block
    TABLE     = "TABLE"      # Structured rows & columns
    CHART     = "CHART"      # Bar/line/pie chart (image-based)
    IMAGE     = "IMAGE"      # Photo, screenshot, logo
    DIAGRAM   = "DIAGRAM"    # Flowchart, architecture diagram
    CAPTION   = "CAPTION"    # Text beneath a figure or table
    OCR_TEXT  = "OCR_TEXT"   # Text from scanned/image page via Tesseract


class ChunkStrategy(str, Enum):
    """How this chunk was created — used for observability."""
    SEMANTIC     = "semantic"     # Fixed-size text with overlap
    TABLE_WHOLE  = "table_whole"  # Entire table as one chunk
    TABLE_ROW    = "table_row"    # One row of a table
    IMAGE_PATCH  = "image_patch"  # Vision-described image/chart
    STRUCTURED   = "structured"   # Exact key-value numeric lookup


# ══════════════════════════════════════════════════════════════════
# PRIMITIVE DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class BBox:
    """Bounding box — where on the page this element appears."""
    x0: float = 0.0   # Left edge
    y0: float = 0.0   # Top edge
    x1: float = 0.0   # Right edge
    y1: float = 0.0   # Bottom edge


@dataclass
class TableData:
    """
    Structured table extracted from a PDF page.

    Example:
        headers = ["Metric", "FY2025-26", "FY2024-25", "Change"]
        rows    = [["Revenue", "482.6", "406.5", "+18.7%"], ...]
    """
    headers:   List[str]       = field(default_factory=list)
    rows:      List[List[str]] = field(default_factory=list)
    title:     Optional[str]   = None    # Table caption if detected
    units:     Optional[str]   = None    # "crore", "million", "%"
    currency:  Optional[str]   = None    # "INR", "$", "EUR"
    footnotes: List[str]       = field(default_factory=list)

    def to_markdown(self) -> str:
        """Format as a Markdown table for the LLM context window."""
        lines = []
        if self.title:
            lines.append(f"**{self.title}**")
        if self.units:
            lines.append(f"*Units: {self.units}*")
        if self.currency:
            lines.append(f"*Currency: {self.currency}*")
        if self.headers:
            lines.append("| " + " | ".join(str(h) for h in self.headers) + " |")
            lines.append("|" + "|".join(["---"] * len(self.headers)) + "|")
        for row in self.rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        for fn in self.footnotes:
            lines.append(f"_* {fn}_")
        return "\n".join(lines)

    def to_search_text(self) -> str:
        """Flat searchable text for embedding/BM25."""
        parts = []
        if self.title:
            parts.append(self.title)
        parts.append(" ".join(str(h) for h in self.headers))
        for row in self.rows:
            parts.append(" | ".join(str(c) for c in row))
        return " | ".join(parts)

    def extract_numbers(self) -> List[Tuple[str, str, str]]:
        """Return (col_header, row_label, value) for all numeric cells."""
        results = []
        num_re = re.compile(r"[\d,]+\.?\d*")
        for row in self.rows:
            row_label = row[0] if row else ""
            for i, cell in enumerate(row[1:], start=1):
                if num_re.search(str(cell)):
                    col = self.headers[i] if i < len(self.headers) else f"col_{i}"
                    results.append((col, row_label, str(cell)))
        return results


# ══════════════════════════════════════════════════════════════════
# STAGE 1 OUTPUT — RawDocument
# ══════════════════════════════════════════════════════════════════

@dataclass
class RawDocument:
    """
    Output of Stage 1 (ingestion.py).
    Represents one extracted element from a source file.
    """
    element_id:   str
    source_file:  str          # Original file path
    doc_name:     str          # Filename only (for citations)
    page_number:  int          # 1-indexed
    element_type: ElementType
    content:      str          # Searchable text representation
    raw_text:     Optional[str]   = None
    image_bytes:  Optional[bytes] = None   # PNG bytes for charts/images
    image_b64:    Optional[str]   = None   # Base64 for LLM APIs
    table_data:   Optional[TableData] = None
    bbox:         Optional[BBox]      = None
    heading_level: Optional[int]      = None   # 1=H1, 2=H2, 3=H3
    confidence:   float               = 1.0    # 0.85 for OCR
    metadata:     Dict[str, Any]      = field(default_factory=dict)

    def citation(self) -> str:
        return f"[{self.doc_name}, p.{self.page_number} — {self.element_type.value}]"


# ══════════════════════════════════════════════════════════════════
# STAGE 2 OUTPUT — ProcessedDocument
# ══════════════════════════════════════════════════════════════════

@dataclass
class ProcessedDocument:
    """
    Output of Stage 2 (preprocessing.py).
    Same as RawDocument but with cleaned text, NER entities,
    and vision descriptions added.
    """
    element_id:          str
    source_file:         str
    doc_name:            str
    page_number:         int
    element_type:        ElementType
    content:             str          # Cleaned, normalized text
    raw_text:            Optional[str]    = None
    image_bytes:         Optional[bytes]  = None
    image_b64:           Optional[str]    = None
    table_data:          Optional[TableData] = None
    bbox:                Optional[BBox]      = None
    heading_level:       Optional[int]       = None
    vision_description:  Optional[str]       = None   # AI-generated chart/image desc
    named_entities:      List[str]           = field(default_factory=list)  # NER
    language:            str                 = "en"
    is_duplicate:        bool                = False
    confidence:          float               = 1.0
    metadata:            Dict[str, Any]      = field(default_factory=dict)

    def citation(self) -> str:
        return f"[{self.doc_name}, p.{self.page_number} — {self.element_type.value}]"


# ══════════════════════════════════════════════════════════════════
# STAGE 3 OUTPUT — Chunk
# ══════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """
    Output of Stage 3 (chunking.py).
    The fundamental unit that enters the embedding + index pipeline.

    One ProcessedDocument can produce multiple Chunks:
      - A long paragraph → several overlapping TEXT chunks
      - A table → 1 whole-table chunk + N row-level chunks
      - A chart → 1 image patch chunk
    """
    chunk_id:         str
    element_id:       str          # Parent element
    doc_name:         str
    page_number:      int
    element_type:     ElementType
    strategy:         ChunkStrategy
    text:             str          # What gets embedded & searched
    image_bytes:      Optional[bytes]    = None
    image_b64:        Optional[str]      = None
    table_data:       Optional[TableData] = None
    structured_vals:  Dict[str, Any]     = field(default_factory=dict)
    parent_heading:   Optional[str]      = None   # Nearest section heading
    metadata:         Dict[str, Any]     = field(default_factory=dict)

    def key(self) -> str:
        """SHA256 fingerprint — used for deduplication."""
        raw = f"{self.doc_name}|{self.page_number}|{self.element_type}|{self.text[:100]}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def citation(self) -> str:
        return f"[{self.doc_name}, p.{self.page_number} — {self.element_type.value}]"


# ══════════════════════════════════════════════════════════════════
# STAGE 6 OUTPUT — EnhancedQuery
# ══════════════════════════════════════════════════════════════════

@dataclass
class EnhancedQuery:
    """
    Output of Stage 6 (query_enhancement.py).
    Contains the original question plus multiple enhanced versions.
    """
    original:          str
    query_type:        str           # "financial" | "text" | "chart" | "diagram"
    hyde_doc:          str           # Hypothetical document for embedding
    expanded:          str           # Original + year/metric terms appended
    keyword_only:      str           # Only important words (no stopwords)
    table_variant:     str           # "... financial table data rows"
    chart_variant:     str           # "... chart graph trend visual"
    year_references:   List[str]     = field(default_factory=list)
    metric_references: List[str]     = field(default_factory=list)
    requires_table:    bool          = False
    requires_chart:    bool          = False
    requires_image:    bool          = False
    requires_diagram:  bool          = False
    requires_calc:     bool          = False


# ══════════════════════════════════════════════════════════════════
# STAGE 7 OUTPUT — ScoredChunk
# ══════════════════════════════════════════════════════════════════

@dataclass
class ScoredChunk:
    """A Chunk with its retrieval/rerank score attached."""
    chunk:  Chunk
    score:  float
    source: str    # Which retrieval path: "dense"|"bm25"|"table"|"chart"|"visual"|"structured"

    def key(self) -> str:
        return self.chunk.key()


# ══════════════════════════════════════════════════════════════════
# STAGE 8 OUTPUT — ContextWindow
# ══════════════════════════════════════════════════════════════════

@dataclass
class ContextWindow:
    """
    Output of Stage 8 (context_assembly.py).
    The formatted context passed to the LLM.
    """
    text:          str            # Full formatted context string
    citations:     List[str]      # All citation strings
    image_bytes:   List[bytes]    # Images to pass to vision LLM
    char_count:    int            = 0
    chunk_count:   int            = 0
    evidence_tags: List[str]      = field(default_factory=list)   # VERIFIED/UNVERIFIED


# ══════════════════════════════════════════════════════════════════
# STAGE 10 INPUT — EvidenceBlock
# ══════════════════════════════════════════════════════════════════

@dataclass
class EvidenceBlock:
    """A top-ranked chunk annotated with verification status."""
    chunk:        Chunk
    rerank_score: float
    verified:     bool       = False   # True = value seen in 2+ sources
    conflicts:    List[str]  = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# FINAL OUTPUT — FinalAnswer
# ══════════════════════════════════════════════════════════════════

@dataclass
class FinalAnswer:
    """
    The complete pipeline output — answer + all metadata.
    Returned by pipeline.py to the caller.
    """
    question:         str
    answer:           str
    citations:        List[str]
    confidence_score: float              = 0.0
    verified:         bool               = False
    hyde_doc:         str                = ""
    query_type:       str                = ""
    retrieval_stats:  Dict[str, int]     = field(default_factory=dict)
    calculations:     Dict[str, Any]     = field(default_factory=dict)
    evidence:         List[Dict[str, Any]] = field(default_factory=list)
    latency:          Dict[str, float]   = field(default_factory=dict)
    warnings:         List[str]          = field(default_factory=list)

    def pretty_print(self) -> None:
        """Print a formatted summary to stdout."""
        print("\n" + "=" * 70)
        print("  ANSWER")
        print("=" * 70)
        print(f"\n{self.answer}\n")
        print("─" * 70)
        print(f"  Confidence : {self.confidence_score:.2f}")
        print(f"  Verified   : {self.verified}")
        print(f"  Query type : {self.query_type}")
        if self.citations:
            print("\n  Sources:")
            for c in self.citations:
                print(f"    {c}")
        if self.calculations.get("growth_rate"):
            gr = self.calculations["growth_rate"]
            print(f"\n  Calculation: {gr['formula']}")
        print("\n  Latency (seconds):")
        for stage, t in self.latency.items():
            print(f"    {stage:<22}: {t:.2f}s")
        print("=" * 70)
