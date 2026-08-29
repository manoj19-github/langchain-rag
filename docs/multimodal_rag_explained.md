# `multimodal_rag.py` — Complete Line-by-Line Explanation

> This document explains **every section and every important line** of the
> multimodal RAG pipeline script. Written for someone who is new to RAG systems.

---

## Table of Contents

1. [Big Picture — What does the script do?](#1-big-picture)
2. [Line 1 — `from __future__ import annotations`](#2-line-1)
3. [Imports](#3-imports)
4. [Configuration Block](#4-configuration-block)
5. [Logging & Debug Helpers](#5-logging--debug-helpers)
6. [Data Models (Enums & Dataclasses)](#6-data-models)
7. [Cache Utilities](#7-cache-utilities)
8. [Vision Utilities](#8-vision-utilities)
9. [Lazy-Loaded Models](#9-lazy-loaded-models)
10. [PDF Text Extraction](#10-pdf-text-extraction)
11. [PDF Table Extraction](#11-pdf-table-extraction)
12. [PDF Image/Chart Extraction](#12-pdf-imagechart-extraction)
13. [Master PDF Extractor — `extract_pdf()`](#13-master-pdf-extractor)
14. [Chunking Engine](#14-chunking-engine)
15. [Index Building — `build_index()`](#15-index-building)
16. [Query Analyzer — `analyze_query()`](#16-query-analyzer)
17. [HyDE Generation — `generate_hyde()`](#17-hyde-generation)
18. [6-Path Retrieval](#18-6-path-retrieval)
19. [RRF Fusion — `rrf_fusion()`](#19-rrf-fusion)
20. [Deduplication — `deduplicate()`](#20-deduplication)
21. [CrossEncoder Reranking — `crossencoder_rerank()`](#21-crossencoder-reranking)
22. [Parent Context Expansion](#22-parent-context-expansion)
23. [Evidence Verification](#23-evidence-verification)
24. [Calculations — `perform_calculations()`](#24-calculations)
25. [Context Builder — `build_final_context()`](#25-context-builder)
26. [Answer Generation — `generate_answer()`](#26-answer-generation)
27. [Master Pipeline — `multimodal_rag()`](#27-master-pipeline)
28. [Main Entry Point](#28-main-entry-point)
29. [Full Architecture Flow Diagram](#29-full-architecture-flow-diagram)

---

## 1. Big Picture

**What does the script actually do?**

Think of it as a very smart research assistant that:

1. **Reads** your PDF — every word, every table, every chart
2. **Stores** all that knowledge in a searchable index
3. **Understands** your question (what type of answer do you need?)
4. **Searches** the index from 6 different angles simultaneously
5. **Ranks** the best evidence using AI
6. **Verifies** the numbers it found
7. **Answers** your question citing exact pages and tables

```
PDF File
   │
   ▼
┌─────────────────────────────────────────┐
│           PDF EXTRACTION                │
│  Text  │  Tables  │  Charts  │  Images  │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│              CHUNKING                   │
│  Split into 252 small searchable pieces │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│         INDEX BUILDING                  │
│  ChromaDB (vectors) + BM25 (keywords)   │
└─────────────────────────────────────────┘
   │
   ▼  User asks a question
┌─────────────────────────────────────────┐
│         6-PATH RETRIEVAL                │
│  Dense + BM25 + Table + Chart +         │
│  Visual + Structured Numeric            │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│    RRF FUSION → RERANK → VERIFY         │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│        LLM GENERATES ANSWER             │
│   with citations, page numbers, units   │
└─────────────────────────────────────────┘
```

---

## 2. Line 1

```python
from __future__ import annotations
```

**Must be the very first line of the file** (Python rule).

It enables "postponed evaluation of type hints". This means you can write:
```python
def foo(x: MyClass) -> MyClass:   # works even if MyClass is defined later
```
Without this line, Python would crash if you reference a type before it's defined.

---

## 3. Imports

```python
import base64      # Encode image bytes → text (for sending images to LLM APIs)
import hashlib     # Create SHA256 fingerprints of files and chunks
import io          # In-memory file objects (e.g., read PNG bytes without saving to disk)
import json        # Parse/write JSON (for vision API responses, cache)
import logging     # Python's built-in logging system
import os          # Read environment variables (API keys)
import pickle      # Save/load Python objects to disk (used for caching)
import re          # Regular expressions (find years like FY2025, numbers, etc.)
import sys         # Access stdout, argv (command-line arguments)
import time        # Measure how long each step takes
import traceback   # Print full error stack traces when something fails
import warnings    # Suppress noisy library warnings
from dataclasses import dataclass, field   # Clean way to define data containers
from enum import Enum                      # Define fixed sets of values (TEXT, TABLE, etc.)
from pathlib import Path                   # Modern file path handling
from typing import Any, Dict, List, Optional, Tuple, Union  # Type hints
```

```python
# Force UTF-8 stdout so ₹/–/' print cleanly on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
```

**Why this matters on Windows:**
Windows uses `cp1252` encoding by default. The Indian Rupee symbol `₹` is NOT in
cp1252 — printing it would crash the script with `UnicodeEncodeError`.
This line forces Python to use UTF-8 everywhere, so any character prints safely.
`errors='replace'` means: if a character still can't be printed, use `?` instead of crashing.

---

## 4. Configuration Block

```python
PDF_PATH: str = r"data/pdf/dummy_company_annual_report_2025_26 (1).pdf"
QUESTION: str = "What was the total revenue of NovaTech Solutions in FY2025-26?"
```
The default PDF and question. You can change these or pass them as command-line args.

```python
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL: str    = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
LLM_TEMPERATURE: float = 0.0
```
- `os.getenv("LLM_PROVIDER", "ollama")` → reads from environment variable.
  If not set, uses `"ollama"` as default.
- `LLM_TEMPERATURE = 0.0` → deterministic answers (no randomness). Good for facts.

```python
VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "ollama")
VISION_MODEL: str    = os.getenv("VISION_MODEL", "llava:13b")
```
Separate model for **describing images/charts** (vision = can see images).
`llava:13b` is a vision-capable local model via Ollama.

```python
HYDE_MODEL: str     = os.getenv("HYDE_MODEL", LLM_MODEL)
HYDE_TEMPERATURE: float = 0.4
HYDE_ENABLED: bool = True
```
HyDE = **Hypothetical Document Embedding**.
Uses a slightly **higher temperature (0.4)** because it needs to be creative when
generating a hypothetical answer passage for retrieval. More on this in Section 17.

```python
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
```
The model that converts text into vectors (numbers). `BAAI/bge-small-en-v1.5` is a
fast, high-quality embedding model from Beijing Academy of AI. It runs locally for free.

```python
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
```
The CrossEncoder model. Unlike embeddings (which score one chunk at a time),
CrossEncoder sees **both the question AND the chunk** together → much more accurate scoring.

### Retrieval Knobs

```python
TOP_K_DENSE: int = 15       # How many chunks to get from dense (vector) search
TOP_K_BM25: int  = 15       # How many chunks from BM25 keyword search
TOP_K_TABLE: int = 10       # How many table chunks to retrieve directly
TOP_K_CHART: int = 8        # How many chart/diagram chunks
TOP_K_VISUAL: int = 8       # How many image chunks
TOP_K_STRUCTURED: int = 5   # How many exact numeric lookups
MMR_FETCH_K: int = 40       # Candidate pool before MMR diversity filtering
MMR_LAMBDA: float = 0.6     # 0=max diversity, 1=max relevance
RERANK_TOP_K: int = 7       # Final number of chunks sent to LLM
```

### RRF Fusion Weights

```python
W_DENSE:      float = 0.35   # Dense vector search contributes 35%
W_BM25:       float = 0.25   # Keyword search contributes 25%
W_TABLE:      float = 0.20   # Table-specific search: 20%
W_CHART:      float = 0.10   # Chart search: 10%
W_VISUAL:     float = 0.07   # Image search: 7%
W_STRUCTURED: float = 0.03   # Exact numeric lookup: 3%
RRF_K: int = 60              # RRF smoothing constant (prevents top-rank domination)
```

These weights determine how much each retrieval path contributes to the final ranking.
Financial questions → table gets more weight. Diagram questions → visual gets more.

### Chunking Settings

```python
TEXT_CHUNK_SIZE: int    = 600   # Max characters per text chunk
TEXT_CHUNK_OVERLAP: int = 80    # Overlap between consecutive chunks (preserves context)
MIN_CHUNK_CHARS: int    = 30    # Ignore chunks shorter than 30 chars (garbage)
MAX_TABLE_ROWS_PER_CHUNK: int = 60  # Max rows to create row-level chunks for
```

### Image Settings

```python
EXTRACT_IMAGES: bool = True      # Toggle image extraction on/off
MIN_IMAGE_WIDTH: int = 80        # Ignore tiny images (icons, bullets)
MIN_IMAGE_HEIGHT: int = 80       # Same for height
MAX_IMAGES_PER_PAGE: int = 10    # Safety limit per page
```

### OCR Settings

```python
OCR_ENABLED: bool = True   # Enable Tesseract OCR for scanned pages
OCR_LANG: str = "eng"      # OCR language
OCR_DPI: int = 300         # Resolution for rendering scanned pages (higher = better OCR)
```

### Cache Settings

```python
CACHE_ENABLED: bool = True    # Cache expensive results to disk
CACHE_DIR: str = ".rag_cache" # Folder where cache files are stored
```

**Why cache?** The first run does:
- PDF extraction (0.33s)
- Embedding 252 chunks into ChromaDB (55s!)
- HyDE generation via LLM (22s)

Subsequent runs load these from cache → total drops from 105s → 45s.

---

## 5. Logging & Debug Helpers

```python
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("multimodal_rag")
if DEBUG:
    logger.setLevel(logging.DEBUG)
```
Standard Python logging. When `DEBUG=True`, all `logger.debug()` calls are visible.

```python
def _safe(text: str) -> str:
    """Replace any character that can't be encoded by the current terminal."""
    try:
        text.encode(sys.stdout.encoding or 'utf-8')
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode('ascii', errors='replace').decode('ascii')
```
Safety wrapper: tries to encode the text for the current terminal.
If it fails (e.g., `₹` on cp1252), replaces bad characters with `?`.

```python
def _dbg(section: str, msg: str = "") -> None:
    if DEBUG:
        print(f"\n{'='*60}")
        print(f"  {section}")
        print(f"{'='*60}")
        if msg:
            print(_safe(msg))
```
Prints a big separator box for major pipeline stages.
Example output:
```
============================================================
  QUERY ANALYSIS
============================================================
```

```python
def _info(msg: str) -> None:
    if DEBUG:
        print(f"  > {_safe(msg)}")
```
Prints a single indented info line:
```
  > Dense: 21 chunks
```

---

## 6. Data Models

Data models are **containers** for structured data. Python `@dataclass` automatically
generates `__init__`, `__repr__`, etc.

### `ElementType` (Enum)

```python
class ElementType(str, Enum):
    TEXT      = "TEXT"       # Regular body text
    HEADING   = "HEADING"    # Section headings (large/bold font)
    PARAGRAPH = "PARAGRAPH"  # Long text blocks
    TABLE     = "TABLE"      # Structured table from pdfplumber
    CHART     = "CHART"      # Bar/line/pie chart (detected by size/aspect ratio)
    IMAGE     = "IMAGE"      # Embedded photo/screenshot
    DIAGRAM   = "DIAGRAM"    # Flowchart, architecture diagram
    CAPTION   = "CAPTION"    # Caption below a figure or table
    OCR_TEXT  = "OCR_TEXT"   # Text extracted via Tesseract from scanned page
    PAGE      = "PAGE"       # Whole-page element (rarely used)
```

### `BBox` — Bounding Box

```python
@dataclass
class BBox:
    x0: float  # Left edge (pixels from left of page)
    y0: float  # Top edge
    x1: float  # Right edge
    y1: float  # Bottom edge
```
Records where on the page an element appears. Used for layout analysis.

### `TableData`

```python
@dataclass
class TableData:
    title: Optional[str]         # Table title (if detected)
    headers: List[str]           # Column headers: ["Metric", "FY2026", "FY2025"]
    rows: List[List[str]]        # Each row: [["Revenue", "482.6", "406.5"], ...]
    units: Optional[str]         # "crore", "million", "%"
    currency: Optional[str]      # "INR", "$", "EUR"
    footnotes: List[str]         # Footnotes below the table
```

**Key methods:**

```python
def to_markdown(self) -> str:
```
Converts the table to Markdown format for the LLM:
```
| Metric   | FY2025-26 | FY2024-25 | Change |
|---|---|---|---|
| Revenue  | 482.6     | 406.5     | +18.7% |
```

```python
def to_search_text(self) -> str:
```
Flat text for embedding/BM25:
```
"Metric FY2025-26 FY2024-25 Change | Revenue 482.6 406.5 +18.7%"
```

```python
def extract_numbers(self) -> List[Tuple[str, str, str]]:
```
Returns `(column_header, row_label, value)` for every numeric cell.
Example: `("FY2025-26", "Revenue", "482.6")`
Used by the structured numeric retrieval path.

### `DocumentElement`

This is the **raw extracted element** from the PDF, before it becomes a chunk.

```python
@dataclass
class DocumentElement:
    element_id: str              # Unique ID like "page_4_table_0"
    document_name: str           # PDF filename
    page_number: int             # Page 1, 2, 3...
    element_type: ElementType    # TEXT, TABLE, CHART, etc.
    content: str                 # Searchable text representation
    raw_text: Optional[str]      # Original extracted text
    image_bytes: Optional[bytes] # PNG image data (for charts/images)
    image_b64: Optional[str]     # Base64 encoded image (for LLM APIs)
    table_data: Optional[TableData]  # Structured table if type=TABLE
    bbox: Optional[BBox]         # Where on the page
    heading_level: Optional[int] # 1=H1, 2=H2, 3=H3
    caption: Optional[str]       # Caption text
    vision_description: Optional[str]  # AI-generated description of chart/image
    confidence: float            # 1.0 for native text, 0.85 for OCR
    metadata: Dict[str, Any]     # Font size, bold flag, etc.
```

### `MultimodalChunk`

**The unit that enters the retrieval index.** One `DocumentElement` can produce multiple chunks (e.g., a large table becomes one whole-table chunk + one chunk per row).

```python
@dataclass
class MultimodalChunk:
    chunk_id: str            # Unique: "page_4_table_0_row_2"
    element_id: str          # Parent element ID
    document_name: str
    page_number: int
    element_type: ElementType
    text: str                # What gets embedded and searched
    image_bytes: Optional[bytes]   # Only for CHART/IMAGE/DIAGRAM chunks
    image_b64: Optional[str]
    table_data: Optional[TableData]
    structured_values: Dict  # {"headers": [...], "numeric_cells": [...]}
    metadata: Dict
    parent_context: Optional[str]  # Nearest heading above this chunk

    def key(self) -> str:
        # SHA256 hash of document+page+type+text — used for deduplication
        raw = f"{self.document_name}|{self.page_number}|{self.element_type}|{self.text[:100]}"
        return hashlib.sha256(raw.encode()).hexdigest()
```

### `ScoredChunk`

A chunk paired with its retrieval score and which path found it:

```python
@dataclass
class ScoredChunk:
    chunk: MultimodalChunk
    score: float    # RRF score or CrossEncoder score
    source: str     # "dense" | "bm25" | "table" | "chart" | "visual" | "structured"
```

### `QueryAnalysis`

The output of `analyze_query()` — tells the retrieval system what to search for:

```python
@dataclass
class QueryAnalysis:
    original_query: str
    query_type: str           # "financial" | "text" | "chart" | "diagram" | "table"
    requires_text: bool       # Always True
    requires_table: bool      # True if query mentions revenue, profit, fy, etc.
    requires_chart: bool      # True if query mentions chart, graph, trend, etc.
    requires_image: bool      # True if query mentions image, photo, etc.
    requires_diagram: bool    # True if query mentions diagram, flow, architecture
    requires_calculation: bool # True if query asks for growth rate, sum, etc.
    year_references: List[str]  # ["FY2025", "FY2024"] extracted from query
    metric_references: List[str] # ["revenue", "profit"] extracted
    expanded_query: str       # Original + years + metrics appended
    keyword_query: str        # Only important keywords (no stopwords)
    table_query: str          # Query variant for table search
    chart_query: str          # Query variant for chart search
    visual_query: str         # Query variant for image search
```

### `EvidenceBlock`

After reranking, each top chunk becomes an evidence block with verification status:

```python
@dataclass
class EvidenceBlock:
    chunk: MultimodalChunk
    rerank_score: float      # CrossEncoder relevance score
    verified: bool           # True = value seen in 2+ sources
    conflicts_with: List[str] # If two sources disagree
```

### `RAGResult`

The final return value of `multimodal_rag()`:

```python
@dataclass
class RAGResult:
    query: str
    hyde_doc: str                    # The hypothetical document generated
    query_analysis: Dict[str, Any]   # What the query analyzer found
    retrieval_stats: Dict[str, int]  # How many chunks each path retrieved
    evidence: List[Dict[str, Any]]   # The top 7 chunks used
    calculations: Dict[str, Any]     # Any Python-computed numbers
    answer: str                      # The final LLM answer text
    citations: List[str]             # ["[report.pdf, p.4 - TABLE]", ...]
    latency: Dict[str, float]        # Time taken per stage
```

---

## 7. Cache Utilities

```python
def _cache_path(key: str) -> Path:
    p = Path(CACHE_DIR)   # ".rag_cache"
    p.mkdir(exist_ok=True)  # Create folder if it doesn't exist
    return p / f"{key}.pkl"  # e.g., ".rag_cache/extracted_abc123.pkl"
```

```python
def _cache_load(key: str) -> Optional[Any]:
    if not CACHE_ENABLED:
        return None
    cp = _cache_path(key)
    if cp.exists():
        try:
            with open(cp, "rb") as f:
                return pickle.load(f)  # Load any Python object from disk
        except Exception:
            return None   # If file is corrupt, just return None (fresh compute)
    return None
```

```python
def _cache_save(key: str, obj: Any) -> None:
    if not CACHE_ENABLED:
        return
    try:
        with open(_cache_path(key), "wb") as f:
            pickle.dump(obj, f)  # Save any Python object to disk
    except Exception:
        pass  # Cache failure is non-fatal
```

```python
def _file_hash(path: str) -> str:
    """Read the PDF in 64KB chunks and compute a SHA256 fingerprint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]  # Use first 16 chars as unique ID
```

**Why hash the file?** If you change the PDF, the hash changes → cache is automatically invalidated. No stale data.

---

## 8. Vision Utilities

```python
def _image_to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")
```
LLM vision APIs don't accept raw bytes — they need images encoded as Base64 text strings.
This converts `b'\x89PNG...'` → `"iVBORw0KGgo..."`.

```python
def _describe_image_with_vision(image_bytes: bytes, prompt: str) -> str:
```
Sends an image + prompt to a vision-capable LLM. Supports 3 providers:

**OpenAI (GPT-4o):**
```python
content = [
    {"type": "text", "text": prompt},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
]
resp = client.chat.completions.create(model=VISION_MODEL, messages=[...])
```

**Anthropic (Claude):**
```python
content = [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
    {"type": "text", "text": prompt}
]
resp = client.messages.create(...)
```

**Ollama (local llava):**
```python
# Direct HTTP call to Ollama API — no SDK needed
payload = json.dumps({"model": VISION_MODEL, "prompt": prompt, "images": [b64]})
req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/generate", ...)
```

If all fail → returns `"[Vision description unavailable]"` gracefully.

### Vision Prompts

Three different prompts for three types of visuals:

```python
VISION_PROMPT_CHART = """Analyze this chart/graph. Return JSON with:
  title, chart_type, x_axis, y_axis, unit, legend, data_points,
  key_observations, visible_text, uncertainty"""
```
Returns structured JSON → parsed and formatted as searchable text.

```python
VISION_PROMPT_IMAGE = """Describe this image from a PDF accurately.
Include: what is shown, visible text/numbers/labels, spatial layout."""
```

```python
VISION_PROMPT_DIAGRAM = """Analyze this diagram. Identify:
diagram type, nodes/labels, arrows/directions, relationships.
Produce: Node A -> Node B -> Node C"""
```

---

## 9. Lazy-Loaded Models

```python
_embeddings_instance = None
_cross_encoder_instance = None
_hyde_llm_instance = None
_rag_llm_instance = None
```
Global variables, initially `None`. Models are loaded **on first use only**
(not at import time) — this avoids loading 1GB+ models if they aren't needed.

```python
def _get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,       # "BAAI/bge-small-en-v1.5"
            model_kwargs={"device": EMBEDDING_DEVICE},  # "cpu" or "cuda"
        )
    return _embeddings_instance
```

```python
def _build_llm(model: str, temperature: float):
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(...)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature, base_url=OLLAMA_BASE_URL)
```
One function builds any LLM. The `if/elif/else` switches between providers.
This is the **provider abstraction** — changing `LLM_PROVIDER=openai` switches the whole system.

---

## 10. PDF Text Extraction

```python
def _detect_if_scanned(page) -> bool:
    blocks = page.get_text("blocks")
    # Count native text characters (type 0 = text block)
    total_chars = sum(len(b[4]) for b in blocks if b[6] == 0)
    return total_chars < 20  # Fewer than 20 chars = likely a scanned image
```
If a page has almost no native text, it was probably scanned. OCR is triggered.

```python
def _ocr_page_bytes(page_image_bytes: bytes) -> str:
    import pytesseract
    from PIL import Image
    img = Image.open(io.BytesIO(page_image_bytes))  # Load PNG from memory
    return pytesseract.image_to_string(img, lang=OCR_LANG).strip()
```
Runs Tesseract OCR on a page rendered as PNG. Returns recognized text.

```python
def _page_to_png(page, dpi: int = OCR_DPI) -> bytes:
    mat = page.get_pixmap(dpi=dpi)  # Render page at 300 DPI
    return mat.tobytes("png")        # Return as PNG bytes
```

```python
def _extract_text_elements(page, page_num, doc_name):
    blocks = page.get_text("dict", flags=11)["blocks"]
    # "dict" mode gives us font size, bold flag, coordinates per span
```

For each text span:
```python
font_size = span.get("size", 12)
is_bold = bool(span.get("flags", 0) & 2**4)  # Bit 4 = bold flag in PDF spec
```

**Heading detection heuristic:**
```python
if font_size >= 14 or (font_size >= 12 and is_bold and len(txt) < 120):
    etype = ElementType.HEADING
elif len(txt) > 80:
    etype = ElementType.PARAGRAPH
else:
    etype = ElementType.TEXT
```
- Large font OR (medium + bold + short) → **HEADING**
- Long text → **PARAGRAPH**
- Short text → **TEXT**

---

## 11. PDF Table Extraction

```python
def _extract_tables_pdfplumber(pdf_path, doc_name):
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            tables = page.extract_tables()
```

`pdfplumber.extract_tables()` detects table borders and returns:
```python
[
    ["Metric", "FY2025-26", "FY2024-25", "Change"],   # headers (row 0)
    ["Revenue", "482.6", "406.5", "+18.7%"],            # row 1
    ["Net Profit", "61.8", "49.2", "+25.6%"],           # row 2
]
```

```python
headers = [str(c).strip() if c else "" for c in table[0]]
rows = [
    [str(c).strip() if c else "" for c in row]
    for row in table[1:]
    if any(c for c in row)   # Skip empty rows
]
```

**Currency and unit detection:**
```python
for sym in ["Rs.", "INR", "USD", "$", "EUR", "GBP"]:
    if sym in all_text:
        currency = sym
        break
for u in ["crore", "Cr", "million", "billion", "lakh", "%"]:
    if u.lower() in all_text.lower():
        units = u
        break
```
Scans all text in the table to auto-detect currency (INR, USD) and units (crore, million).

---

## 12. PDF Image/Chart Extraction

```python
def _extract_images_pymupdf(page, page_num, doc_name):
    img_list = page.get_images(full=True)
    for ii, img in enumerate(img_list[:MAX_IMAGES_PER_PAGE]):
        xref = img[0]                              # Image reference ID in PDF
        base_img = page.parent.extract_image(xref) # Extract raw image data
        img_bytes = base_img.get("image", b"")
        width = base_img.get("width", 0)
        height = base_img.get("height", 0)
```

**Filter tiny images** (bullets, icons, watermarks):
```python
if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
    continue   # Skip — too small to be meaningful
```

**Convert to PNG** for consistent handling:
```python
pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
buf = io.BytesIO()
pil_img.save(buf, format="PNG")
png_bytes = buf.getvalue()
```

**Chart vs Image heuristic:**
```python
aspect = width / max(height, 1)       # Width/height ratio
is_chart = (0.5 < aspect < 4.0) and (width * height > 20000)
# Landscape-ish (not square, not portrait), and large enough
etype = ElementType.CHART if is_chart else ElementType.IMAGE
```
Charts are typically wider than tall (bar charts, line charts). This catches most common chart layouts.

### Vision Enrichment

```python
def _enrich_visual_elements_with_vision(elements, cache_prefix):
    for el in elements:
        if el.element_type not in (CHART, IMAGE, DIAGRAM):
            continue
        # Check cache first
        ckey = f"vision_{cache_prefix}_{el.element_id}"
        cached = _cache_load(ckey)
        if cached is not None:
            el.vision_description = cached
            el.content = cached
            continue
        # Call vision model
        desc = _describe_image_with_vision(el.image_bytes, prompt)
        el.vision_description = desc
        el.content = desc    # Replace "[CHART on page 3]" with actual description
        _cache_save(ckey, desc)
```

For charts, the JSON response is parsed:
```python
data = json.loads(raw)
parts = [
    f"Chart: {data.get('title', 'Unknown')}",
    f"Type: {data.get('chart_type', '')}",
    f"Observations: {data.get('key_observations', '')}",
    f"Data: FY22=300, FY23=350, FY24=407, FY25=483",
]
```

---

## 13. Master PDF Extractor

```python
def extract_pdf(pdf_path: str) -> List[DocumentElement]:
```

**Flow:**
1. Compute `fhash = _file_hash(pdf_path)` — unique ID for this exact PDF
2. Check cache: `_cache_load(f"extracted_{fhash}")` — if found, skip everything
3. **PyMuPDF** → extract text and images page by page
   - Scanned page? → `_ocr_page_bytes()`
   - Normal page? → `_extract_text_elements()`
   - Always → `_extract_images_pymupdf()`
4. **pdfplumber** → extract all tables
5. **Vision enrichment** → describe all charts/images
6. Sort by `(page_number, element_type)` → reading order
7. Save to cache

**Result for our 6-page NovaTech PDF:**
```
Elements: {'HEADING': 16, 'TABLE': 8, 'TEXT': 179, 'PARAGRAPH': 13}
Total: 216 elements
```

---

## 14. Chunking Engine

**Why chunk?** Embedding models have a token limit (~512 tokens). A 50-page PDF
can't be embedded as one unit. We split it into overlapping chunks.

```python
def _chunk_text(text, chunk_size=600, overlap=80) -> List[str]:
    if len(text) <= chunk_size:
        return [text]   # Short enough — keep as is
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to break at a natural boundary (not mid-word/sentence)
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = text.rfind(sep, start + overlap, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        start = end - overlap  # 80 char overlap preserves context
    return chunks
```

**Visual example with overlap:**
```
"The revenue was 482.6 crore. The EBITDA was 96.4 crore. The net profit..."
 ├────────── Chunk 1 ──────────┤
                    ├────────── Chunk 2 ──────────┤
                                        ├──── Chunk 3...
```

### `build_multimodal_chunks()`

```python
def build_multimodal_chunks(elements):
    heading_context: Dict[int, str] = {}  # Tracks last heading per page
    for el in elements:
        if el.element_type == ElementType.HEADING:
            heading_context[page] = el.content  # Update current section name
        parent_ctx = heading_context.get(page, "")  # "3. Financial Highlights"
```

**For TEXT/HEADING/PARAGRAPH:**
```python
for ci, part in enumerate(_chunk_text(el.content)):
    chunks.append(MultimodalChunk(
        text=part,
        parent_context=parent_ctx,  # "What section is this under?"
        ...
    ))
```

**For TABLE — creates TWO types of chunks:**

*Whole-table chunk* (for broad matching):
```python
chunks.append(MultimodalChunk(
    text=td.to_search_text(),    # All headers + all rows flattened
    table_data=td,               # Keep full structure for markdown rendering
    metadata={"markdown": td.to_markdown()}  # For the LLM context
))
```

*Row-level chunks* (for precise row matching):
```python
for ri, row in enumerate(td.rows):
    row_text = " | ".join(f"{h}: {v}" for h, v in zip(td.headers, row))
    # e.g., "n Crore: Revenue | FY2025-26: 482.6 | FY2024-25: 406.5 | Change: +18.7%"
    chunks.append(MultimodalChunk(text=row_text, ...))
```

**For CHART/IMAGE/DIAGRAM:**
```python
desc = el.vision_description or el.content
chunks.append(MultimodalChunk(
    text=desc,            # The AI-generated description is what gets searched
    image_bytes=el.image_bytes,   # Original image bytes kept for LLM context
    image_b64=el.image_b64,
))
```

**Result:** 216 elements → 252 chunks (tables expand into multiple row chunks)

---

## 15. Index Building

```python
def build_index(chunks, pdf_hash):
```

**Step 1: Convert to LangChain Documents**
```python
for c in chunks:
    lc_docs.append(LCDoc(
        page_content=c.text,     # What gets embedded
        metadata={
            "chunk_id": c.chunk_id,
            "page_number": c.page_number,
            "element_type": c.element_type.value,
            ...
        }
    ))
```
LangChain's `Document` is the standard format for LangChain retrievers.

**Step 2: BM25 Index**
```python
bm25_retriever = BM25Retriever.from_documents(lc_docs)
bm25_retriever.k = TOP_K_BM25   # Return top 15 results
```
BM25 = **Best Match 25** — a classical keyword-frequency algorithm.
Better than TF-IDF for short text. Fast and interpretable.
Good for: exact words like "Revenue", "FY2025-26", "EBITDA".

**Step 3: Chroma Vector Store**
```python
vstore = Chroma.from_documents(
    documents=lc_docs,
    embedding=_get_embeddings(),           # BAAI/bge-small-en-v1.5
    collection_name=f"mmrag_{pdf_hash}",   # Unique per PDF
    persist_directory=chroma_dir,          # Saved to disk
)
```
Chroma embeds every chunk into a 384-dimensional vector and stores them.
Semantic similarity search: "profit" will also find "earnings", "income", "net margin".

**Why both?**
| BM25 | Dense (Chroma) |
|------|---------------|
| "revenue FY2026" → finds exact word matches | "how much did they earn?" → semantic match |
| Fast, exact | Slower, fuzzy |
| No context | Understands meaning |
| Complements dense | Complements BM25 |

---

## 16. Query Analyzer

```python
_TABLE_KW = {"revenue", "profit", "margin", "growth", "fy", "financial", "crore", ...}
_CHART_KW = {"chart", "graph", "trend", "bar", "line", "pie", "figure", "plot", ...}
_YEAR_RE = re.compile(r"\bfy[\s\']?\d{2,4}\b|\b20\d\d\b|\b19\d\d\b", re.IGNORECASE)
_METRIC_RE = re.compile(r"\b(revenue|profit|margin|ebitda|eps|cagr|growth|...)\b", re.IGNORECASE)
```

```python
def analyze_query(query: str) -> QueryAnalysis:
    words = set(re.findall(r"\b\w+\b", ql))  # All words in query
    years = _YEAR_RE.findall(query)           # ["FY2025", "FY2024"]
    metrics = _METRIC_RE.findall(query)       # ["revenue"]

    req_table = bool(words & _TABLE_KW) or bool(years)
    # True if any table keyword found, OR if years mentioned
```

**Example:**
| Query | `requires_table` | `requires_chart` | `query_type` |
|-------|----------------|-----------------|--------------|
| "What was revenue in FY2025?" | ✅ True | ❌ False | "financial" |
| "Show me the revenue chart" | ❌ False | ✅ True | "chart" |
| "Describe the org chart" | ❌ False | ❌ False | "diagram" |
| "Who is the CEO?" | ❌ False | ❌ False | "text" |

**Query variants generated:**
```python
expanded_query = "What was revenue in FY2025-26? FY2025 revenue"
# Adds years and metrics → better dense recall

keyword_query = "26 fy2025 novatech revenue solutions total"
# Only meaningful words → better BM25 recall

table_query = "What was revenue in FY2025-26? financial table data rows columns"
chart_query = "What was revenue in FY2025-26? chart graph trend visual figure"
```

---

## 17. HyDE Generation

**HyDE = Hypothetical Document Embedding**

The insight: instead of embedding the *question* for retrieval, generate a *hypothetical answer* and embed that. The hypothetical answer uses the same vocabulary as the actual document → better retrieval.

```python
prompt = (
    "Generate a concise hypothetical passage that WOULD contain the "
    "information to answer the question.\n"
    "This passage is ONLY for embedding similarity search - NOT the final answer.\n"
    "Write as if excerpted from the actual source document.\n\n"
    f"Question: {query}\n\nHypothetical document passage:"
)
resp = llm.invoke([HumanMessage(content=prompt)])
```

**Example:**
```
Query: "What was revenue in FY2025-26?"

HyDE output: "NovaTech Solutions reported revenue of ₹482.6 crore for FY2025-26,
representing 18.7% year-over-year growth from ₹406.5 crore in FY2024-25..."
```

This passage is then embedded and used for dense retrieval alongside the original query.

**Important:** The HyDE passage is marked `[RETRIEVAL AID ONLY]` in the final prompt — the LLM must NOT cite it as evidence.

**Caching:**
```python
ckey = f"hyde_{cache_prefix}_{hashlib.sha256(query.encode()).hexdigest()[:12]}"
# Same question on same PDF → returns cached HyDE immediately (0.02s vs 22s)
```

---

## 18. 6-Path Retrieval

### Path 1: Dense MMR Retrieval

```python
def retrieve_dense(queries, vstore, chunks_by_id, top_k):
    retriever = vstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA},
    )
    for q in [original_query, expanded_query, hyde_doc]:
        docs = retriever.invoke(q)  # Semantic vector search
```

**MMR = Maximal Marginal Relevance**
- `fetch_k=40`: First retrieve 40 candidates by cosine similarity
- `k=15`: Then pick 15 that maximize: relevance AND diversity
- `lambda_mult=0.6`: 60% relevance, 40% diversity

Why diversity? Without it, you'd get 15 nearly-identical chunks from the same table row.

### Path 2: BM25 Sparse Retrieval

```python
def retrieve_bm25(query, bm25_retriever, chunks_by_id):
    docs = bm25_retriever.invoke(query)  # Keyword frequency search
```

Uses the keyword-only query (`"fy2025 revenue novatech"`) → finds exact term matches.

### Path 3: Table Retrieval

```python
def retrieve_tables(qa, chunks):
    priority_tokens = query_tokens | year_tokens | metric_tokens
    for c in chunks:
        if c.element_type != ElementType.TABLE:
            continue
        token_matches = sum(1 for t in priority_tokens if t in c.text.lower())
        score = token_matches / len(priority_tokens)
```

Directly scans all table chunks for token overlap. No embedding needed — fast and precise for financial queries.

### Path 4: Chart Retrieval

```python
def retrieve_charts(qa, chunks):
    for c in chunks:
        if c.element_type not in (CHART, DIAGRAM):
            continue
        hits = sum(1 for t in query_tokens if len(t) > 3 and t in c.text.lower())
```

Scans chart vision descriptions for relevant terms. Only activated when `qa.requires_chart=True`.

### Path 5: Visual (Image) Retrieval

```python
def retrieve_visual(qa, chunks):
    for c in chunks:
        if c.element_type != ElementType.IMAGE:
            continue
```

Same approach but for standalone images. Only when `qa.requires_image=True`.

### Path 6: Structured Numeric Retrieval

```python
def retrieve_structured_numeric(qa, chunks):
    year_pats = [re.compile(r"fy[\s\']?2025", re.IGNORECASE)]
    metric_pats = [re.compile("revenue", re.IGNORECASE)]

    for c in chunks:
        if c.element_type != TABLE:
            continue
        for col, row_label, val in c.table_data.extract_numbers():
            year_match   = any(p.search(col) or p.search(row_label) for p in year_pats)
            metric_match = any(p.search(col) or p.search(row_label) for p in metric_pats)
            if year_match and metric_match:
                results.append(ScoredChunk(chunk=c, score=2.0, source="structured"))
```

**Most precise path**: checks actual table cells for year+metric combinations.
Score = 2.0 (higher than other paths) → structured matches always rank high.

---

## 19. RRF Fusion

After 6 paths return their ranked lists, we must **combine them** into one ranking.

```python
def rrf_fusion(ranked_lists: List[Tuple[List[ScoredChunk], float]], k=60):
    final_scores: Dict[str, float] = {}

    for ranked, weight in ranked_lists:
        for rank, sc in enumerate(ranked):
            key = sc.key()
            # RRF formula: weight / (k + rank + 1)
            rrf_score = weight / (k + rank + 1)
            final_scores[key] += rrf_score
```

**RRF formula explained:**
```
score = weight / (k + rank)

If k=60:
  rank=0 (1st):  1/61  = 0.0164
  rank=1 (2nd):  1/62  = 0.0161
  rank=9 (10th): 1/70  = 0.0143
  rank=99:       1/160 = 0.006
```

The `k=60` constant prevents the 1st-ranked item from dominating too much.
A chunk that ranks 2nd in BM25 and 3rd in Dense gets a combined score — **better than a chunk that only appeared in one list**.

**Example:**
```
TABLE chunk (page 4):
  Dense rank 2:    0.35 / 62  = 0.00565
  BM25 rank 1:     0.25 / 61  = 0.00410
  Table rank 1:    0.20 / 61  = 0.00328
  Structured rank 1: 0.03/61  = 0.00049
  TOTAL:                       = 0.01352  ← High combined score
```

---

## 20. Deduplication

```python
def deduplicate(scored_chunks):
    seen: Dict[str, ScoredChunk] = {}
    for sc in scored_chunks:
        key = sc.key()   # SHA256 of content
        if key not in seen or sc.score > seen[key].score:
            seen[key] = sc   # Keep highest-scored version
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)
```

The same chunk can appear in multiple retrieval paths. Without deduplication, the same
"Revenue: 482.6 | FY2025-26: ..." row would appear 4 times in the evidence. This removes duplicates, keeping the highest RRF score.

---

## 21. CrossEncoder Reranking

**Why rerank?** Embedding similarity is fast but approximate.
CrossEncoder reads (query, chunk) together → much more accurate but slower.

```python
def crossencoder_rerank(query, candidates, top_k=7):
    ce = _get_cross_encoder()  # cross-encoder/ms-marco-MiniLM-L6-v2
    pairs = [
        (query, f"[{sc.chunk.element_type.value}] " + sc.chunk.text[:512])
        for sc in candidates
    ]
    # Note: element type prefix helps the CrossEncoder understand context
    # "[TABLE] Revenue: 482.6 | FY2025-26: ..."

    scores = ce.predict(pairs, batch_size=16)
    # Returns a score per pair — higher = more relevant
```

**Why prefix with element type?**
`"[TABLE] Revenue: 482.6..."` tells the CrossEncoder this is structured financial data,
not a sentence. This improves reranking accuracy.

**Result in our run:**
```
[1] score=2.769  type=TABLE  p=4  src=table       ← Financial highlights table
[2] score=2.635  type=TABLE  p=4  src=dense        ← Same table, different chunk
[3] score=1.398  type=TABLE  p=4  src=structured   ← Row-level chunk
[4] score=0.930  type=PARAGRAPH  p=2  src=bm25    ← Executive summary text
```

---

## 22. Parent Context Expansion

```python
def expand_parent_context(top_chunks, all_chunks):
    page_chunks = {}  # page_number → [all chunks on that page]
    for c in all_chunks:
        page_chunks.setdefault(c.page_number, []).append(c)

    for sc in top_chunks:
        expansion_parts = []
        if c.parent_context:
            expansion_parts.append(f"[Section: {c.parent_context}]")
            # e.g., "[Section: 3. Financial Highlights]"

        if c.element_type == ElementType.CHART:
            # Find related table on same page
            for n in neighbours:
                if n.element_type == ElementType.TABLE:
                    expansion_parts.append(f"[Related table]: {n.text[:300]}")

        if c.element_type == ElementType.TABLE:
            # Find nearby explanatory paragraph
            for n in neighbours:
                if n.element_type == ElementType.PARAGRAPH:
                    expansion_parts.append(f"[Context]: {n.text[:200]}")
```

**Purpose:** A table chunk alone might say `"482.6 | 406.5"` — without context the LLM doesn't know what section this is in. The expanded context adds `[Section: 3. Financial Highlights]` → the LLM can better ground its answer.

---

## 23. Evidence Verification

```python
def verify_evidence(top_chunks, qa):
    # First pass: collect all numbers and which chunks mention them
    all_values: Dict[str, List[str]] = {}
    for sc in top_chunks:
        for n in number_re.findall(sc.chunk.text):
            clean = n.replace(",", "")
            all_values.setdefault(clean, []).append(sc.chunk.citation())
    # e.g., all_values["482.6"] = ["[report.pdf, p.4 - TABLE]", "[report.pdf, p.2 - PARAGRAPH]"]

    # Second pass: mark each chunk as verified/unverified
    for sc in top_chunks:
        verified = True
        for n in number_re.findall(sc.chunk.text):
            clean = n.replace(",", "")
            if float(clean) > 100 and len(all_values.get(clean, [])) < 2:
                verified = False  # Large number seen in only ONE source
```

**Logic:**
- `VERIFIED`: The value `482.6` appears in BOTH the table AND the executive summary paragraph
- `UNVERIFIED`: A value that only appears in one place — flag it so the LLM knows

The LLM sees `[VERIFIED]` or `[UNVERIFIED]` tags next to each evidence block.

---

## 24. Calculations

```python
def perform_calculations(qa, evidence):
    year_re = re.compile(r"\b(fy[\s\']?\d{2,4}|20\d\d|19\d\d)\b", re.IGNORECASE)
    # Finds all years mentioned in each evidence block

    all_numbers: Dict[str, List[float]] = {}
    for eb in evidence:
        years_found = year_re.findall(eb.chunk.text)  # e.g., ["FY2025", "FY2024"]
        nums_found = number_re.findall(eb.chunk.text)  # e.g., ["482.6", "406.5", "18.7"]
        for y in years_found:
            for n in nums_found:
                all_numbers.setdefault(y, []).append(float(n))
```

**Growth rate calculation (Python, not LLM):**
```python
if len(qa.year_references) >= 2:
    y1, y2 = "FY2024", "FY2025"  # from query
    v1, v2 = all_numbers["FY2024"][0], all_numbers["FY2025"][0]
    growth = ((v2 - v1) / v1) * 100
    calculations["growth_rate"] = {
        "formula": f"(({v2} - {v1}) / {v1}) * 100 = {growth:.2f}%"
    }
```

**Why Python, not LLM?** LLMs hallucinate arithmetic (e.g., `482.6 - 406.5 = 75.4`, then calculate wrong %). Python is always exact. The computed result is injected directly into the LLM context as a verified fact.

---

## 25. Context Builder

```python
def build_final_context(evidence, calculations, qa):
    type_order = [TABLE, CHART, TEXT, IMAGE, DIAGRAM, PARAGRAPH, HEADING, OCR_TEXT]
    # Tables always come FIRST — most relevant for financial queries
```

For each evidence block:

```python
if etype == ElementType.TABLE and c.table_data:
    section_lines.append(
        f"[{verified_tag}] {cite}\n"
        f"{c.table_data.to_markdown()}\n"  # Beautiful markdown table
        f"{ctx}"                           # Expanded context (section heading etc.)
    )
```

The LLM receives a fully formatted context like:
```
--- TABLE EVIDENCE ---
[VERIFIED] [report.pdf, p.4 - TABLE]
[Section: 3. Financial Highlights]
| n Crore    | FY2025-26 | FY2024-25 | Change |
|---|---|---|---|
| Revenue    | 482.6     | 406.5     | +18.7% |
| Net Profit | 61.8      | 49.2      | +25.6% |

--- TEXT EVIDENCE ---
[VERIFIED] [report.pdf, p.2 - PARAGRAPH]
Revenue n482.6 Cr  EBITDA n96.4 Cr  Net Profit n61.8 Cr

--- CALCULATIONS (Python-verified) ---
Growth from FY2024 to FY2025:
  ((482.6 - 406.5) / 406.5) * 100 = 18.73%
```

---

## 26. Answer Generation

```python
ANSWER_SYSTEM_PROMPT = """You are a precise document intelligence assistant.

RULES:
1. Answer ONLY from the EVIDENCE blocks. Never use outside knowledge.
2. TABLE evidence IS authoritative. Values in TABLE rows are direct quotes from the PDF.
3. NEVER cite the [HyDE passage] — it is a retrieval aid only.
4. If the HyDE passage says "$1.87 billion" but TABLE says "482.6 Cr", TABLE is correct.
5. State figures in their original units (crore, %, etc.).
6. VERIFIED = seen in multiple sources. UNVERIFIED = single source.
7. Only say "Insufficient evidence" if the answer truly cannot be found.
"""
```

```python
def generate_answer(query, context, citations, image_bytes_list, hyde_doc):
    full_prompt = (
        f"EVIDENCE FROM PDF:\n{context}\n\n"
        f"NOTE: RETRIEVAL AID ONLY:\n[HyDE passage]: {hyde_doc[:150]}\n\n"
        f"Question: {query}\n\n"
        "Answer (cite sources, show calculations, note uncertainty):"
    )
    llm = _get_rag_llm()
    resp = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=full_prompt),
    ])
    return resp.content.strip()
```

**Vision-capable answer** (if OpenAI/Anthropic + images):
The actual chart PNG bytes are sent alongside the text context so the LLM can read chart values directly.

---

## 27. Master Pipeline

```python
def multimodal_rag(pdf_path: str, question: str) -> RAGResult:
```

This is the **orchestrator** — calls everything in order:

```
Step 0:  analyze_query(question)
          → QueryAnalysis with requires_table=True, years=["FY2025"]

Step 1:  extract_pdf(pdf_path)
          → 216 DocumentElements (cached after first run)

Step 2:  build_multimodal_chunks(elements)
          → 252 MultimodalChunks

Step 3:  build_index(chunks, fhash)
          → ChromaDB vectorstore + BM25Retriever

Step 4:  generate_hyde(question, fhash)
          → "NovaTech reported ₹482.6 crore revenue..."

Step 5:  (query enhancement — already in QueryAnalysis)

Step 6:  retrieve_dense([question, expanded, hyde]) → 21 chunks
Step 7:  retrieve_bm25(keyword_query)               → 15 chunks
Step 8:  retrieve_tables(qa, chunks)                → 10 chunks
Step 9:  retrieve_charts(qa, chunks)                →  0 chunks
Step 10: retrieve_visual(qa, chunks)                →  0 chunks
Step 11: retrieve_structured_numeric(qa, chunks)    →  5 chunks

Step 12: rrf_fusion([...]) + deduplicate()
          → 37 unique candidates

Step 13: crossencoder_rerank(question, candidates, top_k=7)
          → 7 final chunks (TABLE p.4 ranked #1 with score 2.769)

Step 14: expand_parent_context(top_chunks, all_chunks)
          → Adds "[Section: 3. Financial Highlights]" to each

Step 15: verify_evidence(top_chunks, qa)
          → 7/7 verified

Step 16: perform_calculations(qa, evidence_blocks)
          → {"growth_rate": {"formula": "((482.6-406.5)/406.5)*100=18.73%"}}

Step 17: build_final_context(evidence, calculations, qa)
          → 3577 chars of formatted markdown context

Step 18: generate_answer(question, context, ...)
          → "The total revenue was ₹482.6 Crore [report.pdf, p.4 - TABLE]"

Return:  RAGResult(answer=..., citations=[...], latency={...})
```

---

## 28. Main Entry Point

```python
if __name__ == "__main__":
```
This block only runs when you execute `python multimodal_rag.py` directly.
It does NOT run if you `import multimodal_rag` from another script.

```python
if len(sys.argv) >= 3:
    _pdf_path = sys.argv[1]          # First argument = PDF path
    _question = " ".join(sys.argv[2:])  # All remaining args = question
elif len(sys.argv) == 2:
    _pdf_path = sys.argv[1]
    _question = input("\nAsk a question: ").strip()  # Interactive prompt
else:
    _pdf_path = PDF_PATH    # Use defaults from config block at top
    _question = QUESTION
```

**Usage examples:**
```bash
# Use defaults from config
python multimodal_rag.py

# Pass PDF + question
python multimodal_rag.py "report.pdf" "What was net profit?"

# Interactive mode
python multimodal_rag.py "report.pdf"
```

**Suggested questions for NovaTech:**
```python
NOVATECH_TEST_QUESTIONS = [
    "What was the total revenue of NovaTech Solutions in FY2025-26?",
    "What was the net profit in FY2025-26 and how does it compare to FY2024-25?",
    "Which segment grew the fastest?",
    ...
]
```

**Number selection shortcut:**
```python
if next_q.isdigit():
    idx = int(next_q) - 1
    next_q = NOVATECH_TEST_QUESTIONS[idx]
# Type "3" → runs question #3
```

---

## 29. Full Architecture Flow Diagram

```
                    USER QUESTION
                         │
                         ▼
              ┌──────────────────────┐
              │   analyze_query()    │
              │  Detect: financial?  │
              │  table? chart? year? │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │     extract_pdf()    │◄──── .rag_cache/extracted_<hash>.pkl
              │  PyMuPDF: text+imgs  │
              │  pdfplumber: tables  │
              │  Vision: charts      │
              └──────────┬───────────┘
                         │ 216 elements
              ┌──────────▼───────────┐
              │ build_multimodal_    │
              │       chunks()       │
              └──────────┬───────────┘
                         │ 252 chunks
              ┌──────────▼───────────┐
              │    build_index()     │◄──── .rag_cache/chroma_<hash>/
              │  ChromaDB + BM25     │◄──── .rag_cache/bm25_<hash>.pkl
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │   generate_hyde()    │◄──── .rag_cache/hyde_<hash>_<q>.pkl
              │  LLM writes fake     │
              │  answer for retrieval│
              └──────────┬───────────┘
                         │ hyde_doc
         ┌───────────────┼───────────────┐
         │               │               │
    Dense MMR        BM25 Sparse      Table Direct
    [original,       [keywords]       [token match]
     expanded,
     hyde_doc]
         │               │               │
    ┌────▼────┐    ┌─────▼────┐   ┌─────▼────┐
    │21 chunks│    │15 chunks │   │10 chunks │
    └────┬────┘    └─────┬────┘   └─────┬────┘
         └───────────────┼───────────────┘
                    + Chart (0)
                    + Visual (0)
                    + Structured (5)
                         │
              ┌──────────▼───────────┐
              │    rrf_fusion()      │
              │ Weighted merge of    │
              │ all 6 ranked lists   │
              └──────────┬───────────┘
                         │ 37 unique candidates
              ┌──────────▼───────────┐
              │   deduplicate()      │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ crossencoder_rerank()│
              │  Score each (query,  │
              │  chunk) pair with CE │
              └──────────┬───────────┘
                         │ Top 7 chunks
              ┌──────────▼───────────┐
              │ expand_parent_       │
              │     context()        │
              │ Add section headings │
              │ Add related tables   │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  verify_evidence()   │
              │  Cross-check numbers │
              │  VERIFIED/UNVERIFIED │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │perform_calculations()│
              │  Python arithmetic   │
              │  Growth rate, ratios │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ build_final_context()│
              │  Markdown tables     │
              │  Verified tags       │
              │  Calculation block   │
              └──────────┬───────────┘
                         │ 3577 chars context
              ┌──────────▼───────────┐
              │  generate_answer()   │
              │  LLM reads context   │
              │  + System prompt     │
              │  + HyDE warning      │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │      RAGResult       │
              │  answer + citations  │
              │  + latency + evidence│
              └──────────────────────┘
```

---

## Summary Table — Every Function

| Function | What it does | Input | Output |
|---|---|---|---|
| `_file_hash()` | SHA256 of PDF | file path | 16-char hex |
| `_cache_load/save()` | Read/write `.pkl` files | key | Python object |
| `_image_to_b64()` | PNG → base64 string | bytes | string |
| `_describe_image_with_vision()` | Call vision LLM | image bytes, prompt | description text |
| `_detect_if_scanned()` | Is page a scan? | PyMuPDF page | bool |
| `_ocr_page_bytes()` | Tesseract OCR | PNG bytes | text |
| `_extract_text_elements()` | Text from PDF page | PyMuPDF page | DocumentElement list |
| `_extract_tables_pdfplumber()` | Tables from PDF | file path | DocumentElement list |
| `_extract_images_pymupdf()` | Images from PDF page | PyMuPDF page | DocumentElement list |
| `_enrich_visual_elements_with_vision()` | Describe charts/images | elements | elements (enriched) |
| `extract_pdf()` | **Master extraction** | file path | DocumentElement list |
| `_chunk_text()` | Split long text | text | List[str] |
| `build_multimodal_chunks()` | Elements → chunks | elements | MultimodalChunk list |
| `build_index()` | Build ChromaDB + BM25 | chunks | (vstore, bm25) |
| `analyze_query()` | Classify question | query string | QueryAnalysis |
| `generate_hyde()` | HyDE passage via LLM | query | string |
| `retrieve_dense()` | Vector search | queries, vstore | ScoredChunk list |
| `retrieve_bm25()` | Keyword search | query, bm25 | ScoredChunk list |
| `retrieve_tables()` | Table token match | qa, chunks | ScoredChunk list |
| `retrieve_charts()` | Chart description match | qa, chunks | ScoredChunk list |
| `retrieve_visual()` | Image description match | qa, chunks | ScoredChunk list |
| `retrieve_structured_numeric()` | Year+metric cell lookup | qa, chunks | ScoredChunk list |
| `rrf_fusion()` | Weighted rank merge | ranked lists | ScoredChunk list |
| `deduplicate()` | Remove duplicates | scored chunks | ScoredChunk list |
| `crossencoder_rerank()` | Precise relevance scoring | query, candidates | ScoredChunk list |
| `expand_parent_context()` | Add section/table context | top chunks | top chunks (enriched) |
| `verify_evidence()` | Cross-check numbers | chunks | EvidenceBlock list |
| `perform_calculations()` | Python arithmetic | evidence | Dict with growth_rate |
| `build_final_context()` | Assemble LLM context | evidence | (text, citations, images) |
| `generate_answer()` | LLM final answer | query + context | answer string |
| `multimodal_rag()` | **Master orchestrator** | pdf_path, question | RAGResult |
