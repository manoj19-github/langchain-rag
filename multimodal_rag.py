from __future__ import annotations
"""
multimodal_rag.py
=================
Production-grade Multimodal RAG pipeline.

Extends HyDE + Dense MMR + BM25 + RRF + CrossEncoder retrieval
into a fully MODALITY-AWARE, PAGE-AWARE PDF retrieval system.

Supports:
  - Normal & scanned PDFs (OCR fallback)
  - Text, headings, paragraphs
  - Tables (structured + searchable)
  - Charts (extracted region + vision description)
  - Images (extracted + vision description)
  - Diagrams (extracted + vision description)
  - Hybrid retrieval across all modalities
  - Weighted RRF fusion
  - CrossEncoder reranking
  - HyDE query enhancement
  - Evidence verification & calculations
  - Grounded citations with page numbers

Requirements (pip install):
    pymupdf>=1.24.0
    pdfplumber>=0.11.0
    pillow>=10.0.0
    pytesseract>=0.3.10
    sentence-transformers>=3.0.0
    rank-bm25>=0.2.2
    chromadb>=0.5.0
    langchain>=0.2.0
    langchain-huggingface>=0.0.3
    langchain-chroma>=0.1.0
    langchain-community>=0.2.0
    langchain-core>=0.2.0
    langchain-ollama>=0.1.0   (swap for openai/anthropic as needed)
    openai>=1.35.0            (for vision + LLM, optional)
    anthropic>=0.30.0         (optional)

Optional system packages:
    tesseract-ocr  (for pytesseract OCR on scanned pages)
"""

# ============================================================
# IMPORTS
# ============================================================
import base64
import hashlib
import io
import json
import logging
import os
import pickle
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Force UTF-8 stdout so ₹/–/’ print cleanly on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION  — edit these, never hardcode secrets
# ============================================================
# Company: NovaTech Solutions Ltd.  |  Annual Report FY2025-26
# PDF: data/pdf/dummy_company_annual_report_2025_26 (1).pdf
PDF_PATH: str = r"data/pdf/dummy_company_annual_report_2025_26 (1).pdf"
QUESTION: str = "What was the total revenue of NovaTech Solutions in FY2025-26?"

# --- LLM Provider: "ollama" | "openai" | "anthropic"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
LLM_TEMPERATURE: float = 0.0

# Vision model (for chart/image description)
VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "ollama")
VISION_MODEL: str = os.getenv("VISION_MODEL", "llava:13b")

# HyDE LLM
HYDE_MODEL: str = os.getenv("HYDE_MODEL", LLM_MODEL)
HYDE_TEMPERATURE: float = 0.4
HYDE_ENABLED: bool = True

# Embedding model
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

# CrossEncoder reranker
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

# Retrieval knobs
TOP_K_DENSE: int = 15
TOP_K_BM25: int = 15
TOP_K_TABLE: int = 10
TOP_K_CHART: int = 8
TOP_K_VISUAL: int = 8
TOP_K_STRUCTURED: int = 5
MMR_FETCH_K: int = 40
MMR_LAMBDA: float = 0.6
RERANK_TOP_K: int = 7
RERANK_BATCH_SIZE: int = 16

# RRF fusion weights
W_DENSE: float = 0.35
W_BM25: float = 0.25
W_TABLE: float = 0.20
W_CHART: float = 0.10
W_VISUAL: float = 0.07
W_STRUCTURED: float = 0.03
RRF_K: int = 60

# Chunking
TEXT_CHUNK_SIZE: int = 600
TEXT_CHUNK_OVERLAP: int = 80
MIN_CHUNK_CHARS: int = 30
MAX_TABLE_ROWS_PER_CHUNK: int = 60

# Image extraction
EXTRACT_IMAGES: bool = True
MIN_IMAGE_WIDTH: int = 80
MIN_IMAGE_HEIGHT: int = 80
MAX_IMAGES_PER_PAGE: int = 10

# OCR
OCR_ENABLED: bool = True
OCR_LANG: str = "eng"
OCR_DPI: int = 300

# Caching
CACHE_ENABLED: bool = True
CACHE_DIR: str = ".rag_cache"

# Debug
DEBUG: bool = True

# API keys (read from environment — never hardcode)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("multimodal_rag")
if DEBUG:
    logger.setLevel(logging.DEBUG)


def _safe(text: str) -> str:
    """Replace any character that can't be encoded by the current terminal."""
    try:
        text.encode(sys.stdout.encoding or 'utf-8')
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode('ascii', errors='replace').decode('ascii')


def _dbg(section: str, msg: str = "") -> None:
    if DEBUG:
        print(f"\n{'='*60}")
        print(f"  {section}")
        print(f"{'='*60}")
        if msg:
            print(_safe(msg))


def _info(msg: str) -> None:
    if DEBUG:
        print(f"  > {_safe(msg)}")


# ============================================================
# ENUMS & DATA MODELS
# ============================================================
class ElementType(str, Enum):
    TEXT = "TEXT"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    CHART = "CHART"
    IMAGE = "IMAGE"
    DIAGRAM = "DIAGRAM"
    CAPTION = "CAPTION"
    OCR_TEXT = "OCR_TEXT"
    PAGE = "PAGE"


@dataclass
class BBox:
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


@dataclass
class TableData:
    title: Optional[str] = None
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    units: Optional[str] = None
    currency: Optional[str] = None
    footnotes: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = []
        if self.title:
            lines.append(f"**{self.title}**")
        if self.units:
            lines.append(f"Units: {self.units}")
        if self.currency:
            lines.append(f"Currency: {self.currency}")
        if self.headers:
            lines.append("| " + " | ".join(str(h) for h in self.headers) + " |")
            lines.append("|" + "|".join(["---"] * len(self.headers)) + "|")
        for row in self.rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        for fn in self.footnotes:
            lines.append(f"* {fn}")
        return "\n".join(lines)

    def to_search_text(self) -> str:
        parts = []
        if self.title:
            parts.append(self.title)
        parts.append(" ".join(str(h) for h in self.headers))
        for row in self.rows:
            parts.append(" ".join(str(c) for c in row))
        return " | ".join(parts)

    def extract_numbers(self) -> List[Tuple[str, str, str]]:
        results = []
        num_re = re.compile(r"[\d,]+\.?\d*")
        for row in self.rows:
            row_label = row[0] if row else ""
            for i, cell in enumerate(row[1:], start=1):
                if num_re.search(str(cell)):
                    col = self.headers[i] if i < len(self.headers) else f"col_{i}"
                    results.append((col, row_label, str(cell)))
        return results


@dataclass
class DocumentElement:
    element_id: str
    document_name: str
    page_number: int
    element_type: ElementType
    content: str
    raw_text: Optional[str] = None
    image_bytes: Optional[bytes] = None
    image_b64: Optional[str] = None
    table_data: Optional[TableData] = None
    bbox: Optional[BBox] = None
    heading_level: Optional[int] = None
    caption: Optional[str] = None
    vision_description: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        return f"[{self.document_name}, p.{self.page_number} - {self.element_type.value}]"


@dataclass
class MultimodalChunk:
    chunk_id: str
    element_id: str
    document_name: str
    page_number: int
    element_type: ElementType
    text: str
    image_bytes: Optional[bytes] = None
    image_b64: Optional[str] = None
    table_data: Optional[TableData] = None
    structured_values: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_context: Optional[str] = None

    def citation(self) -> str:
        return f"[{self.document_name}, p.{self.page_number} - {self.element_type.value}]"

    def key(self) -> str:
        raw = f"{self.document_name}|{self.page_number}|{self.element_type}|{self.text[:100]}"
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class ScoredChunk:
    chunk: MultimodalChunk
    score: float
    source: str

    def key(self) -> str:
        return self.chunk.key()


@dataclass
class QueryAnalysis:
    original_query: str
    query_type: str
    requires_text: bool = True
    requires_table: bool = False
    requires_chart: bool = False
    requires_image: bool = False
    requires_diagram: bool = False
    requires_calculation: bool = False
    numerical_entities: List[str] = field(default_factory=list)
    year_references: List[str] = field(default_factory=list)
    metric_references: List[str] = field(default_factory=list)
    expanded_query: str = ""
    keyword_query: str = ""
    table_query: str = ""
    chart_query: str = ""
    visual_query: str = ""


@dataclass
class EvidenceBlock:
    chunk: MultimodalChunk
    rerank_score: float
    verified: bool = False
    conflicts_with: List[str] = field(default_factory=list)


@dataclass
class RAGResult:
    query: str
    hyde_doc: str
    query_analysis: Dict[str, Any]
    retrieval_stats: Dict[str, int]
    evidence: List[Dict[str, Any]]
    calculations: Dict[str, Any]
    answer: str
    citations: List[str]
    latency: Dict[str, float]


# ============================================================
# CACHE UTILITIES
# ============================================================
def _cache_path(key: str) -> Path:
    p = Path(CACHE_DIR)
    p.mkdir(exist_ok=True)
    return p / f"{key}.pkl"


def _cache_load(key: str) -> Optional[Any]:
    if not CACHE_ENABLED:
        return None
    cp = _cache_path(key)
    if cp.exists():
        try:
            with open(cp, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _cache_save(key: str, obj: Any) -> None:
    if not CACHE_ENABLED:
        return
    try:
        with open(_cache_path(key), "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ============================================================
# VISION UTILITIES
# ============================================================
def _image_to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _describe_image_with_vision(image_bytes: bytes, prompt: str) -> str:
    b64 = _image_to_b64(image_bytes)
    try:
        if VISION_PROVIDER == "openai" and OPENAI_API_KEY:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high"
                        }}
                    ]
                }],
                max_tokens=800,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()

        elif VISION_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model=VISION_MODEL,
                max_tokens=800,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64
                        }},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            return resp.content[0].text.strip()

        elif VISION_PROVIDER == "ollama":
            import urllib.request
            payload = json.dumps({
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
                "options": {"temperature": 0.0}
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data.get("response", "").strip()

    except Exception as e:
        logger.debug(f"Vision description failed: {e}")
    return "[Vision description unavailable]"


VISION_PROMPT_CHART = """Analyze this chart/graph from a PDF. Return JSON:
{
  "title": "chart title or Unknown",
  "chart_type": "bar|line|pie|area|scatter|stacked|other",
  "x_axis": "x-axis label",
  "y_axis": "y-axis label",
  "unit": "unit if visible",
  "legend": ["series1"],
  "data_points": [{"label": "...", "value": "..."}],
  "key_observations": "summary",
  "visible_text": "all visible text/numbers",
  "uncertainty": "low|medium|high"
}
Never invent data not visible in the image."""

VISION_PROMPT_IMAGE = """Describe this image from a PDF accurately and concisely.
Include: what is shown, visible text/numbers/labels, spatial layout, important details.
Do not guess what you cannot see."""

VISION_PROMPT_DIAGRAM = """Analyze this diagram from a PDF.
Identify: diagram type, all visible nodes/labels, arrows and directions, relationships, hierarchy.
Produce a structured text representation like:
  Node A -> Node B -> Node C
and a paragraph description. Do not invent connections not visible."""


# ============================================================
# LAZY-LOADED MODELS
# ============================================================
_embeddings_instance = None
_cross_encoder_instance = None
_hyde_llm_instance = None
_rag_llm_instance = None


def _get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
        )
    return _embeddings_instance


def _get_cross_encoder():
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder_instance = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder_instance


def _build_llm(model: str, temperature: float):
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, api_key=ANTHROPIC_API_KEY)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature, base_url=OLLAMA_BASE_URL)


def _get_hyde_llm():
    global _hyde_llm_instance
    if _hyde_llm_instance is None:
        _hyde_llm_instance = _build_llm(HYDE_MODEL, HYDE_TEMPERATURE)
    return _hyde_llm_instance


def _get_rag_llm():
    global _rag_llm_instance
    if _rag_llm_instance is None:
        _rag_llm_instance = _build_llm(LLM_MODEL, LLM_TEMPERATURE)
    return _rag_llm_instance


# ============================================================
# PDF PROCESSING — TEXT EXTRACTION
# ============================================================
def _detect_if_scanned(page) -> bool:
    blocks = page.get_text("blocks")
    total_chars = sum(len(b[4]) for b in blocks if b[6] == 0)
    return total_chars < 20


def _ocr_page_bytes(page_image_bytes: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(page_image_bytes))
        return pytesseract.image_to_string(img, lang=OCR_LANG).strip()
    except Exception as e:
        logger.debug(f"OCR failed: {e}")
        return ""


def _page_to_png(page, dpi: int = OCR_DPI) -> bytes:
    mat = page.get_pixmap(dpi=dpi)
    return mat.tobytes("png")


def _extract_text_elements(page, page_num: int, doc_name: str) -> List[DocumentElement]:
    elements: List[DocumentElement] = []
    blocks = page.get_text("dict", flags=11)["blocks"]
    current_heading_text = None

    for bi, block in enumerate(blocks):
        if block.get("type") != 0:
            continue
        for span_group in block.get("lines", []):
            for span in span_group.get("spans", []):
                txt = span.get("text", "").strip()
                if not txt or len(txt) < 2:
                    continue
                font_size = span.get("size", 12)
                is_bold = bool(span.get("flags", 0) & 2**4)
                etype = ElementType.TEXT

                if font_size >= 14 or (font_size >= 12 and is_bold and len(txt) < 120):
                    etype = ElementType.HEADING
                    current_heading_text = txt
                elif len(txt) > 80:
                    etype = ElementType.PARAGRAPH

                bbox_raw = span.get("bbox", [0, 0, 0, 0])
                eid = f"page_{page_num}_text_{bi}_{id(span)}"
                el = DocumentElement(
                    element_id=eid,
                    document_name=doc_name,
                    page_number=page_num,
                    element_type=etype,
                    content=txt,
                    raw_text=txt,
                    bbox=BBox(*bbox_raw),
                    heading_level=(1 if font_size >= 18 else 2 if font_size >= 14 else 3)
                    if etype == ElementType.HEADING else None,
                    metadata={
                        "font_size": font_size,
                        "is_bold": is_bold,
                        "heading_context": current_heading_text,
                    }
                )
                elements.append(el)
    return elements


# ============================================================
# PDF PROCESSING — TABLE EXTRACTION
# ============================================================
def _extract_tables_pdfplumber(pdf_path: str, doc_name: str) -> List[DocumentElement]:
    elements: List[DocumentElement] = []
    try:
        import pdfplumber
    except ImportError:
        logger.debug("pdfplumber not installed - skipping table extraction")
        return elements

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                tables = page.extract_tables()
                for ti, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    headers = [str(c).strip() if c else "" for c in table[0]]
                    rows = [
                        [str(c).strip() if c else "" for c in row]
                        for row in table[1:]
                        if any(c for c in row)
                    ]
                    if not rows:
                        continue

                    currency = None
                    units = None
                    all_text = " ".join(headers + [c for r in rows for c in r])
                    for sym in ["Rs.", "INR", "USD", "$", "EUR", "GBP"]:
                        if sym in all_text:
                            currency = sym
                            break
                    for u in ["crore", "Cr", "million", "billion", "lakh", "Mn", "Bn", "%"]:
                        if u.lower() in all_text.lower():
                            units = u
                            break

                    td = TableData(headers=headers, rows=rows, currency=currency, units=units)
                    eid = f"page_{page_num}_table_{ti}"
                    el = DocumentElement(
                        element_id=eid,
                        document_name=doc_name,
                        page_number=page_num,
                        element_type=ElementType.TABLE,
                        content=td.to_search_text(),
                        table_data=td,
                        metadata={"row_count": len(rows), "col_count": len(headers)}
                    )
                    elements.append(el)
    except Exception as e:
        logger.debug(f"pdfplumber error: {e}")
    return elements


# ============================================================
# PDF PROCESSING — IMAGE / CHART / DIAGRAM EXTRACTION
# ============================================================
def _extract_images_pymupdf(page, page_num: int, doc_name: str) -> List[DocumentElement]:
    elements: List[DocumentElement] = []
    if not EXTRACT_IMAGES:
        return elements
    try:
        img_list = page.get_images(full=True)
        for ii, img in enumerate(img_list[:MAX_IMAGES_PER_PAGE]):
            xref = img[0]
            try:
                base_img = page.parent.extract_image(xref)
            except Exception:
                continue
            img_bytes = base_img.get("image", b"")
            width = base_img.get("width", 0)
            height = base_img.get("height", 0)
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
            except Exception:
                png_bytes = img_bytes

            # Heuristic: chart if image is large + landscape
            aspect = width / max(height, 1)
            is_chart = (0.5 < aspect < 4.0) and (width * height > 20000)
            etype = ElementType.CHART if is_chart else ElementType.IMAGE
            eid = f"page_{page_num}_img_{ii}"

            el = DocumentElement(
                element_id=eid,
                document_name=doc_name,
                page_number=page_num,
                element_type=etype,
                content=f"[{etype.value} on page {page_num}]",
                image_bytes=png_bytes,
                image_b64=_image_to_b64(png_bytes),
                metadata={"width": width, "height": height}
            )
            elements.append(el)
    except Exception as e:
        logger.debug(f"Image extraction error page {page_num}: {e}")
    return elements


def _enrich_visual_elements_with_vision(
    elements: List[DocumentElement], cache_prefix: str
) -> List[DocumentElement]:
    for el in elements:
        if el.element_type not in (ElementType.CHART, ElementType.IMAGE, ElementType.DIAGRAM):
            continue
        if el.image_bytes is None:
            continue

        ckey = f"vision_{cache_prefix}_{el.element_id}"
        cached = _cache_load(ckey)
        if cached is not None:
            el.vision_description = cached
            el.content = cached
            continue

        _info(f"Generating vision description for {el.element_id}...")
        if el.element_type == ElementType.CHART:
            raw = _describe_image_with_vision(el.image_bytes, VISION_PROMPT_CHART)
            try:
                data = json.loads(raw)
                el.metadata["chart_data"] = data
                parts = [
                    f"Chart: {data.get('title', 'Unknown')}",
                    f"Type: {data.get('chart_type', '')}",
                    f"X-axis: {data.get('x_axis', '')}",
                    f"Y-axis: {data.get('y_axis', '')}",
                    f"Observations: {data.get('key_observations', '')}",
                    f"Visible text: {data.get('visible_text', '')}",
                ]
                dp = data.get("data_points", [])
                if dp:
                    parts.append("Data: " + ", ".join(
                        f"{d.get('label','?')}={d.get('value','?')}" for d in dp
                    ))
                desc = "\n".join(p for p in parts if p.split(": ", 1)[-1])
            except (json.JSONDecodeError, ValueError):
                desc = raw
        elif el.element_type == ElementType.DIAGRAM:
            desc = _describe_image_with_vision(el.image_bytes, VISION_PROMPT_DIAGRAM)
        else:
            desc = _describe_image_with_vision(el.image_bytes, VISION_PROMPT_IMAGE)

        el.vision_description = desc
        el.content = desc if desc else el.content
        _cache_save(ckey, desc)

    return elements


# ============================================================
# MASTER PDF EXTRACTION
# ============================================================
def extract_pdf(pdf_path: str) -> List[DocumentElement]:
    doc_name = Path(pdf_path).name
    fhash = _file_hash(pdf_path)
    cache_key = f"extracted_{fhash}"
    cached = _cache_load(cache_key)
    if cached is not None:
        _info(f"Loaded {len(cached)} elements from cache.")
        return cached

    _dbg("PDF EXTRACTION", f"File: {pdf_path}")
    elements: List[DocumentElement] = []

    # PyMuPDF text + image
    try:
        import fitz
        doc = fitz.open(pdf_path)
        _info(f"PDF has {len(doc)} pages")
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            is_scanned = _detect_if_scanned(page)
            if is_scanned and OCR_ENABLED:
                _info(f"Page {page_num}: scanned -> running OCR")
                png = _page_to_png(page)
                ocr_text = _ocr_page_bytes(png)
                if ocr_text:
                    elements.append(DocumentElement(
                        element_id=f"page_{page_num}_ocr",
                        document_name=doc_name,
                        page_number=page_num,
                        element_type=ElementType.OCR_TEXT,
                        content=ocr_text,
                        raw_text=ocr_text,
                        confidence=0.85,
                        metadata={"is_ocr": True},
                    ))
            else:
                elements.extend(_extract_text_elements(page, page_num, doc_name))
            elements.extend(_extract_images_pymupdf(page, page_num, doc_name))
        doc.close()
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed - text/image extraction unavailable")
    except Exception as e:
        logger.error(f"PyMuPDF error: {e}")

    # pdfplumber tables
    table_els = _extract_tables_pdfplumber(pdf_path, doc_name)
    elements.extend(table_els)
    _info(f"Extracted {len(table_els)} tables")

    # Vision enrichment
    visual_count = sum(1 for e in elements if e.element_type in (
        ElementType.CHART, ElementType.IMAGE, ElementType.DIAGRAM))
    _info(f"Found {visual_count} visual elements - enriching with vision model")
    _enrich_visual_elements_with_vision(elements, fhash)

    elements.sort(key=lambda e: (e.page_number, e.element_type.value))
    _info(f"Total elements extracted: {len(elements)}")
    _cache_save(cache_key, elements)
    return elements


# ============================================================
# CHUNKING ENGINE
# ============================================================
def _chunk_text(text: str, chunk_size: int = TEXT_CHUNK_SIZE,
                overlap: int = TEXT_CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = text.rfind(sep, start + overlap, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)
    return chunks


def build_multimodal_chunks(elements: List[DocumentElement]) -> List[MultimodalChunk]:
    chunks: List[MultimodalChunk] = []
    heading_context: Dict[int, str] = {}

    for el in elements:
        page = el.page_number
        doc = el.document_name

        if el.element_type == ElementType.HEADING:
            heading_context[page] = el.content

        parent_ctx = heading_context.get(page, "")
        base_meta = {
            "document_name": doc,
            "page_number": page,
            "element_type": el.element_type.value,
            "element_id": el.element_id,
            **el.metadata,
        }

        # TEXT / HEADING / PARAGRAPH / OCR_TEXT
        if el.element_type in (
            ElementType.TEXT, ElementType.HEADING, ElementType.PARAGRAPH,
            ElementType.CAPTION, ElementType.OCR_TEXT
        ):
            for ci, part in enumerate(_chunk_text(el.content)):
                cid = f"{el.element_id}_chunk_{ci}"
                chunks.append(MultimodalChunk(
                    chunk_id=cid, element_id=el.element_id,
                    document_name=doc, page_number=page,
                    element_type=el.element_type, text=part,
                    metadata=base_meta, parent_context=parent_ctx,
                ))

        # TABLE
        elif el.element_type == ElementType.TABLE and el.table_data:
            td = el.table_data
            numbers = td.extract_numbers()
            structured = {
                "headers": td.headers, "row_count": len(td.rows),
                "numeric_cells": numbers[:50], "currency": td.currency,
                "units": td.units, "title": td.title,
            }
            # Whole-table chunk
            chunks.append(MultimodalChunk(
                chunk_id=f"{el.element_id}_whole", element_id=el.element_id,
                document_name=doc, page_number=page,
                element_type=ElementType.TABLE, text=td.to_search_text(),
                table_data=td, structured_values=structured,
                metadata={**base_meta, "markdown": td.to_markdown()},
                parent_context=parent_ctx,
            ))
            # Row-level sub-chunks
            for ri, row in enumerate(td.rows[:MAX_TABLE_ROWS_PER_CHUNK]):
                row_text = " | ".join(
                    f"{h}: {v}" for h, v in zip(td.headers, row)
                )
                chunks.append(MultimodalChunk(
                    chunk_id=f"{el.element_id}_row_{ri}", element_id=el.element_id,
                    document_name=doc, page_number=page,
                    element_type=ElementType.TABLE, text=row_text,
                    table_data=td,
                    structured_values={"row_index": ri, "row_label": row[0] if row else ""},
                    metadata={**base_meta, "is_row_chunk": True, "row_index": ri},
                    parent_context=parent_ctx,
                ))

        # CHART / IMAGE / DIAGRAM
        elif el.element_type in (ElementType.CHART, ElementType.IMAGE, ElementType.DIAGRAM):
            desc = el.vision_description or el.content
            if not desc or len(desc) < MIN_CHUNK_CHARS:
                desc = f"[{el.element_type.value} on page {page}]"
            chunks.append(MultimodalChunk(
                chunk_id=f"{el.element_id}_visual", element_id=el.element_id,
                document_name=doc, page_number=page,
                element_type=el.element_type, text=desc,
                image_bytes=el.image_bytes, image_b64=el.image_b64,
                metadata={**base_meta, "caption": el.caption},
                parent_context=parent_ctx,
            ))

    _info(f"Total multimodal chunks: {len(chunks)}")
    return chunks


# ============================================================
# INDEX BUILDING
# ============================================================
def build_index(chunks: List[MultimodalChunk], pdf_hash: str):
    from langchain_core.documents import Document as LCDoc
    from langchain_chroma import Chroma
    from langchain_community.retrievers import BM25Retriever

    cache_key = f"bm25_{pdf_hash}"
    lc_docs = []
    for c in chunks:
        if len(c.text.strip()) < MIN_CHUNK_CHARS:
            continue
        lc_docs.append(LCDoc(
            page_content=c.text,
            metadata={
                "chunk_id": c.chunk_id,
                "element_id": c.element_id,
                "document_name": c.document_name,
                "page_number": c.page_number,
                "element_type": c.element_type.value,
                "has_image": c.image_b64 is not None,
                "has_table": c.table_data is not None,
            }
        ))

    # BM25 - try to load from cache
    cached_bm25 = _cache_load(cache_key)
    if cached_bm25 is not None:
        _info("Loaded BM25 from cache")
        bm25_retriever = cached_bm25
    else:
        _info("Building BM25 index...")
        bm25_retriever = BM25Retriever.from_documents(lc_docs)
        bm25_retriever.k = TOP_K_BM25
        _cache_save(cache_key, bm25_retriever)

    # Chroma vectorstore
    chroma_dir = f"{CACHE_DIR}/chroma_{pdf_hash}"
    _info("Building vector index...")
    vstore = Chroma.from_documents(
        documents=lc_docs,
        embedding=_get_embeddings(),
        collection_name=f"mmrag_{pdf_hash}",
        persist_directory=chroma_dir,
    )
    _info(f"Index built with {len(lc_docs)} documents")
    return vstore, bm25_retriever


# ============================================================
# QUERY ANALYZER
# ============================================================
_TABLE_KW = {"revenue", "profit", "margin", "growth", "percentage", "fy", "financial",
             "segment", "total", "amount", "quarter", "annual", "crore", "million",
             "billion", "lakh", "ratio", "ebitda", "earnings", "loss", "income",
             "expense", "cost", "sales", "turnover"}
_CHART_KW = {"chart", "graph", "trend", "visual", "shown", "increase", "decrease",
             "highest", "lowest", "peak", "bar", "line", "pie", "distribution",
             "figure", "plot"}
_IMAGE_KW = {"image", "photo", "picture", "screenshot", "illustrate"}
_DIAGRAM_KW = {"diagram", "architecture", "flow", "process", "system",
               "structure", "hierarchy", "network", "pipeline"}
_NUMERIC_KW = {"calculate", "compute", "how much", "how many", "sum",
               "average", "growth rate"}
_YEAR_RE = re.compile(r"\bfy[\s\']?\d{2,4}\b|\b20\d\d\b|\b19\d\d\b", re.IGNORECASE)
_METRIC_RE = re.compile(
    r"\b(revenue|profit|margin|ebitda|eps|cagr|growth|sales|income|cost|expense)\b",
    re.IGNORECASE
)


def analyze_query(query: str) -> QueryAnalysis:
    ql = query.lower()
    words = set(re.findall(r"\b\w+\b", ql))
    years = _YEAR_RE.findall(query)
    metrics = _METRIC_RE.findall(query)

    req_table = bool(words & _TABLE_KW) or bool(years)
    req_chart = bool(words & _CHART_KW)
    req_image = bool(words & _IMAGE_KW)
    req_diagram = bool(words & _DIAGRAM_KW)
    req_calc = bool(words & _NUMERIC_KW) or len(years) >= 2

    if req_chart:
        qtype = "chart"
    elif req_diagram:
        qtype = "diagram"
    elif req_image:
        qtype = "image"
    elif req_table or req_calc:
        qtype = "financial" if (words & {"revenue", "profit", "margin", "fy", "financial"}) else "table"
    else:
        qtype = "text"

    expanded = query
    if years:
        expanded += " " + " ".join(years)
    if metrics:
        expanded += " " + " ".join(metrics)

    stopwords = {"the", "a", "an", "in", "of", "for", "was", "is", "what", "how", "which"}
    keyword_q = " ".join(sorted(words - stopwords))

    return QueryAnalysis(
        original_query=query,
        query_type=qtype,
        requires_text=True,
        requires_table=req_table,
        requires_chart=req_chart,
        requires_image=req_image,
        requires_diagram=req_diagram,
        requires_calculation=req_calc,
        numerical_entities=years + metrics,
        year_references=years,
        metric_references=metrics,
        expanded_query=expanded,
        keyword_query=keyword_q,
        table_query=f"{query} financial table data rows columns",
        chart_query=f"{query} chart graph trend visual figure",
        visual_query=f"{query} image diagram figure visual",
    )


# ============================================================
# HYDE GENERATION
# ============================================================
def generate_hyde(query: str, cache_prefix: str) -> str:
    if not HYDE_ENABLED:
        return query

    ckey = f"hyde_{cache_prefix}_{hashlib.sha256(query.encode()).hexdigest()[:12]}"
    cached = _cache_load(ckey)
    if cached:
        return cached

    prompt = (
        "You are a retrieval enhancement component in a RAG system.\n"
        "Generate a concise hypothetical passage that WOULD contain the information to answer the question.\n"
        "This passage is ONLY for embedding similarity search - NOT the final answer.\n"
        "Include likely terminology, entities, numbers, and relationships from relevant documents.\n"
        "Write as if excerpted from the actual source document. Do not be conversational.\n\n"
        f"Question: {query}\n\n"
        "Hypothetical document passage:"
    )
    try:
        from langchain_core.messages import HumanMessage
        llm = _get_hyde_llm()
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        if not text:
            return query
        _cache_save(ckey, text)
        return text
    except Exception as e:
        logger.debug(f"HyDE failed: {e}")
        return query


# ============================================================
# RETRIEVAL — ALL MODALITY PATHS
# ============================================================
def _lc_doc_to_scored_chunk(
    doc, score: float, source: str,
    chunks_by_id: Dict[str, MultimodalChunk]
) -> Optional[ScoredChunk]:
    cid = doc.metadata.get("chunk_id", "")
    chunk = chunks_by_id.get(cid)
    if chunk is None:
        chunk = MultimodalChunk(
            chunk_id=cid, element_id=doc.metadata.get("element_id", cid),
            document_name=doc.metadata.get("document_name", "Unknown"),
            page_number=doc.metadata.get("page_number", 0),
            element_type=ElementType(doc.metadata.get("element_type", "TEXT")),
            text=doc.page_content, metadata=doc.metadata,
        )
    return ScoredChunk(chunk=chunk, score=score, source=source)


def retrieve_dense(
    queries: List[str], vstore, chunks_by_id: Dict[str, MultimodalChunk], top_k: int
) -> List[ScoredChunk]:
    results: List[ScoredChunk] = []
    seen_ids: set = set()
    retriever = vstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA},
    )
    for q in queries:
        try:
            for doc in retriever.invoke(q):
                sc = _lc_doc_to_scored_chunk(doc, 1.0, "dense", chunks_by_id)
                if sc and sc.key() not in seen_ids:
                    seen_ids.add(sc.key())
                    results.append(sc)
        except Exception as e:
            logger.debug(f"Dense retrieval error: {e}")
    return results


def retrieve_bm25(
    query: str, bm25_retriever, chunks_by_id: Dict[str, MultimodalChunk]
) -> List[ScoredChunk]:
    results = []
    try:
        for doc in bm25_retriever.invoke(query):
            sc = _lc_doc_to_scored_chunk(doc, 1.0, "bm25", chunks_by_id)
            if sc:
                results.append(sc)
    except Exception as e:
        logger.debug(f"BM25 error: {e}")
    return results


def retrieve_tables(qa: QueryAnalysis, chunks: List[MultimodalChunk]) -> List[ScoredChunk]:
    query_tokens = set(re.findall(r"\b\w+\b", qa.original_query.lower()))
    year_tokens = set(y.lower().replace(" ", "") for y in qa.year_references)
    metric_tokens = set(m.lower() for m in qa.metric_references)
    priority_tokens = query_tokens | year_tokens | metric_tokens

    scored = []
    for c in chunks:
        if c.element_type != ElementType.TABLE:
            continue
        text_lower = c.text.lower()
        token_matches = sum(1 for t in priority_tokens if t in text_lower)
        if token_matches > 0:
            scored.append(ScoredChunk(
                chunk=c,
                score=float(token_matches) / max(len(priority_tokens), 1),
                source="table",
            ))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:TOP_K_TABLE]


def retrieve_charts(qa: QueryAnalysis, chunks: List[MultimodalChunk]) -> List[ScoredChunk]:
    query_tokens = set(re.findall(r"\b\w+\b", qa.original_query.lower()))
    scored = []
    for c in chunks:
        if c.element_type not in (ElementType.CHART, ElementType.DIAGRAM):
            continue
        hits = sum(1 for t in query_tokens if len(t) > 3 and t in c.text.lower())
        scored.append(ScoredChunk(chunk=c, score=float(hits), source="chart"))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:TOP_K_CHART]


def retrieve_visual(qa: QueryAnalysis, chunks: List[MultimodalChunk]) -> List[ScoredChunk]:
    query_tokens = set(re.findall(r"\b\w+\b", qa.original_query.lower()))
    scored = []
    for c in chunks:
        if c.element_type != ElementType.IMAGE:
            continue
        hits = sum(1 for t in query_tokens if len(t) > 3 and t in c.text.lower())
        scored.append(ScoredChunk(chunk=c, score=float(hits), source="visual"))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:TOP_K_VISUAL]


def retrieve_structured_numeric(
    qa: QueryAnalysis, chunks: List[MultimodalChunk]
) -> List[ScoredChunk]:
    if not qa.year_references and not qa.metric_references:
        return []

    year_pats = [
        re.compile(r.replace("fy", r"fy[\s\']?"), re.IGNORECASE)
        for r in qa.year_references
    ]
    metric_pats = [re.compile(m, re.IGNORECASE) for m in qa.metric_references]

    results = []
    for c in chunks:
        if c.element_type != ElementType.TABLE or not c.table_data:
            continue
        for col, row_label, val in c.table_data.extract_numbers():
            year_match = any(p.search(col) or p.search(row_label)
                             for p in year_pats) if year_pats else True
            metric_match = any(p.search(col) or p.search(row_label)
                               for p in metric_pats) if metric_pats else True
            if year_match and metric_match:
                results.append(ScoredChunk(chunk=c, score=2.0, source="structured"))
                break

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:TOP_K_STRUCTURED]


# ============================================================
# RRF FUSION
# ============================================================
def rrf_fusion(
    ranked_lists: List[Tuple[List[ScoredChunk], float]], k: int = RRF_K
) -> List[ScoredChunk]:
    final_scores: Dict[str, float] = {}
    chunk_map: Dict[str, ScoredChunk] = {}

    for ranked, weight in ranked_lists:
        for rank, sc in enumerate(ranked):
            key = sc.key()
            rrf_score = weight / (k + rank + 1)
            if key not in final_scores:
                final_scores[key] = 0.0
                chunk_map[key] = sc
            final_scores[key] += rrf_score

    sorted_keys = sorted(final_scores, key=lambda kk: final_scores[kk], reverse=True)
    return [
        ScoredChunk(chunk=chunk_map[key].chunk, score=final_scores[key],
                    source=chunk_map[key].source)
        for key in sorted_keys
    ]


# ============================================================
# DEDUPLICATION
# ============================================================
def deduplicate(scored_chunks: List[ScoredChunk]) -> List[ScoredChunk]:
    seen: Dict[str, ScoredChunk] = {}
    for sc in scored_chunks:
        key = sc.key()
        if key not in seen or sc.score > seen[key].score:
            seen[key] = sc
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)


# ============================================================
# CROSSENCODER RERANKING
# ============================================================
def crossencoder_rerank(
    query: str, candidates: List[ScoredChunk], top_k: int = RERANK_TOP_K
) -> List[ScoredChunk]:
    if not candidates:
        return []
    ce = _get_cross_encoder()
    pairs = [(query, f"[{sc.chunk.element_type.value}] " + sc.chunk.text[:512])
             for sc in candidates]
    try:
        scores = ce.predict(pairs, batch_size=RERANK_BATCH_SIZE)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(chunk=sc.chunk, score=float(s), source=sc.source)
            for sc, s in ranked[:top_k]
        ]
    except Exception as e:
        logger.debug(f"CrossEncoder failed: {e}")
        return candidates[:top_k]


# ============================================================
# PARENT CONTEXT EXPANSION
# ============================================================
def expand_parent_context(
    top_chunks: List[ScoredChunk], all_chunks: List[MultimodalChunk]
) -> List[ScoredChunk]:
    page_chunks: Dict[int, List[MultimodalChunk]] = {}
    for c in all_chunks:
        page_chunks.setdefault(c.page_number, []).append(c)

    for sc in top_chunks:
        c = sc.chunk
        page = c.page_number
        neighbours = page_chunks.get(page, [])
        expansion_parts = []

        if c.parent_context:
            expansion_parts.append(f"[Section: {c.parent_context}]")

        if c.element_type == ElementType.CHART:
            for n in neighbours:
                if n.element_type == ElementType.TABLE and n.chunk_id != c.chunk_id:
                    expansion_parts.append(f"[Related table p.{page}]: {n.text[:300]}")
                    break
            for n in neighbours:
                if n.element_type == ElementType.CAPTION and n.chunk_id != c.chunk_id:
                    expansion_parts.append(f"[Caption]: {n.text}")
                    break

        if c.element_type == ElementType.TABLE:
            for n in neighbours:
                if n.element_type in (ElementType.PARAGRAPH, ElementType.TEXT):
                    expansion_parts.append(f"[Context]: {n.text[:200]}")
                    break

        if expansion_parts:
            c.metadata["expanded_context"] = "\n".join(expansion_parts)

    return top_chunks


# ============================================================
# EVIDENCE VERIFICATION
# ============================================================
def verify_evidence(
    top_chunks: List[ScoredChunk], qa: QueryAnalysis
) -> List[EvidenceBlock]:
    number_re = re.compile(r"[\d,]+\.?\d*")
    all_values: Dict[str, List[str]] = {}

    for sc in top_chunks:
        for n in number_re.findall(sc.chunk.text):
            clean = n.replace(",", "")
            try:
                if float(clean) > 0:
                    all_values.setdefault(clean, []).append(sc.chunk.citation())
            except ValueError:
                pass

    blocks = []
    for sc in top_chunks:
        verified = True
        for n in number_re.findall(sc.chunk.text):
            clean = n.replace(",", "")
            try:
                if float(clean) > 100 and len(all_values.get(clean, [])) < 2:
                    verified = False
                    break
            except ValueError:
                pass
        blocks.append(EvidenceBlock(
            chunk=sc.chunk, rerank_score=sc.score, verified=verified
        ))
    return blocks


# ============================================================
# CALCULATIONS
# ============================================================
def perform_calculations(
    qa: QueryAnalysis, evidence: List[EvidenceBlock]
) -> Dict[str, Any]:
    if not qa.year_references and not qa.metric_references:
        return {}

    number_re = re.compile(r"([\d,]+\.?\d*)")
    year_re = re.compile(r"\b(fy[\s\']?\d{2,4}|20\d\d|19\d\d)\b", re.IGNORECASE)

    all_numbers: Dict[str, List[float]] = {}
    for eb in evidence:
        years_found = year_re.findall(eb.chunk.text)
        nums_found = number_re.findall(eb.chunk.text)
        for y in years_found:
            clean_y = re.sub(r"\s+", "", y).upper()
            for n in nums_found:
                try:
                    val = float(n.replace(",", ""))
                    if val > 0:
                        all_numbers.setdefault(clean_y, []).append(val)
                except ValueError:
                    pass

    calculations: Dict[str, Any] = {}
    if len(qa.year_references) >= 2:
        y1 = re.sub(r"\s+", "", qa.year_references[0]).upper()
        y2 = re.sub(r"\s+", "", qa.year_references[-1]).upper()
        v1_list = all_numbers.get(y1, [])
        v2_list = all_numbers.get(y2, [])
        if v1_list and v2_list:
            v1, v2 = v1_list[0], v2_list[0]
            if v1 != 0:
                growth = ((v2 - v1) / v1) * 100
                calculations["growth_rate"] = {
                    "from_year": y1, "to_year": y2,
                    "from_value": v1, "to_value": v2,
                    "growth_pct": round(growth, 2),
                    "formula": f"(({v2} - {v1}) / {v1}) * 100 = {growth:.2f}%",
                }

    calculations["extracted_values"] = {k: v for k, v in all_numbers.items() if v}
    return calculations


# ============================================================
# CONTEXT BUILDER
# ============================================================
def build_final_context(
    evidence: List[EvidenceBlock],
    calculations: Dict[str, Any],
    qa: QueryAnalysis,
) -> Tuple[str, List[str], List[bytes]]:
    sections: List[str] = []
    citations: List[str] = []
    image_bytes_list: List[bytes] = []
    seen_keys: set = set()

    type_order = [
        ElementType.TABLE, ElementType.CHART, ElementType.TEXT,
        ElementType.IMAGE, ElementType.DIAGRAM, ElementType.PARAGRAPH,
        ElementType.HEADING, ElementType.OCR_TEXT,
    ]
    grouped: Dict[ElementType, List[EvidenceBlock]] = {}
    for eb in evidence:
        grouped.setdefault(eb.chunk.element_type, []).append(eb)

    for etype in type_order:
        ebs = grouped.get(etype, [])
        if not ebs:
            continue
        section_lines = [f"\n--- {etype.value} EVIDENCE ---"]
        for eb in ebs:
            c = eb.chunk
            key = c.key()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            cite = c.citation()
            if cite not in citations:
                citations.append(cite)

            verified_tag = "VERIFIED" if eb.verified else "UNVERIFIED"
            ctx = c.metadata.get("expanded_context", "")

            if etype == ElementType.TABLE and c.table_data:
                section_lines.append(
                    f"[{verified_tag}] {cite}\n{c.table_data.to_markdown()}\n{ctx}"
                )
            elif etype in (ElementType.CHART, ElementType.IMAGE, ElementType.DIAGRAM):
                section_lines.append(
                    f"[{verified_tag}] {cite}\nDescription: {c.text}\n{ctx}"
                )
                if c.image_bytes:
                    image_bytes_list.append(c.image_bytes)
            else:
                section_lines.append(f"[{verified_tag}] {cite}\n{c.text}\n{ctx}")

        sections.append("\n".join(section_lines))

    if calculations.get("growth_rate"):
        gr = calculations["growth_rate"]
        sections.append(
            f"\n--- CALCULATIONS (Python-verified) ---\n"
            f"Growth from {gr['from_year']} to {gr['to_year']}:\n"
            f"  {gr['formula']}"
        )

    return "\n\n".join(sections), citations, image_bytes_list


# ============================================================
# FINAL LLM ANSWER GENERATION
# ============================================================
ANSWER_SYSTEM_PROMPT = """You are a precise document intelligence assistant analyzing an annual report PDF.

RULES — read carefully:
1. Answer ONLY from the EVIDENCE blocks provided below the question. Do not use outside knowledge.
2. TABLE evidence IS authoritative document evidence. Values in TABLE rows are direct quotes from the PDF.
   - If a TABLE row says "Revenue | 482.6 | 406.5" that IS the answer. State it directly.
   - Do NOT say the evidence is insufficient when a TABLE clearly contains the answer.
3. Cite every fact with its source tag, e.g. [Annual Report, p.4 - TABLE].
4. The [HyDE passage] is a RETRIEVAL AID ONLY. NEVER cite it. NEVER treat its numbers as facts.
   - If the HyDE passage says "$1.87 billion" but the TABLE says "482.6 Cr", the TABLE is correct.
5. Currency and units: if the table header says "Crore" or the report mentions "Indian Rupees",
   state the figure in those units. Do not convert or speculate.
6. VERIFIED tag = value appears in multiple evidence sources. UNVERIFIED = single source.
7. Show calculation steps when referencing computed growth/ratio numbers.
8. Only say "Insufficient evidence" when the answer truly cannot be found anywhere in the evidence."""


def generate_answer(
    query: str, context: str, citations: List[str],
    image_bytes_list: List[bytes], hyde_doc: str
) -> str:
    full_prompt = (
        f"EVIDENCE FROM PDF:\n{context}\n\n"
        f"Citations:\n" +
        "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(citations)) +
        f"\n\nNOTE: The following is a RETRIEVAL AID ONLY - do NOT cite it as evidence:\n"
        f"[HyDE passage]: {hyde_doc[:150]}\n\n"
        f"Question: {query}\n\n"
        "Answer (cite sources, show calculations, note uncertainty):"
    )

    use_vision = (
        bool(image_bytes_list) and
        VISION_PROVIDER in ("openai", "anthropic")
    )

    try:
        if use_vision and VISION_PROVIDER == "openai" and OPENAI_API_KEY:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            content: List[dict] = [{"type": "text", "text": full_prompt}]
            for img_bytes in image_bytes_list[:3]:
                b64 = _image_to_b64(img_bytes)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
                })
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=1500, temperature=LLM_TEMPERATURE,
            )
            return resp.choices[0].message.content.strip()

        elif use_vision and VISION_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            msg_content: List[dict] = []
            for img_bytes in image_bytes_list[:3]:
                b64 = _image_to_b64(img_bytes)
                msg_content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64}
                })
            msg_content.append({"type": "text", "text": full_prompt})
            resp = client.messages.create(
                model=LLM_MODEL, max_tokens=1500,
                system=ANSWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": msg_content}]
            )
            return resp.content[0].text.strip()

        else:
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = _get_rag_llm()
            resp = llm.invoke([
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=full_prompt),
            ])
            return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

    except Exception as e:
        logger.error(f"Answer generation failed: {e}\n{traceback.format_exc()}")
        return f"[Error generating answer: {e}]"


# ============================================================
# MASTER PIPELINE
# ============================================================
def multimodal_rag(pdf_path: str, question: str) -> RAGResult:
    """
    Single entry point for the multimodal RAG pipeline.

    Args:
        pdf_path: Path to the PDF file.
        question: User's natural-language question.

    Returns:
        RAGResult with answer, citations, evidence, and latency stats.

    Example:
        result = multimodal_rag("annual_report.pdf", "What was revenue in FY2026?")
        print(result.answer)
    """
    t_total = time.time()
    timings: Dict[str, float] = {}

    fhash = _file_hash(pdf_path)

    # --- STEP 0: Query Analysis ---
    _dbg("QUERY ANALYSIS")
    t0 = time.time()
    qa = analyze_query(question)
    timings["query_analysis"] = time.time() - t0
    _info(f"Type: {qa.query_type}")
    _info(f"table={qa.requires_table} chart={qa.requires_chart} "
          f"image={qa.requires_image} diagram={qa.requires_diagram} "
          f"calc={qa.requires_calculation}")
    _info(f"Years: {qa.year_references}  Metrics: {qa.metric_references}")

    # --- STEP 1: Extract PDF ---
    _dbg("PDF EXTRACTION")
    t0 = time.time()
    elements = extract_pdf(pdf_path)
    timings["extraction"] = time.time() - t0
    by_type: Dict[str, int] = {}
    for e in elements:
        by_type[e.element_type.value] = by_type.get(e.element_type.value, 0) + 1
    _info(f"Elements: {by_type}")

    # --- STEP 2: Build Chunks ---
    t0 = time.time()
    chunks = build_multimodal_chunks(elements)
    timings["chunking"] = time.time() - t0
    chunks_by_id: Dict[str, MultimodalChunk] = {c.chunk_id: c for c in chunks}

    # --- STEP 3: Build Index ---
    _dbg("INDEX BUILDING")
    t0 = time.time()
    vstore, bm25_ret = build_index(chunks, fhash)
    timings["index_build"] = time.time() - t0

    # --- STEP 4: HyDE ---
    _dbg("HYDE GENERATION")
    t0 = time.time()
    hyde_doc = generate_hyde(question, fhash)
    timings["hyde"] = time.time() - t0
    _info(f"HyDE: {hyde_doc[:120]}...")

    # --- STEP 5: Query Enhancement ---
    _dbg("QUERY ENHANCEMENT")
    _info(f"Original:  {question}")
    _info(f"Expanded:  {qa.expanded_query}")
    _info(f"Keywords:  {qa.keyword_query}")

    # --- STEP 6: Dense Retrieval ---
    _dbg("TEXT DENSE RETRIEVAL")
    t0 = time.time()
    dense_results = retrieve_dense(
        [question, qa.expanded_query, hyde_doc],
        vstore, chunks_by_id, TOP_K_DENSE
    )
    timings["dense"] = time.time() - t0
    _info(f"Dense: {len(dense_results)} chunks")

    # --- STEP 7: BM25 ---
    _dbg("BM25 RETRIEVAL")
    t0 = time.time()
    bm25_results = retrieve_bm25(
        qa.keyword_query or question, bm25_ret, chunks_by_id
    )
    timings["bm25"] = time.time() - t0
    _info(f"BM25: {len(bm25_results)} chunks")

    # --- STEP 8: Table Retrieval ---
    _dbg("TABLE RETRIEVAL")
    t0 = time.time()
    table_results = retrieve_tables(qa, chunks) if (qa.requires_table or qa.year_references) else []
    timings["table"] = time.time() - t0
    _info(f"Tables: {len(table_results)} chunks")

    # --- STEP 9: Chart Retrieval ---
    _dbg("CHART RETRIEVAL")
    t0 = time.time()
    chart_results = retrieve_charts(qa, chunks) if qa.requires_chart else []
    timings["chart"] = time.time() - t0
    _info(f"Charts: {len(chart_results)} chunks")

    # --- STEP 10: Visual Retrieval ---
    _dbg("VISUAL RETRIEVAL")
    t0 = time.time()
    visual_results = retrieve_visual(qa, chunks) if qa.requires_image else []
    timings["visual"] = time.time() - t0
    _info(f"Visual: {len(visual_results)} chunks")

    # --- STEP 11: Structured Numeric ---
    t0 = time.time()
    structured_results = retrieve_structured_numeric(qa, chunks)
    timings["structured"] = time.time() - t0
    _info(f"Structured: {len(structured_results)} chunks")

    # --- STEP 12: RRF Fusion ---
    _dbg("RRF FUSION")
    t0 = time.time()
    fused = rrf_fusion([
        (dense_results, W_DENSE),
        (bm25_results, W_BM25),
        (table_results, W_TABLE),
        (chart_results, W_CHART),
        (visual_results, W_VISUAL),
        (structured_results, W_STRUCTURED),
    ])
    deduped = deduplicate(fused)
    timings["rrf_dedup"] = time.time() - t0
    _info(f"After RRF + dedup: {len(deduped)} unique candidates")

    # --- STEP 13: CrossEncoder Reranking ---
    _dbg("CROSSENCODER RERANKING")
    t0 = time.time()
    top_chunks = crossencoder_rerank(question, deduped, top_k=RERANK_TOP_K)
    timings["reranking"] = time.time() - t0
    for i, sc in enumerate(top_chunks):
        _info(f"  [{i+1}] score={sc.score:.4f} type={sc.chunk.element_type.value} "
              f"p={sc.chunk.page_number} src={sc.source}")

    # --- STEP 14: Parent Context Expansion ---
    top_chunks = expand_parent_context(top_chunks, chunks)

    # --- STEP 15: Evidence Verification ---
    _dbg("EVIDENCE VERIFICATION")
    evidence_blocks = verify_evidence(top_chunks, qa)
    verified = sum(1 for e in evidence_blocks if e.verified)
    _info(f"Verified: {verified}/{len(evidence_blocks)}")

    # --- STEP 16: Calculations ---
    _dbg("CALCULATIONS")
    calculations = perform_calculations(qa, evidence_blocks)
    if calculations.get("growth_rate"):
        _info(f"Computed: {calculations['growth_rate']['formula']}")

    # --- STEP 17: Build Context ---
    _dbg("MULTIMODAL CONTEXT")
    context_text, citations, image_bytes_list = build_final_context(
        evidence_blocks, calculations, qa
    )
    _info(f"Context: {len(context_text)} chars | Images: {len(image_bytes_list)} | "
          f"Citations: {len(citations)}")

    # --- STEP 18: Generate Answer ---
    _dbg("ANSWER GENERATION")
    t0 = time.time()
    answer = generate_answer(question, context_text, citations, image_bytes_list, hyde_doc)
    timings["llm"] = time.time() - t0
    timings["total"] = time.time() - t_total

    # --- Print answer ---
    _dbg("ANSWER")
    print(f"\n{answer}\n")

    if citations:
        print(f"\n{'='*60}")
        print("  SOURCES")
        print(f"{'='*60}")
        for c in citations:
            print(f"  {c}")

    _dbg("LATENCY TIMING")
    for stage, t in timings.items():
        _info(f"{stage:<22}: {t:.2f}s")

    return RAGResult(
        query=question,
        hyde_doc=hyde_doc,
        query_analysis={
            "query_type": qa.query_type,
            "requires_text": qa.requires_text,
            "requires_table": qa.requires_table,
            "requires_chart": qa.requires_chart,
            "requires_image": qa.requires_image,
            "requires_diagram": qa.requires_diagram,
            "requires_calculation": qa.requires_calculation,
            "year_references": qa.year_references,
            "metric_references": qa.metric_references,
        },
        retrieval_stats={
            "dense": len(dense_results),
            "bm25": len(bm25_results),
            "table": len(table_results),
            "chart": len(chart_results),
            "visual": len(visual_results),
            "structured": len(structured_results),
            "after_rrf_dedup": len(deduped),
            "after_rerank": len(top_chunks),
        },
        evidence=[
            {
                "rank": i + 1,
                "page": eb.chunk.page_number,
                "type": eb.chunk.element_type.value,
                "citation": eb.chunk.citation(),
                "rerank_score": round(eb.rerank_score, 4),
                "verified": eb.verified,
                "snippet": eb.chunk.text[:200],
            }
            for i, eb in enumerate(evidence_blocks)
        ],
        calculations=calculations,
        answer=answer,
        citations=citations,
        latency=timings,
    )


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys

    # NovaTech Solutions Ltd. - Annual Report FY2025-26
    # Test questions drawn from the actual PDF content
    NOVATECH_TEST_QUESTIONS = [
        "What was the total revenue of NovaTech Solutions in FY2025-26?",
        "What was the net profit in FY2025-26 and how does it compare to FY2024-25?",
        "What was the revenue growth from FY2024-25 to FY2025-26?",
        "Which business segment had the highest revenue in FY2025-26?",
        "Which segment grew the fastest and by how much?",
        "What is the EBITDA margin for FY2025-26 and FY2024-25?",
        "How many employees does NovaTech have and what is the women employee percentage?",
        "What are the key financial ratios for FY2025-26?",
        "What was the voluntary attrition rate and how did it change?",
        "Summarize the financial performance shown on page 4.",
    ]

    if len(sys.argv) >= 3:
        _pdf_path = sys.argv[1]
        _question = " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        _pdf_path = sys.argv[1]
        _question = input("\nAsk a question: ").strip()
    else:
        _pdf_path = PDF_PATH
        _question = QUESTION

    print("\n" + "=" * 70)
    print("  MULTIMODAL RAG  |  NovaTech Solutions Ltd. Annual Report 2025-26")
    print("=" * 70)
    print(f"  PDF:      {_pdf_path}")
    print(f"  Question: {_question}")
    print("=" * 70)

    if not Path(_pdf_path).exists():
        print(f"\nERROR: File not found: {_pdf_path}")
        print("  Usage:  python multimodal_rag.py <path_to.pdf> <your question>")
        sys.exit(1)

    result = multimodal_rag(_pdf_path, _question)

    # Print retrieval trace JSON
    print("\n" + "=" * 70)
    print("  RETRIEVAL TRACE")
    print("=" * 70)
    trace = {
        "query": result.query,
        "hyde": result.hyde_doc[:200] + "..." if len(result.hyde_doc) > 200 else result.hyde_doc,
        "query_analysis": result.query_analysis,
        "retrieval_stats": result.retrieval_stats,
        "calculations": result.calculations,
        "evidence_count": len(result.evidence),
        "citations": result.citations,
        "latency_seconds": {k: round(v, 2) for k, v in result.latency.items()},
    }
    print(json.dumps(trace, indent=2, default=str))

    # Interactive loop
    print("\n" + "-" * 70)
    print("  Suggested questions for NovaTech Annual Report:")
    for i, q in enumerate(NOVATECH_TEST_QUESTIONS, 1):
        print(f"  {i:>2}. {q}")
    print("\n  Type a question or 'quit' to exit.")
    print("-" * 70)

    while True:
        try:
            next_q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if next_q.lower() in ("quit", "exit", "q", ""):
            break
        # Allow selecting by number
        if next_q.isdigit():
            idx = int(next_q) - 1
            if 0 <= idx < len(NOVATECH_TEST_QUESTIONS):
                next_q = NOVATECH_TEST_QUESTIONS[idx]
                print(f"  > Selected: {next_q}")
        multimodal_rag(_pdf_path, next_q)

    print("\nSession complete.")
