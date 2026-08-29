"""
ingestion.py — Stage 1: Data Ingestion
=======================================
Loads documents from disk and extracts raw content.

Supported formats:
  ✅ PDF  (via PyMuPDF for text/images + pdfplumber for tables)
  ✅ Scanned PDF (OCR via Tesseract)

Outputs:
  List[RawDocument]  — one entry per extracted element

Usage:
    from multimodal.ingestion import ingest
    docs = ingest("annual_report.pdf")
    print(f"Extracted {len(docs)} elements")
"""

from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path
from typing import List, Optional

from .config import cfg
from .models import BBox, ElementType, RawDocument, TableData

logger = logging.getLogger("multimodal.ingestion")


# ══════════════════════════════════════════════════════════════════
# HELPER: Detect scanned page
# ══════════════════════════════════════════════════════════════════

def _is_scanned_page(page) -> bool:
    """
    Returns True if the page has almost no native text.
    Scanned pages are images — we need OCR to read them.
    """
    blocks = page.get_text("blocks")
    # Count characters in text blocks (type 0 = text, type 1 = image)
    char_count = sum(len(b[4]) for b in blocks if b[6] == 0)
    return char_count < 20   # Fewer than 20 chars = likely scanned


# ══════════════════════════════════════════════════════════════════
# HELPER: OCR a page
# ══════════════════════════════════════════════════════════════════

def _ocr_page(page, dpi: int = cfg.OCR_DPI) -> str:
    """
    Render the page as an image then run Tesseract OCR on it.
    Returns the recognized text string.
    """
    try:
        import pytesseract
        from PIL import Image

        # Render page → PNG bytes
        mat = page.get_pixmap(dpi=dpi)
        png_bytes = mat.tobytes("png")

        # Tesseract reads the PIL image
        img = Image.open(io.BytesIO(png_bytes))
        text = pytesseract.image_to_string(img, lang=cfg.OCR_LANG)
        return text.strip()
    except Exception as e:
        logger.debug(f"OCR failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# HELPER: Extract text elements from a PyMuPDF page
# ══════════════════════════════════════════════════════════════════

def _extract_text_from_page(page, page_num: int, doc_name: str) -> List[RawDocument]:
    """
    Extracts text spans from a PDF page using PyMuPDF's dict mode.
    Each span has font size, bold flag, bounding box.

    Heuristics used:
      - font_size >= 14 OR (bold + short text) → HEADING
      - long text (> 80 chars) → PARAGRAPH
      - everything else → TEXT
    """
    elements: List[RawDocument] = []
    blocks = page.get_text("dict", flags=11)["blocks"]

    for bi, block in enumerate(blocks):
        if block.get("type") != 0:   # Skip image blocks
            continue
        for span_group in block.get("lines", []):
            for span in span_group.get("spans", []):
                txt = span.get("text", "").strip()
                if not txt or len(txt) < 2:
                    continue

                font_size = span.get("size", 12)
                is_bold   = bool(span.get("flags", 0) & 2**4)

                # Classify element type
                if font_size >= 14 or (font_size >= 12 and is_bold and len(txt) < 120):
                    etype = ElementType.HEADING
                    hlevel = 1 if font_size >= 18 else 2 if font_size >= 14 else 3
                elif len(txt) > 80:
                    etype = ElementType.PARAGRAPH
                    hlevel = None
                else:
                    etype = ElementType.TEXT
                    hlevel = None

                bbox_raw = span.get("bbox", [0, 0, 0, 0])
                eid = f"page_{page_num}_text_{bi}_{id(span)}"

                elements.append(RawDocument(
                    element_id    = eid,
                    source_file   = "",           # Filled by ingest()
                    doc_name      = doc_name,
                    page_number   = page_num,
                    element_type  = etype,
                    content       = txt,
                    raw_text      = txt,
                    bbox          = BBox(*bbox_raw),
                    heading_level = hlevel,
                    metadata      = {"font_size": font_size, "is_bold": is_bold},
                ))
    return elements


# ══════════════════════════════════════════════════════════════════
# HELPER: Extract images from a PyMuPDF page
# ══════════════════════════════════════════════════════════════════

def _extract_images_from_page(page, page_num: int, doc_name: str) -> List[RawDocument]:
    """
    Extracts embedded images from a PDF page.
    Uses aspect ratio + size to guess if something is a CHART vs IMAGE.
    """
    if not cfg.EXTRACT_IMAGES:
        return []

    elements: List[RawDocument] = []
    try:
        img_list = page.get_images(full=True)
        for ii, img_info in enumerate(img_list[:cfg.MAX_IMAGES_PER_PAGE]):
            xref = img_info[0]
            try:
                base_img = page.parent.extract_image(xref)
            except Exception:
                continue

            img_bytes = base_img.get("image", b"")
            width     = base_img.get("width",  0)
            height    = base_img.get("height", 0)

            # Skip tiny images (icons, bullets, watermarks)
            if width < cfg.MIN_IMAGE_WIDTH or height < cfg.MIN_IMAGE_HEIGHT:
                continue

            # Convert to PNG for consistent handling
            try:
                from PIL import Image as PILImage
                pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                png_bytes = buf.getvalue()
            except Exception:
                png_bytes = img_bytes

            # Chart heuristic: landscape + large enough area
            aspect   = width / max(height, 1)
            is_chart = (0.5 < aspect < 4.0) and (width * height > 20_000)
            etype    = ElementType.CHART if is_chart else ElementType.IMAGE

            b64 = base64.b64encode(png_bytes).decode("utf-8")
            eid = f"page_{page_num}_img_{ii}"

            elements.append(RawDocument(
                element_id   = eid,
                source_file  = "",
                doc_name     = doc_name,
                page_number  = page_num,
                element_type = etype,
                content      = f"[{etype.value} on page {page_num}]",
                image_bytes  = png_bytes,
                image_b64    = b64,
                metadata     = {"width": width, "height": height},
            ))
    except Exception as e:
        logger.debug(f"Image extraction error p.{page_num}: {e}")
    return elements


# ══════════════════════════════════════════════════════════════════
# HELPER: Extract tables using pdfplumber
# ══════════════════════════════════════════════════════════════════

def _extract_tables(pdf_path: str, doc_name: str) -> List[RawDocument]:
    """
    Uses pdfplumber to extract structured tables.
    pdfplumber is better than PyMuPDF for table detection because it
    understands table borders and cell boundaries.

    Each table becomes one RawDocument with a TableData object attached.
    """
    elements: List[RawDocument] = []
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed — tables will be skipped")
        return elements

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                tables = page.extract_tables() or []

                for ti, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue

                    # Row 0 = headers, rows 1..N = data
                    headers = [str(c).strip() if c else "" for c in table[0]]
                    rows = [
                        [str(c).strip() if c else "" for c in row]
                        for row in table[1:]
                        if any(c for c in row)
                    ]
                    if not rows:
                        continue

                    # Auto-detect currency & units from cell text
                    all_text = " ".join(headers + [c for r in rows for c in r])
                    currency = next(
                        (s for s in ["Rs.", "INR", "USD", "$", "EUR", "GBP"] if s in all_text),
                        None
                    )
                    units = next(
                        (u for u in ["crore", "Cr", "million", "billion", "lakh", "%"]
                         if u.lower() in all_text.lower()),
                        None
                    )

                    td = TableData(headers=headers, rows=rows,
                                   currency=currency, units=units)
                    eid = f"page_{page_num}_table_{ti}"

                    elements.append(RawDocument(
                        element_id   = eid,
                        source_file  = pdf_path,
                        doc_name     = doc_name,
                        page_number  = page_num,
                        element_type = ElementType.TABLE,
                        content      = td.to_search_text(),
                        table_data   = td,
                        metadata     = {"row_count": len(rows), "col_count": len(headers)},
                    ))
    except Exception as e:
        logger.debug(f"pdfplumber error: {e}")
    return elements


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — ingest()
# ══════════════════════════════════════════════════════════════════

def ingest(source_path: str) -> List[RawDocument]:
    """
    Stage 1 entry point.

    Reads a PDF file and returns all extracted elements:
      - Text spans (TEXT, HEADING, PARAGRAPH)
      - Tables (TABLE)
      - Images & Charts (IMAGE, CHART)
      - OCR text for scanned pages (OCR_TEXT)

    Args:
        source_path: Path to the PDF file.

    Returns:
        List[RawDocument] sorted by (page_number, element_type)
    """
    path     = Path(source_path)
    doc_name = path.name
    elements: List[RawDocument] = []

    logger.info(f"[Stage 1] Ingesting: {source_path}")

    # ── PyMuPDF: text + images ─────────────────────────────────
    try:
        import fitz
        doc = fitz.open(source_path)
        logger.info(f"  PDF has {len(doc)} pages")

        for page_idx in range(len(doc)):
            page     = doc[page_idx]
            page_num = page_idx + 1

            if _is_scanned_page(page) and cfg.OCR_ENABLED:
                logger.info(f"  Page {page_num}: scanned — running OCR")
                ocr_text = _ocr_page(page)
                if ocr_text:
                    elements.append(RawDocument(
                        element_id   = f"page_{page_num}_ocr",
                        source_file  = source_path,
                        doc_name     = doc_name,
                        page_number  = page_num,
                        element_type = ElementType.OCR_TEXT,
                        content      = ocr_text,
                        raw_text     = ocr_text,
                        confidence   = 0.85,
                        metadata     = {"is_ocr": True},
                    ))
            else:
                text_els = _extract_text_from_page(page, page_num, doc_name)
                for el in text_els:
                    el.source_file = source_path
                elements.extend(text_els)

            img_els = _extract_images_from_page(page, page_num, doc_name)
            for el in img_els:
                el.source_file = source_path
            elements.extend(img_els)

        doc.close()
    except ImportError:
        logger.error("PyMuPDF (fitz) not installed — pip install pymupdf")
    except Exception as e:
        logger.error(f"PyMuPDF error: {e}")

    # ── pdfplumber: tables ─────────────────────────────────────
    table_els = _extract_tables(source_path, doc_name)
    elements.extend(table_els)
    logger.info(f"  Tables extracted: {len(table_els)}")

    # Sort by reading order: page then element type
    elements.sort(key=lambda e: (e.page_number, e.element_type.value))

    by_type: dict = {}
    for e in elements:
        by_type[e.element_type.value] = by_type.get(e.element_type.value, 0) + 1
    logger.info(f"  Total elements: {len(elements)} — {by_type}")

    return elements
