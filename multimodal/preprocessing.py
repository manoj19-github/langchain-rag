"""
preprocessing.py — Stage 2: Advanced Preprocessing
====================================================
Cleans and enriches raw extracted elements before chunking.

What this stage does:
  ✅ Text cleaning & normalization (whitespace, encoding)
  ✅ Language detection
  ✅ Duplicate detection (removes repeated boilerplate)
  ✅ Vision enrichment (describes charts/images using a vision LLM)
  ✅ Basic NER (Named Entity Recognition via regex — finds companies,
     years, currencies, percentages)
  ✅ Metadata extraction (date, source, author)

Inputs:  List[RawDocument]
Outputs: List[ProcessedDocument]

Usage:
    from multimodal.preprocessing import preprocess
    processed = preprocess(raw_docs, file_hash="abc123")
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import urllib.request
from typing import List, Optional, Set

from .config import cfg
from .models import ElementType, ProcessedDocument, RawDocument

logger = logging.getLogger("multimodal.preprocessing")


# ══════════════════════════════════════════════════════════════════
# CACHE HELPERS (shared utility — used here for vision cache)
# ══════════════════════════════════════════════════════════════════

import pickle
from pathlib import Path


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
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """
    Normalize whitespace, fix common encoding artifacts.

    Examples:
        "Revenue  n482.6 Cr"  → "Revenue n482.6 Cr"
        "FY2025\u201326"      → "FY2025-26"
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Fix common Unicode dashes/quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Remove null bytes
    text = text.replace("\x00", "")
    return text


# ══════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION (lightweight — no external library needed)
# ══════════════════════════════════════════════════════════════════

# Common Hindi/Devanagari Unicode range
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

def _detect_language(text: str) -> str:
    """
    Simple language detection.
    Returns "hi" for Hindi/Devanagari, "en" otherwise.
    For production use langdetect or lingua instead.
    """
    if _DEVANAGARI_RE.search(text):
        return "hi"
    return "en"


# ══════════════════════════════════════════════════════════════════
# NAMED ENTITY RECOGNITION (regex-based)
# ══════════════════════════════════════════════════════════════════

_YEAR_RE      = re.compile(r"\bFY[\s\']?\d{2,4}\b|\b20\d\d\b|\b19\d\d\b", re.IGNORECASE)
_CURRENCY_RE  = re.compile(r"₹|Rs\.?|INR|USD|\$|EUR|GBP", re.IGNORECASE)
_PERCENT_RE   = re.compile(r"\d+\.?\d*\s*%")
_CRORE_RE     = re.compile(r"\d+\.?\d*\s*(crore|cr|Cr)\b", re.IGNORECASE)
_COMPANY_RE   = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Inc|Corp|Pvt|LLP|PLC)\b")

def _extract_entities(text: str) -> List[str]:
    """
    Extract named entities using regex patterns.
    Returns list of entity strings.
    """
    entities = []
    entities.extend(_YEAR_RE.findall(text))
    entities.extend(_PERCENT_RE.findall(text))
    entities.extend(_CRORE_RE.findall(text))
    entities.extend(_COMPANY_RE.findall(text))
    return list(set(entities))   # Deduplicate


# ══════════════════════════════════════════════════════════════════
# DUPLICATE DETECTION
# ══════════════════════════════════════════════════════════════════

def _content_hash(text: str) -> str:
    """SHA256 of cleaned text — for duplicate detection."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════
# VISION ENRICHMENT (describe charts/images with LLM)
# ══════════════════════════════════════════════════════════════════

_CHART_PROMPT = """Analyze this chart from an annual report PDF. Return JSON:
{
  "title": "chart title or Unknown",
  "chart_type": "bar|line|pie|stacked|area|other",
  "x_axis": "label", "y_axis": "label", "unit": "unit if visible",
  "data_points": [{"label": "FY22", "value": "300"}],
  "key_observations": "one sentence summary",
  "visible_text": "all visible numbers and labels"
}
Never invent data not visible in the image."""

_IMAGE_PROMPT = """Describe this image from a PDF report.
Include: what is shown, all visible text, numbers, labels.
Be specific, factual, and concise."""

_DIAGRAM_PROMPT = """Analyze this diagram from a PDF.
List all visible nodes, arrows, and relationships.
Format: NodeA → NodeB → NodeC
Then write a one-paragraph description."""


def _describe_with_vision(image_bytes: bytes, element_type: ElementType) -> str:
    """
    Send an image to the configured vision model and return a text description.

    Falls back to "[Vision unavailable]" if:
      - No vision model is configured
      - The API call fails
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    if element_type == ElementType.CHART:
        prompt = _CHART_PROMPT
    elif element_type == ElementType.DIAGRAM:
        prompt = _DIAGRAM_PROMPT
    else:
        prompt = _IMAGE_PROMPT

    try:
        if cfg.VISION_PROVIDER == "openai" and cfg.OPENAI_API_KEY:
            import openai
            client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=cfg.VISION_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}", "detail": "high"
                    }},
                ]}],
                max_tokens=600, temperature=0.0,
            )
            return resp.choices[0].message.content.strip()

        elif cfg.VISION_PROVIDER == "ollama":
            payload = json.dumps({
                "model":   cfg.VISION_MODEL,
                "prompt":  prompt,
                "images":  [b64],
                "stream":  False,
                "options": {"temperature": 0.0},
            }).encode()
            req = urllib.request.Request(
                f"{cfg.OLLAMA_BASE_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                raw = data.get("response", "").strip()
                # For charts: parse JSON and reformat as readable text
                if element_type == ElementType.CHART:
                    try:
                        d = json.loads(raw)
                        parts = [
                            f"Chart: {d.get('title', 'Unknown')}",
                            f"Type: {d.get('chart_type', '')}",
                            f"X-axis: {d.get('x_axis', '')}",
                            f"Y-axis: {d.get('y_axis', '')}",
                            f"Observations: {d.get('key_observations', '')}",
                            f"Data: " + ", ".join(
                                f"{p.get('label','?')}={p.get('value','?')}"
                                for p in d.get("data_points", [])
                            ),
                        ]
                        return "\n".join(p for p in parts if p.split(": ", 1)[-1])
                    except (json.JSONDecodeError, AttributeError):
                        return raw
                return raw

    except Exception as e:
        logger.debug(f"Vision description failed for {element_type}: {e}")

    return "[Vision description unavailable]"


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — preprocess()
# ══════════════════════════════════════════════════════════════════

def preprocess(
    raw_docs: List[RawDocument],
    file_hash: str = "",
) -> List[ProcessedDocument]:
    """
    Stage 2 entry point.

    For each RawDocument:
      1. Clean and normalize text
      2. Detect language
      3. Flag duplicates
      4. Extract named entities
      5. Generate vision descriptions for charts/images

    Args:
        raw_docs:  Output of Stage 1 (ingestion.py)
        file_hash: PDF file hash — used as prefix for vision cache keys

    Returns:
        List[ProcessedDocument]
    """
    logger.info(f"[Stage 2] Preprocessing {len(raw_docs)} elements")
    processed: List[ProcessedDocument] = []
    seen_hashes: Set[str] = set()

    visual_types = {ElementType.CHART, ElementType.IMAGE, ElementType.DIAGRAM}

    for doc in raw_docs:
        # 1. Clean text
        clean = _clean_text(doc.content)

        # 2. Detect language
        lang = _detect_language(clean)

        # 3. Duplicate detection (skip repeated boilerplate)
        chash = _content_hash(clean)
        is_dup = chash in seen_hashes
        if not is_dup:
            seen_hashes.add(chash)

        # 4. Extract entities
        entities = _extract_entities(clean)

        # 5. Vision enrichment for charts/images
        vision_desc: Optional[str] = None
        if doc.element_type in visual_types and doc.image_bytes:
            cache_key = f"vision_{file_hash}_{doc.element_id}"
            vision_desc = _cache_load(cache_key)
            if vision_desc is None:
                logger.info(f"  Vision: describing {doc.element_id}...")
                vision_desc = _describe_with_vision(doc.image_bytes, doc.element_type)
                _cache_save(cache_key, vision_desc)
            # Replace placeholder content with real description
            clean = vision_desc if vision_desc else clean

        processed.append(ProcessedDocument(
            element_id         = doc.element_id,
            source_file        = doc.source_file,
            doc_name           = doc.doc_name,
            page_number        = doc.page_number,
            element_type       = doc.element_type,
            content            = clean,
            raw_text           = doc.raw_text,
            image_bytes        = doc.image_bytes,
            image_b64          = doc.image_b64,
            table_data         = doc.table_data,
            bbox               = doc.bbox,
            heading_level      = doc.heading_level,
            vision_description = vision_desc,
            named_entities     = entities,
            language           = lang,
            is_duplicate       = is_dup,
            confidence         = doc.confidence,
            metadata           = {**doc.metadata, "clean_hash": chash},
        ))

    dup_count = sum(1 for p in processed if p.is_duplicate)
    logger.info(f"  Duplicates flagged: {dup_count}")
    logger.info(f"  Preprocessing complete: {len(processed)} documents")
    return processed
