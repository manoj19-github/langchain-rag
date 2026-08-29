"""
query_enhancement.py — Stage 6: Query Enhancement
===================================================
Takes the user's raw question and produces multiple enhanced versions
for better retrieval recall.

Techniques used:
  🔮 HyDE (Hypothetical Document Embedding)
       Generate a fake "answer passage" → embed it → find similar real chunks
       Why? The fake answer uses the same vocabulary as the actual document.

  🔍 Query Rewriting
       Add year/metric terms extracted from the question.
       "What was revenue?" → "What was revenue? FY2025 revenue"

  🔑 Keyword Extraction
       Remove stopwords → keyword-only query for BM25.
       "What was the revenue in FY2025?" → "revenue fy2025"

  🎯 Query Type Detection
       Is this a financial/table/chart/text question?
       Tells the retriever which paths to activate.

Inputs:  question (str), file_hash (str for HyDE caching)
Outputs: EnhancedQuery

Usage:
    from multimodal.query_enhancement import enhance_query
    eq = enhance_query("What was net profit in FY2025-26?", "abc123")
    print(eq.hyde_doc)
    print(eq.requires_table)   # True
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional

from .config import cfg
from .models import EnhancedQuery

logger = logging.getLogger("multimodal.query_enhancement")

# ── LLM singleton ──
_hyde_llm = None


# ══════════════════════════════════════════════════════════════════
# REGEX PATTERNS for query analysis
# ══════════════════════════════════════════════════════════════════

# Matches: FY2025, FY25-26, FY2025-26, 2025, 2024
_YEAR_RE   = re.compile(r"\bfy[\s\']?\d{2,4}\b|\b20\d\d\b|\b19\d\d\b", re.IGNORECASE)
# Matches financial metrics
_METRIC_RE = re.compile(
    r"\b(revenue|profit|margin|ebitda|eps|cagr|growth|sales|income|cost|expense|turnover|loss)\b",
    re.IGNORECASE
)

# Keywords that suggest a TABLE query
_TABLE_KW = {"revenue", "profit", "margin", "growth", "percentage", "fy",
             "financial", "segment", "total", "amount", "annual", "crore",
             "million", "billion", "ratio", "ebitda", "earnings", "income"}

# Keywords that suggest a CHART query
_CHART_KW = {"chart", "graph", "trend", "visual", "highest", "lowest",
             "bar", "line", "pie", "distribution", "figure", "plot", "increase", "decrease"}

# Keywords that suggest an IMAGE query
_IMAGE_KW = {"image", "photo", "picture", "screenshot", "illustrate"}

# Keywords that suggest a DIAGRAM query
_DIAGRAM_KW = {"diagram", "architecture", "flow", "process", "system",
               "structure", "hierarchy", "pipeline", "network"}

# Keywords that suggest CALCULATION needed
_CALC_KW = {"calculate", "compute", "how much", "how many", "sum",
            "average", "growth rate", "difference", "change"}

# English stopwords (removed for keyword-only query)
_STOPWORDS = {"the", "a", "an", "in", "of", "for", "was", "is", "what",
              "how", "which", "where", "when", "who", "did", "do", "does",
              "and", "or", "to", "from", "with", "by", "as", "at", "it"}


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
# HyDE — Hypothetical Document Embedding
# ══════════════════════════════════════════════════════════════════

def _build_hyde_llm():
    """Build the LLM used for HyDE generation (singleton)."""
    global _hyde_llm
    if _hyde_llm is None:
        if cfg.LLM_PROVIDER == "openai":
            from langchain_openai import ChatOpenAI
            _hyde_llm = ChatOpenAI(
                model=cfg.HYDE_MODEL, temperature=cfg.HYDE_TEMPERATURE,
                api_key=cfg.OPENAI_API_KEY
            )
        elif cfg.LLM_PROVIDER == "anthropic":
            from langchain_anthropic import ChatAnthropic
            _hyde_llm = ChatAnthropic(
                model=cfg.HYDE_MODEL, temperature=cfg.HYDE_TEMPERATURE,
                api_key=cfg.ANTHROPIC_API_KEY
            )
        else:
            from langchain_ollama import ChatOllama
            _hyde_llm = ChatOllama(
                model=cfg.HYDE_MODEL, temperature=cfg.HYDE_TEMPERATURE,
                base_url=cfg.OLLAMA_BASE_URL
            )
    return _hyde_llm


def _generate_hyde(question: str, file_hash: str) -> str:
    """
    Ask the LLM to write a hypothetical answer passage for the question.

    This passage is used ONLY for embedding-based retrieval.
    It is NEVER shown to the user as an answer.

    The key insight: a fake answer uses the same words as the real document,
    so embedding the fake answer gives better semantic similarity matches.

    Example:
        Question: "What was total revenue in FY2025-26?"
        HyDE:     "NovaTech reported total revenue of ₹482.6 crore in FY2025-26,
                   representing 18.7% growth over the previous year..."
    """
    if not cfg.HYDE_ENABLED:
        return question

    # Cache by question + file hash
    ckey = f"hyde_{file_hash}_{hashlib.sha256(question.encode()).hexdigest()[:12]}"
    cached = _cache_load(ckey)
    if cached:
        logger.info("  HyDE: loaded from cache ✓")
        return cached

    prompt = (
        "You are a retrieval enhancement component in a RAG system.\n"
        "Write a concise hypothetical passage that WOULD contain the answer to this question.\n"
        "Write it as if excerpted from the source document.\n"
        "Use domain-appropriate terminology, numbers, and entities.\n"
        "This text is ONLY for embedding — NOT the final answer.\n\n"
        f"Question: {question}\n\n"
        "Hypothetical document passage (2-4 sentences):"
    )
    try:
        from langchain_core.messages import HumanMessage
        llm  = _build_hyde_llm()
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        if text:
            _cache_save(ckey, text)
            return text
    except Exception as e:
        logger.debug(f"HyDE generation failed: {e}")

    return question   # Fall back to original question if LLM fails


# ══════════════════════════════════════════════════════════════════
# QUERY ANALYSIS
# ══════════════════════════════════════════════════════════════════

def _analyze(question: str):
    """Determine what modalities and data types this question needs."""
    ql    = question.lower()
    words = set(re.findall(r"\b\w+\b", ql))
    years   = _YEAR_RE.findall(question)
    metrics = _METRIC_RE.findall(question)

    req_table   = bool(words & _TABLE_KW) or bool(years)
    req_chart   = bool(words & _CHART_KW)
    req_image   = bool(words & _IMAGE_KW)
    req_diagram = bool(words & _DIAGRAM_KW)
    req_calc    = bool(words & _CALC_KW) or len(years) >= 2

    if req_chart:       qtype = "chart"
    elif req_diagram:   qtype = "diagram"
    elif req_image:     qtype = "image"
    elif req_table:     qtype = "financial" if (words & {"revenue","profit","margin","ebitda"}) else "table"
    else:               qtype = "text"

    return qtype, req_table, req_chart, req_image, req_diagram, req_calc, years, metrics


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — enhance_query()
# ══════════════════════════════════════════════════════════════════

def enhance_query(question: str, file_hash: str = "") -> EnhancedQuery:
    """
    Stage 6 entry point.

    Produces an EnhancedQuery with multiple variants of the question:
      - hyde_doc:      Hypothetical document for dense retrieval
      - expanded:      Original + year/metric terms
      - keyword_only:  Just important words (for BM25)
      - table_variant: Appended with table-focused suffix
      - chart_variant: Appended with chart-focused suffix

    Args:
        question:  The user's raw question string
        file_hash: PDF file hash (for HyDE cache key)

    Returns:
        EnhancedQuery with all variants filled in
    """
    logger.info(f"[Stage 6] Enhancing query: {question[:80]}")

    (qtype, req_table, req_chart, req_image,
     req_diagram, req_calc, years, metrics) = _analyze(question)

    logger.info(f"  Type: {qtype} | table={req_table} chart={req_chart} "
                f"years={years} metrics={metrics}")

    # ── HyDE ────────────────────────────────────────────────────
    hyde_doc = _generate_hyde(question, file_hash)
    logger.info(f"  HyDE: {hyde_doc[:100]}...")

    # ── Expanded query (original + extracted terms) ──────────────
    expanded = question
    if years:
        expanded += " " + " ".join(years)
    if metrics:
        expanded += " " + " ".join(set(m.lower() for m in metrics))

    # ── Keyword-only query (BM25) ────────────────────────────────
    words    = set(re.findall(r"\b\w+\b", question.lower()))
    keywords = sorted(words - _STOPWORDS)
    keyword_only = " ".join(keywords)

    return EnhancedQuery(
        original          = question,
        query_type        = qtype,
        hyde_doc          = hyde_doc,
        expanded          = expanded,
        keyword_only      = keyword_only,
        table_variant     = f"{question} financial table data rows columns",
        chart_variant     = f"{question} chart graph trend visual figure",
        year_references   = years,
        metric_references = [m.lower() for m in metrics],
        requires_table    = req_table,
        requires_chart    = req_chart,
        requires_image    = req_image,
        requires_diagram  = req_diagram,
        requires_calc     = req_calc,
    )
