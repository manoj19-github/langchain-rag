"""
pipeline.py — Master Orchestrator
===================================
Connects all 12 stages into a single function call: `run(pdf_path, question)`.

This is the ONLY file you need to import from outside.
It wires every stage together and returns a FinalAnswer.

STAGE MAP (from the architecture diagram):
  Stage 1:  ingestion.py        → ingest(pdf_path)
  Stage 2:  preprocessing.py    → preprocess(raw_docs, file_hash)
  Stage 3:  chunking.py         → chunk_documents(processed_docs)
  Stage 4:  embeddings.py       → get_embedding_model()  [used by Stage 5]
  Stage 5:  indexing.py         → build_index(chunks, file_hash)
  Stage 6:  query_enhancement.py → enhance_query(question, file_hash)
  Stage 7:  retrieval.py        → hybrid_retrieve(eq, vstore, bm25, chunks)
  Stage 8:  context_assembly.py  → assemble_context(top_chunks, eq, all_chunks)
  Stage 9:  generation.py       → generate(question, context, eq)
  Stage 10: verification.py     → verify(answer, context, evidence, ...)
  Stage 11: (FinalAnswer.pretty_print) → formatted output + sources + confidence
  Stage 12: observability via logging + latency dict in FinalAnswer

Usage:
    # As a library
    from multimodal.pipeline import run
    result = run("report.pdf", "What was net profit?")
    result.pretty_print()

    # From command line
    python -m multimodal.pipeline
    python -m multimodal.pipeline "report.pdf" "What was revenue?"
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Import all stages ──────────────────────────────────────────────
from .config import cfg
from .models import FinalAnswer
from .ingestion import ingest
from .preprocessing import preprocess
from .chunking import chunk_documents
from .indexing import build_index, build_chunk_lookup
from .query_enhancement import enhance_query
from .retrieval import hybrid_retrieve
from .context_assembly import assemble_context
from .generation import generate
from .verification import verify

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.DEBUG if cfg.DEBUG else logging.WARNING,
    format = "%(message)s",
)
logger = logging.getLogger("multimodal.pipeline")


# ══════════════════════════════════════════════════════════════════
# CACHE HELPERS (for Stage 1–3 results)
# ══════════════════════════════════════════════════════════════════

def _file_hash(path: str) -> str:
    """SHA256 fingerprint of the PDF — used as cache key prefix."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


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
# STAGE BANNER (printed when DEBUG=True)
# ══════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    if cfg.DEBUG:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════════

def run(pdf_path: str, question: str) -> FinalAnswer:
    """
    Run the complete 12-stage multimodal RAG pipeline.

    Args:
        pdf_path: Path to the PDF document
        question: User's natural-language question

    Returns:
        FinalAnswer with answer, citations, confidence, and latency

    Example:
        result = run("annual_report.pdf", "What was EBITDA in FY2025-26?")
        result.pretty_print()
        print(result.confidence_score)  # e.g., 0.85
    """
    t_pipeline = time.time()
    timings:   Dict[str, float] = {}

    # ── Validate PDF ──────────────────────────────────────────────
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    fhash = _file_hash(pdf_path)
    logger.info(f"\n{'═'*60}")
    logger.info(f"  MULTIMODAL RAG PIPELINE")
    logger.info(f"{'═'*60}")
    logger.info(f"  PDF:      {pdf_path}")
    logger.info(f"  Question: {question}")
    logger.info(f"  Hash:     {fhash}")

    # ════════════════════════════════════════════════════════════
    # STAGE 1: Data Ingestion
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 1 — Data Ingestion")
    t = time.time()

    raw_cache_key = f"raw_{fhash}"
    raw_docs = _cache_load(raw_cache_key)
    if raw_docs is not None:
        logger.info(f"  Loaded {len(raw_docs)} raw elements from cache ✓")
    else:
        raw_docs = ingest(pdf_path)
        _cache_save(raw_cache_key, raw_docs)

    timings["1_ingestion"] = time.time() - t
    logger.info(f"  → {len(raw_docs)} elements  [{timings['1_ingestion']:.2f}s]")

    # ════════════════════════════════════════════════════════════
    # STAGE 2: Preprocessing
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 2 — Preprocessing")
    t = time.time()

    proc_cache_key = f"processed_{fhash}"
    processed = _cache_load(proc_cache_key)
    if processed is not None:
        logger.info(f"  Loaded {len(processed)} processed docs from cache ✓")
    else:
        processed = preprocess(raw_docs, file_hash=fhash)
        _cache_save(proc_cache_key, processed)

    timings["2_preprocessing"] = time.time() - t
    logger.info(f"  → {len(processed)} documents  [{timings['2_preprocessing']:.2f}s]")

    # ════════════════════════════════════════════════════════════
    # STAGE 3: Chunking
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 3 — Chunking")
    t = time.time()

    chunk_cache_key = f"chunks_{fhash}"
    chunks = _cache_load(chunk_cache_key)
    if chunks is not None:
        logger.info(f"  Loaded {len(chunks)} chunks from cache ✓")
    else:
        chunks = chunk_documents(processed)
        _cache_save(chunk_cache_key, chunks)

    chunks_by_id = build_chunk_lookup(chunks)
    timings["3_chunking"] = time.time() - t
    logger.info(f"  → {len(chunks)} chunks  [{timings['3_chunking']:.2f}s]")

    # ════════════════════════════════════════════════════════════
    # STAGE 4 + 5: Embedding + Indexing
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 4+5 — Embedding & Indexing")
    t = time.time()

    vstore, bm25_retriever = build_index(chunks, fhash)

    timings["4_5_indexing"] = time.time() - t
    logger.info(f"  → Index ready  [{timings['4_5_indexing']:.2f}s]")

    # ════════════════════════════════════════════════════════════
    # STAGE 6: Query Enhancement
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 6 — Query Enhancement")
    t = time.time()

    eq = enhance_query(question, file_hash=fhash)

    timings["6_query_enhancement"] = time.time() - t
    logger.info(f"  Type: {eq.query_type} | table={eq.requires_table} "
                f"chart={eq.requires_chart} calc={eq.requires_calc}")
    logger.info(f"  [{timings['6_query_enhancement']:.2f}s]")

    # ════════════════════════════════════════════════════════════
    # STAGE 7: Hybrid Retrieval
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 7 — Hybrid Retrieval")
    t = time.time()

    top_chunks, retrieval_stats = hybrid_retrieve(
        eq, vstore, bm25_retriever, chunks, chunks_by_id
    )

    timings["7_retrieval"] = time.time() - t
    logger.info(f"  → {len(top_chunks)} final chunks  [{timings['7_retrieval']:.2f}s]")

    # ════════════════════════════════════════════════════════════
    # STAGE 8: Context Assembly
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 8 — Context Assembly")
    t = time.time()

    context, evidence, calculations = assemble_context(top_chunks, eq, chunks)

    timings["8_context"] = time.time() - t
    logger.info(f"  → {context.char_count} chars  [{timings['8_context']:.2f}s]")

    # ════════════════════════════════════════════════════════════
    # STAGE 9: LLM Generation
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 9 — LLM Generation")
    t = time.time()

    answer_draft = generate(question, context, eq)

    timings["9_generation"] = time.time() - t
    logger.info(f"  → Answer: {len(answer_draft)} chars  [{timings['9_generation']:.2f}s]")

    # ════════════════════════════════════════════════════════════
    # STAGE 10: Verification
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 10 — Verification")
    t = time.time()

    timings["total"] = time.time() - t_pipeline

    final = verify(
        answer_draft    = answer_draft,
        context         = context,
        evidence        = evidence,
        eq              = eq,
        calculations    = calculations,
        retrieval_stats = retrieval_stats,
        latency         = timings,
    )

    timings["10_verification"] = time.time() - t
    timings["total"] = time.time() - t_pipeline

    # ════════════════════════════════════════════════════════════
    # STAGE 11: Final Output
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 11 — Final Output")
    final.pretty_print()

    # ════════════════════════════════════════════════════════════
    # STAGE 12: Observability
    # ════════════════════════════════════════════════════════════
    _banner("STAGE 12 — Observability")
    print("\n  RETRIEVAL TRACE (JSON):")
    trace = {
        "question":       final.question,
        "query_type":     final.query_type,
        "hyde":           final.hyde_doc[:150] + "...",
        "retrieval_stats": final.retrieval_stats,
        "confidence":     final.confidence_score,
        "verified":       final.verified,
        "warnings":       final.warnings,
        "latency":        {k: round(v, 2) for k, v in final.latency.items()},
    }
    print(json.dumps(trace, indent=2, default=str))

    return final


# ══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════

# Default PDF and test questions for NovaTech Annual Report FY2025-26
_DEFAULT_PDF = r"data/pdf/dummy_company_annual_report_2025_26 (1).pdf"
_DEFAULT_Q   = "What was the total revenue of NovaTech Solutions in FY2025-26?"

_NOVATECH_QUESTIONS = [
    "What was the total revenue of NovaTech Solutions in FY2025-26?",
    "What was the net profit in FY2025-26 and how does it compare to FY2024-25?",
    "What was the revenue growth from FY2024-25 to FY2025-26?",
    "Which business segment had the highest revenue?",
    "Which segment grew the fastest and by how much?",
    "What is the EBITDA margin for FY2025-26?",
    "How many employees and what is the women employee percentage?",
    "What are the key financial ratios for FY2025-26?",
    "What was the voluntary attrition rate?",
    "Summarize the financial performance from the Financial Highlights section.",
]


if __name__ == "__main__":
    # Usage:
    #   python -m multimodal.pipeline
    #   python -m multimodal.pipeline "path/to.pdf" "Your question?"

    if len(sys.argv) >= 3:
        _pdf  = sys.argv[1]
        _q    = " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        _pdf  = sys.argv[1]
        _q    = input("\nEnter your question: ").strip()
    else:
        _pdf  = _DEFAULT_PDF
        _q    = _DEFAULT_Q

    print("\n" + "=" * 70)
    print("  MULTIMODAL RAG  |  12-Stage Modular Pipeline")
    print("=" * 70)
    print(f"  PDF:      {_pdf}")
    print(f"  Question: {_q}")
    print("=" * 70)

    if not Path(_pdf).exists():
        print(f"\n❌ ERROR: File not found: {_pdf}")
        sys.exit(1)

    result = run(_pdf, _q)

    # Interactive follow-up
    print("\n" + "-" * 70)
    print("  NovaTech suggested questions (type 1–10 to select):")
    for i, q in enumerate(_NOVATECH_QUESTIONS, 1):
        print(f"  {i:>2}. {q}")
    print("\n  Type a question or 'quit' to exit.")
    print("-" * 70)

    while True:
        try:
            nq = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if nq.lower() in ("quit", "exit", "q", ""):
            break
        if nq.isdigit():
            idx = int(nq) - 1
            if 0 <= idx < len(_NOVATECH_QUESTIONS):
                nq = _NOVATECH_QUESTIONS[idx]
                print(f"  ▶ {nq}")
        run(_pdf, nq)

    print("\nSession complete.")
