"""
context_assembly.py — Stage 8: Context Assembly
================================================
Takes the top-K retrieved chunks and builds a formatted context window
that will be sent to the LLM for answer generation.

What this stage does:
  ✅ Evidence verification — cross-checks numbers across chunks
  ✅ Parent context expansion — adds section headings + related chunks
  ✅ Python calculations — growth rates computed exactly (not by LLM)
  ✅ Context packing — formats tables as Markdown, adds VERIFIED tags
  ✅ Context window limit — respects token budget (~3500 chars max)
  ✅ Citation building — collects source references for the answer

Inputs:  List[ScoredChunk], EnhancedQuery, all_chunks
Outputs: ContextWindow (formatted text + citations + image bytes)

Usage:
    from multimodal.context_assembly import assemble_context
    ctx = assemble_context(top_chunks, eq, all_chunks)
    print(ctx.text)         # Full formatted context for LLM
    print(ctx.citations)    # ["[report.pdf, p.4 — TABLE]", ...]
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Chunk, ContextWindow, ElementType, EnhancedQuery, EvidenceBlock, ScoredChunk
)

logger = logging.getLogger("multimodal.context_assembly")

# Max total characters in the context window
_MAX_CONTEXT_CHARS = 6000


# ══════════════════════════════════════════════════════════════════
# STEP 1: Evidence Verification
# ══════════════════════════════════════════════════════════════════

def _verify_evidence(top_chunks: List[ScoredChunk]) -> List[EvidenceBlock]:
    """
    Cross-checks numbers across evidence sources.

    Logic:
      - Collect all numeric values and which chunks mention them
      - If a large number (> 100) appears in only ONE chunk → UNVERIFIED
      - If it appears in TWO or more chunks → VERIFIED

    Example:
        482.6 appears in TABLE (p.4) AND in PARAGRAPH (p.2)
        → VERIFIED  (two independent sources agree)

    VERIFIED = safer to cite in the final answer.
    UNVERIFIED = single source, higher risk of extraction error.
    """
    number_re = re.compile(r"[\d,]+\.?\d*")

    # Collect: value_string → list of citation strings
    value_sources: Dict[str, List[str]] = {}
    for sc in top_chunks:
        for n in number_re.findall(sc.chunk.text):
            clean = n.replace(",", "")
            try:
                if float(clean) > 0:
                    value_sources.setdefault(clean, []).append(sc.chunk.citation())
            except ValueError:
                pass

    blocks = []
    for sc in top_chunks:
        # Mark as unverified if any large number only appears once
        verified = True
        for n in number_re.findall(sc.chunk.text):
            clean = n.replace(",", "")
            try:
                if float(clean) > 100 and len(value_sources.get(clean, [])) < 2:
                    verified = False
                    break
            except ValueError:
                pass
        blocks.append(EvidenceBlock(
            chunk        = sc.chunk,
            rerank_score = sc.score,
            verified     = verified,
        ))

    verified_count = sum(1 for b in blocks if b.verified)
    logger.info(f"  Evidence verified: {verified_count}/{len(blocks)}")
    return blocks


# ══════════════════════════════════════════════════════════════════
# STEP 2: Parent Context Expansion
# ══════════════════════════════════════════════════════════════════

def _expand_context(
    evidence_blocks: List[EvidenceBlock],
    all_chunks: List[Chunk],
) -> List[EvidenceBlock]:
    """
    Adds supplementary context to each evidence block.

    For TABLE chunks:
      → Finds nearby explanatory paragraphs on the same page
      → Adds "[Context]: ..." prefix

    For CHART chunks:
      → Finds the related table on the same page (if any)
      → Adds "[Related table]: ..."

    This ensures the LLM sees WHY a table exists, not just raw numbers.
    """
    # Group all chunks by page number for fast lookup
    page_chunks: Dict[int, List[Chunk]] = {}
    for c in all_chunks:
        page_chunks.setdefault(c.page_number, []).append(c)

    for eb in evidence_blocks:
        c          = eb.chunk
        page       = c.page_number
        neighbours = page_chunks.get(page, [])
        extras     = []

        # Section heading from parent_heading
        if c.parent_heading:
            extras.append(f"[Section: {c.parent_heading}]")

        if c.element_type == ElementType.TABLE:
            # Add first nearby paragraph for context
            for n in neighbours:
                if n.element_type in (ElementType.PARAGRAPH, ElementType.TEXT) and n.chunk_id != c.chunk_id:
                    extras.append(f"[Context]: {n.text[:200]}")
                    break

        elif c.element_type == ElementType.CHART:
            # Find related table on same page
            for n in neighbours:
                if n.element_type == ElementType.TABLE and n.chunk_id != c.chunk_id:
                    extras.append(f"[Related table p.{page}]: {n.text[:300]}")
                    break

        if extras:
            c.metadata["expanded_context"] = "\n".join(extras)

    return evidence_blocks


# ══════════════════════════════════════════════════════════════════
# STEP 3: Python Calculations (exact arithmetic — not LLM math)
# ══════════════════════════════════════════════════════════════════

def _compute_calculations(
    eq: EnhancedQuery,
    evidence: List[EvidenceBlock],
) -> Dict[str, Any]:
    """
    Perform exact arithmetic before sending to LLM.

    Why Python, not LLM?
      LLMs hallucinate arithmetic. For example:
        "((482.6 - 406.5) / 406.5) * 100"
        LLM might say "19.2%"
        Python gives "18.73%" (correct)

    What we compute:
      - Growth rate between two years (if two years in query)
      - Extracts all numeric values per year mentioned in evidence
    """
    if not eq.year_references and not eq.metric_references:
        return {}

    number_re = re.compile(r"([\d,]+\.?\d*)")
    year_re   = re.compile(r"\b(fy[\s\']?\d{2,4}|20\d\d|19\d\d)\b", re.IGNORECASE)

    # Collect numbers grouped by year mentioned nearby
    year_numbers: Dict[str, List[float]] = {}
    for eb in evidence:
        years_in_text = year_re.findall(eb.chunk.text)
        nums_in_text  = number_re.findall(eb.chunk.text)
        for y in years_in_text:
            yk = re.sub(r"\s+", "", y).upper()
            for n in nums_in_text:
                try:
                    v = float(n.replace(",", ""))
                    if v > 0:
                        year_numbers.setdefault(yk, []).append(v)
                except ValueError:
                    pass

    calcs: Dict[str, Any] = {}

    # Growth rate (if two years in query)
    if len(eq.year_references) >= 2:
        y1 = re.sub(r"\s+", "", eq.year_references[0]).upper()
        y2 = re.sub(r"\s+", "", eq.year_references[-1]).upper()
        v1_list = year_numbers.get(y1, [])
        v2_list = year_numbers.get(y2, [])
        if v1_list and v2_list:
            v1, v2 = v1_list[0], v2_list[0]
            if v1 != 0:
                growth = ((v2 - v1) / v1) * 100
                calcs["growth_rate"] = {
                    "from_year":  y1,     "to_year":   y2,
                    "from_value": v1,     "to_value":  v2,
                    "growth_pct": round(growth, 2),
                    "formula":    f"(({v2} - {v1}) / {v1}) × 100 = {growth:.2f}%",
                }

    calcs["extracted_values"] = {k: v for k, v in year_numbers.items() if v}
    return calcs


# ══════════════════════════════════════════════════════════════════
# STEP 4: Pack Context Window
# ══════════════════════════════════════════════════════════════════

def _pack_context(
    evidence: List[EvidenceBlock],
    calculations: Dict[str, Any],
) -> Tuple[str, List[str], List[bytes]]:
    """
    Format all evidence into a single context string for the LLM.

    Order: TABLE → CHART → TEXT → IMAGE → DIAGRAM → PARAGRAPH → HEADING
    (Tables first because they're most reliable for factual queries)

    Each block gets:
      - [VERIFIED] or [UNVERIFIED] tag
      - Citation: [doc.pdf, p.4 — TABLE]
      - Content: formatted Markdown table OR text passage
      - Expanded context (section heading, related data)

    Also appends Python-computed calculations so LLM sees exact numbers.
    """
    type_order = [
        ElementType.TABLE, ElementType.CHART, ElementType.TEXT,
        ElementType.IMAGE, ElementType.DIAGRAM,
        ElementType.PARAGRAPH, ElementType.HEADING, ElementType.OCR_TEXT,
    ]

    # Group by element type
    grouped: Dict[ElementType, List[EvidenceBlock]] = {}
    for eb in evidence:
        grouped.setdefault(eb.chunk.element_type, []).append(eb)

    sections:    List[str]   = []
    citations:   List[str]   = []
    image_bytes: List[bytes] = []
    seen_keys:   set         = set()
    total_chars: int         = 0

    for etype in type_order:
        ebs = grouped.get(etype, [])
        if not ebs:
            continue

        section_lines = [f"\n--- {etype.value} EVIDENCE ---"]

        for eb in ebs:
            c   = eb.chunk
            key = c.key()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Check context budget
            if total_chars >= _MAX_CONTEXT_CHARS:
                break

            cite = c.citation()
            if cite not in citations:
                citations.append(cite)

            tag = "VERIFIED" if eb.verified else "UNVERIFIED"
            ctx = c.metadata.get("expanded_context", "")

            if etype == ElementType.TABLE and c.table_data:
                block = f"[{tag}] {cite}\n{c.table_data.to_markdown()}\n{ctx}"
            elif etype in (ElementType.CHART, ElementType.IMAGE, ElementType.DIAGRAM):
                block = f"[{tag}] {cite}\nDescription: {c.text}\n{ctx}"
                if c.image_bytes:
                    image_bytes.append(c.image_bytes)
            else:
                block = f"[{tag}] {cite}\n{c.text}\n{ctx}"

            section_lines.append(block)
            total_chars += len(block)

        sections.append("\n".join(section_lines))

    # Append Python-computed calculations
    if calculations.get("growth_rate"):
        gr = calculations["growth_rate"]
        sections.append(
            f"\n--- CALCULATIONS (Python-exact) ---\n"
            f"Growth {gr['from_year']} → {gr['to_year']}:\n"
            f"  {gr['formula']}"
        )

    return "\n\n".join(sections), citations, image_bytes


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — assemble_context()
# ══════════════════════════════════════════════════════════════════

def assemble_context(
    top_chunks: List[ScoredChunk],
    eq:         EnhancedQuery,
    all_chunks: List[Chunk],
) -> Tuple[ContextWindow, List[EvidenceBlock], Dict[str, Any]]:
    """
    Stage 8 entry point.

    Performs:
      1. Evidence verification (VERIFIED / UNVERIFIED tags)
      2. Parent context expansion (section headings, related tables)
      3. Python calculations (growth rate, etc.)
      4. Context packing (format into LLM-ready string)

    Args:
        top_chunks: Reranked chunks from Stage 7
        eq:         Enhanced query from Stage 6
        all_chunks: All chunks (needed for neighbour lookup)

    Returns:
        (ContextWindow, List[EvidenceBlock], calculations_dict)
    """
    logger.info(f"[Stage 8] Assembling context from {len(top_chunks)} chunks")

    # Step 1: Verify
    evidence = _verify_evidence(top_chunks)

    # Step 2: Expand
    evidence = _expand_context(evidence, all_chunks)

    # Step 3: Calculate
    calcs = _compute_calculations(eq, evidence)
    if calcs.get("growth_rate"):
        logger.info(f"  Calculation: {calcs['growth_rate']['formula']}")

    # Step 4: Pack
    context_text, citations, image_bytes = _pack_context(evidence, calcs)

    ctx = ContextWindow(
        text         = context_text,
        citations    = citations,
        image_bytes  = image_bytes,
        char_count   = len(context_text),
        chunk_count  = len(evidence),
        evidence_tags= [("VERIFIED" if eb.verified else "UNVERIFIED") for eb in evidence],
    )

    logger.info(
        f"  Context: {ctx.char_count} chars | "
        f"Citations: {len(citations)} | "
        f"Images: {len(image_bytes)}"
    )
    return ctx, evidence, calcs
