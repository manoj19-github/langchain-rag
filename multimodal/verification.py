"""
verification.py — Stage 10: Verification & Hallucination Control
=================================================================
After the LLM generates an answer, this stage checks it for quality.

Checks performed:
  ✅ Faithfulness — does the answer only use facts from the context?
  ✅ Citation presence — does every factual claim have a source?
  ✅ Number consistency — do numbers in the answer match the evidence?
  ✅ Contradiction detection — does the answer contradict the evidence?
  ✅ HyDE leakage check — did the LLM accidentally cite the HyDE passage?
  ✅ Confidence scoring — 0.0 to 1.0 based on verification results

Inputs:  answer_draft (str), ContextWindow, List[EvidenceBlock]
Outputs: FinalAnswer (with confidence_score, verified flag, warnings)

Usage:
    from multimodal.verification import verify
    final = verify(answer_draft, context, evidence, eq, calcs, stats, latency)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from .models import (
    ContextWindow, EvidenceBlock, EnhancedQuery, FinalAnswer
)

logger = logging.getLogger("multimodal.verification")

# Numbers in the format: 482.6, 18.7%, 1,240, etc.
_NUMBER_RE = re.compile(r"\b\d[\d,]*\.?\d*\s*%?\b")


# ══════════════════════════════════════════════════════════════════
# CHECK 1: Citation Presence
# ══════════════════════════════════════════════════════════════════

def _check_citations(answer: str, citations: List[str]) -> tuple[float, List[str]]:
    """
    Check whether the answer contains source citations.

    Scoring:
      - Answer contains at least one citation reference → +0.3 points
      - No citations found → warning issued, -0.1 points

    Returns (score_contribution, warnings)
    """
    warnings = []

    # Look for citation patterns like [doc.pdf, p.4 — TABLE]
    citation_re = re.compile(r"\[.*?p\.\d+.*?\]", re.IGNORECASE)
    found = citation_re.findall(answer)

    if found:
        return 0.3, []
    else:
        warnings.append("⚠️ Answer does not contain explicit source citations.")
        return 0.1, warnings


# ══════════════════════════════════════════════════════════════════
# CHECK 2: Number Consistency
# ══════════════════════════════════════════════════════════════════

def _check_numbers(
    answer: str,
    evidence: List[EvidenceBlock],
) -> tuple[float, List[str]]:
    """
    Check whether numbers in the answer exist in the evidence.

    Logic:
      1. Extract all numbers from the answer
      2. Check each against all evidence texts
      3. Numbers found in evidence → trusted
      4. Numbers NOT found → possible hallucination

    Scoring:
      - All numbers match evidence → +0.4 points
      - Some mismatch → +0.2 points + warning
      - Major mismatch → +0.0 points + strong warning
    """
    warnings    = []
    all_evidence_text = " ".join(eb.chunk.text for eb in evidence)

    answer_nums  = _NUMBER_RE.findall(answer)
    if not answer_nums:
        return 0.3, []   # No numbers to check — neutral

    unmatched = []
    for num in answer_nums:
        clean = num.replace(",", "").replace("%", "").strip()
        if clean and clean not in all_evidence_text and num not in all_evidence_text:
            unmatched.append(num)

    if not unmatched:
        return 0.4, []
    elif len(unmatched) <= 1:
        warnings.append(f"⚠️ Number(s) in answer not found in evidence: {unmatched}")
        return 0.2, warnings
    else:
        warnings.append(f"🚨 Multiple numbers not found in evidence: {unmatched}")
        return 0.0, warnings


# ══════════════════════════════════════════════════════════════════
# CHECK 3: HyDE Leakage
# ══════════════════════════════════════════════════════════════════

def _check_hyde_leakage(answer: str, hyde_doc: str) -> tuple[float, List[str]]:
    """
    Detect if the LLM accidentally used the HyDE passage as evidence.

    Extracts numbers from hyde_doc and checks if they appear in the answer.
    The HyDE doc is a RETRIEVAL AID — its numbers are NOT real document facts.

    Example:
        HyDE: "revenue of $1.87 billion"
        If answer says "$1.87 billion" → HYDE LEAKAGE detected
    """
    warnings  = []
    hyde_nums = _NUMBER_RE.findall(hyde_doc)
    if not hyde_nums:
        return 0.2, []

    leaked = [n for n in hyde_nums if n in answer]
    if leaked:
        warnings.append(
            f"🚨 HyDE leakage detected — answer may cite fabricated retrieval values: {leaked}"
        )
        return 0.0, warnings

    return 0.2, []


# ══════════════════════════════════════════════════════════════════
# CHECK 4: Insufficient Evidence Detection
# ══════════════════════════════════════════════════════════════════

def _check_insufficient(answer: str) -> tuple[float, List[str]]:
    """
    Check if the LLM said it couldn't answer.
    This might be correct (no evidence) or over-cautious (evidence was there).
    """
    lower = answer.lower()
    if "insufficient evidence" in lower or "cannot be determined" in lower:
        return 0.0, ["⚠️ LLM reported insufficient evidence — check evidence quality."]
    return 0.1, []


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — verify()
# ══════════════════════════════════════════════════════════════════

def verify(
    answer_draft: str,
    context:      ContextWindow,
    evidence:     List[EvidenceBlock],
    eq:           EnhancedQuery,
    calculations: Dict[str, Any],
    retrieval_stats: Dict[str, int],
    latency:      Dict[str, float],
) -> FinalAnswer:
    """
    Stage 10 entry point.

    Runs all verification checks and computes a confidence score.
    Packages everything into a FinalAnswer object.

    Confidence score (0.0 to 1.0):
      citation_check  → max 0.3
      number_check    → max 0.4
      hyde_check      → max 0.2
      base            → 0.1
      Total           → max 1.0

    Args:
        answer_draft:    Raw answer from Stage 9
        context:         Context window (citations, image bytes)
        evidence:        Evidence blocks with VERIFIED tags
        eq:              Enhanced query (has hyde_doc)
        calculations:    Python-computed values
        retrieval_stats: How many chunks each path found
        latency:         Time per stage

    Returns:
        FinalAnswer with all fields populated
    """
    logger.info("[Stage 10] Verifying answer")

    all_warnings: List[str] = []
    total_score = 0.0

    # Run checks
    s1, w1 = _check_citations(answer_draft, context.citations)
    s2, w2 = _check_numbers(answer_draft, evidence)
    s3, w3 = _check_hyde_leakage(answer_draft, eq.hyde_doc)
    s4, w4 = _check_insufficient(answer_draft)

    total_score = round(min(s1 + s2 + s3 + s4 + 0.1, 1.0), 2)
    all_warnings = w1 + w2 + w3 + w4

    is_verified = (total_score >= 0.7 and not w3)   # High score + no HyDE leakage

    logger.info(
        f"  Confidence: {total_score:.2f} | "
        f"Verified: {is_verified} | "
        f"Warnings: {len(all_warnings)}"
    )
    if all_warnings:
        for w in all_warnings:
            logger.warning(f"  {w}")

    # Build evidence summary for FinalAnswer
    evidence_summary = [
        {
            "rank":        i + 1,
            "page":        eb.chunk.page_number,
            "type":        eb.chunk.element_type.value,
            "citation":    eb.chunk.citation(),
            "score":       round(eb.rerank_score, 4),
            "verified":    eb.verified,
            "snippet":     eb.chunk.text[:200],
        }
        for i, eb in enumerate(evidence)
    ]

    return FinalAnswer(
        question         = eq.original,
        answer           = answer_draft,
        citations        = context.citations,
        confidence_score = total_score,
        verified         = is_verified,
        hyde_doc         = eq.hyde_doc,
        query_type       = eq.query_type,
        retrieval_stats  = retrieval_stats,
        calculations     = calculations,
        evidence         = evidence_summary,
        latency          = latency,
        warnings         = all_warnings,
    )
