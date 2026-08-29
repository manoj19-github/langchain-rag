"""
retrieval.py — Stage 7: Hybrid Retrieval
=========================================
Searches the indexes using 5 different paths, then fuses and reranks.

The 5 paths (from the architecture diagram):
  A. Dense Vector Search  — semantic similarity (ChromaDB MMR)
  B. Sparse/Keyword Search — exact term matching (BM25)
  C. Structured/Filtered Search — table-specific token matching
  D. Result Fusion & Reranking — RRF + CrossEncoder
  E. Diversity & Redundancy Control — MMR + deduplication

Why multiple paths?
  Dense alone misses exact keyword matches ("FY2025-26")
  BM25 alone misses semantic matches ("profit" ≠ "earnings")
  Table path alone misses context from paragraphs
  Together they cover all retrieval failure modes.

Inputs:  EnhancedQuery, vstore, bm25_retriever, chunks, chunks_by_id
Outputs: List[ScoredChunk] — the top-K reranked evidence

Usage:
    from multimodal.retrieval import hybrid_retrieve
    scored = hybrid_retrieve(eq, vstore, bm25, chunks, chunk_lookup)
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from .config import cfg
from .models import Chunk, ElementType, EnhancedQuery, ScoredChunk

logger = logging.getLogger("multimodal.retrieval")

# ── CrossEncoder singleton ──
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(cfg.RERANKER_MODEL)
    return _cross_encoder


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _lc_doc_to_scored_chunk(
    doc, score: float, source: str,
    chunks_by_id: Dict[str, Chunk],
) -> Optional[ScoredChunk]:
    """
    Convert a LangChain Document (returned by retrievers) back to a ScoredChunk.
    Uses the chunk_id stored in metadata to look up the original Chunk.
    """
    cid   = doc.metadata.get("chunk_id", "")
    chunk = chunks_by_id.get(cid)
    if chunk is None:
        # Reconstruct a minimal chunk if not found in lookup
        chunk = Chunk(
            chunk_id     = cid,
            element_id   = doc.metadata.get("element_id", cid),
            doc_name     = doc.metadata.get("doc_name", "Unknown"),
            page_number  = doc.metadata.get("page_number", 0),
            element_type = ElementType(doc.metadata.get("element_type", "TEXT")),
            strategy     = cfg,   # Will use default
            text         = doc.page_content,
            metadata     = doc.metadata,
        )
    return ScoredChunk(chunk=chunk, score=score, source=source)


# ══════════════════════════════════════════════════════════════════
# PATH A: Dense Vector Search (ChromaDB + MMR)
# ══════════════════════════════════════════════════════════════════

def retrieve_dense(
    queries: List[str],
    vstore,
    chunks_by_id: Dict[str, Chunk],
    top_k: int = cfg.TOP_K_DENSE,
) -> List[ScoredChunk]:
    """
    Semantic vector search using MMR (Maximal Marginal Relevance).

    Runs multiple query variants through the vector store and merges results.
    MMR balances relevance + diversity — avoids returning 15 near-identical chunks.

    Args:
        queries: List of query strings [original, expanded, hyde_doc]
        vstore:  ChromaDB vector store
        top_k:   Number of chunks to return per query
    """
    retriever = vstore.as_retriever(
        search_type   = "mmr",
        search_kwargs = {
            "k":           top_k,
            "fetch_k":     cfg.MMR_FETCH_K,
            "lambda_mult": cfg.MMR_LAMBDA,
        },
    )
    results: List[ScoredChunk] = []
    seen_keys: set = set()

    for q in queries:
        try:
            for doc in retriever.invoke(q):
                sc = _lc_doc_to_scored_chunk(doc, 1.0, "dense", chunks_by_id)
                if sc and sc.key() not in seen_keys:
                    seen_keys.add(sc.key())
                    results.append(sc)
        except Exception as e:
            logger.debug(f"Dense retrieval error: {e}")

    logger.info(f"  [A] Dense: {len(results)} chunks")
    return results


# ══════════════════════════════════════════════════════════════════
# PATH B: Sparse Keyword Search (BM25)
# ══════════════════════════════════════════════════════════════════

def retrieve_bm25(
    query: str,
    bm25_retriever,
    chunks_by_id: Dict[str, Chunk],
) -> List[ScoredChunk]:
    """
    BM25 keyword search — best for exact term matching.

    BM25 stands for "Best Match 25". It scores documents by:
      - Term frequency in the document
      - Inverse document frequency (rare terms matter more)
      - Document length normalization

    Args:
        query: Keyword-only query (stopwords removed)
    """
    results = []
    try:
        for doc in bm25_retriever.invoke(query):
            sc = _lc_doc_to_scored_chunk(doc, 1.0, "bm25", chunks_by_id)
            if sc:
                results.append(sc)
    except Exception as e:
        logger.debug(f"BM25 error: {e}")

    logger.info(f"  [B] BM25: {len(results)} chunks")
    return results


# ══════════════════════════════════════════════════════════════════
# PATH C: Structured / Filtered Search (Tables)
# ══════════════════════════════════════════════════════════════════

def retrieve_tables(
    eq: EnhancedQuery,
    chunks: List[Chunk],
) -> List[ScoredChunk]:
    """
    Table-specific retrieval via direct token matching.

    Scans all table chunks and scores them by how many query tokens
    they contain. This is faster and more precise than embedding
    for financial queries with specific year/metric terms.

    Only runs if requires_table=True in the EnhancedQuery.
    """
    query_tokens  = set(re.findall(r"\b\w+\b", eq.original.lower()))
    year_tokens   = {y.lower().replace(" ", "") for y in eq.year_references}
    metric_tokens = {m.lower() for m in eq.metric_references}
    priority      = query_tokens | year_tokens | metric_tokens

    scored = []
    for c in chunks:
        if c.element_type != ElementType.TABLE:
            continue
        text_lower = c.text.lower()
        hits = sum(1 for t in priority if t in text_lower)
        if hits > 0:
            scored.append(ScoredChunk(
                chunk  = c,
                score  = hits / max(len(priority), 1),
                source = "table",
            ))

    scored.sort(key=lambda x: x.score, reverse=True)
    result = scored[:cfg.TOP_K_TABLE]
    logger.info(f"  [C] Tables: {len(result)} chunks")
    return result


def retrieve_structured_numeric(
    eq: EnhancedQuery,
    chunks: List[Chunk],
) -> List[ScoredChunk]:
    """
    Exact numeric lookup — finds table rows where a specific
    year AND metric both appear in the same row/column header.

    Example:
        Query mentions: year=FY2025, metric=revenue
        This finds the table cell where row_label="Revenue"
        and column_header="FY2025-26" intersect.
    """
    if not eq.year_references and not eq.metric_references:
        return []

    year_pats   = [re.compile(y.replace("fy", r"fy[\s\']?"), re.IGNORECASE)
                   for y in eq.year_references]
    metric_pats = [re.compile(m, re.IGNORECASE) for m in eq.metric_references]

    results = []
    for c in chunks:
        if c.element_type != ElementType.TABLE or not c.table_data:
            continue
        for col, row_label, val in c.table_data.extract_numbers():
            year_ok   = any(p.search(col) or p.search(row_label) for p in year_pats)   if year_pats   else True
            metric_ok = any(p.search(col) or p.search(row_label) for p in metric_pats) if metric_pats else True
            if year_ok and metric_ok:
                results.append(ScoredChunk(chunk=c, score=2.0, source="structured"))
                break

    results.sort(key=lambda x: x.score, reverse=True)
    result = results[:cfg.TOP_K_STRUCTURED]
    logger.info(f"  [C] Structured numeric: {len(result)} chunks")
    return result


def retrieve_charts(eq: EnhancedQuery, chunks: List[Chunk]) -> List[ScoredChunk]:
    """Chart/diagram retrieval — only when requires_chart=True."""
    if not eq.requires_chart and not eq.requires_diagram:
        return []
    query_tokens = set(re.findall(r"\b\w+\b", eq.original.lower()))
    scored = []
    for c in chunks:
        if c.element_type not in (ElementType.CHART, ElementType.DIAGRAM):
            continue
        hits = sum(1 for t in query_tokens if len(t) > 3 and t in c.text.lower())
        scored.append(ScoredChunk(chunk=c, score=float(hits), source="chart"))
    scored.sort(key=lambda x: x.score, reverse=True)
    result = scored[:cfg.TOP_K_CHART]
    logger.info(f"  [C] Charts: {len(result)} chunks")
    return result


# ══════════════════════════════════════════════════════════════════
# PATH D: Result Fusion — RRF (Reciprocal Rank Fusion)
# ══════════════════════════════════════════════════════════════════

def rrf_fusion(
    ranked_lists: List[Tuple[List[ScoredChunk], float]],
    k: int = cfg.RRF_K,
) -> List[ScoredChunk]:
    """
    Merge multiple ranked result lists into one combined ranking.

    RRF Formula:
        score(chunk) = Σ  weight / (k + rank)
                       over all lists where chunk appears

    Benefits:
      - A chunk that appears in MANY lists gets a high combined score
      - The k=60 constant prevents rank-1 items from dominating too much
      - Works even when lists use different score scales

    Args:
        ranked_lists: List of (ScoredChunk list, weight) tuples
        k:           Smoothing constant (default 60)
    """
    final_scores: Dict[str, float] = {}
    chunk_map:    Dict[str, ScoredChunk] = {}

    for ranked, weight in ranked_lists:
        for rank, sc in enumerate(ranked):
            key = sc.key()
            rrf = weight / (k + rank + 1)
            if key not in final_scores:
                final_scores[key] = 0.0
                chunk_map[key]    = sc
            final_scores[key] += rrf

    sorted_keys = sorted(final_scores, key=lambda kk: final_scores[kk], reverse=True)
    return [
        ScoredChunk(
            chunk  = chunk_map[key].chunk,
            score  = final_scores[key],
            source = chunk_map[key].source,
        )
        for key in sorted_keys
    ]


# ══════════════════════════════════════════════════════════════════
# PATH E: Diversity & Redundancy Control
# ══════════════════════════════════════════════════════════════════

def deduplicate(scored: List[ScoredChunk]) -> List[ScoredChunk]:
    """
    Remove duplicate chunks (same content from different retrieval paths).
    Keeps the version with the highest RRF score.
    """
    seen: Dict[str, ScoredChunk] = {}
    for sc in scored:
        key = sc.key()
        if key not in seen or sc.score > seen[key].score:
            seen[key] = sc
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)


def crossencoder_rerank(
    question: str,
    candidates: List[ScoredChunk],
    top_k: int = cfg.RERANK_TOP_K,
) -> List[ScoredChunk]:
    """
    Precise reranking using a CrossEncoder model.

    Unlike embeddings (which score query and chunk separately),
    the CrossEncoder sees BOTH together:
        Input:  ("[TABLE] Revenue: 482.6 | FY2025-26: ...")
        Query:  "What was total revenue?"
        Output: relevance score (e.g., 2.77)

    This is ~10x more accurate than embedding similarity but slower.
    We only run it on the top candidates (not all chunks).

    The element type prefix ("[TABLE]", "[TEXT]", etc.) helps the
    CrossEncoder understand the content type.
    """
    if not candidates:
        return []

    ce    = _get_cross_encoder()
    pairs = [
        (question, f"[{sc.chunk.element_type.value}] " + sc.chunk.text[:512])
        for sc in candidates
    ]
    try:
        scores = ce.predict(pairs, batch_size=cfg.RERANK_BATCH_SIZE)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(chunk=sc.chunk, score=float(s), source=sc.source)
            for sc, s in ranked[:top_k]
        ]
    except Exception as e:
        logger.debug(f"CrossEncoder failed: {e}")
        return candidates[:top_k]


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — hybrid_retrieve()
# ══════════════════════════════════════════════════════════════════

def hybrid_retrieve(
    eq:           EnhancedQuery,
    vstore,
    bm25_retriever,
    chunks:       List[Chunk],
    chunks_by_id: Dict[str, Chunk],
) -> List[ScoredChunk]:
    """
    Stage 7 entry point. Runs all retrieval paths and returns final top-K.

    Flow:
      1. Dense (A)  → semantic vector search
      2. BM25 (B)   → keyword search
      3. Table (C)  → table token matching + exact numeric lookup
      4. Chart (C)  → chart description matching
      5. RRF (D)    → weighted merge of all lists
      6. Dedup (E)  → remove duplicates
      7. Rerank (D) → CrossEncoder precision scoring

    Returns:
        Top-K ScoredChunks sorted by CrossEncoder score (highest first)
    """
    logger.info("[Stage 7] Hybrid retrieval")

    # ── A: Dense ────────────────────────────────────────────────
    dense = retrieve_dense(
        [eq.original, eq.expanded, eq.hyde_doc],
        vstore, chunks_by_id,
    )

    # ── B: BM25 ─────────────────────────────────────────────────
    bm25 = retrieve_bm25(eq.keyword_only or eq.original, bm25_retriever, chunks_by_id)

    # ── C: Structured ───────────────────────────────────────────
    tables     = retrieve_tables(eq, chunks) if (eq.requires_table or eq.year_references) else []
    structured = retrieve_structured_numeric(eq, chunks)
    charts     = retrieve_charts(eq, chunks)

    # ── D: RRF Fusion ───────────────────────────────────────────
    fused = rrf_fusion([
        (dense,      cfg.W_DENSE),
        (bm25,       cfg.W_BM25),
        (tables,     cfg.W_TABLE),
        (charts,     cfg.W_CHART),
        (structured, cfg.W_STRUCTURED),
    ])

    # ── E: Deduplicate ──────────────────────────────────────────
    deduped = deduplicate(fused)
    logger.info(f"  After RRF + dedup: {len(deduped)} candidates")

    # ── D: CrossEncoder Rerank ──────────────────────────────────
    top_k = crossencoder_rerank(eq.original, deduped)
    for i, sc in enumerate(top_k):
        logger.info(
            f"  [{i+1}] score={sc.score:.4f} "
            f"type={sc.chunk.element_type.value} "
            f"p={sc.chunk.page_number} src={sc.source}"
        )

    stats = {
        "dense": len(dense), "bm25": len(bm25),
        "table": len(tables), "chart": len(charts),
        "structured": len(structured),
        "after_rrf_dedup": len(deduped),
        "after_rerank": len(top_k),
    }
    logger.info(f"  Stats: {stats}")

    return top_k, stats
