"""
generation.py — Stage 9: LLM Answer Generation
================================================
Sends the assembled context + question to the LLM and returns the answer.

What this stage does:
  ✅ Builds the final prompt (system + user + context + HyDE warning)
  ✅ Calls the configured LLM (Ollama / OpenAI / Anthropic)
  ✅ Supports multimodal input (text + images for vision LLMs)
  ✅ Returns raw answer text (verification happens in Stage 10)

The system prompt contains explicit rules to:
  - Use only evidence from the context
  - Never cite the HyDE passage as evidence
  - Accept TABLE values as authoritative document facts
  - State figures in original units (crore, %, etc.)

Inputs:  question, ContextWindow, EnhancedQuery
Outputs: answer_draft (str)

Usage:
    from multimodal.generation import generate
    answer = generate(question, context_window, eq)
"""

from __future__ import annotations

import logging
import traceback
from typing import List

from .config import cfg
from .models import ContextWindow, EnhancedQuery

logger = logging.getLogger("multimodal.generation")

# ── LLM singleton ──
_rag_llm = None


# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise document intelligence assistant analyzing a business report PDF.

RULES — follow these exactly:
1. Answer ONLY using the EVIDENCE blocks provided. Do not use outside knowledge.
2. TABLE evidence IS authoritative — values in table rows are direct quotes from the PDF.
   - If a TABLE row says "Revenue | 482.6 | 406.5", state "Revenue was ₹482.6 Cr."
   - Do NOT say "insufficient evidence" when a table clearly contains the answer.
3. Cite every fact with its source tag: [doc.pdf, p.4 — TABLE]
4. The [HyDE passage] is a RETRIEVAL AID ONLY. NEVER cite it as evidence.
   - If HyDE says "$1.87 billion" but TABLE says "482.6 Cr", the TABLE is correct.
5. State figures in their original units (crore, million, %, etc.). Do not convert.
6. [VERIFIED] = value seen in multiple sources (more reliable).
   [UNVERIFIED] = single source (mention uncertainty).
7. If calculations are shown in the context, use those exact Python-computed values.
8. Only say "Insufficient evidence" if the answer truly cannot be found anywhere.
9. Keep your answer concise, factual, and well-structured."""


# ══════════════════════════════════════════════════════════════════
# BUILD LLM
# ══════════════════════════════════════════════════════════════════

def _get_llm():
    global _rag_llm
    if _rag_llm is None:
        logger.info(f"[Stage 9] Loading LLM: {cfg.LLM_PROVIDER}/{cfg.LLM_MODEL}")
        if cfg.LLM_PROVIDER == "openai":
            from langchain_openai import ChatOpenAI
            _rag_llm = ChatOpenAI(
                model=cfg.LLM_MODEL, temperature=cfg.LLM_TEMPERATURE,
                api_key=cfg.OPENAI_API_KEY,
            )
        elif cfg.LLM_PROVIDER == "anthropic":
            from langchain_anthropic import ChatAnthropic
            _rag_llm = ChatAnthropic(
                model=cfg.LLM_MODEL, temperature=cfg.LLM_TEMPERATURE,
                api_key=cfg.ANTHROPIC_API_KEY,
            )
        else:  # ollama (default)
            from langchain_ollama import ChatOllama
            _rag_llm = ChatOllama(
                model=cfg.LLM_MODEL, temperature=cfg.LLM_TEMPERATURE,
                base_url=cfg.OLLAMA_BASE_URL,
            )
    return _rag_llm


# ══════════════════════════════════════════════════════════════════
# BUILD PROMPT
# ══════════════════════════════════════════════════════════════════

def _build_prompt(
    question:    str,
    context:     ContextWindow,
    hyde_doc:    str,
) -> str:
    """
    Assemble the full user prompt.

    Structure:
      EVIDENCE FROM PDF:
        [context text with tables, paragraphs, calculations]

      Citations:
        [1] [report.pdf, p.4 — TABLE]
        [2] [report.pdf, p.2 — PARAGRAPH]

      NOTE: RETRIEVAL AID ONLY (do not cite):
        [HyDE passage]: NovaTech reported total revenue of...

      Question: What was total revenue in FY2025-26?

      Answer (cite sources, note uncertainty):
    """
    prompt_parts = [
        "EVIDENCE FROM PDF:",
        context.text,
        "\nCitations:",
        "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(context.citations)),
        "\nNOTE: The following is a RETRIEVAL AID — do NOT cite it as evidence:",
        f"[HyDE passage]: {hyde_doc[:150]}",
        f"\nQuestion: {question}",
        "\nAnswer (cite sources, note any uncertainty, show calculations if relevant):",
    ]
    return "\n".join(prompt_parts)


# ══════════════════════════════════════════════════════════════════
# GENERATE (text-only)
# ══════════════════════════════════════════════════════════════════

def _generate_text_only(prompt: str) -> str:
    """Call LLM with text prompt only (Ollama default path)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    llm  = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()


# ══════════════════════════════════════════════════════════════════
# GENERATE (multimodal — text + images)
# ══════════════════════════════════════════════════════════════════

def _generate_with_images(prompt: str, image_bytes_list: List[bytes]) -> str:
    """
    Call OpenAI/Anthropic with both text AND images.
    Images are base64-encoded and passed inline.
    Only used when VISION_PROVIDER is openai or anthropic AND images were retrieved.
    """
    import base64

    if cfg.LLM_PROVIDER == "openai" and cfg.OPENAI_API_KEY:
        import openai
        client  = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)
        content = [{"type": "text", "text": prompt}]
        for img in image_bytes_list[:3]:    # Max 3 images per call
            b64 = base64.b64encode(img).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })
        resp = client.chat.completions.create(
            model    = cfg.LLM_MODEL,
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": content},
            ],
            max_tokens  = 1500,
            temperature = cfg.LLM_TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()

    elif cfg.LLM_PROVIDER == "anthropic" and cfg.ANTHROPIC_API_KEY:
        import anthropic, base64
        client      = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        msg_content = []
        for img in image_bytes_list[:3]:
            b64 = base64.b64encode(img).decode()
            msg_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
        msg_content.append({"type": "text", "text": prompt})
        resp = client.messages.create(
            model    = cfg.LLM_MODEL,
            max_tokens = 1500,
            system   = SYSTEM_PROMPT,
            messages = [{"role": "user", "content": msg_content}],
        )
        return resp.content[0].text.strip()

    # Fall back to text-only
    return _generate_text_only(prompt)


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — generate()
# ══════════════════════════════════════════════════════════════════

def generate(
    question: str,
    context:  ContextWindow,
    eq:       EnhancedQuery,
) -> str:
    """
    Stage 9 entry point.

    Builds the prompt and calls the LLM.
    Automatically uses multimodal (text + images) if:
      - Images were retrieved (context.image_bytes is not empty)
      - LLM provider is openai or anthropic (supports vision)

    Args:
        question: The user's original question
        context:  Assembled context from Stage 8
        eq:       Enhanced query (provides hyde_doc)

    Returns:
        Raw answer string from the LLM
    """
    logger.info(f"[Stage 9] Generating answer ({cfg.LLM_PROVIDER}/{cfg.LLM_MODEL})")

    prompt = _build_prompt(question, context, eq.hyde_doc)

    try:
        # Use vision-capable path only for cloud providers with images
        if (context.image_bytes and
            cfg.LLM_PROVIDER in ("openai", "anthropic")):
            logger.info(f"  Using multimodal path ({len(context.image_bytes)} images)")
            answer = _generate_with_images(prompt, context.image_bytes)
        else:
            answer = _generate_text_only(prompt)

        logger.info(f"  Answer generated ({len(answer)} chars)")
        return answer

    except Exception as e:
        logger.error(f"Answer generation failed: {e}\n{traceback.format_exc()}")
        return f"[Error generating answer: {e}]"
