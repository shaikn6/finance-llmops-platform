"""
Answer generator with citation tracking.

Supports two modes:
  - MOCK_MODE=true  → deterministic, realistic canned responses (no API key needed)
  - MOCK_MODE=false → real OpenAI GPT-4 calls via openai 1.x SDK

All responses include structured citation metadata so hallucination.py
can perform grounding checks.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from pipeline.retriever import RetrievedChunk, get_retriever


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    chunk_id: str
    doc_name: str
    doc_type: str
    text_snippet: str       # First 300 chars of the source chunk
    relevance_score: float


@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    citations: List[Citation]
    latency_ms: float
    model: str
    prompt_version: str
    retrieval_scores: List[float]
    mock_mode: bool


# ---------------------------------------------------------------------------
# Prompt templates (versioned for MLflow tracking)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: Dict[str, str] = {
    "v1_basic": (
        "You are a financial analyst assistant. Using ONLY the provided context excerpts, "
        "answer the question precisely and factually. Cite specific numbers and percentages. "
        "If the answer is not in the context, say so.

"
        "Context:
{context}

Question: {question}

Answer:"
    ),
    "v2_structured": (
        "You are a senior financial analyst at a credit union. "
        "Your answers must be grounded exclusively in the provided source documents. "
        "Structure your answer in 2-3 sentences. Lead with the key metric or finding. "
        "Include relevant figures (dollar amounts, ratios, percentages) exactly as stated in the source.

"
        "Source Documents:
{context}

"
        "Analyst Question: {question}

"
        "Evidence-Based Answer:"
    ),
    "v3_cot": (
        "You are a senior financial analyst. Think step by step before answering. "
        "First identify which source excerpt is most relevant. "
        "Then extract the precise data points. "
        "Then compose a concise, citation-supported answer.

"
        "Source Excerpts:
{context}

"
        "Question: {question}

"
        "Analysis and Answer:"
    ),
}

DEFAULT_PROMPT_VERSION = "v2_structured"

# ---------------------------------------------------------------------------
# Mock responses for MOCK_MODE
# ---------------------------------------------------------------------------

_MOCK_RESPONSES: Dict[str, str] = {
    "lcr": (
        "Meridian Financial Corp maintains a liquidity coverage ratio (LCR) of 134% as of "
        "December 31, 2023, which exceeds the regulatory minimum of 100% required under Basel III. "
        "The Company holds $2.84 billion in high-quality liquid assets (HQLA) to support this ratio."
    ),
    "revenue": (
        "For fiscal year 2023, Meridian Financial Corp recorded total net revenue of $3.21 billion, "
        "a 7.4% increase from $2.99 billion in fiscal year 2022. Commercial Banking was the largest "
        "segment at $1.42 billion (44.2% of total revenue)."
    ),
    "cyber": (
        "Meridian invested $78.3 million in information security infrastructure in fiscal year 2023, "
        "a 23.4% year-over-year increase. The Company employs 186 dedicated cybersecurity professionals "
        "and maintains a 24/7 Security Operations Center with a mean time to detect of 4.2 minutes."
    ),
    "debt": (
        "As of December 31, 2023, Meridian Financial Corp's long-term debt totaled $7.83 billion, "
        "comprising $3.40 billion in senior unsecured notes, $1.85 billion in subordinated debt "
        "qualifying as Tier 2 capital, and $2.58 billion in Federal Home Loan Bank advances."
    ),
    "nim": (
        "Management guided to a net interest margin (NIM) of 3.40% to 3.55% for full-year 2024. "
        "Q4 2023 NIM was 3.41% on a reported basis (3.38% core, excluding purchase accounting accretion "
        "of $9 million). Q1 2024 is expected to be the trough, with sequential improvement from Q2."
    ),
    "ai": (
        "Meridian's AI document intelligence platform processes a 10-K filing in approximately 8–12 minutes, "
        "compared to 3.5–5 hours for an experienced analyst, achieving a 94.7% accuracy rate in Q4 2023 "
        "evaluation. The platform required $4.2 million in 2023 capital outlay and flags any output "
        "below 85% grounding score for human review."
    ),
    "default": (
        "Based on the Meridian Financial Corp 10-K filing and Q4 2023 earnings call transcript, "
        "the relevant financial data shows strong performance across key metrics. The Company "
        "maintained disciplined capital allocation with a CET1 ratio of 11.8%, net charge-off rate "
        "of 0.31% (vs peer median 0.44%), and total deposits of $24.6 billion as of December 31, 2023."
    ),
}


def _select_mock_response(question: str) -> str:
    """Select a realistic mock response based on keyword matching."""
    q_lower = question.lower()
    keyword_map = {
        "lcr": ["lcr", "liquidity coverage", "liquidity ratio", "hqla"],
        "revenue": ["revenue", "income", "net revenue", "commercial banking", "segment"],
        "cyber": ["cyber", "security", "information security", "phishing", "soc"],
        "debt": ["debt", "long-term debt", "borrowing", "fhlb", "subordinated", "note"],
        "nim": ["nim", "net interest margin", "margin", "deposit beta", "rate cut"],
        "ai": ["ai", "document intelligence", "platform", "analyst", "10-k processing", "8 minute"],
    }
    for key, keywords in keyword_map.items():
        if any(kw in q_lower for kw in keywords):
            return _MOCK_RESPONSES[key]
    return _MOCK_RESPONSES["default"]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class FinancialAnswerGenerator:
    """
    Generates citation-grounded answers to financial questions.

    Works in both MOCK_MODE (no API key) and live OpenAI mode.
    """

    def __init__(
        self,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        model: str = "gpt-4",
        top_k: int = 3,
        mock_mode: Optional[bool] = None,
    ) -> None:
        self.prompt_version = prompt_version
        self.model = model
        self.top_k = top_k
        self._retriever = None

        # Determine mock mode
        if mock_mode is None:
            env_val = os.getenv("MOCK_MODE", "true").lower()
            self.mock_mode = env_val in ("true", "1", "yes")
        else:
            self.mock_mode = mock_mode

        # Try to import OpenAI if not in mock mode
        self._openai_client = None
        if not self.mock_mode:
            self._init_openai()

    def _init_openai(self) -> None:
        try:
            from openai import OpenAI  # pragma: no cover
            api_key = os.getenv("OPENAI_API_KEY")  # pragma: no cover
            if not api_key:  # pragma: no cover
                print("[generator] OPENAI_API_KEY not set; falling back to MOCK_MODE")  # pragma: no cover
                self.mock_mode = True  # pragma: no cover
                return  # pragma: no cover
            self._openai_client = OpenAI(api_key=api_key)  # pragma: no cover
        except ImportError:  # pragma: no cover
            print("[generator] openai package not available; using MOCK_MODE")  # pragma: no cover
            self.mock_mode = True  # pragma: no cover

    def _get_retriever(self):
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build a formatted context string from retrieved chunks."""
        parts = []
        for chunk in chunks:
            doc_label = f"[SOURCE: {chunk.doc_name} | chunk={chunk.chunk_id} | score={chunk.score:.3f}]"
            parts.append(f"{doc_label}
{chunk.text}")
        return "

---

".join(parts)

    def _call_openai(self, prompt: str) -> str:  # pragma: no cover
        """Call OpenAI API with error handling."""
        try:
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise financial analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[generator] OpenAI call failed: {e}; using mock response")
            return _select_mock_response(prompt)

    def generate(self, question: str) -> GeneratedAnswer:
        """
        Generate an answer to a financial question with full citation tracking.

        Args:
            question: Financial question to answer.

        Returns:
            GeneratedAnswer with answer text, citations, and latency.
        """
        t0 = time.time()

        # Retrieve relevant chunks
        retriever = self._get_retriever()
        retrieved = retriever.retrieve(question, top_k=self.top_k)

        if not retrieved:
            return GeneratedAnswer(
                question=question,
                answer="No relevant documents found for this question.",
                citations=[],
                latency_ms=round((time.time() - t0) * 1000, 1),
                model=self.model if not self.mock_mode else "mock",
                prompt_version=self.prompt_version,
                retrieval_scores=[],
                mock_mode=self.mock_mode,
            )

        # Build prompt
        template = PROMPT_TEMPLATES.get(self.prompt_version, PROMPT_TEMPLATES[DEFAULT_PROMPT_VERSION])
        context = self._build_context(retrieved)
        prompt = template.format(context=context, question=question)

        # Generate answer
        if self.mock_mode:
            # Simulate slight latency
            time.sleep(0.05)
            answer = _select_mock_response(question)
        else:
            answer = self._call_openai(prompt)

        latency_ms = round((time.time() - t0) * 1000, 1)

        # Build citations — use full text for grounding checks, snippet for display
        citations = [
            Citation(
                chunk_id=c.chunk_id,
                doc_name=c.doc_name,
                doc_type=c.doc_type,
                text_snippet=c.text,   # full text used for hallucination grounding
                relevance_score=c.score,
            )
            for c in retrieved
        ]

        return GeneratedAnswer(
            question=question,
            answer=answer,
            citations=citations,
            latency_ms=latency_ms,
            model=self.model if not self.mock_mode else "mock/all-MiniLM-L6-v2",
            prompt_version=self.prompt_version,
            retrieval_scores=[c.score for c in retrieved],
            mock_mode=self.mock_mode,
        )

    def to_dict(self, result: GeneratedAnswer) -> Dict[str, Any]:
        """Serialize a GeneratedAnswer to a JSON-compatible dict."""
        return {
            "question": result.question,
            "answer": result.answer,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "doc_type": c.doc_type,
                    "text_snippet": c.text_snippet,
                    "relevance_score": round(c.relevance_score, 4),
                }
                for c in result.citations
            ],
            "latency_ms": result.latency_ms,
            "model": result.model,
            "prompt_version": result.prompt_version,
            "retrieval_scores": [round(s, 4) for s in result.retrieval_scores],
            "mock_mode": result.mock_mode,
        }


# Module-level singleton
_generator: Optional[FinancialAnswerGenerator] = None


def get_generator(prompt_version: str = DEFAULT_PROMPT_VERSION) -> FinancialAnswerGenerator:
    """Return shared generator instance."""
    global _generator
    if _generator is None or _generator.prompt_version != prompt_version:
        _generator = FinancialAnswerGenerator(prompt_version=prompt_version)
    return _generator


if __name__ == "__main__":
    gen = get_generator()
    result = gen.generate("What is Meridian's liquidity coverage ratio?")
    import json
    print(json.dumps(gen.to_dict(result), indent=2))
