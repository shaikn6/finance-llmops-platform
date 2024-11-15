"""
RAG Evaluation Harness — RAGAS-style metrics from scratch.

Metrics:
  - faithfulness_score   : answer is grounded in retrieved docs (0–1)
  - relevance_score      : retrieved docs match the query (0–1)
  - answer_similarity    : answer overlaps with gold reference (0–1)
  - context_precision    : fraction of retrieved chunks that are relevant (0–1)
  - context_recall       : fraction of gold references covered by context (0–1)

Runs 20 test Q&A pairs and produces a scores CSV at evaluation/results/.

Usage:
    python -m evaluation.eval_harness
    python -m evaluation.eval_harness --output /tmp/eval_results.csv
"""

from __future__ import annotations

import csv
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Test Q&A pairs (20 total)
# ---------------------------------------------------------------------------

TEST_QA_PAIRS: List[Dict[str, str]] = [
    {
        "query": "What is Meridian Financial Corp's liquidity coverage ratio?",
        "gold_answer": (
            "Meridian Financial Corp maintains an LCR of 134% as of December 31, 2023, "
            "exceeding the 100% regulatory minimum under Basel III."
        ),
    },
    {
        "query": "What was Meridian's total net revenue in fiscal year 2023?",
        "gold_answer": (
            "Total net revenue was $3.21 billion for fiscal year 2023, "
            "a 7.4% increase from $2.99 billion in fiscal year 2022."
        ),
    },
    {
        "query": "What is the CET1 capital ratio?",
        "gold_answer": (
            "The CET1 ratio stands at 11.8%, above the 6.5% well-capitalized threshold."
        ),
    },
    {
        "query": "How much did Meridian invest in cybersecurity in 2023?",
        "gold_answer": (
            "Meridian invested $78.3 million in information security infrastructure "
            "in fiscal year 2023, a 23.4% year-over-year increase."
        ),
    },
    {
        "query": "What is the net interest margin guidance for 2024?",
        "gold_answer": (
            "Management guided to a NIM of 3.40% to 3.55% for full-year 2024, "
            "with Q1 2024 expected to be the trough."
        ),
    },
    {
        "query": "What is Meridian's long-term debt position?",
        "gold_answer": (
            "Long-term debt totaled $7.83 billion, comprising $3.40 billion in senior "
            "unsecured notes, $1.85 billion in subordinated debt, and $2.58 billion in FHLB advances."
        ),
    },
    {
        "query": "What deposit beta does Meridian assume in its NIM guidance?",
        "gold_answer": (
            "Meridian modeled a cumulative deposit beta of 52% on interest-bearing deposits."
        ),
    },
    {
        "query": "What is the net charge-off rate?",
        "gold_answer": (
            "The net charge-off rate was 0.31%, below the peer median of 0.44%."
        ),
    },
    {
        "query": "How accurate is the AI document intelligence platform?",
        "gold_answer": (
            "The platform achieves a 94.7% accuracy rate and processes a 10-K in "
            "approximately 8–12 minutes, versus 3.5–5 hours for an experienced analyst."
        ),
    },
    {
        "query": "What are Meridian's annual long-term debt maturities through 2028?",
        "gold_answer": (
            "Annual principal maturities for 2024–2028 are $420 million, $1.24 billion, "
            "$890 million, $640 million, and $720 million respectively."
        ),
    },
    {
        "query": "What is the total deposit base?",
        "gold_answer": (
            "Total deposits were $24.6 billion as of December 31, 2023, "
            "reflecting continued franchise strength."
        ),
    },
    {
        "query": "What did Meridian report for Q4 2023 NIM?",
        "gold_answer": (
            "Q4 2023 NIM was 3.41% reported and 3.38% core, excluding $9 million of "
            "purchase accounting accretion."
        ),
    },
    {
        "query": "What is the allowance for credit losses ratio?",
        "gold_answer": (
            "The ACL ratio stood at 1.12% of total loans, providing 3.6x coverage of "
            "non-performing loans."
        ),
    },
    {
        "query": "How many cybersecurity professionals does Meridian employ?",
        "gold_answer": (
            "Meridian employs 186 dedicated cybersecurity professionals and maintains a "
            "24/7 Security Operations Center."
        ),
    },
    {
        "query": "What capital return did Meridian execute in 2023?",
        "gold_answer": (
            "The Company repurchased $320 million in common shares and paid $186 million "
            "in common dividends during fiscal year 2023."
        ),
    },
    {
        "query": "What revenue did Commercial Banking contribute?",
        "gold_answer": (
            "Commercial Banking was the largest segment at $1.42 billion, "
            "representing 44.2% of total net revenue."
        ),
    },
    {
        "query": "What is the mean time to detect security incidents?",
        "gold_answer": (
            "The Security Operations Center has a mean time to detect of 4.2 minutes."
        ),
    },
    {
        "query": "What percentage of loans are non-performing?",
        "gold_answer": (
            "Non-performing loans represented 0.31% of total loans as of December 31, 2023."
        ),
    },
    {
        "query": "What is the risk-weighted asset base?",
        "gold_answer": (
            "Risk-weighted assets totaled $42.7 billion as of year-end 2023."
        ),
    },
    {
        "query": "What is the HQLA buffer relative to 30-day net cash outflows?",
        "gold_answer": (
            "The Company holds $2.84 billion in HQLA, covering projected 30-day net cash "
            "outflows by a factor of 1.34x (LCR = 134%)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Tokenization helpers (shared with hallucination module style)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r'\b[\w%.,$]+\b', _normalize(text)))


# ---------------------------------------------------------------------------
# Individual metric scorers
# ---------------------------------------------------------------------------

def faithfulness_score(answer: str, context_texts: List[str]) -> float:
    """
    Measure how well the answer is grounded in the retrieved context.

    Implementation:
        1. Extract factual claims (numbers, percentages, acronyms) from answer.
        2. For each claim, compute max token-overlap against any context chunk.
        3. faithfulness = fraction of claims with overlap >= 0.5.

    Returns float in [0, 1].
    """
    if not answer.strip():
        return 1.0
    if not context_texts:
        return 0.0

    # Extract verifiable tokens: numbers, %, $, known abbreviations
    claim_pattern = re.compile(
        r'\$[\d,.]+\s*(?:billion|million|thousand)?|'
        r'\d+(?:\.\d+)?\s*%|'
        r'\d+(?:\.\d+)?\s*(?:basis points?|bps)|'
        r'\b(?:LCR|CET1|NIM|NCO|HQLA|FHLB|ACL|AFS|OCC|FDIC)\b|'
        r'(?:Q[1-4]\s+\d{4}|\d{4})',
        re.IGNORECASE,
    )
    claims = list({m.group().strip() for m in claim_pattern.finditer(answer)})

    if not claims:
        # No verifiable claims — conservative default
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(" ".join(context_texts))
        if not answer_tokens:
            return 1.0
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        return min(1.0, overlap * 1.5)  # scale up since general tokens are common

    grounded = 0
    for claim in claims:
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            grounded += 1
            continue
        best = 0.0
        for ctx in context_texts:
            ctx_tokens = _tokenize(ctx)
            ratio = len(claim_tokens & ctx_tokens) / len(claim_tokens)
            if ratio > best:
                best = ratio
        if best >= 0.50:
            grounded += 1

    return round(grounded / len(claims), 4)


def relevance_score(query: str, context_texts: List[str]) -> float:
    """
    Measure how relevant the retrieved context is to the query.

    Implementation:
        For each context chunk, compute token overlap with the query.
        relevance = average of top-k chunk scores.

    Returns float in [0, 1].
    """
    if not context_texts:
        return 0.0

    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.5

    scores: List[float] = []
    for ctx in context_texts:
        ctx_tokens = _tokenize(ctx)
        if not ctx_tokens:
            scores.append(0.0)
            continue
        # Jaccard similarity
        intersection = len(query_tokens & ctx_tokens)
        union = len(query_tokens | ctx_tokens)
        jaccard = intersection / union if union > 0 else 0.0
        scores.append(jaccard)

    return round(sum(scores) / len(scores), 4)


def answer_similarity(generated: str, gold: str) -> float:
    """
    Token-level F1 similarity between generated answer and gold reference.

    Returns float in [0, 1].
    """
    gen_tokens = _tokenize(generated)
    gold_tokens = _tokenize(gold)

    if not gen_tokens or not gold_tokens:
        return 0.0

    common = len(gen_tokens & gold_tokens)
    precision = common / len(gen_tokens)
    recall = common / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def context_precision(context_texts: List[str], gold_answer: str) -> float:
    """
    Fraction of retrieved context chunks that are relevant to the gold answer.

    A chunk is "relevant" if its token overlap with the gold answer >= 0.05.

    Returns float in [0, 1].
    """
    if not context_texts:
        return 0.0

    gold_tokens = _tokenize(gold_answer)
    if not gold_tokens:
        return 0.5

    relevant = 0
    for ctx in context_texts:
        ctx_tokens = _tokenize(ctx)
        overlap = len(ctx_tokens & gold_tokens) / len(gold_tokens)
        if overlap >= 0.05:
            relevant += 1

    return round(relevant / len(context_texts), 4)


def context_recall(context_texts: List[str], gold_answer: str) -> float:
    """
    Fraction of the gold answer's key facts covered by the retrieved context.

    Key facts are extracted as: numbers, percentages, dollar amounts, dates.

    Returns float in [0, 1].
    """
    if not gold_answer.strip():
        return 1.0
    if not context_texts:
        return 0.0

    fact_pattern = re.compile(
        r'\$[\d,.]+\s*(?:billion|million|thousand)?|'
        r'\d+(?:\.\d+)?\s*%|'
        r'\b(?:LCR|CET1|NIM|NCO|HQLA|FHLB)\b|'
        r'(?:Q[1-4]\s+\d{4}|\d{4})',
        re.IGNORECASE,
    )
    gold_facts = list({m.group().strip() for m in fact_pattern.finditer(gold_answer)})

    if not gold_facts:
        # Fall back to general token overlap
        gold_tokens = _tokenize(gold_answer)
        ctx_tokens = _tokenize(" ".join(context_texts))
        if not gold_tokens:
            return 1.0
        return round(min(1.0, len(gold_tokens & ctx_tokens) / len(gold_tokens)), 4)

    covered = 0
    all_ctx = " ".join(context_texts)
    for fact in gold_facts:
        fact_tokens = _tokenize(fact)
        ctx_tokens = _tokenize(all_ctx)
        if len(fact_tokens & ctx_tokens) / max(len(fact_tokens), 1) >= 0.5:
            covered += 1

    return round(covered / len(gold_facts), 4)


# ---------------------------------------------------------------------------
# Per-sample result
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    sample_id: int
    query: str
    gold_answer: str
    generated_answer: str
    retrieved_chunks: List[str]

    # RAGAS-style metrics
    faithfulness: float = 0.0
    relevance: float = 0.0
    answer_similarity: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Composite
    ragas_score: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("retrieved_chunks", None)
        return d


@dataclass
class EvalReport:
    run_timestamp: str
    num_samples: int
    samples: List[EvalSample] = field(default_factory=list)

    # Aggregate metrics
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_answer_similarity: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_ragas_score: float = 0.0
    avg_latency_ms: float = 0.0

    output_csv: Optional[Path] = None


# ---------------------------------------------------------------------------
# Generator helper (uses pipeline mock)
# ---------------------------------------------------------------------------

def _generate_answer_and_context(
    query: str, top_k: int = 3
) -> Tuple[str, List[str]]:
    """
    Generate an answer + retrieve context for a given query.
    Uses multi-agent pipeline in mock mode (no API key required).
    """
    try:
        from agents.multi_agent_rag import MultiAgentRAGPipeline
        pipeline = MultiAgentRAGPipeline(top_k=top_k)
        result = pipeline.run(query)
        state = result.state
        answer = state.get("final_answer", "")
        context = [c["text"] for c in state.get("retrieved_chunks", [])]
        return answer, context
    except Exception:
        pass

    # Fallback: use generator + mock retriever chunks
    try:
        from pipeline.generator import FinancialAnswerGenerator
        gen = FinancialAnswerGenerator(mock_mode=True)
        gen_result = gen.generate(query)
        answer = gen_result.answer
        context = [c.text_snippet for c in gen_result.citations]
        return answer, context
    except Exception:
        pass

    return f"Unable to generate answer for: {query}", []


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

class RAGEvalHarness:
    """
    Runs RAGAS-style evaluation on the 20 test Q&A pairs.

    Usage:
        harness = RAGEvalHarness()
        report = harness.run()
        print(f"Avg RAGAS score: {report.avg_ragas_score:.3f}")
    """

    def __init__(
        self,
        qa_pairs: Optional[List[Dict[str, str]]] = None,
        top_k: int = 3,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.qa_pairs = qa_pairs or TEST_QA_PAIRS
        self.top_k = top_k
        self.output_dir = output_dir or RESULTS_DIR

    def _score_sample(
        self,
        sample_id: int,
        qa: Dict[str, str],
    ) -> EvalSample:
        t0 = time.time()
        generated, context = _generate_answer_and_context(qa["query"], top_k=self.top_k)
        latency_ms = round((time.time() - t0) * 1000, 1)

        faith = faithfulness_score(generated, context)
        relev = relevance_score(qa["query"], context)
        sim = answer_similarity(generated, qa["gold_answer"])
        prec = context_precision(context, qa["gold_answer"])
        rec = context_recall(context, qa["gold_answer"])

        # RAGAS composite = harmonic mean of all 5 metrics
        metrics = [faith, relev, sim, prec, rec]
        valid = [m for m in metrics if m > 0]
        if valid:
            ragas = len(valid) / sum(1.0 / m for m in valid)
        else:
            ragas = 0.0

        return EvalSample(
            sample_id=sample_id,
            query=qa["query"],
            gold_answer=qa["gold_answer"],
            generated_answer=generated,
            retrieved_chunks=context,
            faithfulness=faith,
            relevance=relev,
            answer_similarity=sim,
            context_precision=prec,
            context_recall=rec,
            ragas_score=round(ragas, 4),
            latency_ms=latency_ms,
        )

    def run(self, verbose: bool = True) -> EvalReport:
        """
        Run evaluation on all Q&A pairs.

        Returns:
            EvalReport with per-sample and aggregate metrics.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        report = EvalReport(
            run_timestamp=timestamp,
            num_samples=len(self.qa_pairs),
        )

        if verbose:
            print(f"[eval] Starting RAG evaluation — {len(self.qa_pairs)} samples")

        for i, qa in enumerate(self.qa_pairs):
            if verbose:
                print(f"[eval] Sample {i+1:02d}/{len(self.qa_pairs)}: {qa['query'][:60]}...")
            sample = self._score_sample(i + 1, qa)
            report.samples.append(sample)

            if verbose:
                print(
                    f"       faithful={sample.faithfulness:.3f} "
                    f"relev={sample.relevance:.3f} "
                    f"sim={sample.answer_similarity:.3f} "
                    f"prec={sample.context_precision:.3f} "
                    f"rec={sample.context_recall:.3f} "
                    f"ragas={sample.ragas_score:.3f} "
                    f"[{sample.latency_ms:.0f}ms]"
                )

        # Aggregate
        n = len(report.samples)
        if n > 0:
            report.avg_faithfulness = round(sum(s.faithfulness for s in report.samples) / n, 4)
            report.avg_relevance = round(sum(s.relevance for s in report.samples) / n, 4)
            report.avg_answer_similarity = round(sum(s.answer_similarity for s in report.samples) / n, 4)
            report.avg_context_precision = round(sum(s.context_precision for s in report.samples) / n, 4)
            report.avg_context_recall = round(sum(s.context_recall for s in report.samples) / n, 4)
            report.avg_ragas_score = round(sum(s.ragas_score for s in report.samples) / n, 4)
            report.avg_latency_ms = round(sum(s.latency_ms for s in report.samples) / n, 1)

        if verbose:
            print(f"\n[eval] --- Aggregate Results ---")
            print(f"  Faithfulness:       {report.avg_faithfulness:.3f}")
            print(f"  Relevance:          {report.avg_relevance:.3f}")
            print(f"  Answer Similarity:  {report.avg_answer_similarity:.3f}")
            print(f"  Context Precision:  {report.avg_context_precision:.3f}")
            print(f"  Context Recall:     {report.avg_context_recall:.3f}")
            print(f"  RAGAS Score:        {report.avg_ragas_score:.3f}")
            print(f"  Avg Latency:        {report.avg_latency_ms:.0f}ms")

        # Save CSV
        report.output_csv = self._save_csv(report)
        return report

    def _save_csv(self, report: EvalReport) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / f"eval_{report.run_timestamp}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if not report.samples:
                return csv_path

            fieldnames = list(report.samples[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample in report.samples:
                writer.writerow(sample.to_dict())

        print(f"[eval] Results saved to {csv_path}")
        return csv_path

    def get_scores_dataframe(self, report: EvalReport):
        """Return evaluation results as a pandas DataFrame."""
        import pandas as pd
        rows = [s.to_dict() for s in report.samples]
        return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation harness")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=3, help="Retrieval top-k")
    args = parser.parse_args()

    harness = RAGEvalHarness(
        top_k=args.top_k,
        output_dir=args.output.parent if args.output else RESULTS_DIR,
    )
    report = harness.run(verbose=True)
    print(f"\n[eval] CSV: {report.output_csv}")
