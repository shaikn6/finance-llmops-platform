"""
Tests for the RAG evaluation harness (evaluation/eval_harness.py).

All tests are deterministic and require no external API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.eval_harness import (
    faithfulness_score,
    relevance_score,
    answer_similarity,
    context_precision,
    context_recall,
    RAGEvalHarness,
    EvalSample,
    EvalReport,
    TEST_QA_PAIRS,
    _normalize,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

class TestTokenization:
    def test_normalize_lowercases(self):
        assert _normalize("HELLO WORLD") == "hello world"

    def test_normalize_collapses_whitespace(self):
        assert _normalize("  hello   world  ") == "hello world"

    def test_tokenize_returns_set(self):
        result = _tokenize("Hello world")
        assert isinstance(result, set)

    def test_tokenize_non_empty(self):
        result = _tokenize("revenue was $3.21 billion")
        assert len(result) > 0

    def test_tokenize_empty_string(self):
        result = _tokenize("")
        assert result == set()


# ---------------------------------------------------------------------------
# Faithfulness score tests
# ---------------------------------------------------------------------------

class TestFaithfulnessScore:
    GOOD_SOURCE = (
        "Meridian Financial Corp maintains an LCR of 134% as of December 31, 2023. "
        "The Company holds $2.84 billion in HQLA. CET1 ratio is 11.8%."
    )

    def test_perfect_grounding_returns_high_score(self):
        answer = "The LCR is 134%. HQLA is $2.84 billion."
        score = faithfulness_score(answer, [self.GOOD_SOURCE])
        assert score >= 0.7, f"Expected high faithfulness, got {score}"

    def test_hallucinated_figures_lower_score(self):
        answer = "Revenue was $999 trillion in 2099."
        score = faithfulness_score(answer, [self.GOOD_SOURCE])
        assert score <= 0.5

    def test_empty_answer_returns_one(self):
        score = faithfulness_score("", [self.GOOD_SOURCE])
        assert score == 1.0

    def test_no_context_returns_zero(self):
        score = faithfulness_score("The LCR is 134%.", [])
        assert score == 0.0

    def test_score_in_range(self):
        answer = "The LCR is 134% and the CET1 is 11.8%."
        score = faithfulness_score(answer, [self.GOOD_SOURCE])
        assert 0.0 <= score <= 1.0

    def test_answer_fully_from_source_high_faithfulness(self):
        answer = self.GOOD_SOURCE
        score = faithfulness_score(answer, [self.GOOD_SOURCE])
        assert score >= 0.7


# ---------------------------------------------------------------------------
# Relevance score tests
# ---------------------------------------------------------------------------

class TestRelevanceScore:
    def test_relevant_context_returns_positive_score(self):
        query = "What is the LCR ratio?"
        context = ["The LCR ratio is 134% as of December 31, 2023."]
        score = relevance_score(query, context)
        assert score > 0.0

    def test_irrelevant_context_returns_lower_score(self):
        query = "What is the LCR ratio?"
        irrelevant = ["The weather in New York is sunny. Temperatures are warm."]
        score = relevance_score(query, irrelevant)
        relevant = ["The LCR (liquidity coverage ratio) is 134% for Meridian Corp."]
        score_relevant = relevance_score(query, relevant)
        assert score < score_relevant

    def test_empty_context_returns_zero(self):
        score = relevance_score("What is the LCR?", [])
        assert score == 0.0

    def test_score_in_range(self):
        score = relevance_score("LCR ratio", ["The LCR is 134%."])
        assert 0.0 <= score <= 1.0

    def test_multiple_chunks_averaged(self):
        query = "What is the LCR?"
        contexts = [
            "LCR is 134%.",
            "Revenue was $3.21 billion.",
            "Unrelated text about weather.",
        ]
        score = relevance_score(query, contexts)
        assert 0.0 < score <= 1.0


# ---------------------------------------------------------------------------
# Answer similarity tests
# ---------------------------------------------------------------------------

class TestAnswerSimilarity:
    def test_identical_answers_return_one(self):
        text = "The LCR is 134% as of December 31, 2023."
        score = answer_similarity(text, text)
        assert score == 1.0

    def test_completely_different_answers_return_low(self):
        gen = "Revenue was $3.21 billion in fiscal 2023."
        gold = "The cybersecurity budget exceeded expectations."
        score = answer_similarity(gen, gold)
        assert score < 0.5

    def test_partially_overlapping_returns_intermediate(self):
        gen = "The LCR is 134% as of December 2023."
        gold = "Meridian maintains an LCR of 134% in December 2023."
        score = answer_similarity(gen, gold)
        assert 0.3 < score < 1.0

    def test_empty_generated_returns_zero(self):
        score = answer_similarity("", "The LCR is 134%.")
        assert score == 0.0

    def test_empty_gold_returns_zero(self):
        score = answer_similarity("The LCR is 134%.", "")
        assert score == 0.0

    def test_score_in_range(self):
        score = answer_similarity("The LCR is 134%.", "The LCR ratio is 134%.")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Context precision tests
# ---------------------------------------------------------------------------

class TestContextPrecision:
    def test_relevant_chunks_return_high_precision(self):
        gold = "The LCR is 134%. HQLA is $2.84 billion."
        contexts = [
            "LCR was 134% as of December 31, 2023.",
            "HQLA totaled $2.84 billion.",
        ]
        score = context_precision(contexts, gold)
        assert score >= 0.5

    def test_irrelevant_chunks_return_low_precision(self):
        gold = "The LCR is 134%."
        contexts = [
            "The weather today is sunny.",
            "Stock prices moved higher yesterday.",
        ]
        score = context_precision(contexts, gold)
        # "The" and "is" are stop-word overlap so score can be at most 0.5;
        # the precision ceiling for truly unrelated contexts should be <= 0.5
        assert score <= 0.5

    def test_empty_context_returns_zero(self):
        score = context_precision([], "The LCR is 134%.")
        assert score == 0.0

    def test_score_in_range(self):
        score = context_precision(["LCR is 134%."], "LCR is 134%.")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Context recall tests
# ---------------------------------------------------------------------------

class TestContextRecall:
    def test_context_covers_gold_facts_high_recall(self):
        gold = "The LCR is 134%. Revenue was $3.21 billion."
        contexts = [
            "The LCR (liquidity coverage ratio) is 134% as of December 31, 2023.",
            "Total net revenue was $3.21 billion for fiscal year 2023.",
        ]
        score = context_recall(contexts, gold)
        assert score >= 0.5

    def test_empty_context_returns_zero(self):
        score = context_recall([], "The LCR is 134%.")
        assert score == 0.0

    def test_empty_gold_returns_one(self):
        score = context_recall(["LCR is 134%."], "")
        assert score == 1.0

    def test_score_in_range(self):
        score = context_recall(["Revenue was $3.21 billion."], "Revenue was $3.21 billion.")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TEST_QA_PAIRS tests
# ---------------------------------------------------------------------------

class TestTestQAPairs:
    def test_has_twenty_pairs(self):
        assert len(TEST_QA_PAIRS) == 20

    def test_each_pair_has_query(self):
        for pair in TEST_QA_PAIRS:
            assert "query" in pair
            assert len(pair["query"]) > 10

    def test_each_pair_has_gold_answer(self):
        for pair in TEST_QA_PAIRS:
            assert "gold_answer" in pair
            assert len(pair["gold_answer"]) > 10

    def test_queries_are_financial(self):
        all_queries = " ".join(p["query"] for p in TEST_QA_PAIRS)
        financial_terms = [
            "ratio", "revenue", "capital", "margin", "loan",
            "liquidity", "debt", "earnings", "risk", "credit",
        ]
        assert any(term in all_queries.lower() for term in financial_terms)


# ---------------------------------------------------------------------------
# RAGEvalHarness integration tests (fast — uses mock pipeline)
# ---------------------------------------------------------------------------

class TestRAGEvalHarness:
    @pytest.fixture
    def harness(self, tmp_path):
        from evaluation.eval_harness import TEST_QA_PAIRS
        # Use only 3 pairs for speed
        return RAGEvalHarness(
            qa_pairs=TEST_QA_PAIRS[:3],
            top_k=2,
            output_dir=tmp_path / "eval_results",
        )

    def test_harness_runs(self, harness):
        report = harness.run(verbose=False)
        assert isinstance(report, EvalReport)

    def test_harness_correct_sample_count(self, harness):
        report = harness.run(verbose=False)
        assert report.num_samples == 3

    def test_harness_samples_populated(self, harness):
        report = harness.run(verbose=False)
        assert len(report.samples) == 3

    def test_harness_aggregate_metrics_in_range(self, harness):
        report = harness.run(verbose=False)
        for metric in [
            report.avg_faithfulness, report.avg_relevance,
            report.avg_answer_similarity, report.avg_context_precision,
            report.avg_context_recall, report.avg_ragas_score,
        ]:
            assert 0.0 <= metric <= 1.0, f"Metric out of range: {metric}"

    def test_harness_produces_csv(self, harness):
        report = harness.run(verbose=False)
        assert report.output_csv is not None
        assert report.output_csv.exists()

    def test_harness_csv_has_content(self, harness):
        report = harness.run(verbose=False)
        content = report.output_csv.read_text()
        assert "faithfulness" in content
        assert "ragas_score" in content

    def test_get_scores_dataframe(self, harness):
        import pandas as pd
        report = harness.run(verbose=False)
        df = harness.get_scores_dataframe(report)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "faithfulness" in df.columns
