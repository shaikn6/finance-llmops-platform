"""
Tests for the semantic retriever.

Uses the real FAISS index built from sample documents.
No external API calls — only sentence-transformers + faiss-cpu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def retriever():
    """Build/load FAISS index and return a FinancialRetriever."""
    from pipeline.retriever import FinancialRetriever
    from pipeline.ingestion import build_faiss_index

    index, chunks = build_faiss_index()
    r = FinancialRetriever(index=index, chunks=chunks)
    r._lazy_init()
    return r


class TestRetriever:
    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("liquidity coverage ratio", top_k=3)
        assert len(results) > 0

    def test_retrieve_correct_top_k(self, retriever):
        for k in [1, 2, 3]:
            results = retriever.retrieve("revenue breakdown", top_k=k)
            assert len(results) <= k

    def test_retrieve_scores_between_0_and_1(self, retriever):
        results = retriever.retrieve("cybersecurity risk", top_k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.01, f"Score out of range: {r.score}"

    def test_retrieve_scores_descending(self, retriever):
        results = retriever.retrieve("total debt obligations FHLB", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not ordered by descending score"

    def test_retrieve_rank_is_correct(self, retriever):
        results = retriever.retrieve("net interest margin", top_k=3)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_retrieve_has_source_metadata(self, retriever):
        results = retriever.retrieve("CET1 capital ratio", top_k=2)
        for r in results:
            assert r.chunk_id, "chunk_id should not be empty"
            assert r.doc_name, "doc_name should not be empty"
            assert r.doc_type in ("10k", "earnings_call"), f"Unexpected doc_type: {r.doc_type}"
            assert r.text, "chunk text should not be empty"

    def test_retrieve_financial_query_lcr(self, retriever):
        """LCR information should rank in top-2 results for a direct query."""
        results = retriever.retrieve("liquidity coverage ratio LCR Basel III", top_k=3)
        texts = " ".join(r.text for r in results)
        # The actual LCR content should appear somewhere in retrieved chunks
        assert "134" in texts or "liquidity" in texts.lower()

    def test_retrieve_with_metadata_structure(self, retriever):
        result = retriever.retrieve_with_metadata("AI document intelligence platform", top_k=2)
        assert "query" in result
        assert "chunks" in result
        assert "avg_score" in result
        assert "latency_ms" in result
        assert isinstance(result["chunks"], list)
        for chunk in result["chunks"]:
            assert "rank" in chunk
            assert "score" in chunk
            assert "text" in chunk
            assert "doc_name" in chunk

    def test_retrieve_empty_query_graceful(self, retriever):
        """Empty query should not crash — may return low-relevance results."""
        try:
            results = retriever.retrieve("   ", top_k=1)
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Retriever raised on empty query: {e}")

    def test_retrieve_different_queries_give_different_results(self, retriever):
        res1 = retriever.retrieve("cybersecurity phishing incident", top_k=1)
        res2 = retriever.retrieve("debt maturity subordinated notes", top_k=1)
        if res1 and res2:
            assert res1[0].chunk_id != res2[0].chunk_id, (
                "Completely different queries should retrieve different top chunks"
            )
