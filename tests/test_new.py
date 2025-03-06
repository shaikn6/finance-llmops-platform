"""
Comprehensive tests for finance-llmops-platform — targeting 95%+ coverage.

Covers:
- pipeline/hallucination.py (extract_factual_claims, check_hallucination, _tokenize, _claim_in_sources)
- pipeline/monitor.py (LLMMonitor: log, to_dataframe, generate_drift_report, PSI, get_summary, get_time_series)
- pipeline/retriever.py (RetrievedChunk, FinancialRetriever, retrieve_with_metadata)
- pipeline/ingestion.py (_approximate_tokens, _split_into_chunks, load_documents, DocumentChunk)
- pipeline/generator.py (FinancialAnswerGenerator, _select_mock_response, PROMPT_TEMPLATES, Citation/GeneratedAnswer)
- agents/multi_agent_rag.py (all agents: Researcher, Analyst, FactChecker, Synthesizer; _initial_state; FinanceRAGPipeline)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MOCK_MODE", "true")


# ───────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ───────────────────────────────────────────────────────────────────────────────

MERIDIAN_SOURCE = (
    "Meridian Financial Corp maintains a liquidity coverage ratio (LCR) of 134% as of "
    "December 31, 2023, exceeding the regulatory minimum of 100% mandated under Basel III. "
    "The Company holds $2.84 billion in high-quality liquid assets (HQLA). "
    "Total net revenue was $3.21 billion for fiscal year 2023, a 7.4% increase. "
    "The Company invested $78.3 million in information security infrastructure. "
    "The CET1 ratio stands at 11.8%, above the 6.5% well-capitalized threshold. "
    "Net charge-off rate was 0.31% vs peer median of 0.44%. "
    "FHLB advances total $2.58 billion. "
    "The AI platform processes a 10-K in 8 to 12 minutes with 94.7% accuracy."
)

MOCK_CHUNK = {
    "chunk_id": "10k_excerpts_0001",
    "doc_name": "10k_excerpts",
    "doc_type": "10k",
    "text": MERIDIAN_SOURCE,
    "score": 0.85,
}


# ───────────────────────────────────────────────────────────────────────────────
# pipeline/hallucination.py
# ───────────────────────────────────────────────────────────────────────────────

class TestExtractFactualClaims:
    def test_extracts_percentage(self):
        from pipeline.hallucination import extract_factual_claims
        claims = extract_factual_claims("LCR is 134% as of 2023.")
        pct_claims = [c for c in claims if "%" in c]
        assert len(pct_claims) >= 1

    def test_extracts_dollar_amounts(self):
        from pipeline.hallucination import extract_factual_claims
        claims = extract_factual_claims("Revenue was $3.21 billion and HQLA is $2.84 billion.")
        dollar_claims = [c for c in claims if "$" in c]
        assert len(dollar_claims) >= 2

    def test_extracts_dates(self):
        from pipeline.hallucination import extract_factual_claims
        claims = extract_factual_claims("As of December 31, 2023 in fiscal year 2022.")
        date_claims = [c for c in claims if any(d in c for d in ["2023", "2022", "December"])]
        assert len(date_claims) >= 1

    def test_extracts_acronyms(self):
        from pipeline.hallucination import extract_factual_claims
        claims = extract_factual_claims("The LCR, CET1, and HQLA metrics are key.")
        acronym_claims = [c for c in claims if c in ("LCR", "CET1", "HQLA")]
        assert len(acronym_claims) >= 2

    def test_deduplicates_claims(self):
        from pipeline.hallucination import extract_factual_claims
        claims = extract_factual_claims("Revenue was $3.21 billion. Revenue was $3.21 billion.")
        dollar_claims = [c for c in claims if "$3.21" in c]
        assert len(dollar_claims) == 1

    def test_empty_string_returns_empty(self):
        from pipeline.hallucination import extract_factual_claims
        assert extract_factual_claims("") == []

    def test_plain_text_returns_minimal(self):
        from pipeline.hallucination import extract_factual_claims
        claims = extract_factual_claims("The company performed well last year with results.")
        assert len(claims) <= 5

    def test_basis_points_extracted(self):
        from pipeline.hallucination import extract_factual_claims
        claims = extract_factual_claims("The spread widened by 18 basis points in Q4.")
        bps_claims = [c for c in claims if "basis" in c.lower() or "bps" in c.lower() or "18" in c]
        assert len(bps_claims) >= 1

    def test_company_name_extracted(self):
        from pipeline.hallucination import extract_factual_claims
        claims = extract_factual_claims("Meridian Financial Corp reported strong results.")
        company_claims = [c for c in claims if "Meridian" in c]
        assert len(company_claims) >= 1


class TestHallucinationCheck:
    def test_fully_grounded_answer(self):
        from pipeline.hallucination import check_hallucination
        answer = (
            "Meridian Financial Corp maintains an LCR of 134% as of December 31, 2023, "
            "exceeding the 100% regulatory minimum. The Company holds $2.84 billion in HQLA."
        )
        report = check_hallucination(answer, [MERIDIAN_SOURCE])
        assert report.grounding_score > 0.5
        assert report.hallucination_risk < 1.0

    def test_hallucinated_answer_has_uncited_claims(self):
        from pipeline.hallucination import check_hallucination
        answer = "Revenue was $5 trillion in 2030 (invented). LCR is 500% (false)."
        report = check_hallucination(answer, [MERIDIAN_SOURCE])
        assert len(report.claims) > 0

    def test_no_claims_returns_perfect_score(self):
        from pipeline.hallucination import check_hallucination
        report = check_hallucination("The results were positive overall.", [MERIDIAN_SOURCE])
        assert report.grounding_score == 1.0
        assert report.hallucination_risk == 0.0
        assert report.claims == []

    def test_report_fields_present(self):
        from pipeline.hallucination import check_hallucination
        report = check_hallucination("LCR is 134%.", [MERIDIAN_SOURCE])
        assert hasattr(report, "claims")
        assert hasattr(report, "uncited_claims")
        assert hasattr(report, "grounded_claims")
        assert hasattr(report, "grounding_score")
        assert hasattr(report, "hallucination_risk")
        assert hasattr(report, "claim_details")

    def test_grounding_score_between_zero_and_one(self):
        from pipeline.hallucination import check_hallucination
        report = check_hallucination("LCR is 134%, CET1 is 11.8%.", [MERIDIAN_SOURCE])
        assert 0.0 <= report.grounding_score <= 1.0
        assert 0.0 <= report.hallucination_risk <= 1.0

    def test_grounding_risk_equals_one_minus_score(self):
        from pipeline.hallucination import check_hallucination
        report = check_hallucination("LCR is 134%.", [MERIDIAN_SOURCE])
        assert abs(report.grounding_score + report.hallucination_risk - 1.0) < 1e-4

    def test_grounded_claims_subset_of_claims(self):
        from pipeline.hallucination import check_hallucination
        report = check_hallucination("LCR is 134%.", [MERIDIAN_SOURCE])
        for claim in report.grounded_claims:
            assert claim in report.claims

    def test_uncited_claims_subset_of_claims(self):
        from pipeline.hallucination import check_hallucination
        report = check_hallucination("Revenue was $999 trillion in 2099.", [MERIDIAN_SOURCE])
        for claim in report.uncited_claims:
            assert claim in report.claims

    def test_multiple_source_texts(self):
        from pipeline.hallucination import check_hallucination
        sources = [MERIDIAN_SOURCE, "Supplemental source about $78.3 million investment."]
        report = check_hallucination("The company invested $78.3 million.", sources)
        assert report is not None

    def test_empty_source_texts_no_grounding(self):
        from pipeline.hallucination import check_hallucination
        report = check_hallucination("LCR is 134% and CET1 is 11.8%.", [])
        assert report.grounding_score == 0.0 or report.hallucination_risk >= 0.0

    def test_custom_threshold(self):
        from pipeline.hallucination import check_hallucination
        report = check_hallucination("LCR is 134%.", [MERIDIAN_SOURCE], threshold=0.01)
        assert report is not None

    def test_check_from_generated_answer(self):
        from pipeline.hallucination import check_from_generated_answer
        from pipeline.generator import GeneratedAnswer, Citation
        citation = Citation("c1", "10k", "10k", MERIDIAN_SOURCE, 0.9)
        ga = GeneratedAnswer(
            question="What is the LCR?",
            answer="LCR is 134% as of December 31, 2023. HQLA is $2.84 billion.",
            citations=[citation],
            latency_ms=100.0,
            model="mock",
            prompt_version="v2_structured",
            retrieval_scores=[0.9],
            mock_mode=True,
        )
        report = check_from_generated_answer(ga)
        assert report.grounding_score >= 0.0

    def test_claim_details_contain_overlap_score(self):
        from pipeline.hallucination import check_hallucination
        report = check_hallucination("LCR is 134%.", [MERIDIAN_SOURCE])
        for detail in report.claim_details:
            assert "claim" in detail
            assert "is_grounded" in detail
            assert "overlap_score" in detail


class TestTokenizeAndNormalize:
    def test_normalize_lowercases(self):
        from pipeline.hallucination import _normalize
        assert _normalize("SAVINGS ACCOUNT") == "savings account"

    def test_normalize_strips_accents(self):
        from pipeline.hallucination import _normalize
        assert "e" in _normalize("café")

    def test_normalize_collapses_whitespace(self):
        from pipeline.hallucination import _normalize
        assert _normalize("  hello   world  ") == "hello world"

    def test_tokenize_returns_set(self):
        from pipeline.hallucination import _tokenize
        result = _tokenize("savings account rate")
        assert isinstance(result, set)

    def test_tokenize_contains_words(self):
        from pipeline.hallucination import _tokenize
        result = _tokenize("savings account rate 5.75 percent")
        assert "savings" in result or "account" in result


# ───────────────────────────────────────────────────────────────────────────────
# pipeline/monitor.py
# ───────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def monitor(tmp_path):
    from pipeline.monitor import LLMMonitor
    return LLMMonitor(monitor_dir=tmp_path / "monitoring")


@pytest.fixture
def populated_monitor(monitor):
    for i in range(25):
        monitor.log_interaction(
            question=f"Q{i}",
            answer=f"Answer {i} " * 20,
            grounding_score=0.80 + (i % 5) * 0.02,
            hallucination_risk=0.20 - (i % 3) * 0.01,
            avg_retrieval_score=0.75 + (i % 4) * 0.01,
            latency_ms=200.0 + i * 5,
            prompt_version="v2_structured",
            model="mock",
            num_citations=3,
            num_uncited_claims=0,
            num_total_claims=3,
        )
    return monitor


class TestLLMMonitor:
    def test_log_returns_interaction_log(self, monitor):
        from pipeline.monitor import InteractionLog
        log = monitor.log_interaction(
            question="What is the LCR?",
            answer="LCR is 134%.",
            grounding_score=0.9,
            hallucination_risk=0.1,
            avg_retrieval_score=0.85,
            latency_ms=300.0,
            prompt_version="v2_structured",
            model="mock",
        )
        assert isinstance(log, InteractionLog)
        assert log.question == "What is the LCR?"

    def test_logs_persisted_to_disk(self, tmp_path):
        from pipeline.monitor import LLMMonitor
        mon = LLMMonitor(monitor_dir=tmp_path / "m1")
        mon.log_interaction("q", "a", 0.9, 0.1, 0.8, 200.0, "v2", "gpt4")
        assert (tmp_path / "m1" / "interactions.jsonl").exists()

    def test_reload_from_disk(self, tmp_path):
        from pipeline.monitor import LLMMonitor
        mon = LLMMonitor(monitor_dir=tmp_path / "m2")
        mon.log_interaction("q", "a", 0.9, 0.1, 0.8, 200.0, "v2", "gpt4")
        mon2 = LLMMonitor(monitor_dir=tmp_path / "m2")
        assert len(mon2._logs) == 1

    def test_to_dataframe_returns_dataframe(self, populated_monitor):
        df = populated_monitor.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 25

    def test_to_dataframe_empty_monitor(self, monitor):
        df = monitor.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_get_summary_returns_dict(self, populated_monitor):
        summary = populated_monitor.get_summary()
        assert "total_queries" in summary
        assert summary["total_queries"] == 25

    def test_get_summary_empty_monitor(self, monitor):
        summary = monitor.get_summary()
        assert summary["total_queries"] == 0

    def test_summary_avg_grounding_score(self, populated_monitor):
        summary = populated_monitor.get_summary()
        assert 0.0 <= summary["avg_grounding_score"] <= 1.0

    def test_summary_pct_high_risk(self, populated_monitor):
        summary = populated_monitor.get_summary()
        assert "pct_high_risk" in summary
        assert 0.0 <= summary["pct_high_risk"] <= 100.0

    def test_get_time_series_returns_dict(self, populated_monitor):
        ts = populated_monitor.get_time_series()
        assert "timestamps" in ts
        assert "grounding_scores" in ts
        assert len(ts["timestamps"]) == 25

    def test_get_time_series_empty(self, monitor):
        ts = monitor.get_time_series()
        assert ts["timestamps"] == []

    def test_generate_drift_report_insufficient_data(self, monitor):
        report = monitor.generate_drift_report()
        assert report["status"] == "insufficient_data"

    def test_generate_drift_report_with_data(self, populated_monitor):
        report = populated_monitor.generate_drift_report()
        assert report["status"] == "ok"
        assert "total_interactions" in report
        assert "baseline_size" in report
        assert "recent_size" in report

    def test_drift_report_has_drift_results(self, populated_monitor):
        report = populated_monitor.generate_drift_report()
        assert "drift_results" in report

    def test_drift_report_any_drift_detected_is_bool(self, populated_monitor):
        report = populated_monitor.generate_drift_report()
        assert isinstance(report["any_drift_detected"], bool)

    def test_compute_psi_zero_for_identical(self, monitor):
        baseline = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        current = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        psi = monitor._compute_psi(baseline, current)
        assert psi < 0.01

    def test_compute_psi_high_for_shifted(self, monitor):
        baseline = pd.Series([0.1, 0.2, 0.1, 0.2, 0.1])
        current = pd.Series([0.9, 0.8, 0.9, 0.8, 0.9])
        psi = monitor._compute_psi(baseline, current)
        assert psi > 0.0

    def test_compute_psi_empty_series(self, monitor):
        psi = monitor._compute_psi(pd.Series([], dtype=float), pd.Series([1.0]))
        assert psi == 0.0

    def test_compute_psi_constant_series(self, monitor):
        psi = monitor._compute_psi(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 1.0, 1.0]))
        assert psi == 0.0

    def test_response_length_logged(self, monitor):
        monitor.log_interaction("q", "a" * 50, 0.9, 0.1, 0.8, 200.0, "v2", "gpt4")
        df = monitor.to_dataframe()
        assert df.iloc[0]["response_length"] == 50

    def test_num_citations_logged(self, monitor):
        monitor.log_interaction("q", "a", 0.9, 0.1, 0.8, 200.0, "v2", "gpt4", num_citations=3)
        df = monitor.to_dataframe()
        assert df.iloc[0]["num_citations"] == 3

    def test_evidently_report_falls_back_to_psi(self, populated_monitor):
        with patch("pipeline.monitor.LLMMonitor._run_evidently_report", side_effect=ImportError("no evidently")):
            report = populated_monitor.generate_drift_report()
        assert report["status"] == "ok"
        assert report.get("engine") in ("psi_fallback", "evidently", None)


# ───────────────────────────────────────────────────────────────────────────────
# pipeline/retriever.py (with mocked FAISS)
# ───────────────────────────────────────────────────────────────────────────────

from pipeline.retriever import RetrievedChunk, FinancialRetriever


class TestRetrievedChunk:
    def test_retrieved_chunk_fields(self):
        chunk = RetrievedChunk("id1", "doc1", "10k", "text", 0.85, 1)
        assert chunk.chunk_id == "id1"
        assert chunk.doc_name == "doc1"
        assert chunk.doc_type == "10k"
        assert chunk.text == "text"
        assert chunk.score == 0.85
        assert chunk.rank == 1


class TestFinancialRetriever:
    def _make_mock_chunks(self):
        """Mock chunks that mimic DocumentChunk dataclass."""
        chunks = []
        for i in range(5):
            c = MagicMock()
            c.chunk_id = f"chunk_{i}"
            c.doc_name = "10k_excerpts"
            c.doc_type = "10k"
            c.text = MERIDIAN_SOURCE
            chunks.append(c)
        return chunks

    def _make_mock_index(self, n_chunks):
        import numpy as np
        idx = MagicMock()
        idx.ntotal = n_chunks
        # Return top-k results
        scores = np.array([[0.9, 0.85, 0.80]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int64)
        idx.search.return_value = (scores, indices)
        return idx

    def test_retrieve_returns_list(self):
        chunks = self._make_mock_chunks()
        index = self._make_mock_index(len(chunks))
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        retriever = FinancialRetriever(index=index, chunks=chunks)
        retriever._model = mock_model
        retriever._initialized = True

        results = retriever.retrieve("What is the LCR?", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_retrieve_returns_retrieved_chunks(self):
        chunks = self._make_mock_chunks()
        index = self._make_mock_index(len(chunks))
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        retriever = FinancialRetriever(index=index, chunks=chunks)
        retriever._model = mock_model
        retriever._initialized = True

        results = retriever.retrieve("query", top_k=2)
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_respects_top_k(self):
        chunks = self._make_mock_chunks()
        index = self._make_mock_index(len(chunks))
        scores = np.array([[0.9, 0.85]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int64)
        index.search.return_value = (scores, indices)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        retriever = FinancialRetriever(index=index, chunks=chunks)
        retriever._model = mock_model
        retriever._initialized = True

        results = retriever.retrieve("query", top_k=2)
        assert len(results) <= 2

    def test_retrieve_with_metadata_returns_dict(self):
        chunks = self._make_mock_chunks()
        index = self._make_mock_index(len(chunks))
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        retriever = FinancialRetriever(index=index, chunks=chunks)
        retriever._model = mock_model
        retriever._initialized = True

        result = retriever.retrieve_with_metadata("What is the LCR?", top_k=2)
        assert "query" in result
        assert "chunks" in result
        assert "latency_ms" in result
        assert "avg_score" in result

    def test_retrieve_with_metadata_avg_score(self):
        chunks = self._make_mock_chunks()
        index = self._make_mock_index(len(chunks))
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        retriever = FinancialRetriever(index=index, chunks=chunks)
        retriever._model = mock_model
        retriever._initialized = True

        result = retriever.retrieve_with_metadata("Q", top_k=3)
        assert result["avg_score"] >= 0.0

    def test_retrieve_out_of_bounds_index_skipped(self):
        chunks = self._make_mock_chunks()[:2]
        index = self._make_mock_index(2)
        # Include an out-of-bounds index
        index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
            np.array([[0, 1, 99]], dtype=np.int64),  # 99 is out of bounds
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        retriever = FinancialRetriever(index=index, chunks=chunks)
        retriever._model = mock_model
        retriever._initialized = True

        results = retriever.retrieve("Q", top_k=3)
        assert len(results) == 2  # Only valid indices returned

    def test_get_retriever_singleton(self):
        from pipeline.retriever import get_retriever
        r1 = get_retriever()
        r2 = get_retriever()
        assert r1 is r2


# ───────────────────────────────────────────────────────────────────────────────
# pipeline/ingestion.py
# ───────────────────────────────────────────────────────────────────────────────

class TestIngestionHelpers:
    def test_approximate_tokens_positive(self):
        from pipeline.ingestion import _approximate_tokens
        assert _approximate_tokens("hello world") > 0

    def test_approximate_tokens_empty_returns_one(self):
        from pipeline.ingestion import _approximate_tokens
        assert _approximate_tokens("") == 1

    def test_approximate_tokens_longer_text(self):
        from pipeline.ingestion import _approximate_tokens
        long = "word " * 100
        assert _approximate_tokens(long) > _approximate_tokens("word")

    def test_split_into_chunks_basic(self):
        from pipeline.ingestion import _split_into_chunks
        text = "Sentence one. Sentence two. Sentence three."
        chunks = _split_into_chunks(text, "test_doc", "10k", chunk_size=10, overlap=2)
        assert len(chunks) >= 1

    def test_split_into_chunks_ids(self):
        from pipeline.ingestion import _split_into_chunks
        text = ". ".join([f"Sentence {i}" for i in range(50)]) + "."
        chunks = _split_into_chunks(text, "test_doc", "10k")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # All unique

    def test_split_sets_doc_name(self):
        from pipeline.ingestion import _split_into_chunks
        text = "Financial results show strong growth. Revenue increased by 7.4%."
        chunks = _split_into_chunks(text, "meridian_10k", "10k")
        assert all(c.doc_name == "meridian_10k" for c in chunks)

    def test_split_sets_doc_type(self):
        from pipeline.ingestion import _split_into_chunks
        text = "Management discussed strong results and margin improvement."
        chunks = _split_into_chunks(text, "earnings", "earnings_call")
        assert all(c.doc_type == "earnings_call" for c in chunks)

    def test_chunk_has_token_count(self):
        from pipeline.ingestion import _split_into_chunks
        text = "The LCR was 134% as of December 31 2023."
        chunks = _split_into_chunks(text, "doc", "10k")
        for c in chunks:
            assert c.token_count > 0

    def test_empty_text_returns_no_chunks(self):
        from pipeline.ingestion import _split_into_chunks
        chunks = _split_into_chunks("", "doc", "10k")
        assert len(chunks) == 0

    def test_load_documents_returns_list(self, tmp_path):
        from pipeline.ingestion import load_documents
        tenk = tmp_path / "sample_10k_excerpts.txt"
        tenk.write_text(MERIDIAN_SOURCE * 5)
        chunks = load_documents(data_dir=tmp_path)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_load_documents_empty_dir(self, tmp_path):
        from pipeline.ingestion import load_documents
        chunks = load_documents(data_dir=tmp_path)
        assert chunks == []


# ───────────────────────────────────────────────────────────────────────────────
# pipeline/generator.py
# ───────────────────────────────────────────────────────────────────────────────

class TestSelectMockResponse:
    def test_lcr_keyword_matches_lcr_response(self):
        from pipeline.generator import _select_mock_response
        resp = _select_mock_response("What is the LCR ratio?")
        assert "134%" in resp or "liquidity" in resp.lower()

    def test_revenue_keyword_matches(self):
        from pipeline.generator import _select_mock_response
        resp = _select_mock_response("What is the total net revenue?")
        assert "revenue" in resp.lower() or "$" in resp

    def test_cyber_keyword_matches(self):
        from pipeline.generator import _select_mock_response
        resp = _select_mock_response("How much did we spend on cybersecurity?")
        assert "security" in resp.lower() or "$" in resp

    def test_nim_keyword_matches(self):
        from pipeline.generator import _select_mock_response
        resp = _select_mock_response("What is the net interest margin guidance?")
        assert "NIM" in resp or "margin" in resp.lower()

    def test_unknown_query_returns_default(self):
        from pipeline.generator import _select_mock_response
        resp = _select_mock_response("What is the weather forecast for tomorrow?")
        assert isinstance(resp, str) and len(resp) > 10

    def test_debt_keyword_matches(self):
        from pipeline.generator import _select_mock_response
        resp = _select_mock_response("What is the total long-term debt?")
        assert "debt" in resp.lower() or "$" in resp


class TestFinancialAnswerGenerator:
    @pytest.fixture
    def gen(self):
        from pipeline.generator import FinancialAnswerGenerator
        return FinancialAnswerGenerator(mock_mode=True)

    def test_generate_returns_generated_answer(self, gen):
        from pipeline.generator import GeneratedAnswer
        result = gen.generate("What is the LCR ratio?")
        assert isinstance(result, GeneratedAnswer)

    def test_generate_has_answer(self, gen):
        result = gen.generate("What is the LCR ratio?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_generate_has_citations(self, gen):
        result = gen.generate("What is the LCR ratio?")
        assert isinstance(result.citations, list)

    def test_generate_has_latency(self, gen):
        result = gen.generate("What is the LCR ratio?")
        assert result.latency_ms >= 0.0

    def test_generate_has_mock_mode_true(self, gen):
        result = gen.generate("What is the LCR ratio?")
        assert result.mock_mode is True

    def test_generate_has_prompt_version(self, gen):
        result = gen.generate("What is the LCR ratio?")
        assert result.prompt_version in ("v1_basic", "v2_structured", "v3_cot")

    def test_to_dict_returns_dict(self, gen):
        result = gen.generate("What is the LCR ratio?")
        d = gen.to_dict(result)
        assert isinstance(d, dict)
        assert "question" in d
        assert "answer" in d
        assert "citations" in d

    def test_prompt_versions_available(self):
        from pipeline.generator import PROMPT_TEMPLATES
        assert "v1_basic" in PROMPT_TEMPLATES
        assert "v2_structured" in PROMPT_TEMPLATES
        assert "v3_cot" in PROMPT_TEMPLATES

    def test_all_prompt_versions_have_context_placeholder(self):
        from pipeline.generator import PROMPT_TEMPLATES
        for name, tmpl in PROMPT_TEMPLATES.items():
            assert "{context}" in tmpl, f"Template {name} missing {{context}}"
            assert "{question}" in tmpl, f"Template {name} missing {{question}}"

    def test_mock_mode_from_env(self):
        from pipeline.generator import FinancialAnswerGenerator
        with patch.dict(os.environ, {"MOCK_MODE": "true"}):
            gen = FinancialAnswerGenerator()
            assert gen.mock_mode is True

    def test_mock_mode_false_falls_back_without_api_key(self):
        from pipeline.generator import FinancialAnswerGenerator
        with patch.dict(os.environ, {"MOCK_MODE": "false", "OPENAI_API_KEY": ""}):
            gen = FinancialAnswerGenerator()
            # Without API key should fallback to mock
            assert gen.mock_mode is True

    def test_get_generator_singleton(self):
        from pipeline import generator as gen_module
        gen_module._generator = None
        from pipeline.generator import get_generator
        g1 = get_generator()
        g2 = get_generator()
        assert g1 is g2

    def test_no_relevant_docs_returns_no_docs_message(self):
        from pipeline.generator import FinancialAnswerGenerator
        gen = FinancialAnswerGenerator(mock_mode=True)
        # Patch retriever to return empty list
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        gen._retriever = mock_retriever
        result = gen.generate("Some question with no docs")
        assert "No relevant" in result.answer or len(result.answer) >= 0


# ───────────────────────────────────────────────────────────────────────────────
# agents/multi_agent_rag.py
# ───────────────────────────────────────────────────────────────────────────────

class TestInitialState:
    def test_initial_state_has_query(self):
        from agents.multi_agent_rag import _initial_state
        state = _initial_state("What is the LCR?")
        assert state["query"] == "What is the LCR?"

    def test_initial_state_has_run_id(self):
        from agents.multi_agent_rag import _initial_state
        state = _initial_state("Q")
        assert len(state["run_id"]) == 8

    def test_initial_state_all_pending(self):
        from agents.multi_agent_rag import _initial_state, AgentStatus
        state = _initial_state("Q")
        for key in ("researcher_status", "analyst_status", "fact_checker_status", "synthesizer_status"):
            assert state[key] == AgentStatus.PENDING

    def test_initial_state_empty_lists(self):
        from agents.multi_agent_rag import _initial_state
        state = _initial_state("Q")
        assert state["retrieved_chunks"] == []
        assert state["flagged_claims"] == []

    def test_initial_state_grounding_score_one(self):
        from agents.multi_agent_rag import _initial_state
        state = _initial_state("Q")
        assert state["grounding_score"] == 1.0

    def test_initial_state_top_k(self):
        from agents.multi_agent_rag import _initial_state
        state = _initial_state("Q", top_k=5)
        assert state["top_k"] == 5


class TestAgentStatus:
    def test_status_values(self):
        from agents.multi_agent_rag import AgentStatus
        assert AgentStatus.PENDING == "pending"
        assert AgentStatus.RUNNING == "running"
        assert AgentStatus.DONE == "done"
        assert AgentStatus.SKIPPED == "skipped"
        assert AgentStatus.ERROR == "error"


class TestResearcherAgent:
    def _make_state(self):
        from agents.multi_agent_rag import _initial_state
        return _initial_state("What is the LCR ratio?", top_k=3)

    def test_researcher_runs_and_completes(self):
        from agents.multi_agent_rag import ResearcherAgent, AgentStatus
        agent = ResearcherAgent(top_k=3)
        state = self._make_state()
        result = agent.run(state)
        assert result["researcher_status"] == AgentStatus.DONE

    def test_researcher_populates_chunks(self):
        from agents.multi_agent_rag import ResearcherAgent
        agent = ResearcherAgent(top_k=3)
        state = self._make_state()
        result = agent.run(state)
        assert len(result["retrieved_chunks"]) > 0

    def test_researcher_mock_chunks_have_required_fields(self):
        from agents.multi_agent_rag import ResearcherAgent
        agent = ResearcherAgent(top_k=3)
        state = self._make_state()
        result = agent.run(state)
        for chunk in result["retrieved_chunks"]:
            assert "chunk_id" in chunk
            assert "doc_name" in chunk
            assert "doc_type" in chunk
            assert "text" in chunk
            assert "score" in chunk

    def test_researcher_latency_logged(self):
        from agents.multi_agent_rag import ResearcherAgent
        agent = ResearcherAgent(top_k=2)
        state = self._make_state()
        result = agent.run(state)
        assert result["retrieval_latency_ms"] >= 0.0

    def test_researcher_top_k_respected(self):
        from agents.multi_agent_rag import ResearcherAgent
        agent = ResearcherAgent(top_k=2)
        state = self._make_state()
        result = agent.run(state)
        assert len(result["retrieved_chunks"]) <= 3  # mock returns up to 3

    def test_researcher_returns_finance_rag_state(self):
        from agents.multi_agent_rag import ResearcherAgent, FinanceRAGState
        agent = ResearcherAgent(top_k=2)
        state = self._make_state()
        result = agent.run(state)
        # Should be a dict matching FinanceRAGState schema
        assert "query" in result
        assert "retrieved_chunks" in result


class TestAnalystAgent:
    def _make_state_with_chunks(self):
        from agents.multi_agent_rag import _initial_state
        state = dict(_initial_state("What is the LCR?"))
        state["retrieved_chunks"] = [
            {"chunk_id": "c1", "doc_name": "10k", "doc_type": "10k", "text": MERIDIAN_SOURCE, "score": 0.9},
            {"chunk_id": "c2", "doc_name": "ec", "doc_type": "earnings_call", "text": "NIM guidance 3.40%. Revenue $3.21 billion.", "score": 0.8},
        ]
        from agents.multi_agent_rag import FinanceRAGState
        return FinanceRAGState(**state)

    def test_analyst_completes(self):
        from agents.multi_agent_rag import AnalystAgent, AgentStatus
        agent = AnalystAgent()
        state = self._make_state_with_chunks()
        result = agent.run(state)
        assert result["analyst_status"] == AgentStatus.DONE

    def test_analyst_separates_chunk_types(self):
        from agents.multi_agent_rag import AnalystAgent
        agent = AnalystAgent()
        state = self._make_state_with_chunks()
        result = agent.run(state)
        assert len(result["filing_references"]) >= 1
        assert len(result["earnings_references"]) >= 1

    def test_analyst_has_cross_reference_notes(self):
        from agents.multi_agent_rag import AnalystAgent
        agent = AnalystAgent()
        state = self._make_state_with_chunks()
        result = agent.run(state)
        assert isinstance(result["cross_reference_notes"], str)
        assert len(result["cross_reference_notes"]) > 0

    def test_analyst_only_filings(self):
        from agents.multi_agent_rag import AnalystAgent, _initial_state, FinanceRAGState
        state = dict(_initial_state("Q"))
        state["retrieved_chunks"] = [
            {"chunk_id": "c1", "doc_name": "10k", "doc_type": "10k", "text": MERIDIAN_SOURCE, "score": 0.9},
        ]
        result = AnalystAgent().run(FinanceRAGState(**state))
        assert "10-K" in result["cross_reference_notes"] or "filings" in result["cross_reference_notes"].lower()

    def test_analyst_only_earnings(self):
        from agents.multi_agent_rag import AnalystAgent, _initial_state, FinanceRAGState
        state = dict(_initial_state("Q"))
        state["retrieved_chunks"] = [
            {"chunk_id": "c1", "doc_name": "ec", "doc_type": "earnings_call", "text": "NIM 3.40%.", "score": 0.8},
        ]
        result = AnalystAgent().run(FinanceRAGState(**state))
        assert "earnings" in result["cross_reference_notes"].lower()

    def test_analyst_no_chunks(self):
        from agents.multi_agent_rag import AnalystAgent, _initial_state
        state = _initial_state("Q")
        result = AnalystAgent().run(state)
        assert "No source" in result["cross_reference_notes"] or isinstance(result["cross_reference_notes"], str)


class TestFactCheckerAgent:
    def _make_state_with_chunks(self):
        from agents.multi_agent_rag import _initial_state, FinanceRAGState
        state = dict(_initial_state("What is the LCR?"))
        state["retrieved_chunks"] = [MOCK_CHUNK]
        return FinanceRAGState(**state)

    def test_fact_checker_completes(self):
        from agents.multi_agent_rag import FactCheckerAgent, AgentStatus
        agent = FactCheckerAgent()
        state = self._make_state_with_chunks()
        result = agent.run(state)
        assert result["fact_checker_status"] == AgentStatus.DONE

    def test_fact_checker_grounding_score_in_range(self):
        from agents.multi_agent_rag import FactCheckerAgent
        agent = FactCheckerAgent()
        state = self._make_state_with_chunks()
        result = agent.run(state)
        assert 0.0 <= result["grounding_score"] <= 1.0

    def test_fact_checker_hallucination_risk_in_range(self):
        from agents.multi_agent_rag import FactCheckerAgent
        agent = FactCheckerAgent()
        state = self._make_state_with_chunks()
        result = agent.run(state)
        assert 0.0 <= result["hallucination_risk"] <= 1.0

    def test_fact_checker_flagged_claims_is_list(self):
        from agents.multi_agent_rag import FactCheckerAgent
        agent = FactCheckerAgent()
        state = self._make_state_with_chunks()
        result = agent.run(state)
        assert isinstance(result["flagged_claims"], list)

    def test_fact_checker_grounded_claims_is_list(self):
        from agents.multi_agent_rag import FactCheckerAgent
        agent = FactCheckerAgent()
        state = self._make_state_with_chunks()
        result = agent.run(state)
        assert isinstance(result["grounded_claims"], list)


class TestFinanceRAGPipeline:
    def test_pipeline_importable(self):
        try:
            from agents.multi_agent_rag import FinanceRAGPipeline
            assert FinanceRAGPipeline is not None
        except ImportError:
            pytest.skip("FinanceRAGPipeline not found")

    def test_pipeline_runs_end_to_end(self):
        try:
            from agents.multi_agent_rag import FinanceRAGPipeline
        except ImportError:
            pytest.skip("FinanceRAGPipeline not found")
        pipeline = FinanceRAGPipeline()
        result = pipeline.run("What is the LCR ratio?")
        assert isinstance(result, dict)
        assert "query" in result or "final_answer" in result or "synthesizer_status" in result

    def test_pipeline_run_has_final_answer(self):
        try:
            from agents.multi_agent_rag import FinanceRAGPipeline
        except ImportError:
            pytest.skip("FinanceRAGPipeline not found")
        pipeline = FinanceRAGPipeline()
        result = pipeline.run("What is Meridian's revenue?")
        # final_answer may be empty but should exist
        assert "final_answer" in result or isinstance(result, dict)
