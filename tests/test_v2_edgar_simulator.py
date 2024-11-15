"""
Tests for the SEC EDGAR simulator (data/edgar_simulator.py).

All tests are deterministic — pure in-memory, no network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.edgar_simulator import (
    EdgarSimulator,
    EdgarFiling,
    COMPANY_REGISTRY,
    _FILINGS,
)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestCompanyRegistry:
    def test_registry_has_ten_companies(self):
        assert len(COMPANY_REGISTRY) == 10

    def test_all_tickers_are_uppercase(self):
        for ticker in COMPANY_REGISTRY:
            assert ticker == ticker.upper()

    def test_each_company_has_required_fields(self):
        required = {"name", "sector", "fiscal_year_end", "total_assets", "cik"}
        for ticker, info in COMPANY_REGISTRY.items():
            missing = required - set(info.keys())
            assert not missing, f"{ticker} missing fields: {missing}"

    def test_all_tickers_in_filings(self):
        for ticker in COMPANY_REGISTRY:
            assert ticker in _FILINGS, f"{ticker} missing from _FILINGS"

    def test_all_filings_have_three_sections(self):
        expected = {"risk_factors", "mda", "financial_statements"}
        for ticker, sections in _FILINGS.items():
            assert set(sections.keys()) == expected, f"{ticker}: unexpected sections"


# ---------------------------------------------------------------------------
# EdgarSimulator.get_filing tests
# ---------------------------------------------------------------------------

class TestGetFiling:
    @pytest.fixture
    def sim(self):
        return EdgarSimulator()

    def test_get_filing_returns_edgarfiling(self, sim):
        filing = sim.get_filing("APEX", "risk_factors")
        assert isinstance(filing, EdgarFiling)

    def test_get_filing_correct_ticker(self, sim):
        filing = sim.get_filing("BRKR", "mda")
        assert filing.ticker == "BRKR"

    def test_get_filing_correct_section(self, sim):
        filing = sim.get_filing("APEX", "financial_statements")
        assert filing.section == "financial_statements"

    def test_get_filing_text_non_empty(self, sim):
        for ticker in COMPANY_REGISTRY:
            for section in ("risk_factors", "mda", "financial_statements"):
                filing = sim.get_filing(ticker, section)
                assert filing is not None
                assert len(filing.text) > 50, f"{ticker}/{section} text too short"

    def test_get_filing_unknown_ticker_returns_none(self, sim):
        result = sim.get_filing("ZZZZ", "mda")
        assert result is None

    def test_get_filing_unknown_section_returns_none(self, sim):
        result = sim.get_filing("APEX", "unknown_section")
        assert result is None

    def test_get_filing_case_insensitive_ticker(self, sim):
        filing = sim.get_filing("apex", "mda")
        assert filing is not None
        assert filing.ticker == "APEX"

    def test_get_filing_has_cik(self, sim):
        filing = sim.get_filing("APEX", "mda")
        assert filing.cik.startswith("000")

    def test_get_filing_has_doc_id(self, sim):
        filing = sim.get_filing("APEX", "risk_factors")
        assert filing.doc_id.startswith("EDGAR-")

    def test_get_filing_has_retrieved_at(self, sim):
        filing = sim.get_filing("APEX", "mda")
        assert "Z" in filing.retrieved_at  # ISO format ends with Z

    def test_get_filing_stable_doc_id(self, sim):
        """Same ticker+section always yields the same doc_id."""
        f1 = sim.get_filing("BRKR", "mda")
        f2 = sim.get_filing("BRKR", "mda")
        assert f1.doc_id == f2.doc_id


# ---------------------------------------------------------------------------
# EdgarSimulator.get_company_filings tests
# ---------------------------------------------------------------------------

class TestGetCompanyFilings:
    @pytest.fixture
    def sim(self):
        return EdgarSimulator()

    def test_returns_three_sections(self, sim):
        filings = sim.get_company_filings("APEX")
        assert len(filings) == 3

    def test_all_sections_present(self, sim):
        filings = sim.get_company_filings("DVRT")
        sections = {f.section for f in filings}
        assert sections == {"risk_factors", "mda", "financial_statements"}

    def test_all_same_ticker(self, sim):
        filings = sim.get_company_filings("ENCR")
        for filing in filings:
            assert filing.ticker == "ENCR"


# ---------------------------------------------------------------------------
# EdgarSimulator.get_all_filings tests
# ---------------------------------------------------------------------------

class TestGetAllFilings:
    @pytest.fixture
    def sim(self):
        return EdgarSimulator()

    def test_returns_thirty_filings(self, sim):
        filings = sim.get_all_filings()
        assert len(filings) == 30

    def test_all_tickers_represented(self, sim):
        filings = sim.get_all_filings()
        tickers_present = {f.ticker for f in filings}
        assert tickers_present == set(COMPANY_REGISTRY.keys())

    def test_no_duplicate_filings(self, sim):
        filings = sim.get_all_filings()
        doc_ids = [f.doc_id for f in filings]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_ids found"


# ---------------------------------------------------------------------------
# EdgarSimulator.list_companies tests
# ---------------------------------------------------------------------------

class TestListCompanies:
    @pytest.fixture
    def sim(self):
        return EdgarSimulator()

    def test_returns_ten_companies(self, sim):
        companies = sim.list_companies()
        assert len(companies) == 10

    def test_each_company_has_ticker(self, sim):
        for co in sim.list_companies():
            assert "ticker" in co

    def test_companies_sorted_alphabetically(self, sim):
        companies = sim.list_companies()
        tickers = [co["ticker"] for co in companies]
        assert tickers == sorted(tickers)


# ---------------------------------------------------------------------------
# EdgarSimulator.search_filings tests
# ---------------------------------------------------------------------------

class TestSearchFilings:
    @pytest.fixture
    def sim(self):
        return EdgarSimulator()

    def test_lcr_keyword_finds_filings(self, sim):
        results = sim.search_filings(["LCR", "liquidity coverage"])
        assert len(results) > 0

    def test_search_returns_edgarfiling_objects(self, sim):
        results = sim.search_filings(["CET1"])
        for r in results:
            assert isinstance(r, EdgarFiling)

    def test_nonexistent_keyword_returns_empty(self, sim):
        results = sim.search_filings(["XYZXYZ_NONEXISTENT_TOKEN_12345"])
        assert results == []

    def test_multiple_keywords_union(self, sim):
        single = sim.search_filings(["LCR"])
        multi = sim.search_filings(["LCR", "CET1"])
        # More keywords => at least as many results
        assert len(multi) >= len(single)


# ---------------------------------------------------------------------------
# EdgarSimulator.to_document_chunks tests
# ---------------------------------------------------------------------------

class TestToDocumentChunks:
    @pytest.fixture
    def sim(self):
        return EdgarSimulator()

    def test_all_chunks_returns_thirty(self, sim):
        chunks = sim.to_document_chunks()
        assert len(chunks) == 30

    def test_single_company_returns_three(self, sim):
        chunks = sim.to_document_chunks("APEX")
        assert len(chunks) == 3

    def test_chunks_have_required_keys(self, sim):
        chunks = sim.to_document_chunks("APEX")
        required = {"chunk_id", "doc_name", "doc_type", "text"}
        for chunk in chunks:
            assert required.issubset(chunk.keys())

    def test_doc_type_is_10k(self, sim):
        chunks = sim.to_document_chunks("APEX")
        for chunk in chunks:
            assert chunk["doc_type"] == "10k"

    def test_chunk_text_non_empty(self, sim):
        chunks = sim.to_document_chunks()
        for chunk in chunks:
            assert len(chunk["text"]) > 50
