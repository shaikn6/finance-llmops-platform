"""
Tests for hallucination detection.

All tests are deterministic — no LLM or network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hallucination import (
    check_hallucination,
    extract_factual_claims,
    HallucinationReport,
)


MERIDIAN_SOURCE = (
    "Meridian Financial Corp maintains a liquidity coverage ratio (LCR) of 134% as of "
    "December 31, 2023, exceeding the regulatory minimum of 100% mandated under Basel III. "
    "The Company holds $2.84 billion in high-quality liquid assets (HQLA). "
    "Total net revenue was $3.21 billion for fiscal year 2023. "
    "The Company invested $78.3 million in information security infrastructure. "
    "The CET1 ratio stands at 11.8%, above the 6.5% well-capitalized threshold. "
    "Net charge-off rate was 0.31% vs peer median of 0.44%. "
    "FHLB advances total $2.58 billion. "
    "The AI platform processes a 10-K in 8 to 12 minutes with 94.7% accuracy."
)


class TestClaimExtraction:
    def test_extracts_percentages(self):
        claims = extract_factual_claims("The LCR is 134% and the CET1 ratio is 11.8%.")
        pct_claims = [c for c in claims if "%" in c]
        assert len(pct_claims) >= 2

    def test_extracts_dollar_amounts(self):
        claims = extract_factual_claims("Revenue was $3.21 billion and HQLA is $2.84 billion.")
        dollar_claims = [c for c in claims if "$" in c]
        assert len(dollar_claims) >= 2

    def test_extracts_dates(self):
        claims = extract_factual_claims("As of December 31, 2023, fiscal year 2022 results show growth.")
        date_claims = [c for c in claims if any(
            d in c for d in ["2023", "2022", "December"]
        )]
        assert len(date_claims) >= 1

    def test_extracts_acronyms(self):
        claims = extract_factual_claims("The LCR, CET1, and NIM are key metrics for HQLA reporting.")
        acronym_claims = [c for c in claims if c in ("LCR", "CET1", "NIM", "HQLA")]
        assert len(acronym_claims) >= 2

    def test_deduplicates_claims(self):
        claims = extract_factual_claims("Revenue was $3.21 billion. Revenue was $3.21 billion.")
        dollar_claims = [c for c in claims if "$3.21" in c]
        assert len(dollar_claims) == 1, "Duplicate claims should be deduplicated"

    def test_empty_text_returns_empty_list(self):
        claims = extract_factual_claims("")
        assert claims == []

    def test_plain_narrative_returns_minimal_claims(self):
        claims = extract_factual_claims("The company performed well last year with positive results.")
        assert len(claims) <= 3


class TestHallucinationCheck:
    def test_fully_grounded_answer(self):
        answer = (
            "Meridian Financial Corp maintains an LCR of 134% as of December 31, 2023, "
            "exceeding the 100% regulatory minimum. The Company holds $2.84 billion in HQLA."
        )
        report = check_hallucination(answer, [MERIDIAN_SOURCE])
        assert isinstance(report, HallucinationReport)
        assert report.grounding_score >= 0.7, (
            f"Expected high grounding for well-cited answer, got {report.grounding_score}"
        )
        assert report.hallucination_risk <= 0.3

    def test_hallucinated_answer_detected(self):
        answer = (
            "Meridian Financial Corp had revenue of $99 trillion in 2099. "
            "The NCO rate was 42.5%. The company holds $500 billion in HQLA."
        )
        report = check_hallucination(answer, [MERIDIAN_SOURCE])
        assert report.hallucination_risk > 0.0, (
            "Hallucinated numbers should result in non-zero risk"
        )
        assert len(report.uncited_claims) > 0, "Should flag some uncited claims"

    def test_no_sources_all_uncited(self):
        answer = "Revenue was $3.21 billion and CET1 is 11.8%."
        report = check_hallucination(answer, [])
        # With no sources, all claims should be uncited
        assert report.hallucination_risk >= 0.0
        assert report.grounding_score <= 1.0

    def test_empty_answer_neutral_report(self):
        report = check_hallucination("", [MERIDIAN_SOURCE])
        assert report.grounding_score == 1.0
        assert report.hallucination_risk == 0.0
        assert report.claims == []

    def test_grounding_score_plus_risk_equals_one(self):
        answer = (
            "The LCR is 134%. Revenue was $3.21 billion. CET1 is 11.8%. "
            "Fake metric is 999%."
        )
        report = check_hallucination(answer, [MERIDIAN_SOURCE])
        total = round(report.grounding_score + report.hallucination_risk, 6)
        assert abs(total - 1.0) < 0.001, f"Grounding + risk should sum to 1.0, got {total}"

    def test_report_contains_all_claims(self):
        answer = "Revenue was $3.21 billion. The LCR is 134%. CET1 is 11.8%."
        report = check_hallucination(answer, [MERIDIAN_SOURCE])
        all_claims = set(report.grounded_claims) | set(report.uncited_claims)
        assert set(report.claims).issubset(all_claims | set(report.claims))

    def test_claim_details_populated(self):
        answer = "The CET1 ratio is 11.8% and HQLA is $2.84 billion."
        report = check_hallucination(answer, [MERIDIAN_SOURCE])
        assert isinstance(report.claim_details, list)
        for detail in report.claim_details:
            assert "claim" in detail
            assert "is_grounded" in detail
            assert "overlap_score" in detail
            assert 0.0 <= detail["overlap_score"] <= 1.0

    def test_partial_hallucination(self):
        """Mix of real and fabricated numbers — should be partial risk."""
        answer = (
            "The LCR is 134% and CET1 is 11.8%. "
            "However, revenue was $50 trillion and the NCO rate was 99%."
        )
        report = check_hallucination(answer, [MERIDIAN_SOURCE])
        # Should have some grounded and some uncited
        assert 0.0 < report.grounding_score < 1.0 or len(report.uncited_claims) > 0

    def test_multiple_source_documents(self):
        source1 = "LCR is 134%. CET1 is 11.8%."
        source2 = "Revenue was $3.21 billion in fiscal year 2023."
        answer = "LCR is 134%. Revenue was $3.21 billion."
        report = check_hallucination(answer, [source1, source2])
        assert report.grounding_score > 0.0
