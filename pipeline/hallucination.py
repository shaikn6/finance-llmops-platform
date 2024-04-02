"""
Hallucination detection via citation overlap analysis.

Strategy:
  1. Extract factual claims from the generated answer (numbers, percentages,
     named entities, dates, dollar amounts).
  2. For each extracted claim, check whether it appears in the cited source
     chunks (token overlap >= threshold).
  3. Return a grounding score (0–1) and a list of uncited claims.

This is a deterministic, regex + token-overlap approach — no LLM call needed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

# Minimum token overlap ratio for a claim to be considered "grounded"
GROUNDING_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Claim extraction patterns
# ---------------------------------------------------------------------------

# Dollar amounts: $3.21 billion, $847 million, $78.3 million, etc.
_DOLLAR_PATTERN = re.compile(
    r'\$[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|bn|mn|k)?',
    re.IGNORECASE
)

# Percentages: 7.4%, 134%, 3.47%, etc.
_PERCENT_PATTERN = re.compile(
    r'\d+(?:\.\d+)?\s*%'
)

# Basis points: 18 basis points, +200 bps
_BPS_PATTERN = re.compile(
    r'\d+(?:\.\d+)?\s*(?:basis points?|bps)',
    re.IGNORECASE
)

# Dates: December 31, 2023 / Q4 2023 / fiscal year 2023
_DATE_PATTERN = re.compile(
    r'(?:Q[1-4]\s+\d{4}|(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}|'
    r'fiscal\s+year\s+\d{4}|\d{4})',
    re.IGNORECASE
)

# Ratios: 11.8%, LCR, CET1, NIM, NCO, NII, HQLA (keep as-is since they're acronyms/terms)
_RATIO_ABBREV_PATTERN = re.compile(
    r'\b(?:LCR|CET1|NIM|NCO|NII|HQLA|AUM|FHLB|SOFR|AFS|ACL|RIA|OCC)\b'
)

# Named entities (all-caps multi-word or specific company names)
_COMPANY_PATTERN = re.compile(
    r'Meridian Financial Corp(?:oration)?'
)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    """Simple word tokenization — returns a set of unique tokens."""
    tokens = re.findall(r'\b[\w%.,$]+\b', _normalize(text))
    return set(tokens)


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def extract_factual_claims(text: str) -> List[str]:
    """
    Extract atomic factual claims from generated answer text.

    Each claim is a small string (number, date, percentage, acronym) that
    can be independently verified against source chunks.
    """
    claims: List[str] = []

    for pattern in [
        _DOLLAR_PATTERN,
        _PERCENT_PATTERN,
        _BPS_PATTERN,
        _DATE_PATTERN,
        _RATIO_ABBREV_PATTERN,
        _COMPANY_PATTERN,
    ]:
        matches = pattern.findall(text)
        claims.extend(m.strip() for m in matches if m.strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for c in claims:
        key = _normalize(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

def _claim_in_sources(claim: str, source_texts: List[str]) -> Tuple[bool, float]:
    """
    Check if a claim is sufficiently covered by at least one source chunk.

    Uses token overlap: overlap_ratio = |claim_tokens ∩ source_tokens| / |claim_tokens|

    Returns (is_grounded, best_overlap_score).
    """
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return True, 1.0  # Empty claim considered grounded

    best_overlap = 0.0
    for source_text in source_texts:
        source_tokens = _tokenize(source_text)
        overlap = claim_tokens & source_tokens
        overlap_ratio = len(overlap) / len(claim_tokens)
        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio

    return best_overlap >= GROUNDING_THRESHOLD, best_overlap


@dataclass
class HallucinationReport:
    """Result of a hallucination check on a generated answer."""
    answer: str
    claims: List[str]
    uncited_claims: List[str]
    grounded_claims: List[str]
    grounding_score: float          # fraction of claims that are grounded
    hallucination_risk: float       # 1 - grounding_score (higher = worse)
    claim_details: List[dict]       # per-claim overlap details


def check_hallucination(
    answer: str,
    source_texts: List[str],
    threshold: float = GROUNDING_THRESHOLD,
) -> HallucinationReport:
    """
    Check a generated answer for hallucinated (uncited) factual claims.

    Args:
        answer: The generated answer text.
        source_texts: List of source chunk texts used to generate the answer.
        threshold: Minimum token overlap to consider a claim grounded.

    Returns:
        HallucinationReport with grounding score and uncited claim list.
    """
    claims = extract_factual_claims(answer)

    if not claims:
        # No factual claims to check — neutral report
        return HallucinationReport(
            answer=answer,
            claims=[],
            uncited_claims=[],
            grounded_claims=[],
            grounding_score=1.0,
            hallucination_risk=0.0,
            claim_details=[],
        )

    uncited: List[str] = []
    grounded: List[str] = []
    claim_details: List[dict] = []

    for claim in claims:
        is_grounded, overlap = _claim_in_sources(claim, source_texts)
        detail = {
            "claim": claim,
            "is_grounded": is_grounded,
            "overlap_score": round(overlap, 3),
        }
        claim_details.append(detail)
        if is_grounded:
            grounded.append(claim)
        else:
            uncited.append(claim)

    grounding_score = len(grounded) / len(claims) if claims else 1.0
    hallucination_risk = 1.0 - grounding_score

    return HallucinationReport(
        answer=answer,
        claims=claims,
        uncited_claims=uncited,
        grounded_claims=grounded,
        grounding_score=round(grounding_score, 4),
        hallucination_risk=round(hallucination_risk, 4),
        claim_details=claim_details,
    )


def check_from_generated_answer(generated_answer) -> HallucinationReport:
    """
    Convenience wrapper that takes a GeneratedAnswer object directly.
    """
    source_texts = [c.text_snippet for c in generated_answer.citations]
    return check_hallucination(
        answer=generated_answer.answer,
        source_texts=source_texts,
    )


if __name__ == "__main__":
    import json

    sample_answer = (
        "Meridian Financial Corp maintains an LCR of 134% as of December 31, 2023, "
        "exceeding the 100% regulatory minimum. The Company holds $2.84 billion in HQLA. "
        "Revenue was $5 trillion in 2025 (hallucinated)."
    )
    sample_source = (
        "Meridian Financial Corp maintains a liquidity coverage ratio (LCR) of 134% as of "
        "December 31, 2023, exceeding the regulatory minimum of 100% mandated under Basel III. "
        "The Company holds $2.84 billion in high-quality liquid assets (HQLA)."
    )
    report = check_hallucination(sample_answer, [sample_source])
    print(f"Claims: {report.claims}")
    print(f"Grounded: {report.grounded_claims}")
    print(f"Uncited: {report.uncited_claims}")
    print(f"Grounding score: {report.grounding_score:.1%}")
    print(f"Hallucination risk: {report.hallucination_risk:.1%}")
