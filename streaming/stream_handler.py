"""
FastAPI + Streamlit streaming handler for LLM output.

Features:
  - FastAPI endpoint at /api/v2/stream that emits Server-Sent Events (SSE)
  - Token-by-token word generator with realistic finance vocabulary
  - Streamlit helper compatible with st.write_stream()
  - Mock streamer works without any external API key

Server-Sent Events format:
    data: {"token": "word", "done": false, "index": 0}
    data: {"token": "", "done": true, "index": 42}

Usage (FastAPI):
    uvicorn streaming.stream_handler:app --port 8001

Usage (Streamlit):
    from streaming.stream_handler import streamlit_word_stream
    st.write_stream(streamlit_word_stream("What is the LCR?"))
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import AsyncIterator, Iterator, Optional

# ---------------------------------------------------------------------------
# Finance vocabulary for mock streamer
# ---------------------------------------------------------------------------

_FINANCE_WORDS = [
    # Ratios and metrics
    "liquidity", "coverage", "ratio", "LCR", "CET1", "NIM", "NCO",
    "HQLA", "Basel", "III", "Tier-1", "capital", "adequacy",
    # Financial terms
    "revenue", "earnings", "net", "interest", "margin", "deposit",
    "beta", "yield", "spread", "credit", "loss", "provision",
    "charge-off", "delinquency", "forbearance",
    # Common financial verbs and modifiers
    "exceeded", "maintained", "reported", "grew", "declined",
    "year-over-year", "quarterly", "annualized", "adjusted",
    "core", "reported", "regulatory", "minimum", "threshold",
    # Companies and entities
    "Meridian", "Financial", "Corp", "Basel", "III", "FDIC", "OCC",
    # Numbers and units (often appear in finance text)
    "$2.84", "billion", "$3.21", "134%", "11.8%", "3.41%", "7.4%",
    "December", "31,", "2023", "Q4", "fiscal", "year",
]

_SENTENCE_STARTERS = [
    "Meridian Financial Corp",
    "The Company",
    "Based on the 10-K filing,",
    "According to the Q4 2023 earnings call,",
    "As of December 31, 2023,",
    "The regulatory framework requires",
    "Cross-referencing both sources confirms",
]

_SENTENCE_MIDDLES = [
    "maintains an LCR of 134%, exceeding the 100% regulatory minimum.",
    "reported total net revenue of $3.21 billion, a 7.4% increase year-over-year.",
    "holds $2.84 billion in high-quality liquid assets (HQLA).",
    "invested $78.3 million in cybersecurity infrastructure during fiscal 2023.",
    "carries a CET1 ratio of 11.8%, well above the 6.5% well-capitalized threshold.",
    "guided NIM to 3.40%–3.55% for full-year 2024 with Q1 as the trough.",
    "processed 10-K filings in 8–12 minutes with 94.7% accuracy via the AI platform.",
]


def _build_mock_response(query: str) -> str:
    """Build a realistic mock finance response for the given query."""
    q_lower = query.lower()

    keyword_responses = {
        "lcr": (
            "Meridian Financial Corp maintains a liquidity coverage ratio of 134% as of "
            "December 31, 2023, exceeding the 100% regulatory minimum required under Basel III. "
            "The Company holds $2.84 billion in high-quality liquid assets. "
            "This buffer provides a 34 percentage-point cushion above the regulatory floor, "
            "reflecting disciplined balance-sheet management."
        ),
        "revenue": (
            "Total net revenue for fiscal year 2023 was $3.21 billion, representing a 7.4% "
            "increase from $2.99 billion in fiscal year 2022. Commercial Banking contributed "
            "$1.42 billion or 44.2% of total revenue, while Wealth Management added $0.68 billion. "
            "Management expects low-to-mid single-digit growth in 2024."
        ),
        "cet1": (
            "The Common Equity Tier 1 ratio stands at 11.8% as of December 31, 2023, "
            "comfortably above the 6.5% well-capitalized threshold and the Company's internal "
            "target of 10.5%. Earnings retention and controlled risk-weighted-asset growth "
            "drove 40 basis points of CET1 expansion during fiscal 2023."
        ),
        "nim": (
            "Management guided to a net interest margin of 3.40% to 3.55% for full-year 2024. "
            "Q4 2023 NIM was 3.41% reported (3.38% core), with Q1 2024 expected to be the trough "
            "before sequential improvement beginning in Q2. "
            "Deposit beta assumptions embed a cumulative rate of 52% on interest-bearing deposits."
        ),
    }

    for key, response in keyword_responses.items():
        if key in q_lower:
            return response

    # Default response
    return (
        "Based on Meridian Financial Corp's 10-K filing and Q4 2023 earnings call transcript, "
        "the Company demonstrated strong performance across key operating metrics. "
        "The CET1 ratio of 11.8% and LCR of 134% both exceed regulatory minimums by a comfortable margin. "
        "Total deposits of $24.6 billion reflect continued franchise strength, "
        "while net charge-offs of 0.31% remain well below the peer median of 0.44%."
    )


# ---------------------------------------------------------------------------
# Sync word-by-word generator
# ---------------------------------------------------------------------------

def word_stream(
    query: str,
    delay_range: tuple[float, float] = (0.03, 0.12),
    rng_seed: Optional[int] = None,
) -> Iterator[str]:
    """
    Synchronous generator that yields one word at a time with realistic
    inter-word delay, simulating LLM token streaming.

    Compatible with Streamlit's st.write_stream() when used as-is.

    Args:
        query: The finance question to answer.
        delay_range: (min_delay, max_delay) seconds between words.
        rng_seed: Optional seed for reproducible output.

    Yields:
        One word (plus trailing space) per iteration.
        Final yield is an empty string to signal completion.
    """
    rng = random.Random(rng_seed)
    text = _build_mock_response(query)
    words = text.split()

    for i, word in enumerate(words):
        delay = rng.uniform(*delay_range)
        time.sleep(delay)
        # Add trailing space after every word except last
        yield word + (" " if i < len(words) - 1 else "")

    yield ""


# ---------------------------------------------------------------------------
# Async SSE generator
# ---------------------------------------------------------------------------

async def sse_token_generator(
    query: str,
    delay_range: tuple[float, float] = (0.03, 0.12),
    rng_seed: Optional[int] = None,
) -> AsyncIterator[str]:
    """
    Async generator that emits Server-Sent Events for the streaming endpoint.

    Each SSE message is a JSON object:
        {"token": "<word>", "done": false, "index": <n>}

    The final message is:
        {"token": "", "done": true, "index": <total>}

    Yields:
        SSE-formatted strings including the "data: " prefix and double newline.
    """
    rng = random.Random(rng_seed)
    text = _build_mock_response(query)
    words = text.split()

    for i, word in enumerate(words):
        delay = rng.uniform(*delay_range)
        await asyncio.sleep(delay)
        payload = json.dumps({"token": word + " ", "done": False, "index": i})
        yield f"data: {payload}\n\n"

    done_payload = json.dumps({"token": "", "done": True, "index": len(words)})
    yield f"data: {done_payload}\n\n"


# ---------------------------------------------------------------------------
# Streamlit helper
# ---------------------------------------------------------------------------

def streamlit_word_stream(query: str) -> Iterator[str]:
    """
    Streamlit-compatible stream generator.

    Usage:
        import streamlit as st
        from streaming.stream_handler import streamlit_word_stream
        st.write_stream(streamlit_word_stream("What is the LCR?"))

    Yields:
        One word at a time (no trailing newline — Streamlit handles that).
    """
    for token in word_stream(query, delay_range=(0.04, 0.10)):
        if token:
            yield token


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def _make_app():
    """
    Lazily create the FastAPI app so importing this module does NOT require
    fastapi to be installed unless the app is actually launched.
    """
    try:
        from fastapi import FastAPI, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "fastapi is required to run the streaming server. "
            "Install it with: pip install fastapi uvicorn"
        ) from exc

    application = FastAPI(
        title="Finance LLMOps Streaming API",
        description="Token-by-token SSE streaming for financial Q&A",
        version="2.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @application.get("/health")
    async def health():
        return {"status": "ok", "version": "2.0.0"}

    @application.get(
        "/api/v2/stream",
        summary="Stream a finance Q&A answer token-by-token via SSE",
        response_description="Server-Sent Events stream of JSON tokens",
    )
    async def stream_answer(
        query: str = Query(..., description="Financial question to answer"),
        seed: Optional[int] = Query(None, description="RNG seed for reproducible output"),
    ):
        """
        Stream an LLM-style answer word-by-word using Server-Sent Events.

        Connect with EventSource:
            const es = new EventSource('/api/v2/stream?query=What+is+the+LCR');
            es.onmessage = (e) => {
              const data = JSON.parse(e.data);
              if (!data.done) console.log(data.token);
            };
        """

        async def event_stream():
            async for chunk in sse_token_generator(query, rng_seed=seed):
                yield chunk

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @application.post(
        "/api/v2/stream",
        summary="POST variant of the streaming endpoint",
    )
    async def stream_answer_post(body: dict):
        query = body.get("query", "")
        seed = body.get("seed", None)

        async def event_stream():
            async for chunk in sse_token_generator(query, rng_seed=seed):
                yield chunk

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return application


# Expose a module-level `app` for uvicorn
try:
    app = _make_app()
except RuntimeError:
    app = None  # FastAPI not installed; sync helpers still work


if __name__ == "__main__":
    import sys

    # Demo: stream to terminal
    query_arg = " ".join(sys.argv[1:]) or "What is Meridian's liquidity coverage ratio?"
    print(f"Streaming answer for: {query_arg!r}\n")
    for token in word_stream(query_arg):
        if token:
            print(token, end="", flush=True)
    print("\n\n[Stream complete]")
