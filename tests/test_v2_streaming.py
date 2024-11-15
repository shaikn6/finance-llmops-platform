"""
Tests for the SSE streaming handler (streaming/stream_handler.py).

All tests are synchronous and require no network calls or external API keys.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from streaming.stream_handler import (
    word_stream,
    streamlit_word_stream,
    _build_mock_response,
    sse_token_generator,
)


# ---------------------------------------------------------------------------
# Mock response builder tests
# ---------------------------------------------------------------------------

class TestBuildMockResponse:
    def test_lcr_query_mentions_lcr(self):
        response = _build_mock_response("What is the LCR?")
        assert "134%" in response or "liquidity coverage" in response.lower()

    def test_revenue_query_mentions_revenue(self):
        response = _build_mock_response("What was net revenue in 2023?")
        assert "3.21" in response or "revenue" in response.lower()

    def test_cet1_query_mentions_cet1(self):
        response = _build_mock_response("What is the CET1 ratio?")
        assert "11.8%" in response or "CET1" in response

    def test_nim_query_mentions_nim(self):
        response = _build_mock_response("What is the NIM guidance?")
        assert "3.4" in response or "NIM" in response or "net interest margin" in response.lower()

    def test_unknown_query_returns_default(self):
        response = _build_mock_response("Tell me about unicorns.")
        assert len(response) > 50  # Should return default financial content

    def test_response_is_non_empty_string(self):
        for query in ["", "  ", "What is X?", "ABCDEF"]:
            r = _build_mock_response(query)
            assert isinstance(r, str)
            assert len(r) > 0


# ---------------------------------------------------------------------------
# word_stream generator tests
# ---------------------------------------------------------------------------

class TestWordStream:
    def test_stream_yields_tokens(self):
        tokens = list(word_stream("What is the LCR?", delay_range=(0, 0)))
        assert len(tokens) > 0

    def test_stream_terminates_with_empty_string(self):
        tokens = list(word_stream("What is the LCR?", delay_range=(0, 0)))
        # Last token is empty string sentinel
        assert tokens[-1] == ""

    def test_stream_tokens_are_strings(self):
        tokens = list(word_stream("What is the LCR?", delay_range=(0, 0)))
        for t in tokens:
            assert isinstance(t, str)

    def test_stream_non_empty_tokens_not_just_whitespace(self):
        tokens = [t for t in word_stream("What is the LCR?", delay_range=(0, 0)) if t]
        for t in tokens:
            assert t.strip() or t == " "  # word + trailing space

    def test_stream_reproducible_with_seed(self):
        tokens1 = list(word_stream("What is the LCR?", delay_range=(0, 0), rng_seed=42))
        tokens2 = list(word_stream("What is the LCR?", delay_range=(0, 0), rng_seed=42))
        assert tokens1 == tokens2

    def test_stream_different_seeds_same_content(self):
        """Content should be the same (same query), only timing differs."""
        tokens1 = [t for t in word_stream("What is the LCR?", delay_range=(0, 0), rng_seed=1) if t]
        tokens2 = [t for t in word_stream("What is the LCR?", delay_range=(0, 0), rng_seed=99) if t]
        # Same words (order/content fixed by query, not seed)
        joined1 = "".join(tokens1).strip()
        joined2 = "".join(tokens2).strip()
        assert joined1 == joined2

    def test_stream_reconstructs_full_response(self):
        query = "What is the CET1 ratio?"
        tokens = [t for t in word_stream(query, delay_range=(0, 0)) if t]
        full_text = "".join(tokens).strip()
        expected = _build_mock_response(query)
        assert full_text == expected

    def test_stream_with_zero_delay_is_fast(self):
        t0 = time.time()
        tokens = list(word_stream("What is the LCR?", delay_range=(0, 0)))
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"Zero-delay stream took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# streamlit_word_stream tests
# ---------------------------------------------------------------------------

class TestStreamlitWordStream:
    def test_streamlit_stream_yields_words(self):
        words = list(streamlit_word_stream("What is the LCR?"))
        assert len(words) > 0

    def test_streamlit_stream_no_empty_strings(self):
        """Streamlit stream should not yield empty strings (unlike word_stream)."""
        words = list(streamlit_word_stream("What is the LCR?"))
        for w in words:
            assert w, "Streamlit stream should not yield empty strings"

    def test_streamlit_stream_words_are_strings(self):
        for w in streamlit_word_stream("Test query"):
            assert isinstance(w, str)


# ---------------------------------------------------------------------------
# SSE async generator tests
# ---------------------------------------------------------------------------

class TestSSEGenerator:
    def _collect(self, query: str) -> list:
        """Helper: collect all SSE events from async generator into a list."""
        import asyncio

        async def _run():
            events = []
            async for chunk in sse_token_generator(query, delay_range=(0, 0)):
                events.append(chunk)
            return events

        return asyncio.get_event_loop().run_until_complete(_run())

    def test_sse_events_have_data_prefix(self):
        events = self._collect("What is the LCR?")
        for event in events:
            assert event.startswith("data: ")

    def test_sse_events_have_double_newline(self):
        events = self._collect("What is the LCR?")
        for event in events:
            assert event.endswith("\n\n")

    def test_sse_last_event_has_done_true(self):
        import json
        events = self._collect("What is the LCR?")
        last = json.loads(events[-1].removeprefix("data: ").strip())
        assert last["done"] is True

    def test_sse_non_last_events_have_done_false(self):
        import json
        events = self._collect("What is the LCR?")
        for event in events[:-1]:
            data = json.loads(event.removeprefix("data: ").strip())
            assert data["done"] is False

    def test_sse_events_have_index(self):
        import json
        events = self._collect("What is the LCR?")
        for event in events:
            data = json.loads(event.removeprefix("data: ").strip())
            assert "index" in data
            assert isinstance(data["index"], int)

    def test_sse_indices_are_sequential(self):
        import json
        events = self._collect("What is the LCR?")
        indices = [json.loads(e.removeprefix("data: ").strip())["index"] for e in events]
        non_final = indices[:-1]
        for i, idx in enumerate(non_final):
            assert idx == i

    def test_sse_tokens_reconstruct_response(self):
        import json
        events = self._collect("What is the LCR?")
        tokens = [
            json.loads(e.removeprefix("data: ").strip())["token"]
            for e in events[:-1]  # exclude done event
        ]
        reconstructed = "".join(tokens).strip()
        expected = _build_mock_response("What is the LCR?")
        assert reconstructed == expected
