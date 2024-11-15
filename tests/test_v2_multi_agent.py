"""
Tests for the multi-agent RAG pipeline (agents/multi_agent_rag.py).

All tests are deterministic and require no external API keys or FAISS index.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.multi_agent_rag import (
    AgentStatus,
    FinanceRAGState,
    ResearcherAgent,
    AnalystAgent,
    FactCheckerAgent,
    SynthesizerAgent,
    MultiAgentRAGPipeline,
    AgentPipelineResult,
    _initial_state,
    get_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_query():
    return "What is Meridian's liquidity coverage ratio?"


@pytest.fixture
def initial_state(sample_query):
    return _initial_state(sample_query, top_k=3)


@pytest.fixture
def pipeline():
    return MultiAgentRAGPipeline(top_k=3)


# ---------------------------------------------------------------------------
# FinanceRAGState tests
# ---------------------------------------------------------------------------

class TestFinanceRAGState:
    def test_initial_state_has_query(self, sample_query):
        state = _initial_state(sample_query)
        assert state["query"] == sample_query

    def test_initial_state_has_run_id(self, initial_state):
        assert len(initial_state["run_id"]) == 8

    def test_initial_state_agents_pending(self, initial_state):
        for key in ("researcher_status", "analyst_status",
                    "fact_checker_status", "synthesizer_status"):
            assert initial_state[key] == AgentStatus.PENDING

    def test_initial_state_empty_chunks(self, initial_state):
        assert initial_state["retrieved_chunks"] == []

    def test_initial_state_top_k(self):
        state = _initial_state("query", top_k=5)
        assert state["top_k"] == 5

    def test_initial_state_pipeline_start_set(self, initial_state):
        assert initial_state["pipeline_start_ms"] > 0

    def test_two_states_have_different_run_ids(self, sample_query):
        s1 = _initial_state(sample_query)
        s2 = _initial_state(sample_query)
        assert s1["run_id"] != s2["run_id"]


# ---------------------------------------------------------------------------
# ResearcherAgent tests
# ---------------------------------------------------------------------------

class TestResearcherAgent:
    def test_researcher_returns_state(self, initial_state):
        agent = ResearcherAgent(top_k=3)
        new_state = agent.run(initial_state)
        assert isinstance(new_state, dict)

    def test_researcher_status_done(self, initial_state):
        agent = ResearcherAgent(top_k=3)
        new_state = agent.run(initial_state)
        assert new_state["researcher_status"] == AgentStatus.DONE

    def test_researcher_returns_chunks(self, initial_state):
        agent = ResearcherAgent(top_k=3)
        new_state = agent.run(initial_state)
        assert isinstance(new_state["retrieved_chunks"], list)
        assert len(new_state["retrieved_chunks"]) > 0

    def test_researcher_chunk_has_required_keys(self, initial_state):
        agent = ResearcherAgent(top_k=3)
        new_state = agent.run(initial_state)
        for chunk in new_state["retrieved_chunks"]:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "score" in chunk
            assert "doc_name" in chunk

    def test_researcher_latency_positive(self, initial_state):
        agent = ResearcherAgent(top_k=3)
        new_state = agent.run(initial_state)
        assert new_state["retrieval_latency_ms"] >= 0

    def test_researcher_top_k_respected(self):
        agent = ResearcherAgent(top_k=2)
        state = _initial_state("What is the LCR?", top_k=2)
        new_state = agent.run(state)
        assert len(new_state["retrieved_chunks"]) <= 2

    def test_researcher_mock_chunks_realistic_content(self, initial_state):
        agent = ResearcherAgent(top_k=3)
        new_state = agent.run(initial_state)
        texts = " ".join(c["text"] for c in new_state["retrieved_chunks"])
        # Should contain financial content
        assert any(kw in texts.lower() for kw in ["liquidity", "revenue", "capital", "meridian"])


# ---------------------------------------------------------------------------
# AnalystAgent tests
# ---------------------------------------------------------------------------

class TestAnalystAgent:
    @pytest.fixture
    def researcher_state(self, initial_state):
        researcher = ResearcherAgent(top_k=3)
        return researcher.run(initial_state)

    def test_analyst_returns_state(self, researcher_state):
        agent = AnalystAgent()
        new_state = agent.run(researcher_state)
        assert isinstance(new_state, dict)

    def test_analyst_status_done(self, researcher_state):
        agent = AnalystAgent()
        new_state = agent.run(researcher_state)
        assert new_state["analyst_status"] == AgentStatus.DONE

    def test_analyst_produces_cross_reference_notes(self, researcher_state):
        agent = AnalystAgent()
        new_state = agent.run(researcher_state)
        assert isinstance(new_state["cross_reference_notes"], str)
        assert len(new_state["cross_reference_notes"]) > 0

    def test_analyst_separates_earnings_vs_filings(self, researcher_state):
        agent = AnalystAgent()
        new_state = agent.run(researcher_state)
        assert isinstance(new_state["earnings_references"], list)
        assert isinstance(new_state["filing_references"], list)

    def test_analyst_notes_mention_source_type(self, researcher_state):
        agent = AnalystAgent()
        new_state = agent.run(researcher_state)
        notes = new_state["cross_reference_notes"].lower()
        # Should describe what sources were found
        assert any(kw in notes for kw in ["10-k", "earnings", "filings", "chunks", "cross-referencing", "only"])


# ---------------------------------------------------------------------------
# FactCheckerAgent tests
# ---------------------------------------------------------------------------

class TestFactCheckerAgent:
    @pytest.fixture
    def analyst_state(self, initial_state):
        researcher = ResearcherAgent(top_k=3)
        analyst = AnalystAgent()
        s1 = researcher.run(initial_state)
        return analyst.run(s1)

    def test_fact_checker_returns_state(self, analyst_state):
        agent = FactCheckerAgent()
        new_state = agent.run(analyst_state)
        assert isinstance(new_state, dict)

    def test_fact_checker_status_done(self, analyst_state):
        agent = FactCheckerAgent()
        new_state = agent.run(analyst_state)
        assert new_state["fact_checker_status"] == AgentStatus.DONE

    def test_fact_checker_grounding_score_in_range(self, analyst_state):
        agent = FactCheckerAgent()
        new_state = agent.run(analyst_state)
        assert 0.0 <= new_state["grounding_score"] <= 1.0

    def test_fact_checker_risk_in_range(self, analyst_state):
        agent = FactCheckerAgent()
        new_state = agent.run(analyst_state)
        assert 0.0 <= new_state["hallucination_risk"] <= 1.0

    def test_fact_checker_grounding_plus_risk_equals_one(self, analyst_state):
        agent = FactCheckerAgent()
        new_state = agent.run(analyst_state)
        total = round(new_state["grounding_score"] + new_state["hallucination_risk"], 5)
        assert abs(total - 1.0) < 0.01

    def test_fact_checker_claims_are_lists(self, analyst_state):
        agent = FactCheckerAgent()
        new_state = agent.run(analyst_state)
        assert isinstance(new_state["flagged_claims"], list)
        assert isinstance(new_state["grounded_claims"], list)


# ---------------------------------------------------------------------------
# SynthesizerAgent tests
# ---------------------------------------------------------------------------

class TestSynthesizerAgent:
    @pytest.fixture
    def fact_checked_state(self, initial_state):
        researcher = ResearcherAgent(top_k=3)
        analyst = AnalystAgent()
        fact_checker = FactCheckerAgent()
        s1 = researcher.run(initial_state)
        s2 = analyst.run(s1)
        return fact_checker.run(s2)

    def test_synthesizer_returns_state(self, fact_checked_state):
        agent = SynthesizerAgent()
        new_state = agent.run(fact_checked_state)
        assert isinstance(new_state, dict)

    def test_synthesizer_status_done(self, fact_checked_state):
        agent = SynthesizerAgent()
        new_state = agent.run(fact_checked_state)
        assert new_state["synthesizer_status"] == AgentStatus.DONE

    def test_synthesizer_produces_final_answer(self, fact_checked_state):
        agent = SynthesizerAgent()
        new_state = agent.run(fact_checked_state)
        assert isinstance(new_state["final_answer"], str)
        assert len(new_state["final_answer"]) > 50

    def test_synthesizer_sets_total_latency(self, fact_checked_state):
        agent = SynthesizerAgent()
        new_state = agent.run(fact_checked_state)
        assert new_state["total_latency_ms"] > 0

    def test_synthesizer_answer_contains_fact_check_status(self, fact_checked_state):
        agent = SynthesizerAgent()
        new_state = agent.run(fact_checked_state)
        answer = new_state["final_answer"].lower()
        # Should mention grounding or fact-check
        assert any(kw in answer for kw in ["grounding", "fact-check", "verified", "grounded"])


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------

class TestMultiAgentPipeline:
    def test_pipeline_runs_successfully(self, pipeline, sample_query):
        result = pipeline.run(sample_query)
        assert isinstance(result, AgentPipelineResult)

    def test_pipeline_final_state_has_answer(self, pipeline, sample_query):
        result = pipeline.run(sample_query)
        assert len(result.state["final_answer"]) > 30

    def test_pipeline_all_agents_done(self, pipeline, sample_query):
        result = pipeline.run(sample_query)
        state = result.state
        for key in ("researcher_status", "analyst_status",
                    "fact_checker_status", "synthesizer_status"):
            assert state[key] == AgentStatus.DONE, f"{key} not DONE"

    def test_pipeline_timings_populated(self, pipeline, sample_query):
        result = pipeline.run(sample_query)
        for name in ("researcher", "analyst", "fact_checker", "synthesizer"):
            assert name in result.agent_timings
            assert result.agent_timings[name] >= 0

    def test_pipeline_streamed_yields_four_steps(self, pipeline, sample_query):
        steps = list(pipeline.run_streamed(sample_query))
        assert len(steps) == 4

    def test_pipeline_streamed_yields_correct_agent_names(self, pipeline, sample_query):
        agent_names = [name for name, _ in pipeline.run_streamed(sample_query)]
        assert agent_names == ["researcher", "analyst", "fact_checker", "synthesizer"]

    def test_pipeline_intermediate_states_on_yield(self, pipeline, sample_query):
        result = pipeline.run(sample_query, yield_intermediate=True)
        assert len(result.step_states) == 4

    def test_get_pipeline_singleton(self):
        p1 = get_pipeline()
        p2 = get_pipeline()
        assert p1 is p2

    def test_pipeline_different_queries_different_answers(self, pipeline):
        r1 = pipeline.run("What is the LCR ratio?")
        r2 = pipeline.run("What is the total debt outstanding?")
        # Answers should differ (they use keyword dispatch)
        assert r1.state["final_answer"] != r2.state["final_answer"]
