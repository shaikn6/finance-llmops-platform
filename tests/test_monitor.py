"""
Tests for LLM output monitoring.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.monitor import LLMMonitor, InteractionLog


@pytest.fixture
def monitor(tmp_path):
    """Return a fresh LLMMonitor writing to a temporary directory."""
    return LLMMonitor(monitor_dir=tmp_path / "monitoring")


def _log_n_interactions(monitor: LLMMonitor, n: int) -> None:
    for i in range(n):
        monitor.log_interaction(
            question=f"Test question {i}",
            answer=f"Test answer {i} with some financial data like $1.{i} billion",
            grounding_score=0.7 + 0.01 * (i % 10),
            hallucination_risk=0.3 - 0.01 * (i % 10),
            avg_retrieval_score=0.55 + 0.005 * (i % 5),
            latency_ms=200 + i * 2,
            prompt_version="v2_structured",
            model="mock",
            num_citations=3,
            num_uncited_claims=0 if i % 3 != 0 else 1,
            num_total_claims=5,
        )


class TestMonitorLogging:
    def test_log_single_interaction(self, monitor):
        log = monitor.log_interaction(
            question="What is the LCR?",
            answer="The LCR is 134%.",
            grounding_score=0.9,
            hallucination_risk=0.1,
            avg_retrieval_score=0.62,
            latency_ms=245.3,
            prompt_version="v2_structured",
            model="mock",
        )
        assert isinstance(log, InteractionLog)
        assert log.question == "What is the LCR?"
        assert log.grounding_score == 0.9
        assert log.hallucination_risk == 0.1
        assert log.latency_ms == 245.3

    def test_log_persists_to_disk(self, monitor, tmp_path):
        monitor.log_interaction(
            question="Persisted?",
            answer="Yes, $1 billion.",
            grounding_score=0.8,
            hallucination_risk=0.2,
            avg_retrieval_score=0.5,
            latency_ms=100,
            prompt_version="v1_basic",
            model="mock",
        )
        log_file = monitor.log_path
        assert log_file.exists()
        content = log_file.read_text()
        assert "Persisted?" in content

    def test_reload_logs_from_disk(self, tmp_path):
        dir1 = tmp_path / "mon1"
        m1 = LLMMonitor(monitor_dir=dir1)
        m1.log_interaction(
            question="Reload test",
            answer="$2.84 billion HQLA",
            grounding_score=0.85,
            hallucination_risk=0.15,
            avg_retrieval_score=0.6,
            latency_ms=180,
            prompt_version="v2_structured",
            model="mock",
        )
        # Create new monitor pointing to same dir — should reload
        m2 = LLMMonitor(monitor_dir=dir1)
        assert len(m2._logs) == 1
        assert m2._logs[0].question == "Reload test"

    def test_response_length_computed(self, monitor):
        answer = "This is a test answer."
        log = monitor.log_interaction(
            question="Q?",
            answer=answer,
            grounding_score=1.0,
            hallucination_risk=0.0,
            avg_retrieval_score=0.5,
            latency_ms=100,
            prompt_version="v2_structured",
            model="mock",
        )
        assert log.response_length == len(answer)


class TestMonitorDataFrame:
    def test_empty_monitor_returns_empty_df(self, monitor):
        df = monitor.to_dataframe()
        assert df.empty

    def test_to_dataframe_correct_columns(self, monitor):
        _log_n_interactions(monitor, 3)
        df = monitor.to_dataframe()
        required = [
            "question", "answer", "grounding_score", "hallucination_risk",
            "latency_ms", "prompt_version",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_to_dataframe_row_count(self, monitor):
        _log_n_interactions(monitor, 7)
        df = monitor.to_dataframe()
        assert len(df) == 7


class TestMonitorDriftReport:
    def test_insufficient_data_report(self, monitor):
        monitor.log_interaction(
            question="Q",
            answer="A",
            grounding_score=0.9,
            hallucination_risk=0.1,
            avg_retrieval_score=0.5,
            latency_ms=100,
            prompt_version="v2_structured",
            model="mock",
        )
        report = monitor.generate_drift_report()
        assert report["status"] == "insufficient_data"

    def test_drift_report_with_sufficient_data(self, monitor):
        _log_n_interactions(monitor, 20)
        report = monitor.generate_drift_report()
        assert report["status"] == "ok"
        assert "baseline_size" in report
        assert "recent_size" in report
        assert "drift_results" in report

    def test_drift_report_has_all_metrics(self, monitor):
        _log_n_interactions(monitor, 20)
        report = monitor.generate_drift_report()
        if report["status"] == "ok":
            expected_metrics = {
                "grounding_score", "hallucination_risk",
                "response_length", "latency_ms",
            }
            found = set(report.get("drift_results", {}).keys())
            assert found >= expected_metrics, f"Missing metrics: {expected_metrics - found}"

    def test_stable_data_no_drift(self, monitor):
        """Uniform data should not trigger drift."""
        for _ in range(20):
            monitor.log_interaction(
                question="Q",
                answer="$3.21 billion revenue",
                grounding_score=0.85,
                hallucination_risk=0.15,
                avg_retrieval_score=0.60,
                latency_ms=200,
                prompt_version="v2_structured",
                model="mock",
            )
        report = monitor.generate_drift_report()
        if report["status"] == "ok":
            assert not report.get("any_drift_detected", True), (
                "Stable data should not show drift"
            )


class TestMonitorSummary:
    def test_summary_empty(self, monitor):
        summary = monitor.get_summary()
        assert summary.get("total_queries", 0) == 0

    def test_summary_counts(self, monitor):
        _log_n_interactions(monitor, 5)
        summary = monitor.get_summary()
        assert summary["total_queries"] == 5
        assert 0.0 <= summary["avg_grounding_score"] <= 1.0
        assert 0.0 <= summary["avg_hallucination_risk"] <= 1.0

    def test_time_series_keys(self, monitor):
        _log_n_interactions(monitor, 3)
        ts = monitor.get_time_series()
        expected_keys = [
            "timestamps", "grounding_scores", "hallucination_risks",
            "response_lengths", "retrieval_scores",
        ]
        for key in expected_keys:
            assert key in ts, f"Missing time series key: {key}"
        assert len(ts["grounding_scores"]) == 3
