"""
LLM output monitoring using Evidently AI.

Tracks:
  - Response length distribution
  - Grounding score trend
  - Hallucination risk trend
  - Retrieval score trend

Generates drift reports comparing baseline vs. recent queries.
Falls back to pandas-based statistics if Evidently is unavailable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

MONITOR_DIR = Path(__file__).parent.parent / "data" / "monitoring"
BASELINE_THRESHOLD = 10   # Minimum interactions to establish baseline
RECENT_WINDOW = 50        # Last N interactions for drift detection


# ---------------------------------------------------------------------------
# Interaction log entry
# ---------------------------------------------------------------------------

@dataclass
class InteractionLog:
    """Single query-response interaction logged for monitoring."""
    timestamp: float
    question: str
    answer: str
    response_length: int        # chars
    num_citations: int
    avg_retrieval_score: float
    grounding_score: float
    hallucination_risk: float
    latency_ms: float
    prompt_version: str
    model: str
    num_uncited_claims: int
    num_total_claims: int


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class LLMMonitor:
    """
    Monitors LLM pipeline outputs and detects distributional drift.

    Persists interaction logs to disk as JSONL for reproducibility.
    """

    def __init__(self, monitor_dir: Path = MONITOR_DIR) -> None:
        self.monitor_dir = monitor_dir
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = monitor_dir / "interactions.jsonl"
        self._logs: List[InteractionLog] = self._load_logs()

    def _load_logs(self) -> List[InteractionLog]:
        """Load existing logs from disk."""
        if not self.log_path.exists():
            return []
        logs = []
        for line in self.log_path.read_text().splitlines():
            try:
                d = json.loads(line)
                logs.append(InteractionLog(**d))
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        return logs

    def log_interaction(
        self,
        question: str,
        answer: str,
        grounding_score: float,
        hallucination_risk: float,
        avg_retrieval_score: float,
        latency_ms: float,
        prompt_version: str,
        model: str,
        num_citations: int = 0,
        num_uncited_claims: int = 0,
        num_total_claims: int = 0,
    ) -> InteractionLog:
        """Log a single LLM interaction for monitoring."""
        log = InteractionLog(
            timestamp=time.time(),
            question=question,
            answer=answer,
            response_length=len(answer),
            num_citations=num_citations,
            avg_retrieval_score=round(avg_retrieval_score, 4),
            grounding_score=round(grounding_score, 4),
            hallucination_risk=round(hallucination_risk, 4),
            latency_ms=round(latency_ms, 1),
            prompt_version=prompt_version,
            model=model,
            num_uncited_claims=num_uncited_claims,
            num_total_claims=num_total_claims,
        )
        self._logs.append(log)
        # Append to JSONL
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(log)) + "
")
        return log

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all logs to a pandas DataFrame."""
        if not self._logs:
            return pd.DataFrame()
        return pd.DataFrame([asdict(l) for l in self._logs])

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute summary statistics for a DataFrame slice."""
        if df.empty:
            return {}
        numeric_cols = [
            "response_length", "grounding_score", "hallucination_risk",
            "avg_retrieval_score", "latency_ms",
        ]
        stats = {}
        for col in numeric_cols:
            if col in df.columns:
                stats[f"{col}_mean"] = round(df[col].mean(), 4)
                stats[f"{col}_std"] = round(df[col].std(), 4)
                stats[f"{col}_p50"] = round(df[col].quantile(0.50), 4)
                stats[f"{col}_p95"] = round(df[col].quantile(0.95), 4)
        return stats

    def generate_drift_report(self) -> Dict[str, Any]:
        """
        Compare baseline vs. recent interactions using statistical drift detection.

        Returns a report with drift scores per metric.
        Uses Evidently AI if available, otherwise falls back to PSI computation.
        """
        df = self.to_dataframe()

        if df.empty or len(df) < 2:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 2 interactions. Have {len(df)}.",
                "total_interactions": len(df),
            }

        # Split into baseline and current
        n = len(df)
        midpoint = max(1, n // 2)
        baseline_df = df.iloc[:midpoint].copy()
        recent_df = df.iloc[midpoint:].copy()

        # Try Evidently AI first
        try:
            report = self._run_evidently_report(baseline_df, recent_df)
            report["engine"] = "evidently"
            return report
        except Exception as e:  # pragma: no cover
            # Fall back to pandas-based PSI computation
            return self._run_psi_report(baseline_df, recent_df)  # pragma: no cover

    def _run_evidently_report(
        self, baseline_df: pd.DataFrame, recent_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run Evidently AI drift detection."""
        from evidently import ColumnMapping
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import ColumnDriftMetric

        numerical_features = [
            "response_length", "grounding_score", "hallucination_risk",
            "avg_retrieval_score", "latency_ms",
        ]

        available = [c for c in numerical_features if c in baseline_df.columns]

        report = Report(metrics=[
            ColumnDriftMetric(column_name=col)
            for col in available
        ])

        column_mapping = ColumnMapping(numerical_features=available)
        report.run(
            reference_data=baseline_df[available],
            current_data=recent_df[available],
            column_mapping=column_mapping,
        )

        result = report.as_dict()

        # Extract drift results per column
        drift_results = {}
        for metric_result in result.get("metrics", []):
            col = metric_result.get("metric", {}).get("column_name", "")
            if col:
                drift_results[col] = {
                    "drift_score": metric_result.get("result", {}).get("drift_score", 0.0),
                    "is_drifted": metric_result.get("result", {}).get("drift_detected", False),
                }

        return {
            "status": "ok",
            "total_interactions": len(baseline_df) + len(recent_df),
            "baseline_size": len(baseline_df),
            "recent_size": len(recent_df),
            "baseline_stats": self._compute_stats(baseline_df),
            "recent_stats": self._compute_stats(recent_df),
            "drift_results": drift_results,
            "any_drift_detected": any(
                v.get("is_drifted", False) for v in drift_results.values()
            ),
        }

    def _run_psi_report(
        self, baseline_df: pd.DataFrame, recent_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        PSI (Population Stability Index) fallback.
        PSI > 0.2 indicates significant distribution shift.
        """
        metrics = [
            "response_length", "grounding_score", "hallucination_risk",
            "avg_retrieval_score", "latency_ms",
        ]
        drift_results = {}
        for col in metrics:
            if col not in baseline_df.columns or col not in recent_df.columns:
                continue
            psi = self._compute_psi(baseline_df[col].dropna(), recent_df[col].dropna())
            drift_results[col] = {
                "drift_score": round(psi, 4),
                "is_drifted": psi > 0.2,
                "method": "PSI",
            }

        return {
            "status": "ok",
            "engine": "psi_fallback",
            "total_interactions": len(baseline_df) + len(recent_df),
            "baseline_size": len(baseline_df),
            "recent_size": len(recent_df),
            "baseline_stats": self._compute_stats(baseline_df),
            "recent_stats": self._compute_stats(recent_df),
            "drift_results": drift_results,
            "any_drift_detected": any(
                v.get("is_drifted", False) for v in drift_results.values()
            ),
        }

    @staticmethod
    def _compute_psi(
        baseline: pd.Series, current: pd.Series, bins: int = 10
    ) -> float:
        """Compute Population Stability Index between two distributions."""
        if len(baseline) == 0 or len(current) == 0:
            return 0.0
        combined = pd.concat([baseline, current])
        min_val, max_val = combined.min(), combined.max()
        if min_val == max_val:
            return 0.0

        boundaries = np.linspace(min_val, max_val, bins + 1)
        base_counts, _ = np.histogram(baseline, bins=boundaries)
        curr_counts, _ = np.histogram(current, bins=boundaries)

        base_pct = (base_counts + 1e-8) / len(baseline)
        curr_pct = (curr_counts + 1e-8) / len(current)

        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return float(psi)

    def get_time_series(self) -> Dict[str, List]:
        """Return time series data for dashboard charting."""
        df = self.to_dataframe()
        if df.empty:
            return {
                "timestamps": [],
                "grounding_scores": [],
                "hallucination_risks": [],
                "response_lengths": [],
                "retrieval_scores": [],
                "latencies_ms": [],
            }
        return {
            "timestamps": df["timestamp"].tolist(),
            "grounding_scores": df["grounding_score"].tolist(),
            "hallucination_risks": df["hallucination_risk"].tolist(),
            "response_lengths": df["response_length"].tolist(),
            "retrieval_scores": df["avg_retrieval_score"].tolist(),
            "latencies_ms": df["latency_ms"].tolist(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics for dashboard display."""
        df = self.to_dataframe()
        if df.empty:
            return {"total_queries": 0}
        return {
            "total_queries": len(df),
            "avg_grounding_score": round(df["grounding_score"].mean(), 3),
            "avg_hallucination_risk": round(df["hallucination_risk"].mean(), 3),
            "avg_latency_ms": round(df["latency_ms"].mean(), 1),
            "avg_retrieval_score": round(df["avg_retrieval_score"].mean(), 3),
            "pct_high_risk": round(
                (df["hallucination_risk"] > 0.3).mean() * 100, 1
            ),
        }


# Module-level singleton
_monitor: Optional[LLMMonitor] = None


def get_monitor() -> LLMMonitor:
    global _monitor
    if _monitor is None:
        _monitor = LLMMonitor()
    return _monitor
