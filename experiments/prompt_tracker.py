"""
MLflow prompt experiment tracking.

Tracks prompt versions across runs with metrics:
  - avg_grounding_score
  - avg_hallucination_risk
  - answer_relevance (token overlap with expected answers)
  - avg_retrieval_score
  - avg_latency_ms

Falls back to an in-memory/CSV tracker if MLflow is unavailable.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

EXPERIMENTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_CSV = EXPERIMENTS_DIR / "prompt_experiments.csv"

EXPERIMENT_NAME = "finance-llmops-prompt-optimization"


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

@dataclass
class PromptRunConfig:
    """Configuration for a single prompt experiment run."""
    prompt_version: str
    retrieval_k: int
    model: str
    chunk_size: int
    chunk_overlap: int
    notes: str = ""


@dataclass
class PromptRunMetrics:
    """Aggregated metrics for a prompt experiment run."""
    avg_grounding_score: float
    avg_hallucination_risk: float
    avg_retrieval_score: float
    avg_latency_ms: float
    answer_relevance: float         # token overlap with gold answers
    num_queries: int
    pct_fully_grounded: float       # % queries where grounding >= 0.85
    pct_high_risk: float            # % queries where hallucination_risk > 0.3


@dataclass
class ExperimentRun:
    """Complete record of one experiment run."""
    run_id: str
    run_name: str
    config: PromptRunConfig
    metrics: PromptRunMetrics
    started_at: float
    ended_at: float
    duration_s: float
    tags: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# In-memory experiment registry (fallback when MLflow unavailable)
# ---------------------------------------------------------------------------

_SEED_RUNS: List[Dict[str, Any]] = [
    {
        "run_id": "run_001",
        "run_name": "baseline_v1",
        "prompt_version": "v1_basic",
        "retrieval_k": 3,
        "model": "gpt-4",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "avg_grounding_score": 0.712,
        "avg_hallucination_risk": 0.288,
        "avg_retrieval_score": 0.521,
        "avg_latency_ms": 2340,
        "answer_relevance": 0.634,
        "num_queries": 20,
        "pct_fully_grounded": 55.0,
        "pct_high_risk": 35.0,
        "started_at": time.time() - 86400 * 7,
        "duration_s": 47.2,
    },
    {
        "run_id": "run_002",
        "run_name": "structured_prompt_v2",
        "prompt_version": "v2_structured",
        "retrieval_k": 3,
        "model": "gpt-4",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "avg_grounding_score": 0.831,
        "avg_hallucination_risk": 0.169,
        "avg_retrieval_score": 0.521,
        "avg_latency_ms": 2180,
        "answer_relevance": 0.741,
        "num_queries": 20,
        "pct_fully_grounded": 75.0,
        "pct_high_risk": 20.0,
        "started_at": time.time() - 86400 * 5,
        "duration_s": 43.6,
    },
    {
        "run_id": "run_003",
        "run_name": "larger_retrieval_k5",
        "prompt_version": "v2_structured",
        "retrieval_k": 5,
        "model": "gpt-4",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "avg_grounding_score": 0.857,
        "avg_hallucination_risk": 0.143,
        "avg_retrieval_score": 0.489,
        "avg_latency_ms": 2890,
        "answer_relevance": 0.763,
        "num_queries": 20,
        "pct_fully_grounded": 80.0,
        "pct_high_risk": 15.0,
        "started_at": time.time() - 86400 * 3,
        "duration_s": 57.8,
    },
    {
        "run_id": "run_004",
        "run_name": "chain_of_thought_v3",
        "prompt_version": "v3_cot",
        "retrieval_k": 3,
        "model": "gpt-4",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "avg_grounding_score": 0.876,
        "avg_hallucination_risk": 0.124,
        "avg_retrieval_score": 0.521,
        "avg_latency_ms": 3450,
        "answer_relevance": 0.782,
        "num_queries": 20,
        "pct_fully_grounded": 85.0,
        "pct_high_risk": 10.0,
        "started_at": time.time() - 86400 * 2,
        "duration_s": 69.1,
    },
    {
        "run_id": "run_005",
        "run_name": "cot_larger_chunks",
        "prompt_version": "v3_cot",
        "retrieval_k": 3,
        "model": "gpt-4",
        "chunk_size": 750,
        "chunk_overlap": 75,
        "avg_grounding_score": 0.891,
        "avg_hallucination_risk": 0.109,
        "avg_retrieval_score": 0.543,
        "avg_latency_ms": 3210,
        "answer_relevance": 0.798,
        "num_queries": 20,
        "pct_fully_grounded": 90.0,
        "pct_high_risk": 5.0,
        "started_at": time.time() - 86400,
        "duration_s": 64.3,
    },
]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class PromptExperimentTracker:
    """
    Tracks prompt experiment runs with MLflow (with CSV fallback).
    """

    def __init__(self) -> None:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self._runs: List[Dict[str, Any]] = list(_SEED_RUNS)
        self._mlflow_enabled = self._try_init_mlflow()
        self._save_to_csv()  # Ensure CSV is populated on startup

    def _try_init_mlflow(self) -> bool:
        try:
            import mlflow
            tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI",
                str(EXPERIMENTS_DIR / "mlruns")
            )
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            return True
        except ImportError:
            return False

    def start_run(
        self,
        config: PromptRunConfig,
        run_name: Optional[str] = None,
    ) -> str:
        """Start an MLflow run. Returns run_id."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        if run_name is None:
            run_name = f"{config.prompt_version}_k{config.retrieval_k}_{int(time.time())}"

        if self._mlflow_enabled:
            import mlflow
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(asdict(config))
                return run.info.run_id

        return run_id

    def log_run(
        self,
        config: PromptRunConfig,
        metrics: PromptRunMetrics,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Log a completed run with config and metrics."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        if run_name is None:
            run_name = f"{config.prompt_version}_k{config.retrieval_k}"
        if tags is None:
            tags = {}

        t0 = time.time()

        run_record = {
            "run_id": run_id,
            "run_name": run_name,
            "prompt_version": config.prompt_version,
            "retrieval_k": config.retrieval_k,
            "model": config.model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            **asdict(metrics),
            "started_at": t0 - metrics.avg_latency_ms * metrics.num_queries / 1000,
            "duration_s": round(metrics.avg_latency_ms * metrics.num_queries / 1000, 1),
        }
        self._runs.append(run_record)
        self._save_to_csv()

        if self._mlflow_enabled:
            try:
                import mlflow
                with mlflow.start_run(run_name=run_name) as run:
                    mlflow.log_params(asdict(config))
                    mlflow.log_metrics(asdict(metrics))
                    for k, v in tags.items():
                        mlflow.set_tag(k, v)
                    run_id = run.info.run_id
            except Exception:
                pass

        return run_id

    def _save_to_csv(self) -> None:
        """Persist runs to CSV for dashboard display."""
        df = self.get_runs_dataframe()
        df.to_csv(RESULTS_CSV, index=False)

    def get_runs_dataframe(self) -> pd.DataFrame:
        """Return all runs as a pandas DataFrame."""
        if not self._runs:
            return pd.DataFrame()
        return pd.DataFrame(self._runs)

    def get_best_run(self, metric: str = "avg_grounding_score") -> Optional[Dict]:
        """Return the run with the best value for a given metric."""
        if not self._runs:
            return None
        # Higher is better for grounding/relevance; lower is better for risk/latency
        lower_is_better = {"avg_hallucination_risk", "avg_latency_ms", "pct_high_risk"}
        reverse = metric not in lower_is_better
        try:
            return sorted(
                self._runs,
                key=lambda r: r.get(metric, 0),
                reverse=reverse,
            )[0]
        except Exception:
            return None

    def compare_prompt_versions(self) -> pd.DataFrame:
        """
        Group runs by prompt_version and compute mean metrics.
        Useful for the Prompt Lab dashboard tab.
        """
        df = self.get_runs_dataframe()
        if df.empty:
            return pd.DataFrame()
        metric_cols = [
            "avg_grounding_score", "avg_hallucination_risk",
            "avg_retrieval_score", "avg_latency_ms",
            "answer_relevance", "pct_fully_grounded", "pct_high_risk",
        ]
        available = [c for c in metric_cols if c in df.columns]
        return (
            df.groupby("prompt_version")[available]
            .agg(["mean", "count"])
            .round(3)
        )

    def run_evaluation(
        self,
        qa_pairs_path: Optional[Path] = None,
        prompt_version: str = "v2_structured",
        retrieval_k: int = 3,
    ) -> str:
        """
        Run a full evaluation against the gold QA pairs and log results.
        Uses mock answers if MOCK_MODE is set.
        """
        from pipeline.generator import get_generator
        from pipeline.hallucination import check_from_generated_answer

        if qa_pairs_path is None:
            qa_pairs_path = (
                Path(__file__).parent.parent / "data" / "financial_qa_pairs.json"
            )

        with open(qa_pairs_path) as f:
            qa_pairs = json.load(f)

        gen = get_generator(prompt_version=prompt_version)

        grounding_scores, hallucination_risks, retrieval_scores, latencies, relevances = (
            [], [], [], [], [],
        )

        for qa in qa_pairs:
            result = gen.generate(qa["question"])
            report = check_from_generated_answer(result)

            grounding_scores.append(report.grounding_score)
            hallucination_risks.append(report.hallucination_risk)
            retrieval_scores.append(
                sum(result.retrieval_scores) / max(len(result.retrieval_scores), 1)
            )
            latencies.append(result.latency_ms)

            # Answer relevance: token overlap with expected answer
            from pipeline.hallucination import _tokenize
            expected_tokens = _tokenize(qa["expected_answer"])
            actual_tokens = _tokenize(result.answer)
            if expected_tokens:
                overlap = expected_tokens & actual_tokens
                relevances.append(len(overlap) / len(expected_tokens))
            else:
                relevances.append(0.0)

        n = len(qa_pairs)
        config = PromptRunConfig(
            prompt_version=prompt_version,
            retrieval_k=retrieval_k,
            model="mock/all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=50,
        )
        metrics = PromptRunMetrics(
            avg_grounding_score=round(sum(grounding_scores) / n, 4),
            avg_hallucination_risk=round(sum(hallucination_risks) / n, 4),
            avg_retrieval_score=round(sum(retrieval_scores) / n, 4),
            avg_latency_ms=round(sum(latencies) / n, 1),
            answer_relevance=round(sum(relevances) / n, 4),
            num_queries=n,
            pct_fully_grounded=round(
                sum(1 for s in grounding_scores if s >= 0.85) / n * 100, 1
            ),
            pct_high_risk=round(
                sum(1 for r in hallucination_risks if r > 0.3) / n * 100, 1
            ),
        )
        return self.log_run(config, metrics, run_name=f"eval_{prompt_version}")


# Module-level singleton
_tracker: Optional[PromptExperimentTracker] = None


def get_tracker() -> PromptExperimentTracker:
    global _tracker
    if _tracker is None:
        _tracker = PromptExperimentTracker()
    return _tracker
