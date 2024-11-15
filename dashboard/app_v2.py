"""
Finance LLMOps Platform V2 — Streamlit Dashboard

Tab 1: Multi-Agent Pipeline Visualizer — shows which agent is active, state flow
Tab 2: Streaming Q&A Interface — token-by-token output via st.write_stream()
Tab 3: RAG Evaluation Dashboard — RAGAS metrics on 20 test pairs
Tab 4: V1 Platform (preserved) — original 5-tab Ask/Evidence/Hallucination/Monitoring/PromptLab

Launches standalone:
    streamlit run dashboard/app_v2.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config (first Streamlit call) ───────────────────────────────────────
st.set_page_config(
    page_title="Finance LLMOps V2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (dark luxury + V2 badge) ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0d16 0%, #0f1520 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .v2-header {
        background: linear-gradient(135deg, #0a1628 0%, #0f2a52 50%, #0a1628 100%);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #1e3a6b;
        position: relative;
    }
    .v2-badge {
        position: absolute; top: 1rem; right: 1.5rem;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: #fff; font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.1em; padding: 0.2rem 0.6rem;
        border-radius: 20px;
    }
    .v2-header h1 { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .v2-header p  { color: #7fb3f5; margin: 0.25rem 0 0; font-size: 0.88rem; }

    /* Agent pipeline cards */
    .agent-card {
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 0.3rem 0;
        border: 1px solid #1e293b;
        font-size: 0.84rem;
    }
    .agent-pending  { background: #0f1117; border-color: #1e293b; color: #64748b; }
    .agent-running  { background: #0f2044; border-color: #3b82f6;
                      color: #93c5fd; animation: pulse 1.5s infinite; }
    .agent-done     { background: #052e16; border-color: #10b981; color: #6ee7b7; }
    .agent-error    { background: #450a0a; border-color: #ef4444; color: #fca5a5; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }

    .agent-label { font-weight: 700; font-size: 0.88rem; margin-bottom: 0.2rem; }
    .agent-desc  { font-size: 0.78rem; color: #94a3b8; }

    /* Streaming token display */
    .stream-box {
        background: #050d1f;
        border: 1px solid #1e3a6b;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        min-height: 100px;
        font-family: 'Georgia', serif;
        font-size: 0.95rem;
        line-height: 1.8;
        color: #cbd5e1;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #0f1117; border: 1px solid #1e293b;
        border-radius: 8px; padding: 0.75rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; }

    /* Eval score color bands */
    .score-excellent { color: #10b981; font-weight: 700; }
    .score-good      { color: #84cc16; font-weight: 700; }
    .score-fair      { color: #f59e0b; font-weight: 700; }
    .score-poor      { color: #ef4444; font-weight: 700; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #1e293b; }
    .stTabs [data-baseweb="tab"] { font-size: 0.82rem; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Header ────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown(
        """
        <div class="v2-header">
          <span class="v2-badge">V2</span>
          <h1>Finance LLMOps Platform V2</h1>
          <p>Multi-Agent LangGraph · SSE Streaming · RAGAS Evaluation ·
             SEC EDGAR Simulator · Hallucination Grounding</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='padding:1rem 0 0.5rem'>"
            "<span style='font-size:1.3rem'>🚀</span> "
            "<span style='font-size:0.95rem;font-weight:700;color:#e2e8f0'>Finance LLMOps V2</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:0.75rem;color:#64748b;margin:0 0 1rem'>"
            "Multi-Agent RAG Pipeline</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        top_k = st.slider("Top-K retrieval chunks", 1, 5, 3)
        stream_speed = st.select_slider(
            "Stream speed",
            options=["Slow", "Normal", "Fast"],
            value="Normal",
        )
        speed_map = {"Slow": (0.08, 0.18), "Normal": (0.03, 0.10), "Fast": (0.01, 0.04)}
        delay_range = speed_map[stream_speed]

        st.divider()
        mock_mode = os.getenv("MOCK_MODE", "true").lower() in ("true", "1", "yes")
        st.markdown(
            f"<span style='font-size:0.78rem;color:{'#f59e0b' if mock_mode else '#10b981'}'>"
            f"{'⚡ MOCK MODE' if mock_mode else '🔴 LIVE (OpenAI)'}</span>",
            unsafe_allow_html=True,
        )

    return top_k, delay_range


# ── Tab 1: Multi-Agent Pipeline Visualizer ────────────────────────────────────
_AGENT_META = {
    "researcher": {
        "label": "ResearcherAgent",
        "icon": "🔍",
        "desc": "Retrieves top-k SEC document chunks from FAISS index",
    },
    "analyst": {
        "label": "AnalystAgent",
        "icon": "📊",
        "desc": "Cross-references earnings-call data vs 10-K filings",
    },
    "fact_checker": {
        "label": "FactCheckerAgent",
        "icon": "🛡",
        "desc": "Flags hallucinated claims via token-overlap grounding",
    },
    "synthesizer": {
        "label": "SynthesizerAgent",
        "icon": "✨",
        "desc": "Composes final citation-grounded answer",
    },
}


def _agent_card_html(name: str, status: str, extra: str = "") -> str:
    meta = _AGENT_META.get(name, {"label": name, "icon": "·", "desc": ""})
    css_class = f"agent-{status}"
    icon = meta["icon"]
    running_indicator = " ◎" if status == "running" else ""
    return (
        f"<div class='agent-card {css_class}'>"
        f"<div class='agent-label'>{icon} {meta['label']}{running_indicator}</div>"
        f"<div class='agent-desc'>{meta['desc']}</div>"
        f"{'<div style=\"margin-top:0.4rem;font-size:0.78rem\">' + extra + '</div>' if extra else ''}"
        f"</div>"
    )


def render_pipeline_tab(top_k: int):
    st.markdown("#### Multi-Agent Pipeline Visualizer")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>Run the four-agent pipeline "
        "on a financial question and watch each agent activate in sequence.</p>",
        unsafe_allow_html=True,
    )

    SUGGESTED = [
        "What is Meridian's liquidity coverage ratio?",
        "What was the net revenue breakdown by segment in 2023?",
        "What is the CET1 capital ratio and regulatory buffer?",
        "How much did Meridian invest in cybersecurity?",
        "What NIM guidance was provided for 2024?",
    ]

    with st.expander("Suggested queries", expanded=False):
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            if cols[i % 2].button(q, key=f"pipe_sugg_{i}", use_container_width=True):
                st.session_state["pipeline_query"] = q

    query = st.text_input(
        "Financial question",
        key="pipeline_query",
        placeholder="e.g. What is Meridian's long-term debt structure?",
    )

    run_clicked = st.button("Run Pipeline", type="primary", key="run_pipeline")

    # Agent status display
    col_agents, col_output = st.columns([1, 2])

    with col_agents:
        st.markdown(
            "<p style='color:#64748b;font-size:0.78rem;font-weight:600;"
            "text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.5rem'>"
            "Agent Status</p>",
            unsafe_allow_html=True,
        )

        if "pipeline_agent_states" not in st.session_state:
            statuses = {name: "pending" for name in _AGENT_META}
        else:
            statuses = st.session_state["pipeline_agent_states"]

        agent_placeholder = st.empty()

        def render_agent_cards(statuses, extras=None):
            html = ""
            for name in _AGENT_META:
                extra = (extras or {}).get(name, "")
                html += _agent_card_html(name, statuses.get(name, "pending"), extra)
            agent_placeholder.markdown(html, unsafe_allow_html=True)

        render_agent_cards(statuses)

    with col_output:
        st.markdown(
            "<p style='color:#64748b;font-size:0.78rem;font-weight:600;"
            "text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.5rem'>"
            "Pipeline Output</p>",
            unsafe_allow_html=True,
        )
        output_placeholder = st.empty()

    if run_clicked and query.strip():
        try:
            from agents.multi_agent_rag import MultiAgentRAGPipeline
            pipeline = MultiAgentRAGPipeline(top_k=top_k)

            statuses = {name: "pending" for name in _AGENT_META}
            extras: dict = {}
            render_agent_cards(statuses, extras)
            output_placeholder.info("Pipeline starting…")

            for agent_name, partial_state in pipeline.run_streamed(query):
                # Mark current as done, next as running
                statuses[agent_name] = "done"
                agent_list = list(_AGENT_META.keys())
                idx = agent_list.index(agent_name)
                if idx + 1 < len(agent_list):
                    statuses[agent_list[idx + 1]] = "running"

                # Build extras for done agent
                if agent_name == "researcher":
                    n = len(partial_state.get("retrieved_chunks", []))
                    extras["researcher"] = f"Retrieved {n} chunks · {partial_state.get('retrieval_latency_ms', 0):.0f}ms"
                elif agent_name == "analyst":
                    extras["analyst"] = partial_state.get("cross_reference_notes", "")[:120]
                elif agent_name == "fact_checker":
                    gs = partial_state.get("grounding_score", 1.0)
                    flagged = len(partial_state.get("flagged_claims", []))
                    extras["fact_checker"] = f"Grounding: {gs:.0%} · Flagged: {flagged} claim(s)"
                elif agent_name == "synthesizer":
                    extras["synthesizer"] = f"Total latency: {partial_state.get('total_latency_ms', 0):.0f}ms"

                render_agent_cards(statuses, extras)
                time.sleep(0.1)  # brief pause so UI updates are visible

            # All done
            final_state = partial_state
            st.session_state["pipeline_agent_states"] = statuses
            st.session_state["pipeline_last_state"] = final_state

            # Display final answer
            output_placeholder.empty()
            with col_output:
                st.markdown(
                    f"<div style='background:#050d1f;border:1px solid #1e3a6b;"
                    f"border-radius:10px;padding:1rem;color:#cbd5e1;line-height:1.7;"
                    f"font-size:0.9rem'>{final_state.get('final_answer', '')}</div>",
                    unsafe_allow_html=True,
                )

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Chunks", len(final_state.get("retrieved_chunks", [])))
                m2.metric("Grounding", f"{final_state.get('grounding_score', 1.0):.0%}")
                m3.metric("Risk", f"{final_state.get('hallucination_risk', 0.0):.0%}")
                m4.metric("Latency", f"{final_state.get('total_latency_ms', 0):.0f}ms")

        except Exception as exc:
            output_placeholder.error(f"Pipeline error: {exc}")

    # Show last result if already run
    elif "pipeline_last_state" in st.session_state and not run_clicked:
        last = st.session_state["pipeline_last_state"]
        with col_output:
            st.markdown(
                f"<div style='background:#050d1f;border:1px solid #1e3a6b;"
                f"border-radius:10px;padding:1rem;color:#64748b;line-height:1.7;"
                f"font-size:0.85rem;font-style:italic'>"
                f"Previous result (run again to refresh):<br><br>"
                f"<span style='color:#cbd5e1;font-style:normal'>"
                f"{last.get('final_answer', '')[:600]}{'...' if len(last.get('final_answer','')) > 600 else ''}"
                f"</span></div>",
                unsafe_allow_html=True,
            )


# ── Tab 2: Streaming Q&A ──────────────────────────────────────────────────────

def render_streaming_tab(delay_range: tuple):
    st.markdown("#### Streaming Q&A Interface")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>Token-by-token LLM response "
        "streaming via Server-Sent Events. Each word is displayed as it arrives, "
        "simulating real-time LLM output.</p>",
        unsafe_allow_html=True,
    )

    STREAM_SUGGESTED = [
        "What is Meridian's liquidity coverage ratio?",
        "Describe Meridian's AI document intelligence platform.",
        "What is the NIM guidance for 2024?",
        "How does the cybersecurity investment compare year-over-year?",
    ]

    with st.expander("Suggested streaming queries", expanded=False):
        scols = st.columns(2)
        for i, q in enumerate(STREAM_SUGGESTED):
            if scols[i % 2].button(q, key=f"stream_sugg_{i}", use_container_width=True):
                st.session_state["stream_query"] = q

    stream_query = st.text_input(
        "Question to stream",
        key="stream_query",
        placeholder="e.g. What is the CET1 capital ratio?",
    )

    col_ask, col_clear = st.columns([1, 4])
    stream_clicked = col_ask.button("Stream Answer", type="primary", key="stream_btn")
    if col_clear.button("Clear", key="stream_clear"):
        st.session_state.pop("stream_history", None)

    if "stream_history" not in st.session_state:
        st.session_state["stream_history"] = []

    if stream_clicked and stream_query.strip():
        from streaming.stream_handler import streamlit_word_stream

        st.markdown("---")
        st.markdown(
            f"<div style='background:#0f1117;border-left:3px solid #3b82f6;"
            f"border-radius:4px;padding:0.5rem 0.75rem;color:#93c5fd;"
            f"font-size:0.85rem;margin-bottom:0.5rem'>{stream_query}</div>",
            unsafe_allow_html=True,
        )

        # Use st.write_stream for live display
        def _stream_gen():
            from streaming.stream_handler import word_stream
            for token in word_stream(stream_query, delay_range=delay_range):
                if token:
                    yield token

        full_response = st.write_stream(_stream_gen())

        # Save to history
        st.session_state["stream_history"].append({
            "query": stream_query,
            "response": full_response or "",
        })

    # Show history
    if st.session_state.get("stream_history"):
        st.markdown("---")
        st.markdown(
            "<p style='color:#64748b;font-size:0.78rem;font-weight:600;"
            "text-transform:uppercase;letter-spacing:0.1em'>Stream History</p>",
            unsafe_allow_html=True,
        )
        for item in reversed(st.session_state["stream_history"][:-1]):  # skip last (already shown)
            with st.expander(f"Q: {item['query'][:60]}…", expanded=False):
                st.markdown(
                    f"<div class='stream-box'>{item['response']}</div>",
                    unsafe_allow_html=True,
                )

    # SSE endpoint info
    with st.expander("SSE Endpoint Details", expanded=False):
        st.code(
            "# Launch the FastAPI streaming server:\n"
            "uvicorn streaming.stream_handler:app --port 8001\n\n"
            "# Connect via EventSource (JavaScript):\n"
            "const es = new EventSource('/api/v2/stream?query=What+is+the+LCR');\n"
            "es.onmessage = (e) => {\n"
            "  const data = JSON.parse(e.data);\n"
            "  if (!data.done) document.body.innerText += data.token;\n"
            "};",
            language="bash",
        )


# ── Tab 3: RAG Evaluation Dashboard ──────────────────────────────────────────

def _score_color_class(score: float) -> str:
    if score >= 0.75:
        return "score-excellent"
    if score >= 0.55:
        return "score-good"
    if score >= 0.35:
        return "score-fair"
    return "score-poor"


def render_eval_tab():
    st.markdown("#### RAG Evaluation Dashboard")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>RAGAS-style evaluation metrics "
        "on 20 financial Q&A test pairs. Metrics are computed from scratch without "
        "external API calls: faithfulness, relevance, answer similarity, "
        "context precision, and context recall.</p>",
        unsafe_allow_html=True,
    )

    col_btn, col_info = st.columns([1, 3])
    run_eval = col_btn.button("Run Evaluation", type="primary", key="run_eval")
    col_info.markdown(
        "<p style='color:#64748b;font-size:0.82rem;margin-top:0.5rem'>"
        "Scores 20 Q&A pairs on 5 RAGAS metrics. Takes ~10–30 seconds.</p>",
        unsafe_allow_html=True,
    )

    if run_eval:
        with st.spinner("Running RAG evaluation on 20 test pairs…"):
            try:
                from evaluation.eval_harness import RAGEvalHarness
                harness = RAGEvalHarness()
                report = harness.run(verbose=False)
                st.session_state["eval_report"] = report
                st.success(f"Evaluation complete — RAGAS score: {report.avg_ragas_score:.3f}")
            except Exception as exc:
                st.error(f"Evaluation error: {exc}")
                return

    report = st.session_state.get("eval_report")

    if report is None:
        # Show placeholder with metric descriptions
        st.info("Click 'Run Evaluation' to score the 20 test Q&A pairs.")

        st.markdown("#### Metric Definitions")
        metrics_info = {
            "Faithfulness": "Answer claims are grounded in retrieved source documents (token overlap).",
            "Relevance": "Retrieved context chunks match the query (Jaccard similarity).",
            "Answer Similarity": "Token-level F1 between generated and gold reference answers.",
            "Context Precision": "Fraction of retrieved chunks relevant to the gold answer.",
            "Context Recall": "Fraction of gold answer facts covered by retrieved context.",
            "RAGAS Score": "Harmonic mean of all five metrics.",
        }
        for metric, desc in metrics_info.items():
            st.markdown(
                f"<div style='padding:0.4rem 0;border-bottom:1px solid #1e293b'>"
                f"<span style='color:#e2e8f0;font-weight:600'>{metric}</span> "
                f"<span style='color:#64748b;font-size:0.85rem'>— {desc}</span></div>",
                unsafe_allow_html=True,
            )
        return

    # Aggregate metrics
    st.markdown("#### Aggregate Scores")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Faithfulness", f"{report.avg_faithfulness:.3f}")
    c2.metric("Relevance", f"{report.avg_relevance:.3f}")
    c3.metric("Ans Similarity", f"{report.avg_answer_similarity:.3f}")
    c4.metric("Ctx Precision", f"{report.avg_context_precision:.3f}")
    c5.metric("Ctx Recall", f"{report.avg_context_recall:.3f}")
    c6.metric("RAGAS Score", f"{report.avg_ragas_score:.3f}")

    st.metric("Avg Latency", f"{report.avg_latency_ms:.0f}ms")

    # Radar chart of aggregate scores
    metrics_labels = ["Faithfulness", "Relevance", "Ans Similarity", "Ctx Precision", "Ctx Recall"]
    metrics_values = [
        report.avg_faithfulness,
        report.avg_relevance,
        report.avg_answer_similarity,
        report.avg_context_precision,
        report.avg_context_recall,
    ]
    metrics_values_closed = metrics_values + [metrics_values[0]]
    metrics_labels_closed = metrics_labels + [metrics_labels[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=metrics_values_closed,
        theta=metrics_labels_closed,
        fill="toself",
        line_color="#6366f1",
        fillcolor="rgba(99,102,241,0.2)",
        name="RAG Metrics",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0a0f1a",
            radialaxis=dict(range=[0, 1], gridcolor="#1e293b", tickfont=dict(color="#64748b")),
            angularaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8")),
        ),
        paper_bgcolor="#0f1117",
        font_color="#e2e8f0",
        height=360,
        showlegend=False,
        margin=dict(t=30, b=30, l=30, r=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Per-sample breakdown
    st.markdown("#### Per-Sample Results")
    try:
        from evaluation.eval_harness import RAGEvalHarness
        harness = RAGEvalHarness()
        df = harness.get_scores_dataframe(report)

        # Format for display
        display_cols = [
            "sample_id", "query", "faithfulness", "relevance",
            "answer_similarity", "context_precision", "context_recall",
            "ragas_score", "latency_ms",
        ]
        available = [c for c in display_cols if c in df.columns]
        df_display = df[available].copy()
        df_display["query"] = df_display["query"].str[:60] + "…"

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # RAGAS score distribution
        if "ragas_score" in df.columns:
            fig_hist = px.histogram(
                df,
                x="ragas_score",
                nbins=10,
                title="RAGAS Score Distribution (20 samples)",
                color_discrete_sequence=["#6366f1"],
            )
            fig_hist.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#0a0f1a",
                font_color="#94a3b8", height=280,
                xaxis=dict(gridcolor="#1e293b", range=[0, 1]),
                yaxis=dict(gridcolor="#1e293b"),
                margin=dict(t=40, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as exc:
        st.warning(f"Could not render per-sample breakdown: {exc}")

    if report.output_csv:
        st.markdown(
            f"<p style='color:#64748b;font-size:0.82rem'>"
            f"Results saved to: <code>{report.output_csv}</code></p>",
            unsafe_allow_html=True,
        )


# ── Tab 4: V1 Platform (preserved) ───────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing V1 pipeline…")
def load_v1_pipeline():
    from pipeline.ingestion import build_faiss_index
    from pipeline.retriever import FinancialRetriever
    from pipeline.generator import FinancialAnswerGenerator
    from pipeline.monitor import get_monitor
    from experiments.prompt_tracker import get_tracker

    index, chunks = build_faiss_index()
    retriever = FinancialRetriever(index=index, chunks=chunks)
    retriever._lazy_init()
    generator = FinancialAnswerGenerator()
    generator._retriever = retriever
    monitor = get_monitor()
    tracker = get_tracker()
    return retriever, generator, monitor, tracker


def render_v1_tab():
    st.markdown("#### V1 Platform — Original RAG Interface")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>The original V1 platform is "
        "fully preserved below. All five V1 tabs are accessible via the import.</p>",
        unsafe_allow_html=True,
    )

    try:
        retriever, generator, monitor, tracker = load_v1_pipeline()

        v1_tab1, v1_tab2, v1_tab3, v1_tab4, v1_tab5 = st.tabs([
            "Ask", "Source Evidence", "Hallucination Check", "LLM Monitoring", "Prompt Lab"
        ])

        # Import V1 render functions
        from dashboard.app import (
            render_ask_tab,
            render_evidence_tab,
            render_hallucination_tab,
            render_monitoring_tab,
            render_prompt_lab_tab,
        )

        with v1_tab1:
            render_ask_tab(generator, monitor, retriever, "v2_structured", 3)
        with v1_tab2:
            render_evidence_tab(retriever, 3)
        with v1_tab3:
            render_hallucination_tab()
        with v1_tab4:
            render_monitoring_tab(monitor)
        with v1_tab5:
            render_prompt_lab_tab(tracker)

    except Exception as exc:
        st.info(
            f"V1 pipeline not available in this environment ({exc}). "
            "Run `streamlit run dashboard/app.py` to launch V1 standalone."
        )
        st.markdown(
            "<p style='color:#64748b;font-size:0.85rem;'>V1 features include: "
            "Ask (cited answers), Source Evidence explorer, Hallucination grounding gauge, "
            "LLM Monitoring drift charts, and Prompt Lab experiment comparison.</p>",
            unsafe_allow_html=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    render_header()
    top_k, delay_range = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Multi-Agent Pipeline",
        "Streaming Q&A",
        "RAG Evaluation",
        "V1 Platform",
    ])

    with tab1:
        render_pipeline_tab(top_k=top_k)

    with tab2:
        render_streaming_tab(delay_range=delay_range)

    with tab3:
        render_eval_tab()

    with tab4:
        render_v1_tab()


if __name__ == "__main__":
    main()
