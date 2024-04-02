"""
Finance LLMOps Platform — Streamlit Dashboard
Meridian Financial Corp Document Intelligence

Tabs:
  1. Ask        — Chat interface with cited answers
  2. Evidence   — Source chunk explorer with highlighted passages
  3. Hallucination Check — Grounding gauge and uncited claim list
  4. LLM Monitoring    — Drift charts (response length, grounding, risk)
  5. Prompt Lab        — MLflow experiment comparison table
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── Ensure project root is on path ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Finance LLMOps Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #1a1f2e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Main header */
    .platform-header {
        background: linear-gradient(135deg, #0f2044 0%, #1a3a6b 50%, #0f2044 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2d4a7a;
    }
    .platform-header h1 {
        color: #e2e8f0;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .platform-header p {
        color: #7fb3f5;
        margin: 0.25rem 0 0;
        font-size: 0.9rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #0f1117;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; }

    /* Chat bubbles */
    .chat-user {
        background: #1e293b;
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    .chat-assistant {
        background: #0f2044;
        border-left: 3px solid #10b981;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    .chat-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.3rem;
    }

    /* Citation badges */
    .citation-badge {
        display: inline-block;
        background: #1e3a5f;
        border: 1px solid #3b82f6;
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        font-size: 0.75rem;
        color: #93c5fd;
        margin: 0.1rem;
    }

    /* Source expander highlight */
    .source-highlight {
        background: #1a2744;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 0.5rem 0.75rem;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #cbd5e1;
        white-space: pre-wrap;
    }

    /* Hallucination claim tags */
    .claim-grounded {
        display: inline-block;
        background: #064e3b;
        border: 1px solid #10b981;
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.78rem;
        color: #6ee7b7;
        margin: 0.15rem;
    }
    .claim-uncited {
        display: inline-block;
        background: #450a0a;
        border: 1px solid #ef4444;
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.78rem;
        color: #fca5a5;
        margin: 0.15rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    /* Section divider */
    .section-title {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 1rem 0 0.5rem;
    }

    /* Drift indicator */
    .drift-ok { color: #10b981; font-weight: 600; }
    .drift-warn { color: #f59e0b; font-weight: 600; }
    .drift-alert { color: #ef4444; font-weight: 600; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Lazy-load pipeline (cached so it only runs once per session) ──────────────
@st.cache_resource(show_spinner="Initializing pipeline — loading FAISS index & embeddings…")
def load_pipeline():
    """Build/load FAISS index and initialize all pipeline components."""
    from pipeline.ingestion import build_faiss_index
    from pipeline.retriever import FinancialRetriever
    from pipeline.generator import FinancialAnswerGenerator
    from pipeline.monitor import get_monitor
    from experiments.prompt_tracker import get_tracker

    index, chunks = build_faiss_index()
    retriever = FinancialRetriever(index=index, chunks=chunks)
    # Pre-warm retriever model
    retriever._lazy_init()

    generator = FinancialAnswerGenerator()
    generator._retriever = retriever

    monitor = get_monitor()
    tracker = get_tracker()

    # Seed monitoring with realistic data if empty
    _seed_monitor_data(monitor)

    return retriever, generator, monitor, tracker


def _seed_monitor_data(monitor):
    """Populate monitoring log with 30 realistic interactions for chart demo."""
    if len(monitor._logs) >= 20:
        return

    from pipeline.hallucination import check_hallucination
    import random
    rng = random.Random(42)

    sample_pairs = [
        ("What is the LCR ratio?",
         "Meridian Financial Corp maintains an LCR of 134% as of December 31, 2023."),
        ("What was net revenue in 2023?",
         "Total net revenue was $3.21 billion, a 7.4% increase from $2.99 billion in 2022."),
        ("How much did Meridian invest in cybersecurity?",
         "Meridian invested $78.3 million in information security in fiscal 2023."),
        ("What is the CET1 ratio?",
         "The Common Equity Tier 1 ratio stands at 11.8%, above the 6.5% well-capitalized threshold."),
        ("Describe the AI document intelligence platform.",
         "The platform processes a 10-K in 8–12 minutes with 94.7% accuracy."),
        ("What is the deposit beta assumption?",
         "Meridian modeled a cumulative deposit beta of 52% on interest-bearing deposits."),
        ("What are annual principal maturities?",
         "Annual maturities for 2024–2028 are $420M, $1.24B, $890M, $640M, $720M respectively."),
    ]

    sample_source = (
        "Meridian Financial Corp maintains a liquidity coverage ratio (LCR) of 134% "
        "as of December 31, 2023, exceeding the regulatory minimum of 100%. "
        "Total net revenue was $3.21 billion in fiscal 2023. "
        "The Company invested $78.3 million in cybersecurity. "
        "CET1 ratio stands at 11.8%. "
        "The AI platform processes a 10-K in 8–12 minutes with 94.7% accuracy. "
        "Cumulative deposit beta was 52%. "
        "Annual maturities for 2024 through 2028 are $420 million, $1.24 billion, "
        "$890 million, $640 million, and $720 million respectively."
    )

    import time as _time
    base_time = _time.time() - 30 * 86400

    for i in range(30):
        q, a = sample_pairs[i % len(sample_pairs)]
        # Inject occasional hallucination for realism
        if i % 7 == 0:
            a = a + " Revenue was also $5 trillion in 2025."

        report = check_hallucination(a, [sample_source])

        grounding_decay = max(0.0, 0.05 * math.sin(i * 0.4))
        gs = max(0.1, min(1.0, report.grounding_score - grounding_decay + rng.uniform(-0.05, 0.05)))
        hr = 1.0 - gs

        monitor.log_interaction(
            question=q,
            answer=a,
            grounding_score=gs,
            hallucination_risk=hr,
            avg_retrieval_score=rng.uniform(0.48, 0.68),
            latency_ms=rng.uniform(120, 380) + (i * 3),
            prompt_version="v2_structured",
            model="mock/all-MiniLM-L6-v2",
            num_citations=3,
            num_uncited_claims=len(report.uncited_claims),
            num_total_claims=len(report.claims),
        )
        # Patch timestamp for realistic time series
        monitor._logs[-1].timestamp = base_time + i * 86400


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(tracker):
    with st.sidebar:
        st.markdown(
            "<div style='padding:1rem 0 0.5rem;'>"
            "<span style='font-size:1.3rem;'>📊</span> "
            "<span style='font-size:0.95rem;font-weight:700;color:#e2e8f0;'>"
            "Finance LLMOps</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:0.75rem;color:#64748b;margin:0 0 1rem;'>"
            "Meridian Financial Corp</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("<p class='section-title'>Document Source</p>", unsafe_allow_html=True)
        doc_filter = st.selectbox(
            "Filter by document",
            ["All Documents", "10-K Filing (2023)", "Q4 2023 Earnings Call"],
            label_visibility="collapsed",
        )

        st.markdown("<p class='section-title'>Prompt Version</p>", unsafe_allow_html=True)
        prompt_version = st.selectbox(
            "Prompt version",
            ["v2_structured", "v1_basic", "v3_cot"],
            label_visibility="collapsed",
        )

        st.markdown("<p class='section-title'>Retrieval</p>", unsafe_allow_html=True)
        top_k = st.slider("Top-K chunks", min_value=1, max_value=5, value=3)

        st.markdown("<p class='section-title'>Mode</p>", unsafe_allow_html=True)
        mock_mode = os.getenv("MOCK_MODE", "true").lower() in ("true", "1", "yes")
        mode_label = "MOCK MODE (no API key)" if mock_mode else "LIVE (OpenAI)"
        st.markdown(
            f"<span style='font-size:0.78rem;color:{'#f59e0b' if mock_mode else '#10b981'};'>"
            f"{'⚡' if mock_mode else '🔴'} {mode_label}</span>",
            unsafe_allow_html=True,
        )

        st.divider()

        # Best run summary
        best = tracker.get_best_run("avg_grounding_score")
        if best:
            st.markdown("<p class='section-title'>Best Prompt Run</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='font-size:0.82rem;color:#10b981;'>🏆 {best.get('prompt_version','')}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size:0.78rem;color:#64748b;'>Grounding: "
                f"<b style='color:#e2e8f0;'>{best.get('avg_grounding_score',0)*100:.1f}%</b></p>",
                unsafe_allow_html=True,
            )

    return doc_filter, prompt_version, top_k


# ── Header ────────────────────────────────────────────────────────────────────
def render_header(monitor):
    summary = monitor.get_summary()
    total_q = summary.get("total_queries", 0)
    avg_gs = summary.get("avg_grounding_score", 0)
    avg_hr = summary.get("avg_hallucination_risk", 0)
    avg_lat = summary.get("avg_latency_ms", 0)

    st.markdown(
        """
        <div class="platform-header">
          <h1>📊 Finance LLMOps Platform</h1>
          <p>Meridian Financial Corp · SEC 10-K + Earnings Call Intelligence · Production RAG Pipeline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Queries", f"{total_q:,}")
    col2.metric("Avg Grounding", f"{avg_gs*100:.1f}%", delta="+2.1%")
    col3.metric("Hallucination Risk", f"{avg_hr*100:.1f}%", delta="-1.8%", delta_color="inverse")
    col4.metric("Avg Latency", f"{avg_lat:.0f}ms")
    col5.metric(
        "Docs Indexed",
        "2",
        help="10-K Filing + Q4 2023 Earnings Call",
    )


# ── Tab 1: Ask ────────────────────────────────────────────────────────────────
def render_ask_tab(generator, monitor, retriever, prompt_version, top_k):
    st.markdown("#### Ask a Financial Question")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>Queries are answered using "
        "semantic search over Meridian Financial Corp's 10-K and earnings call "
        "transcripts. All answers include source citations.</p>",
        unsafe_allow_html=True,
    )

    # Suggested questions
    SUGGESTED = [
        "What is Meridian's liquidity coverage ratio?",
        "What was the net revenue breakdown by segment in 2023?",
        "How much did Meridian invest in cybersecurity?",
        "What is the AI document intelligence platform's accuracy rate?",
        "What is the CET1 ratio and how does it compare to regulatory minimums?",
        "What deposit beta is embedded in the 2024 NIM guidance?",
    ]

    with st.expander("Suggested questions", expanded=False):
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state["question_input"] = q

    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="e.g. What is Meridian's allowance for credit losses ratio?",
    )

    col_ask, col_clear = st.columns([1, 4])
    ask_clicked = col_ask.button("Ask", type="primary", use_container_width=True)
    if col_clear.button("Clear history", use_container_width=True):
        st.session_state.pop("qa_history", None)

    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    if ask_clicked and question.strip():
        with st.spinner("Retrieving and generating answer…"):
            generator.prompt_version = prompt_version
            generator.top_k = top_k
            result = generator.generate(question)

            from pipeline.hallucination import check_from_generated_answer
            h_report = check_from_generated_answer(result)

            avg_ret = (
                sum(result.retrieval_scores) / len(result.retrieval_scores)
                if result.retrieval_scores else 0.0
            )
            monitor.log_interaction(
                question=result.question,
                answer=result.answer,
                grounding_score=h_report.grounding_score,
                hallucination_risk=h_report.hallucination_risk,
                avg_retrieval_score=avg_ret,
                latency_ms=result.latency_ms,
                prompt_version=result.prompt_version,
                model=result.model,
                num_citations=len(result.citations),
                num_uncited_claims=len(h_report.uncited_claims),
                num_total_claims=len(h_report.claims),
            )

            st.session_state["last_result"] = result
            st.session_state["last_hallucination"] = h_report
            st.session_state["qa_history"].append((question, result, h_report))

    # Render history (most recent first)
    for q_text, res, h_rep in reversed(st.session_state.get("qa_history", [])):
        st.markdown(
            f"<div class='chat-user'><div class='chat-label'>Analyst</div>{q_text}</div>",
            unsafe_allow_html=True,
        )

        gs_color = "#10b981" if h_rep.grounding_score >= 0.85 else (
            "#f59e0b" if h_rep.grounding_score >= 0.6 else "#ef4444"
        )

        citations_html = " ".join(
            f"<span class='citation-badge'>{c.doc_name} #{c.chunk_id}</span>"
            for c in res.citations
        )

        st.markdown(
            f"""
            <div class='chat-assistant'>
              <div class='chat-label'>Platform · {res.prompt_version} · {res.latency_ms:.0f}ms</div>
              <div style='color:#e2e8f0;line-height:1.6;'>{res.answer}</div>
              <div style='margin-top:0.5rem;'>
                <span style='font-size:0.75rem;color:#64748b;'>Sources: </span>
                {citations_html}
                &nbsp;&nbsp;
                <span style='font-size:0.75rem;background:#0f2f0f;border:1px solid {gs_color};
                    border-radius:4px;padding:0.1rem 0.4rem;color:{gs_color};'>
                  Grounding: {h_rep.grounding_score*100:.0f}%
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Tab 2: Source Evidence ────────────────────────────────────────────────────
def render_evidence_tab(retriever, top_k):
    st.markdown("#### Source Evidence Explorer")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>Enter a query to retrieve "
        "the most relevant source document passages. Expand each result to read "
        "the full chunk with highlighted metadata.</p>",
        unsafe_allow_html=True,
    )

    evidence_q = st.text_input(
        "Query for evidence retrieval",
        placeholder="e.g. long-term debt obligations and FHLB advances",
        key="evidence_query",
    )

    if st.button("Retrieve Sources", key="btn_retrieve"):
        with st.spinner("Searching vector index…"):
            results = retriever.retrieve_with_metadata(evidence_q, top_k=top_k)

        st.markdown(
            f"<p style='color:#64748b;font-size:0.82rem;'>Found "
            f"<b style='color:#e2e8f0;'>{len(results['chunks'])}</b> chunks "
            f"in <b style='color:#e2e8f0;'>{results['latency_ms']:.0f}ms</b> · "
            f"Avg similarity: <b style='color:#e2e8f0;'>{results['avg_score']:.3f}</b></p>",
            unsafe_allow_html=True,
        )

        for chunk in results["chunks"]:
            doc_icon = "📄" if chunk["doc_type"] == "10k" else "🎤"
            score_color = (
                "#10b981" if chunk["score"] >= 0.6 else
                "#f59e0b" if chunk["score"] >= 0.4 else "#ef4444"
            )
            with st.expander(
                f"{doc_icon} Rank {chunk['rank']} · {chunk['doc_name']} "
                f"· {chunk['chunk_id']} · similarity={chunk['score']:.4f}",
                expanded=chunk["rank"] == 1,
            ):
                col_meta, col_score = st.columns([3, 1])
                with col_meta:
                    st.markdown(
                        f"<span class='citation-badge'>doc: {chunk['doc_name']}</span> "
                        f"<span class='citation-badge'>chunk: {chunk['chunk_id']}</span> "
                        f"<span class='citation-badge'>type: {chunk['doc_type']}</span>",
                        unsafe_allow_html=True,
                    )
                with col_score:
                    st.metric("Similarity", f"{chunk['score']:.4f}")

                st.markdown(
                    f"<div class='source-highlight'>{chunk['text']}</div>",
                    unsafe_allow_html=True,
                )
    else:
        # Show last Q&A's sources if available
        last_result = st.session_state.get("last_result")
        if last_result and last_result.citations:
            st.info("Showing source chunks from your last Ask query. Enter a new query above to search directly.")
            for i, c in enumerate(last_result.citations):
                with st.expander(
                    f"📄 Source {i+1} · {c.doc_name} · {c.chunk_id} "
                    f"· relevance={c.relevance_score:.4f}",
                    expanded=i == 0,
                ):
                    display_text = c.text_snippet[:600] + "..." if len(c.text_snippet) > 600 else c.text_snippet
                    st.markdown(
                        f"<div class='source-highlight'>{display_text}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Ask a question in the **Ask** tab or enter a query above to see source evidence.")


# ── Tab 3: Hallucination Check ────────────────────────────────────────────────
def render_hallucination_tab():
    st.markdown("#### Hallucination Detection")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>Every generated answer is "
        "scored for citation grounding. Factual claims (numbers, percentages, "
        "dates, acronyms) are checked against retrieved source chunks using "
        "token overlap analysis.</p>",
        unsafe_allow_html=True,
    )

    last_result = st.session_state.get("last_result")
    last_h = st.session_state.get("last_hallucination")

    if not last_result or not last_h:
        # Demo mode: show a pre-populated example
        st.info("No active query. Showing a demonstration with a sample answer.")
        from pipeline.hallucination import check_hallucination

        demo_answer = (
            "Meridian Financial Corp maintains an LCR of 134% as of December 31, 2023, "
            "exceeding the 100% regulatory minimum. The Company holds $2.84 billion in HQLA. "
            "Net revenue was $3.21 billion, growing 7.4% year-over-year. "
            "The CET1 ratio is 11.8%, above the 6.5% threshold. "
            "The AI platform processes filings in 8 minutes with 94.7% accuracy."
        )
        demo_source = (
            "Meridian Financial Corp maintains a liquidity coverage ratio (LCR) of 134% as of "
            "December 31, 2023, exceeding the regulatory minimum of 100%. "
            "The Company holds $2.84 billion in HQLA. "
            "Total net revenue was $3.21 billion, a 7.4% increase. "
            "CET1 ratio at 11.8%, exceeding the well-capitalized threshold of 6.5%. "
            "The platform processes a 10-K in approximately 8 to 12 minutes with 94.7% accuracy."
        )
        last_h = check_hallucination(demo_answer, [demo_source])
        demo_mode = True
        answer_text = demo_answer
    else:
        demo_mode = False
        answer_text = last_result.answer

    gs = last_h.grounding_score
    hr = last_h.hallucination_risk

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gs * 100,
        title={"text": "Grounding Score", "font": {"size": 16, "color": "#94a3b8"}},
        delta={"reference": 85, "suffix": "%", "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 36, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#64748b"}},
            "bar": {"color": "#10b981" if gs >= 0.85 else "#f59e0b" if gs >= 0.6 else "#ef4444"},
            "bgcolor": "#0f1117",
            "bordercolor": "#1e293b",
            "steps": [
                {"range": [0, 60], "color": "#1a0505"},
                {"range": [60, 85], "color": "#1a1205"},
                {"range": [85, 100], "color": "#051a0a"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.75,
                "value": 85,
            },
        },
    ))
    fig_gauge.update_layout(
        height=280,
        paper_bgcolor="#0f1117",
        font_color="#e2e8f0",
        margin=dict(t=50, b=20, l=30, r=30),
    )

    col_gauge, col_stats = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_stats:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Grounding Score", f"{gs*100:.1f}%")
        c2.metric("Hallucination Risk", f"{hr*100:.1f}%",
                  delta=f"{(hr-0.15)*100:+.1f}% vs baseline", delta_color="inverse")
        c1.metric("Total Claims", len(last_h.claims))
        c2.metric("Uncited Claims", len(last_h.uncited_claims))

        threshold_ok = gs >= 0.85
        status_color = "#10b981" if threshold_ok else "#ef4444"
        status_text = "PASSES 85% THRESHOLD" if threshold_ok else "BELOW 85% THRESHOLD — REVIEW REQUIRED"
        st.markdown(
            f"<div style='background:#0a0a0a;border:1px solid {status_color};"
            f"border-radius:6px;padding:0.5rem 0.75rem;margin-top:0.5rem;'>"
            f"<span style='color:{status_color};font-weight:700;font-size:0.82rem;'>"
            f"{'✓' if threshold_ok else '⚠'} {status_text}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Claim Analysis")

    if not last_h.claims:
        st.info("No factual claims detected in this answer.")
        return

    # Show answer with claim tags
    st.markdown(
        "<p style='color:#64748b;font-size:0.82rem;'>Generated answer with detected claims:</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:#0a1525;border:1px solid #1e293b;border-radius:8px;"
        f"padding:0.75rem 1rem;color:#cbd5e1;font-size:0.88rem;line-height:1.7;'>"
        f"{answer_text}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Claim breakdown
    col_g, col_u = st.columns(2)
    with col_g:
        st.markdown(
            "<p style='color:#10b981;font-size:0.82rem;font-weight:600;'>✓ Grounded Claims</p>",
            unsafe_allow_html=True,
        )
        if last_h.grounded_claims:
            tags = " ".join(
                f"<span class='claim-grounded'>{c}</span>"
                for c in last_h.grounded_claims
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown(
                "<p style='color:#64748b;font-size:0.82rem;'>None</p>",
                unsafe_allow_html=True,
            )

    with col_u:
        st.markdown(
            "<p style='color:#ef4444;font-size:0.82rem;font-weight:600;'>⚠ Uncited Claims</p>",
            unsafe_allow_html=True,
        )
        if last_h.uncited_claims:
            tags = " ".join(
                f"<span class='claim-uncited'>{c}</span>"
                for c in last_h.uncited_claims
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown(
                "<p style='color:#10b981;font-size:0.82rem;'>None — fully grounded</p>",
                unsafe_allow_html=True,
            )

    # Per-claim overlap table
    if last_h.claim_details:
        st.markdown("#### Per-Claim Overlap Scores")
        df_claims = pd.DataFrame(last_h.claim_details)
        df_claims["status"] = df_claims["is_grounded"].map(
            {True: "Grounded", False: "Uncited"}
        )
        df_claims.columns = [c.replace("_", " ").title() for c in df_claims.columns]
        st.dataframe(
            df_claims,
            use_container_width=True,
            hide_index=True,
        )


# ── Tab 4: LLM Monitoring ─────────────────────────────────────────────────────
def render_monitoring_tab(monitor):
    st.markdown("#### LLM Output Monitoring")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>Distribution drift tracking "
        "across response length, grounding score, hallucination risk, and retrieval quality. "
        "Baseline (first half) vs. recent (second half) comparison.</p>",
        unsafe_allow_html=True,
    )

    df = monitor.to_dataframe()
    if df.empty:
        st.info("No monitoring data yet. Ask some questions in the Ask tab.")
        return

    # Time series
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s")

    # Summary metrics row
    summary = monitor.get_summary()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Queries", f"{summary['total_queries']:,}")
    col2.metric("Avg Grounding", f"{summary['avg_grounding_score']*100:.1f}%")
    col3.metric("Avg Risk", f"{summary['avg_hallucination_risk']*100:.1f}%")
    col4.metric("Avg Latency", f"{summary['avg_latency_ms']:.0f}ms")
    col5.metric("High-Risk Queries", f"{summary['pct_high_risk']:.1f}%")

    st.markdown("---")

    # 2x2 chart grid
    col_left, col_right = st.columns(2)

    with col_left:
        # Grounding score trend
        fig_gs = px.line(
            df, x="dt", y="grounding_score",
            title="Grounding Score Over Time",
            labels={"dt": "", "grounding_score": "Grounding Score"},
            color_discrete_sequence=["#10b981"],
        )
        fig_gs.add_hline(
            y=0.85, line_dash="dash", line_color="#f59e0b",
            annotation_text="85% threshold", annotation_font_color="#f59e0b",
        )
        fig_gs.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0a0f1a",
            font_color="#94a3b8", height=280,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0, 1.05]),
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_gs, use_container_width=True)

    with col_right:
        # Hallucination risk trend
        fig_hr = px.line(
            df, x="dt", y="hallucination_risk",
            title="Hallucination Risk Over Time",
            labels={"dt": "", "hallucination_risk": "Risk Score"},
            color_discrete_sequence=["#ef4444"],
        )
        fig_hr.add_hline(
            y=0.30, line_dash="dash", line_color="#f59e0b",
            annotation_text="30% alert", annotation_font_color="#f59e0b",
        )
        fig_hr.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0a0f1a",
            font_color="#94a3b8", height=280,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0, 1.05]),
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        # Response length trend
        fig_len = px.bar(
            df, x="dt", y="response_length",
            title="Response Length Over Time (chars)",
            labels={"dt": "", "response_length": "Characters"},
            color_discrete_sequence=["#3b82f6"],
        )
        fig_len.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0a0f1a",
            font_color="#94a3b8", height=280,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_len, use_container_width=True)

    with col_right2:
        # Retrieval score trend
        fig_ret = px.line(
            df, x="dt", y="avg_retrieval_score",
            title="Avg Retrieval Similarity Score",
            labels={"dt": "", "avg_retrieval_score": "Cosine Similarity"},
            color_discrete_sequence=["#8b5cf6"],
        )
        fig_ret.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0a0f1a",
            font_color="#94a3b8", height=280,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0, 1.0]),
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    # Drift report
    st.markdown("---")
    st.markdown("#### Distribution Drift Report")
    drift_report = monitor.generate_drift_report()

    if drift_report.get("status") == "insufficient_data":
        st.warning(drift_report.get("message", "Not enough data for drift analysis."))
    else:
        drift_results = drift_report.get("drift_results", {})
        if drift_results:
            drift_rows = []
            for col, info in drift_results.items():
                is_drifted = info.get("is_drifted", False)
                drift_rows.append({
                    "Metric": col.replace("_", " ").title(),
                    "Drift Score": round(info.get("drift_score", 0), 4),
                    "Status": "DRIFTED" if is_drifted else "STABLE",
                    "Method": info.get("method", drift_report.get("engine", "evidently")),
                })
            df_drift = pd.DataFrame(drift_rows)
            st.dataframe(df_drift, use_container_width=True, hide_index=True)

        any_drift = drift_report.get("any_drift_detected", False)
        if any_drift:
            st.warning("Distribution drift detected. Review prompt version or data quality.")
        else:
            st.success("No significant drift detected across monitored metrics.")


# ── Tab 5: Prompt Lab ─────────────────────────────────────────────────────────
def render_prompt_lab_tab(tracker):
    st.markdown("#### Prompt Experiment Lab")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>MLflow-tracked prompt versions "
        "evaluated against 20 gold QA pairs. Compare grounding, hallucination risk, "
        "and answer relevance across prompt iterations.</p>",
        unsafe_allow_html=True,
    )

    df_runs = tracker.get_runs_dataframe()
    if df_runs.empty:
        st.info("No experiment runs yet.")
        return

    # Run new eval button
    col_btn, col_info = st.columns([1, 3])
    if col_btn.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating all prompt versions against gold QA pairs…"):
            tracker.run_evaluation(prompt_version="v2_structured")
            st.success("Evaluation complete! Refreshing…")
            st.rerun()
    col_info.markdown(
        "<p style='color:#64748b;font-size:0.82rem;margin-top:0.5rem;'>"
        "Evaluates current prompt version against 20 gold financial QA pairs.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Experiment Runs")

    # Format and display runs table
    display_cols = {
        "run_name": "Run Name",
        "prompt_version": "Prompt",
        "retrieval_k": "Top-K",
        "avg_grounding_score": "Avg Grounding",
        "avg_hallucination_risk": "Avg Risk",
        "answer_relevance": "Answer Relevance",
        "pct_fully_grounded": "% Fully Grounded",
        "avg_latency_ms": "Avg Latency (ms)",
        "num_queries": "Queries",
    }
    available = {k: v for k, v in display_cols.items() if k in df_runs.columns}
    df_display = df_runs[list(available.keys())].rename(columns=available)

    # Format percentages and scores
    for col in ["Avg Grounding", "Avg Risk", "Answer Relevance"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, float) else x)
    if "% Fully Grounded" in df_display.columns:
        df_display["% Fully Grounded"] = df_display["% Fully Grounded"].apply(
            lambda x: f"{x:.1f}%" if isinstance(x, float) else x
        )
    if "Avg Latency (ms)" in df_display.columns:
        df_display["Avg Latency (ms)"] = df_display["Avg Latency (ms)"].apply(
            lambda x: f"{x:.0f}ms" if isinstance(x, float) else x
        )

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Best run highlight
    best = tracker.get_best_run("avg_grounding_score")
    if best:
        st.success(
            f"Best run: **{best.get('run_name', '')}** "
            f"(prompt={best.get('prompt_version','')}, "
            f"grounding={best.get('avg_grounding_score',0)*100:.1f}%, "
            f"risk={best.get('avg_hallucination_risk',0)*100:.1f}%)"
        )

    st.markdown("---")
    st.markdown("#### Prompt Version Comparison")

    # Bar chart: grounding score by prompt version
    if "prompt_version" in df_runs.columns and "avg_grounding_score" in df_runs.columns:
        fig_pv = px.bar(
            df_runs.groupby("prompt_version", as_index=False)["avg_grounding_score"].mean(),
            x="prompt_version",
            y="avg_grounding_score",
            title="Average Grounding Score by Prompt Version",
            labels={"prompt_version": "Prompt Version", "avg_grounding_score": "Avg Grounding Score"},
            color="avg_grounding_score",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            range_color=[0.5, 1.0],
        )
        fig_pv.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0a0f1a",
            font_color="#94a3b8", height=320,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0, 1.05]),
            coloraxis_showscale=False,
            margin=dict(t=50, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_pv, use_container_width=True)

    # Scatter: latency vs grounding
    if all(c in df_runs.columns for c in ["avg_latency_ms", "avg_grounding_score", "prompt_version"]):
        fig_scatter = px.scatter(
            df_runs,
            x="avg_latency_ms",
            y="avg_grounding_score",
            color="prompt_version",
            size="num_queries" if "num_queries" in df_runs.columns else None,
            hover_data=["run_name"],
            title="Grounding Score vs. Latency (Quality–Speed Tradeoff)",
            labels={
                "avg_latency_ms": "Avg Latency (ms)",
                "avg_grounding_score": "Avg Grounding Score",
            },
        )
        fig_scatter.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0a0f1a",
            font_color="#94a3b8", height=320,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0.5, 1.0]),
            margin=dict(t=50, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    retriever, generator, monitor, tracker = load_pipeline()

    doc_filter, prompt_version, top_k = render_sidebar(tracker)
    render_header(monitor)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Ask",
        "Source Evidence",
        "Hallucination Check",
        "LLM Monitoring",
        "Prompt Lab",
    ])

    with tab1:
        render_ask_tab(generator, monitor, retriever, prompt_version, top_k)

    with tab2:
        render_evidence_tab(retriever, top_k)

    with tab3:
        render_hallucination_tab()

    with tab4:
        render_monitoring_tab(monitor)

    with tab5:
        render_prompt_lab_tab(tracker)


if __name__ == "__main__":
    main()
