# Changelog

## v2.0.0 — 2026-05-30

### What's New
- Multi-agent LangGraph pipeline: Researcher → Analyst → FactChecker → Synthesizer
- Streaming responses: token-by-token output via Server-Sent Events
- RAG evaluation harness: faithfulness + relevance scoring on 20 test pairs
- SEC EDGAR simulator: 10 companies with realistic 10-K excerpts

### Improvements
- Dashboard extended to 4 tabs with agent pipeline visualizer
- Hallucination detection now runs as dedicated agent step

### Under the Hood
- Added LangGraph multi-agent orchestration layer
- SSE streaming endpoint at /api/v2/stream
- +35 tests (total: 80+)

## v1.0.0 — 2026-05-30
- RAG over synthetic SEC 10-K + hallucination grounding + MLflow versioning
