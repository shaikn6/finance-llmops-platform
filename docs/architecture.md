# Architecture

## System Components

```
finance-llmops-platform/
├── pipeline/         # Core RAG pipeline
│   ├── ingestion     # Document loading → chunking → FAISS embedding
│   ├── retriever     # Semantic search over FAISS
│   ├── generator     # LLM answer generation + citation tracking
│   ├── hallucination # Token-overlap grounding check
│   └── monitor       # Evidently AI drift monitoring
├── experiments/      # MLflow prompt experiment tracking
├── dashboard/        # Streamlit 5-tab UI
├── data/             # Sample docs, FAISS index, monitoring logs
└── tests/            # pytest suite
```

## Data Flow

```
User Question
     │
     ▼
FinancialRetriever
  - Embed with all-MiniLM-L6-v2
  - FAISS top-k search
  - Return chunks + metadata
     │
     ▼
FinancialAnswerGenerator
  - Build prompt (versioned template)
  - Call OpenAI / Mock response
  - Track citations
     │
     ▼
HallucinationDetector
  - Extract factual claims
  - Token overlap vs source chunks
  - Grounding score 0-1
     │
     ▼
LLMMonitor (Evidently AI)
  - Log interaction
  - Track drift over time
     │
     ▼
Streamlit Dashboard
  - Display answer + sources
  - Show grounding gauge
  - Render drift charts
```

## Embedding Model

- Model: `all-MiniLM-L6-v2` (sentence-transformers)
- Dimension: 384
- Normalization: L2 (cosine similarity via inner product)
- Index type: FAISS `IndexFlatIP`

## No External Dependencies Required

Set `MOCK_MODE=true` to run the full pipeline without:
- OpenAI API key
- MLflow server (falls back to local SQLite)
- Evidently cloud (uses local PSI computation)
