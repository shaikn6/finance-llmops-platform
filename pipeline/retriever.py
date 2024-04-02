"""
Semantic retriever over the FAISS index.

Embeds a query using sentence-transformers and returns the top-k most
similar document chunks with source metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Any

import numpy as np


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with its similarity score."""
    chunk_id: str
    doc_name: str
    doc_type: str
    text: str
    score: float            # cosine similarity (0–1)
    rank: int


class FinancialRetriever:
    """
    Semantic search over a FAISS index of financial documents.

    Usage:
        retriever = FinancialRetriever()
        results = retriever.retrieve("What is the LCR ratio?", top_k=3)
    """

    def __init__(
        self,
        index: Optional[Any] = None,
        chunks: Optional[list] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._index = index
        self._chunks = chunks
        self._model = None
        self._model_name = model_name
        self._initialized = False

    def _lazy_init(self) -> None:
        """Load index and model on first call (avoids import-time overhead)."""
        if self._initialized:
            return

        from sentence_transformers import SentenceTransformer
        from pipeline.ingestion import build_faiss_index

        if self._index is None or self._chunks is None:
            print("[retriever] Initializing FAISS index...")
            self._index, self._chunks = build_faiss_index()

        print(f"[retriever] Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)
        self._initialized = True

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language question about the financial documents.
            top_k: Number of chunks to return.

        Returns:
            List of RetrievedChunk objects ordered by descending similarity.
        """
        self._lazy_init()

        # Embed query
        query_embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Search FAISS
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        results: List[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_name=chunk.doc_name,
                    doc_type=chunk.doc_type,
                    text=chunk.text,
                    score=float(score),
                    rank=rank + 1,
                )
            )

        return results

    def retrieve_with_metadata(
        self, query: str, top_k: int = 3
    ) -> dict:
        """
        Retrieve chunks and return rich metadata dict for downstream use.
        """
        t0 = time.time()
        chunks = self.retrieve(query, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000

        return {
            "query": query,
            "top_k": top_k,
            "latency_ms": round(latency_ms, 1),
            "chunks": [
                {
                    "rank": c.rank,
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "doc_type": c.doc_type,
                    "score": round(c.score, 4),
                    "text": c.text,
                    "text_snippet": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                }
                for c in chunks
            ],
            "avg_score": round(
                sum(c.score for c in chunks) / len(chunks), 4
            ) if chunks else 0.0,
        }


# Module-level singleton for reuse
_retriever: Optional[FinancialRetriever] = None


def get_retriever() -> FinancialRetriever:
    """Return the shared retriever instance, initializing on first call."""
    global _retriever
    if _retriever is None:
        _retriever = FinancialRetriever()
    return _retriever


if __name__ == "__main__":
    r = get_retriever()
    result = r.retrieve_with_metadata(
        "What is Meridian's liquidity coverage ratio?", top_k=3
    )
    print(f"Query: {result['query']}")
    print(f"Latency: {result['latency_ms']}ms | Avg score: {result['avg_score']}")
    for chunk in result["chunks"]:
        print(f"\n  [{chunk['rank']}] {chunk['doc_name']} (score={chunk['score']})")
        print(f"      {chunk['text_snippet']}")
