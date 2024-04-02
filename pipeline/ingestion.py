"""
Document ingestion pipeline.

Loads financial documents, chunks them, embeds with sentence-transformers,
and stores in a FAISS index. No external API required.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = DATA_DIR / "faiss_index"

CHUNK_SIZE = 500        # tokens (approximate — 1 token ≈ 4 chars)
CHUNK_OVERLAP = 50      # tokens
CHARS_PER_TOKEN = 4


@dataclass
class DocumentChunk:
    """A single chunk of a financial document."""
    chunk_id: str
    doc_name: str
    doc_type: str           # "10k" | "earnings_call"
    text: str
    char_start: int
    char_end: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def _approximate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_into_chunks(
    text: str,
    doc_name: str,
    doc_type: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Split text into overlapping chunks by approximate token count.
    Splits on sentence boundaries where possible.
    """
    # Split into sentences (crude but effective for financial text)
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_pattern.split(text)

    chunks: List[DocumentChunk] = []
    current_sentences: List[str] = []
    current_chars = 0
    char_cursor = 0
    chunk_start = 0
    chunk_index = 0

    for sent in sentences:
        sent_chars = len(sent)
        sent_tokens = _approximate_tokens(sent)

        # If adding this sentence would exceed chunk size, finalize current chunk
        if current_chars + sent_chars > chunk_size * CHARS_PER_TOKEN and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = DocumentChunk(
                chunk_id=f"{doc_name}_{chunk_index:04d}",
                doc_name=doc_name,
                doc_type=doc_type,
                text=chunk_text,
                char_start=chunk_start,
                char_end=char_cursor,
                token_count=_approximate_tokens(chunk_text),
            )
            chunks.append(chunk)
            chunk_index += 1

            # Keep overlap: retain last `overlap` tokens worth of sentences
            overlap_chars = overlap * CHARS_PER_TOKEN
            kept: List[str] = []
            kept_chars = 0
            for s in reversed(current_sentences):
                if kept_chars + len(s) <= overlap_chars:
                    kept.insert(0, s)
                    kept_chars += len(s)
                else:
                    break
            current_sentences = kept
            current_chars = kept_chars
            chunk_start = char_cursor - kept_chars

        current_sentences.append(sent)
        current_chars += sent_chars
        char_cursor += sent_chars + 1  # +1 for space

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        if chunk_text.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_name}_{chunk_index:04d}",
                    doc_name=doc_name,
                    doc_type=doc_type,
                    text=chunk_text,
                    char_start=chunk_start,
                    char_end=char_cursor,
                    token_count=_approximate_tokens(chunk_text),
                )
            )

    return chunks


def load_documents(data_dir: Path = DATA_DIR) -> List[DocumentChunk]:
    """Load all financial documents and return chunked corpus."""
    all_chunks: List[DocumentChunk] = []

    # Load 10-K excerpts
    tenk_path = data_dir / "sample_10k_excerpts.txt"
    if tenk_path.exists():
        text = tenk_path.read_text(encoding="utf-8")
        chunks = _split_into_chunks(
            text, doc_name="10k_excerpts", doc_type="10k"
        )
        all_chunks.extend(chunks)
        print(f"[ingestion] Loaded 10-K: {len(chunks)} chunks")

    # Load earnings call transcripts
    ec_path = data_dir / "sample_earnings_calls.txt"
    if ec_path.exists():
        text = ec_path.read_text(encoding="utf-8")
        chunks = _split_into_chunks(
            text, doc_name="earnings_calls", doc_type="earnings_call"
        )
        all_chunks.extend(chunks)
        print(f"[ingestion] Loaded earnings calls: {len(chunks)} chunks")

    print(f"[ingestion] Total chunks: {len(all_chunks)}")
    return all_chunks


def build_faiss_index(
    chunks: Optional[List[DocumentChunk]] = None,
    index_dir: Path = INDEX_DIR,
    force_rebuild: bool = False,
) -> tuple[Any, List[DocumentChunk]]:
    """
    Build or load a FAISS index from document chunks.

    Returns (faiss_index, chunks).
    Caches to disk so rebuilds are fast.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.pkl"
    meta_path = index_dir / "meta.json"

    if not force_rebuild and index_path.exists() and chunks_path.exists():
        print("[ingestion] Loading cached FAISS index...")
        index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            cached_chunks = pickle.load(f)
        print(f"[ingestion] Loaded {index.ntotal} vectors from cache")
        return index, cached_chunks

    if chunks is None:
        chunks = load_documents()

    if not chunks:
        raise ValueError("No document chunks to index.")

    print("[ingestion] Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c.text for c in chunks]
    print(f"[ingestion] Embedding {len(texts)} chunks...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    elapsed = time.time() - t0
    print(f"[ingestion] Embedding complete in {elapsed:.1f}s")

    dim = embeddings.shape[1]
    # Inner product index (cosine similarity since embeddings are normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Persist
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    meta = {
        "num_chunks": len(chunks),
        "embedding_dim": dim,
        "model": "all-MiniLM-L6-v2",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[ingestion] FAISS index built: {index.ntotal} vectors (dim={dim})")
    return index, chunks


if __name__ == "__main__":
    chunks = load_documents()
    index, chunks = build_faiss_index(chunks, force_rebuild=True)
    print(f"Index ready: {index.ntotal} vectors")
