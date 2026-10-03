"""Embedding backends.

local  : sentence-transformers on your own machine (free, offline, heavy install)
gemini : Gemini embedding API (light install; uses the visitor's/your API key)

Default: 'gemini' on a shared deployment (STUDYRAG_MULTIUSER=1), otherwise 'local'.
Override with STUDYRAG_EMBEDDINGS=local|gemini.
"""
import importlib.util
import math
import os
import time
from functools import lru_cache

LOCAL_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("STUDYRAG_EMBED_MODEL", "gemini-embedding-001")
GEMINI_DIM = 768  # smaller vectors = less memory; plenty for study documents
BATCH = 50


def backend() -> str:
    choice = os.getenv("STUDYRAG_EMBEDDINGS", "").lower()
    if choice in ("local", "gemini"):
        return choice
    if os.getenv("STUDYRAG_MULTIUSER") == "1":
        return "gemini"
    # single-user mode: use local if installed, otherwise fall back to Gemini
    return "local" if importlib.util.find_spec("sentence_transformers") else "gemini"


@lru_cache(maxsize=1)
def _local_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "Local embeddings need sentence-transformers: pip install -r requirements-local.txt"
        ) from e
    return SentenceTransformer(LOCAL_MODEL)


def _normalize(v) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _embed_gemini(texts: list[str], query: bool) -> list[list[float]]:
    from google.genai import types

    from core import llm

    task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        for attempt in range(5):
            try:
                resp = llm._client().models.embed_content(
                    model=GEMINI_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task, output_dimensionality=GEMINI_DIM
                    ),
                )
                break
            except Exception as e:
                transient = any(k in str(e) for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
                if transient and attempt < 4:
                    time.sleep(5 * 2**attempt)  # wait and retry on free-tier rate limits
                    continue
                raise
        out += [_normalize(e.values) for e in resp.embeddings]
    if len(out) != len(texts):
        raise RuntimeError("Embedding service returned an unexpected number of vectors.")
    return out


def embed(texts: list[str], query: bool = False) -> list[list[float]]:
    """Embed texts. query=True for search questions, False for stored passages."""
    if backend() == "gemini":
        return _embed_gemini(texts, query)
    return _local_model().encode(texts, normalize_embeddings=True).tolist()
