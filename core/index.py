"""Embeddings + vector store (local, free).

Two collections keep sources separate:
  - "documents": uploaded course files
  - "notes":     notes the student writes
Every chunk carries a 'folder' (track) so searches can be limited to one track.
"""
import hashlib
import random
from functools import lru_cache
from pathlib import Path

import chromadb

from core.embeddings import backend, embed
from core.folders import DEFAULT_FOLDER
from core.workspace import current_id

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma")


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=DB_PATH)


def _collection(name: str):
    uid = current_id()  # 'local' = single user; otherwise each visitor gets own collections
    full = name if uid == "local" else f"u{uid}_{name}"
    if backend() == "gemini":
        full += "_g"  # different embedding size, so keep it apart from local-model collections
    return _client().get_or_create_collection(name=full, metadata={"hnsw:space": "cosine"})


def _embed(texts: list[str], query: bool = False) -> list[list[float]]:
    return embed(texts, query=query)


def ensure_folder_metadata(collection: str) -> None:
    """One-time upgrade: chunks indexed before folders existed go into 'General'."""
    col = _collection(collection)
    data = col.get(include=["metadatas"])
    ids, metas = [], []
    for i, m in zip(data["ids"], data["metadatas"]):
        if "folder" not in m:
            ids.append(i)
            metas.append({**m, "folder": DEFAULT_FOLDER})
    if ids:
        col.update(ids=ids, metadatas=metas)


def add_chunks(collection: str, chunks: list[dict]) -> int:
    """Embed and store chunks [{'text','source','location','folder'}].

    Re-adding a file to the same folder replaces its old chunks (no duplicates).
    """
    if not chunks:
        return 0
    for src, fld in {(c["source"], c.get("folder", DEFAULT_FOLDER)) for c in chunks}:
        delete_source(collection, src, fld)
    ids = [
        hashlib.md5(
            f"{c.get('folder', DEFAULT_FOLDER)}|{c['source']}|{c['location']}|{i}|{c['text'][:50]}".encode()
        ).hexdigest()
        for i, c in enumerate(chunks)
    ]
    texts = [c["text"] for c in chunks]
    _collection(collection).upsert(
        ids=ids,
        documents=texts,
        embeddings=_embed(texts),
        metadatas=[
            {
                "source": c["source"],
                "location": c["location"],
                "folder": c.get("folder", DEFAULT_FOLDER),
            }
            for c in chunks
        ],
    )
    return len(chunks)


def search(
    collection: str,
    query: str,
    k: int = 5,
    min_score: float = 0.30,
    rel_cutoff: float = 0.75,
    folders: list[str] | None = None,
) -> list[dict]:
    """Return up to k relevant chunks: [{'text','source','location','folder','score'}].

    folders: only search inside these folders (None = all folders).
    Weak matches are dropped:
      - min_score:  absolute floor on cosine similarity
      - rel_cutoff: keep only chunks scoring >= rel_cutoff * best score
    """
    col = _collection(collection)
    if col.count() == 0:
        return []
    where = {"folder": {"$in": list(folders)}} if folders else None
    n = min(max(k * 3, 10), col.count())
    res = col.query(query_embeddings=_embed([query], query=True), n_results=n, where=where)
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(
            {
                "text": text,
                "source": meta["source"],
                "location": meta["location"],
                "folder": meta.get("folder", DEFAULT_FOLDER),
                "score": round(1 - dist, 3),  # cosine similarity
            }
        )
    if not hits:
        return []
    best = hits[0]["score"]
    hits = [h for h in hits if h["score"] >= min_score and h["score"] >= best * rel_cutoff]
    return hits[:k]


def list_sources(collection: str) -> list[dict]:
    """[{'source','folder','chunks'}] sorted by folder then name."""
    data = _collection(collection).get(include=["metadatas"])
    counts: dict[tuple[str, str], int] = {}
    for meta in data["metadatas"]:
        key = (meta.get("folder", DEFAULT_FOLDER), meta["source"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"folder": f, "source": s, "chunks": n}
        for (f, s), n in sorted(counts.items(), key=lambda kv: (kv[0][0].lower(), kv[0][1].lower()))
    ]


def delete_source(collection: str, source: str, folder: str | None = None) -> None:
    """Delete a source's chunks (in one folder, or everywhere if folder is None)."""
    if folder is None:
        where = {"source": source}
    else:
        where = {"$and": [{"source": source}, {"folder": folder}]}
    _collection(collection).delete(where=where)


def move_source(collection: str, source: str, old_folder: str, new_folder: str) -> None:
    """Move a source's chunks to another folder (no re-embedding needed)."""
    col = _collection(collection)
    data = col.get(
        where={"$and": [{"source": source}, {"folder": old_folder}]}, include=["metadatas"]
    )
    if data["ids"]:
        col.update(
            ids=data["ids"],
            metadatas=[{**m, "folder": new_folder} for m in data["metadatas"]],
        )


def sample_chunks(collection: str, n: int, folders: list[str] | None = None) -> list[dict]:
    """A varied sample of n chunks (spread across files) for quizzes/flashcards."""
    col = _collection(collection)
    where = {"folder": {"$in": list(folders)}} if folders else None
    data = col.get(where=where, include=["documents", "metadatas"])
    by_source: dict[str, list[dict]] = {}
    for text, meta in zip(data["documents"], data["metadatas"]):
        by_source.setdefault(meta["source"], []).append(
            {
                "text": text,
                "source": meta["source"],
                "location": meta["location"],
                "folder": meta.get("folder", DEFAULT_FOLDER),
            }
        )
    for chunks in by_source.values():
        random.shuffle(chunks)
    out: list[dict] = []
    while len(out) < n and any(by_source.values()):  # round-robin across files
        for src in list(by_source):
            if by_source[src] and len(out) < n:
                out.append(by_source[src].pop())
    return out
