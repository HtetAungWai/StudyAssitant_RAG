"""Small evaluation helpers: retrieval quality and answer correctness (keyword-based)."""


def parse_keywords(s) -> list[str]:
    """'240, hours' -> ['240', 'hours'] (lowercased). Empty/NaN -> []."""
    if not isinstance(s, str):
        return []
    return [k.strip().lower() for k in s.split(",") if k.strip()]


def contains_all(text: str, keywords: list[str]) -> bool:
    t = (text or "").lower()
    return bool(keywords) and all(k in t for k in keywords)


def evaluate_question(question, keywords, hits, rag_answer, plain_answer) -> dict:
    """Score one question.

    retrieval hit: some retrieved chunk contains all expected keywords (rank = its position)
    rag/plain correct: the answer text contains all expected keywords
    """
    rank = next((i for i, h in enumerate(hits, start=1) if contains_all(h["text"], keywords)), None)
    return {
        "question": question,
        "retrieval_hit": rank is not None,
        "rank": rank,
        "rag_correct": contains_all(rag_answer, keywords),
        "plain_correct": contains_all(plain_answer, keywords),
        "rag_answer": rag_answer,
        "plain_answer": plain_answer,
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0, "hit_at_k": 0.0, "mrr": 0.0, "rag_acc": 0.0, "plain_acc": 0.0}
    return {
        "n": n,
        "hit_at_k": sum(r["retrieval_hit"] for r in results) / n,
        "mrr": sum(1 / r["rank"] for r in results if r["rank"]) / n,
        "rag_acc": sum(r["rag_correct"] for r in results) / n,
        "plain_acc": sum(r["plain_correct"] for r in results) / n,
    }
