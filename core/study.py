"""Quiz and flashcard generation from your own material."""
import json
import math
import re

from core import llm
from core.index import sample_chunks, search

QUIZ_SYSTEM = """You write study quizzes using ONLY the numbered sources provided.
Return ONLY valid JSON in exactly this shape, with no extra text:
{"questions": [{"question": "...", "options": ["A", "B", "C", "D"], "answer_index": 0, "explanation": "...", "source": 1}]}
Rules:
- Every question must be answerable from the sources. Do not use outside knowledge.
- Exactly 4 options, exactly one correct. Distractors must be plausible but clearly wrong according to the sources.
- Vary which position holds the correct answer. answer_index is 0-3.
- explanation: one or two sentences saying why the answer is correct.
- source: the number of the source the question is based on.
- Cover different parts of the sources; do not repeat questions."""

CARD_SYSTEM = """You write study flashcards using ONLY the numbered sources provided.
Return ONLY valid JSON in exactly this shape, with no extra text:
{"cards": [{"front": "...", "back": "...", "source": 1}]}
Rules:
- front: a short term, question or prompt. back: a clear, concise answer (1-3 sentences).
- Everything must come from the sources. Do not use outside knowledge.
- source: the number of the source the card is based on.
- Cover different parts of the sources; do not repeat cards."""


def gather(kinds, folders, topic: str, limit: int = 12) -> list[dict]:
    """Pick source chunks: relevant to a topic if given, otherwise a varied sample."""
    pairs = [(k, c) for k, c in (("document", "documents"), ("note", "notes")) if k in kinds]
    if not pairs:
        return []
    per = math.ceil(limit / len(pairs))
    chunks: list[dict] = []
    for kind, coll in pairs:
        if topic.strip():
            hits = search(coll, topic, k=per, min_score=0.15, rel_cutoff=0.5, folders=folders)
        else:
            hits = sample_chunks(coll, per, folders)
        chunks += [dict(h, kind=kind) for h in hits]
    return chunks


def _context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({llm._label(c)})\n{c['text'][:900]}" for i, c in enumerate(chunks, start=1)
    )


def _parse_json(text: str) -> dict | None:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _clean_quiz(raw: list, n_sources: int) -> list[dict]:
    items = []
    for q in raw or []:
        try:
            opts = [str(o).strip() for o in q["options"]]
            idx = int(q["answer_index"])
            if len(opts) != 4 or not 0 <= idx <= 3 or not str(q["question"]).strip():
                continue
            src = int(q.get("source", 0))
            items.append(
                {
                    "question": str(q["question"]).strip(),
                    "options": opts,
                    "answer_index": idx,
                    "explanation": str(q.get("explanation", "")).strip(),
                    "source": src if 1 <= src <= n_sources else None,
                }
            )
        except Exception:
            continue
    return items


def _clean_cards(raw: list, n_sources: int) -> list[dict]:
    items = []
    for c in raw or []:
        try:
            front, back = str(c["front"]).strip(), str(c["back"]).strip()
            if not front or not back:
                continue
            src = int(c.get("source", 0))
            items.append({"front": front, "back": back, "source": src if 1 <= src <= n_sources else None})
        except Exception:
            continue
    return items


def generate_items(kind: str, chunks: list[dict], n: int, model: str | None = None) -> dict:
    """kind: 'quiz' or 'flashcards'. Returns {'items', 'model', 'error'}."""
    system = QUIZ_SYSTEM if kind == "quiz" else CARD_SYSTEM
    key = "questions" if kind == "quiz" else "cards"
    clean = _clean_quiz if kind == "quiz" else _clean_cards
    noun = "multiple-choice questions" if kind == "quiz" else "flashcards"
    prompt = f"SOURCES:\n{_context(chunks)}\n\nCreate {n} {noun}."

    used = model or llm._model()
    for attempt in range(2):
        try:
            text, used = llm.generate(prompt, system, model, json_mode=True, temperature=0.4)
        except Exception as e:
            return {"items": [], "model": used, "error": llm.friendly_error(e, model)}
        data = _parse_json(text)
        items = clean((data or {}).get(key), len(chunks)) if isinstance(data, dict) else []
        if items:
            return {"items": items[:n], "model": used, "error": None}
    return {
        "items": [],
        "model": used,
        "error": "The model did not return usable items. Try again, or pick another model.",
    }
