"""Gemini helpers: grounded answers with citations, plain answers, JSON generation."""
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Default model. Override in the app sidebar, or set GEMINI_MODEL in .env.
DEFAULT_MODEL = "gemini-2.5-flash"

KIND_NAMES = {
    "document": "uploaded documents",
    "note": "personal notes",
    "web": "web search results",
}

SYSTEM_TEMPLATE = """You are StudyRAG, a study assistant for university students.
Answer the question using ONLY the numbered sources provided. Each source is labeled with its type ({names}).
Rules:
- If the sources do not contain the answer, say clearly that you could not find it in the provided sources. Do not guess or use outside knowledge.
- If answering needs information from more than one source (for example a requirement in a document and the student's progress in a note), combine them and show the simple calculation or reasoning briefly.
- Cite every claim with the source number in square brackets, like [1] or [2][3]. Use a separate bracket per source.
- Only cite sources you actually used.
{extra}- Be concise and clear. Use short bullet points when listing items."""

PLAIN_SYSTEM = "You are a helpful study assistant. Answer the question concisely and accurately."


def _names(kinds) -> str:
    names = [KIND_NAMES[k] for k in kinds]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _system(kinds) -> str:
    extra = ""
    if "web" in kinds:
        extra = "- If web sources disagree or look unreliable, say so briefly.\n"
    return SYSTEM_TEMPLATE.format(names=_names(kinds), extra=extra)


def _label(c: dict) -> str:
    kind = c.get("kind", "document")
    if kind == "note":
        return f"Note: {c['source']}"
    if kind == "web":
        return f"Web: {c['source']}, {c['location']}"
    return f"Document: {c['source']}, {c['location']}"


_resolved: dict[str, str] = {}  # remembers an auto-picked model for this session


def _model() -> str:
    return _resolved.get("model") or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def _api_key() -> str | None:
    """The visitor's own key (pasted in the sidebar) wins over the server's key."""
    try:
        import streamlit as st

        k = st.session_state.get("user_api_key")
        if isinstance(k, str) and k.strip():
            return k.strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


@lru_cache(maxsize=32)
def _client_for(key: str) -> genai.Client:
    # Cached on purpose: if a Client object is garbage-collected mid-request,
    # google-genai closes its HTTP connection ("client has been closed").
    return genai.Client(api_key=key)


def _client() -> genai.Client:
    key = _api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found")
    return _client_for(key)


def list_models() -> list[str]:
    """Names of models your key can use for text generation."""
    names = []
    for m in _client().models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            names.append(m.name.replace("models/", ""))
    return sorted(names)


def _is_not_found(e: Exception) -> bool:
    msg = str(e)
    return "404" in msg or "NOT_FOUND" in msg


def _switch_model() -> bool:
    """Auto mode only: if the default model is missing, pick a current Flash model."""
    try:
        skip = ("image", "tts", "live", "audio", "embedding", "robotics", "computer")
        names = [
            n for n in list_models()
            if n.startswith("gemini") and "flash" in n and not any(x in n for x in skip)
        ]
        names = [n for n in names if "lite" not in n] or names
        if not names:
            return False
        choice = sorted(names, reverse=True)[0]
        if choice == _model():
            return False
        _resolved["model"] = choice
        return True
    except Exception:
        return False


def _call(prompt: str, model: str, system: str, json_mode: bool = False, temperature: float = 0.2):
    kwargs = {"response_mime_type": "application/json"} if json_mode else {}
    return _client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, **kwargs
        ),
    )


def generate(
    prompt: str,
    system: str,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
) -> tuple[str, str]:
    """Call Gemini. Returns (text, model_used). Raises on failure.

    model: manually chosen name. None = auto (GEMINI_MODEL, else default,
    with automatic fallback to an available Flash model).
    """
    manual = bool(model)
    used = model or _model()
    try:
        resp = _call(prompt, used, system, json_mode, temperature)
    except Exception as e:
        if _is_not_found(e) and not manual and not os.getenv("GEMINI_MODEL") and _switch_model():
            used = _model()
            resp = _call(prompt, used, system, json_mode, temperature)
        else:
            raise
    return (resp.text or "").strip(), used


def friendly_error(e: Exception, model: str | None = None) -> str:
    msg = str(e)
    used = model or _model()
    if "GEMINI_API_KEY not found" in msg:
        return (
            "No Gemini API key yet. Paste your free key in the sidebar "
            "(get one at aistudio.google.com/apikey), or set GEMINI_API_KEY in .env."
        )
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return (
            f"Rate limit or quota reached for '{used}'. "
            "Wait a minute, or pick a different model in the sidebar."
        )
    if _is_not_found(e):
        return (
            f"Model '{used}' is not available for your key. "
            "Pick another one under 'Gemini model' in the sidebar."
        )
    if "API key" in msg or "API_KEY" in msg or "403" in msg or "401" in msg:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
    return f"Gemini error with '{used}': {msg}"


def answer_plain(question: str, model: str | None = None) -> dict:
    """Plain Gemini answer with NO retrieval (baseline for evaluation)."""
    try:
        text, used = generate(question, PLAIN_SYSTEM, model)
        return {"answer": text, "error": False, "model": used}
    except Exception as e:
        return {"answer": friendly_error(e, model), "error": True, "model": model or _model()}


def answer_from_chunks(
    question: str,
    chunks: list[dict],
    model: str | None = None,
    kinds: tuple = ("document",),
) -> dict:
    """Return {'answer', 'sources', 'error', 'model'}.

    kinds: which source types were searched: "document", "note", "web".
    Each chunk may carry kind= one of those (used to label it for the model).
    'sources' only contains chunks the model actually cited.
    """
    if not chunks:
        return {
            "answer": (
                "I couldn't get web results for that right now. Try rephrasing, or check your connection."
                if tuple(kinds) == ("web",)
                else f"I couldn't find anything relevant to that in your {_names(kinds)}."
            ),
            "sources": [],
            "error": False,
            "model": model or _model(),
        }

    context = "\n\n".join(
        f"[{i}] ({_label(c)})\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )
    prompt = f"SOURCES:\n{context}\n\nQUESTION: {question}"

    try:
        text, used_model = generate(prompt, _system(kinds), model)
    except Exception as e:
        return {"answer": friendly_error(e, model), "sources": [], "error": True,
                "model": model or _model()}

    if not text:
        return {
            "answer": "The model returned an empty answer. Try rephrasing or pick another model.",
            "sources": [],
            "error": True,
            "model": used_model,
        }

    cited = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        cited.update(int(n) for n in re.split(r"\s*,\s*", group))
    used = [dict(chunks[n - 1], n=n) for n in sorted(cited) if 1 <= n <= len(chunks)]
    return {"answer": text, "sources": used, "error": False, "model": used_model}
