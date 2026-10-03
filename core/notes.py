"""Student notes: stored as JSON (so they can be edited) and indexed for search."""
import json
from pathlib import Path

from core.folders import DEFAULT_FOLDER
from core.index import add_chunks, delete_source
from core.parsers import chunk_text
from core.workspace import data_dir

NOTES_COLLECTION = "notes"


def _path() -> Path:
    return data_dir() / "notes.json"


def load_notes() -> dict[str, dict]:
    """Map of note title -> {'text': str, 'folder': str}. Reads older formats too."""
    try:
        raw = json.loads(_path().read_text(encoding="utf-8"))
    except Exception:
        return {}
    notes = {}
    for title, v in raw.items():
        if isinstance(v, str):  # older format: plain text only
            notes[title] = {"text": v, "folder": DEFAULT_FOLDER}
        else:
            notes[title] = {"text": v.get("text", ""), "folder": v.get("folder", DEFAULT_FOLDER)}
    return notes


def _write(notes: dict[str, dict]) -> None:
    _path().write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


def save_note(title: str, text: str, folder: str = DEFAULT_FOLDER) -> int:
    """Create or overwrite a note and (re)index it. Returns number of chunks."""
    notes = load_notes()
    notes[title] = {"text": text, "folder": folder}
    _write(notes)
    delete_source(NOTES_COLLECTION, title)  # remove old version, whatever folder it was in
    chunks = [
        {"text": c, "source": title, "location": "Note", "folder": folder}
        for c in chunk_text(text)
    ]
    return add_chunks(NOTES_COLLECTION, chunks)


def delete_note(title: str) -> None:
    notes = load_notes()
    notes.pop(title, None)
    _write(notes)
    delete_source(NOTES_COLLECTION, title)
