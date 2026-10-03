"""Folders (tracks) such as 'Data Science'. Documents and notes each belong to one."""
import json
from pathlib import Path

from core.workspace import data_dir

DEFAULT_FOLDER = "General"


def _path() -> Path:
    return data_dir() / "folders.json"


def _load() -> list[str]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return [str(x) for x in data]
    except Exception:
        return []


def _save(names: list[str]) -> None:
    _path().write_text(json.dumps(sorted(set(names)), ensure_ascii=False, indent=2), encoding="utf-8")


def list_folders(extra=()) -> list[str]:
    """All folder names, General first. 'extra' adds folders found in the data itself."""
    names = set(_load()) | {e for e in extra if e}
    names.discard(DEFAULT_FOLDER)
    return [DEFAULT_FOLDER] + sorted(names, key=str.lower)


def add_folder(name: str) -> str | None:
    """Create a folder. Returns the cleaned name (None if the name is empty)."""
    name = " ".join(name.split())[:40]
    if not name:
        return None
    existing = {n.lower(): n for n in list_folders()}
    if name.lower() in existing:
        return existing[name.lower()]
    _save(_load() + [name])
    return name


def delete_folder(name: str) -> None:
    """Remove a folder from the registry (caller must make sure it is empty)."""
    _save([n for n in _load() if n != name])
