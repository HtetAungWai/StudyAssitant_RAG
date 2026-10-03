"""Per-visitor workspaces.

On a shared deployment every visitor gets a private workspace id, so people never
see each other's files or notes. Running locally (no id) uses the plain data/ folder.
"""
import re
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_VALID = re.compile(r"[0-9a-f]{12}")


def valid_id(uid) -> bool:
    """Workspace ids are 12 hex characters (also keeps paths and names safe)."""
    return isinstance(uid, str) and bool(_VALID.fullmatch(uid))


def current_id() -> str:
    """'local' in single-user mode, otherwise this visitor's workspace id."""
    try:
        import streamlit as st

        uid = st.session_state.get("uid")
    except Exception:
        uid = None
    return uid if valid_id(uid) else "local"


def data_dir() -> Path:
    """Folder for this visitor's notes.json / folders.json."""
    uid = current_id()
    d = DATA_ROOT if uid == "local" else DATA_ROOT / "users" / uid
    d.mkdir(parents=True, exist_ok=True)
    return d
