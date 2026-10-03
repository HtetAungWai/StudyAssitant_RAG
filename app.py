import csv
import io
import os
import random
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from core import notes as notes_store
from core.evaluate import evaluate_question, parse_keywords, summarize
from core.folders import DEFAULT_FOLDER, add_folder, delete_folder, list_folders
from core.index import (
    add_chunks,
    delete_source,
    ensure_folder_metadata,
    list_sources,
    move_source,
    search,
)
from core.llm import answer_from_chunks, answer_plain, friendly_error, list_models
from core.parsers import SUPPORTED_TYPES, file_to_chunks
from core.study import gather, generate_items
from core.web import search_web
from core.workspace import valid_id

DOCS, NOTES = "documents", "notes"
PAGES = ["💬 Chat", "🧠 Study", "📁 Documents", "📝 Notes", "📊 Evaluation"]
SOURCE_OPTIONS = {"📄 Documents": "document", "📝 Notes": "note", "🌐 Web": "web"}

st.set_page_config(page_title="StudyRAG", page_icon="📚", layout="wide")
st.title("📚 StudyRAG")

# Streamlit Community Cloud runs apps from /mount/src/: treat that as a shared deployment
if Path(__file__).resolve().as_posix().startswith("/mount/src/"):
    os.environ.setdefault("STUDYRAG_MULTIUSER", "1")

# Shared deployment: give each visitor a private workspace (kept in the page URL ?u=...)
MULTI_USER = os.getenv("STUDYRAG_MULTIUSER") == "1"
if MULTI_USER and "uid" not in st.session_state:
    uid = st.query_params.get("u")
    if not valid_id(uid):
        uid = uuid.uuid4().hex[:12]
        st.query_params["u"] = uid
    st.session_state.uid = uid

# one-time upgrade of data indexed before folders existed
if "setup_done" not in st.session_state:
    ensure_folder_metadata(DOCS)
    ensure_folder_metadata(NOTES)
    st.session_state.setup_done = True
for key, default in (("chat_msgs", []), ("note_nonce", 0), ("study_id", 0), ("model_list", [])):
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Sidebar: navigation + model picker (all pages) ----------
with st.sidebar:
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    if MULTI_USER:
        st.text_input(
            "🔑 Your Gemini API key",
            type="password",
            key="user_api_key",
            help="Free at aistudio.google.com/apikey. Used only in your session, never stored.",
        )
        if not st.session_state.get("user_api_key") and not os.getenv("GEMINI_API_KEY"):
            st.warning("Paste a free Gemini API key to ask questions.")
        st.caption(
            "🔒 This is your private workspace. Bookmark this page's URL to come back to it. "
            "Files may be cleared when the server restarts. Text excerpts from your files are "
            "sent to Google's Gemini API to write answers, so check its terms before uploading "
            "anything sensitive."
        )
    with st.expander("⚙️ Gemini model"):
        if st.button("Load / refresh model list"):
            try:
                st.session_state.model_list = list_models()
                if not st.session_state.model_list:
                    st.warning("No models returned for this key.")
            except Exception as e:
                st.error(str(e))
        picked_model = st.selectbox("Choose model", ["Auto"] + st.session_state.model_list)
        custom = st.text_input("Or type a model name", placeholder="e.g. gemini-3-flash")
        # Priority: typed name > dropdown choice > Auto
        chosen_model = custom.strip() or (None if picked_model == "Auto" else picked_model)
        st.caption(f"Using: {chosen_model or 'Auto (.env or default)'}")
    st.divider()


def err_text(e: Exception) -> str:
    """Friendly text for key/rate-limit problems; raw message otherwise."""
    msg = str(e)
    keys = ("GEMINI_API_KEY", "429", "RESOURCE_EXHAUSTED", "API key", "API_KEY", "403", "401")
    return friendly_error(e) if any(k in msg for k in keys) else msg


def tuning_controls():
    with st.sidebar.expander("Retrieval tuning"):
        top_k = st.slider("Max chunks per source type", 1, 10, 5)
        min_score = st.slider("Minimum similarity", 0.0, 0.8, 0.30, 0.05)
        rel_cutoff = st.slider("Keep chunks within % of best match", 0.3, 1.0, 0.75, 0.05)
    return top_k, min_score, rel_cutoff


doc_sources = list_sources(DOCS)
note_map = notes_store.load_notes()
folders = list_folders(
    extra=[s["folder"] for s in doc_sources] + [n["folder"] for n in note_map.values()]
)


def source_label(s: dict) -> str:
    kind = s.get("kind", "document")
    if kind == "web":
        return f"🌐 {s['source']}"
    if kind == "note":
        return f"📝 {s['source']} · 📁 {s.get('folder', DEFAULT_FOLDER)}"
    return f"📄 {s['source']} · {s['location']} · 📁 {s.get('folder', DEFAULT_FOLDER)}"


def show_sources(sources: list[dict]):
    if not sources:
        return
    st.markdown("**Sources**")
    for s in sources:
        with st.expander(f"[{s['n']}] {source_label(s)}"):
            if s.get("kind") == "web":
                st.markdown(f"[{s['location']}]({s['location']})")
            st.write(s["text"])


# =====================================================================
# 💬 CHAT
# =====================================================================
if page == "💬 Chat":
    web_n, read_pages = 5, True
    with st.sidebar:
        st.subheader("Answer from")
        picked = st.multiselect(
            "Sources", list(SOURCE_OPTIONS), default=["📄 Documents", "📝 Notes"],
            label_visibility="collapsed",
        )
        folder_filter = st.multiselect(
            "Limit to folders (empty = all)", folders, help="Applies to documents and notes."
        )
        if "🌐 Web" in picked:
            web_n = st.slider("Web results", 3, 8, 5)
            read_pages = st.checkbox("Read full web pages (slower, better)", value=True)
    top_k, min_score, rel_cutoff = tuning_controls()
    if st.sidebar.button("Clear chat"):
        st.session_state.chat_msgs = []
        st.rerun()

    kinds = tuple(SOURCE_OPTIONS[p] for p in picked)
    if picked:
        where = ", ".join(folder_filter) if folder_filter else "all folders"
        extra = "" if kinds == ("web",) else f" · folders: {where}"
        st.caption(f"Answering from: {', '.join(picked)}{extra}")
    else:
        st.warning("Pick at least one source in the sidebar (Documents, Notes, or Web).")

    for m in st.session_state.chat_msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_sources(m.get("sources", []))
                if m.get("caption"):
                    st.caption(m["caption"])

    question = st.chat_input("Ask a question...", disabled=not picked)
    if question:
        st.session_state.chat_msgs.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            hits, engine, retrieval_error = [], "", None
            flt = folder_filter or None
            with st.spinner("Searching..."):
                try:
                    for kind, coll in (("document", DOCS), ("note", NOTES)):
                        if kind in kinds:
                            hits += [
                                dict(h, kind=kind)
                                for h in search(coll, question, k=top_k, min_score=min_score,
                                                rel_cutoff=rel_cutoff, folders=flt)
                            ]
                    if "web" in kinds:
                        web_hits, engine = search_web(question, n=web_n, read_pages=read_pages)
                        hits += [dict(h, kind="web") for h in web_hits]
                except Exception as e:
                    retrieval_error = err_text(e)
            if retrieval_error:
                result = {"answer": retrieval_error, "sources": [], "model": chosen_model or "-"}
            else:
                with st.spinner("Writing answer..."):
                    result = answer_from_chunks(question, hits, model=chosen_model, kinds=kinds)
            st.markdown(result["answer"])
            show_sources(result["sources"])
            caption = f"Model: {result['model']}" + (f" · Search: {engine}" if engine else "")
            st.caption(caption)

        st.session_state.chat_msgs.append(
            {"role": "assistant", "content": result["answer"],
             "sources": result["sources"], "caption": caption}
        )


# =====================================================================
# 🧠 STUDY (quiz + flashcards)
# =====================================================================
elif page == "🧠 Study":
    st.subheader("🧠 Quiz & Flashcards")
    st.caption("Generated only from your own documents and notes.")

    c1, c2 = st.columns(2)
    with c1:
        study_picked = st.multiselect(
            "Study from", ["📄 Documents", "📝 Notes"], default=["📄 Documents"]
        )
        study_folders = st.multiselect("Limit to folders (empty = all)", folders)
    with c2:
        kind_label = st.radio("Create", ["Quiz", "Flashcards"], horizontal=True)
        count = st.slider("How many?", 3, 15, 8)
        topic = st.text_input(
            "Topic (optional)", placeholder="e.g. gradient descent. Leave empty for a mix."
        )

    if st.button("✨ Generate", type="primary", disabled=not study_picked):
        study_kinds = tuple(SOURCE_OPTIONS[p] for p in study_picked)
        try:
            with st.spinner("Picking material..."):
                chunks = gather(study_kinds, study_folders or None, topic)
        except Exception as e:
            chunks = None
            st.session_state.pop("study", None)
            st.error(err_text(e))
        if chunks is None:
            pass
        elif not chunks:
            st.session_state.pop("study", None)
            st.warning("Nothing found. Index some documents or notes first, or change the topic/folder.")
        else:
            with st.spinner(f"Writing your {kind_label.lower()}..."):
                res = generate_items(kind_label.lower(), chunks, count, chosen_model)
            if res["error"]:
                st.session_state.pop("study", None)
                st.error(res["error"])
            else:
                st.session_state.study_id += 1
                st.session_state.study = {
                    "id": st.session_state.study_id,
                    "type": kind_label.lower(),
                    "items": res["items"],
                    "chunks": chunks,
                    "model": res["model"],
                    "submitted": False,
                    "chosen": [],
                    "i": 0,
                    "show": False,
                    "known": set(),
                }

    study = st.session_state.get("study")

    def src_caption(item, chunks):
        n = item.get("source")
        if n:
            c = chunks[n - 1]
            return source_label(dict(c, kind=c.get("kind", "document")))
        return ""

    if study and study["type"] == "quiz":
        sid, items, chunks = study["id"], study["items"], study["chunks"]
        st.divider()
        with st.form(f"quizform_{sid}"):
            for i, q in enumerate(items):
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                st.radio("Answer", q["options"], index=None, key=f"q_{sid}_{i}",
                         label_visibility="collapsed")
            submitted = st.form_submit_button("Submit answers", type="primary")
        if submitted:
            study["chosen"] = [st.session_state.get(f"q_{sid}_{i}") for i in range(len(items))]
            study["submitted"] = True
        if study["submitted"]:
            correct = 0
            for i, q in enumerate(items):
                right = q["options"][q["answer_index"]]
                chosen = study["chosen"][i] if i < len(study["chosen"]) else None
                ok = chosen == right
                correct += ok
                with st.expander(
                    f"{'✅' if ok else '❌'} Q{i + 1}. {q['question']}", expanded=not ok
                ):
                    st.write(f"Your answer: {chosen or '(no answer)'}")
                    if not ok:
                        st.write(f"Correct answer: **{right}**")
                    if q["explanation"]:
                        st.write(q["explanation"])
                    cap = src_caption(q, chunks)
                    if cap:
                        st.caption(f"Source: {cap}")
            st.success(f"Score: {correct} / {len(items)}")
        st.caption(f"Model: {study['model']}")

    elif study and study["type"] == "flashcards":
        items, chunks = study["items"], study["chunks"]

        def _fc(action: str):
            s = st.session_state.study
            last = len(s["items"]) - 1
            if action == "next":
                s["i"], s["show"] = min(s["i"] + 1, last), False
            elif action == "prev":
                s["i"], s["show"] = max(s["i"] - 1, 0), False
            elif action == "flip":
                s["show"] = not s["show"]
            elif action == "known":
                s["known"].add(s["i"])
                s["i"], s["show"] = min(s["i"] + 1, last), False
            elif action == "shuffle":
                random.shuffle(s["items"])
                s["i"], s["show"], s["known"] = 0, False, set()

        st.divider()
        i = study["i"]
        card = items[i]
        st.progress((i + 1) / len(items), text=f"Card {i + 1} of {len(items)} · ✅ known: {len(study['known'])}")
        with st.container(border=True):
            st.markdown(f"### {card['front']}")
            if study["show"]:
                st.divider()
                st.markdown(card["back"])
                cap = src_caption(card, chunks)
                if cap:
                    st.caption(f"Source: {cap}")
            else:
                st.caption("Think of the answer, then press Flip.")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.button("⬅ Prev", on_click=_fc, args=("prev",))
        b2.button("🔄 Flip", on_click=_fc, args=("flip",), type="primary")
        b3.button("✅ Knew it", on_click=_fc, args=("known",))
        b4.button("Next ➡", on_click=_fc, args=("next",))
        b5.button("🔀 Shuffle", on_click=_fc, args=("shuffle",))

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Front", "Back"])
        for c in items:
            w.writerow([c["front"], c["back"]])
        st.download_button("⬇ Download as CSV (Anki-ready)", buf.getvalue(),
                           file_name="flashcards.csv", mime="text/csv")
        st.caption(f"Model: {study['model']}")


# =====================================================================
# 📁 DOCUMENTS
# =====================================================================
elif page == "📁 Documents":
    st.subheader("📁 Documents & folders")
    st.caption("Organise files into folders (tracks), e.g. Data Science. Chat can be limited to a folder.")

    with st.expander("➕ Create a folder"):
        new_name = st.text_input("Folder name", placeholder="e.g. Data Science", key="new_folder")
        if st.button("Create folder", disabled=not new_name.strip()):
            created = add_folder(new_name)
            st.session_state.flash_docs = f"Folder '{created}' is ready."
            st.rerun()
    if st.session_state.get("flash_docs"):
        st.success(st.session_state.pop("flash_docs"))

    st.markdown("**Upload files**")
    up_folder = st.selectbox("Upload into folder", folders)
    uploads = st.file_uploader("Files", type=SUPPORTED_TYPES, accept_multiple_files=True)
    if st.button("Index files", type="primary", disabled=not uploads):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    chunks = file_to_chunks(f.name, f.getvalue())
                    for c in chunks:
                        c["folder"] = up_folder
                    n = add_chunks(DOCS, chunks)
                    if n:
                        st.success(f"{f.name} → {up_folder}: {n} chunks indexed")
                    else:
                        st.warning(f"{f.name}: no readable text found")
                except Exception as e:
                    st.error(f"{f.name}: {err_text(e)}")
        doc_sources = list_sources(DOCS)

    st.divider()
    st.markdown("**Your library**")

    def _move(name: str, old: str, key: str):
        new = st.session_state[key]
        if new != old:
            move_source(DOCS, name, old, new)

    for folder in folders:
        docs = [s for s in doc_sources if s["folder"] == folder]
        notes_here = [t for t, n in note_map.items() if n["folder"] == folder]
        with st.expander(f"📁 {folder}  ({len(docs)} files, {len(notes_here)} notes)", expanded=bool(docs)):
            if not docs:
                st.caption("No documents in this folder.")
            for d in docs:
                c1, c2, c3 = st.columns([4, 3, 1])
                c1.write(f"{d['source']}  \n`{d['chunks']} chunks`")
                key = f"mv_{folder}_{d['source']}"
                c2.selectbox(
                    "Move to", folders, index=folders.index(folder), key=key,
                    label_visibility="collapsed", on_change=_move,
                    args=(d["source"], folder, key),
                )
                if c3.button("🗑", key=f"del_{folder}_{d['source']}"):
                    delete_source(DOCS, d["source"], folder)
                    st.rerun()
            if folder != DEFAULT_FOLDER and not docs and not notes_here:
                if st.button("Delete empty folder", key=f"delf_{folder}"):
                    delete_folder(folder)
                    st.rerun()


# =====================================================================
# 📝 NOTES
# =====================================================================
elif page == "📝 Notes":
    st.subheader("📝 My notes")
    st.caption("Write notes (progress, summaries, reminders). Chat can combine them with your documents.")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))

    titles = [None] + sorted(note_map)
    sel = st.selectbox(
        "Note", titles,
        format_func=lambda t: "➕ New note" if t is None else f"{t}   ·   📁 {note_map[t]['folder']}",
    )
    nonce = st.session_state.note_nonce
    cur = note_map.get(sel)
    title = st.text_input("Title", value=sel or "", key=f"nt_{sel}_{nonce}")
    folder_default = cur["folder"] if cur else DEFAULT_FOLDER
    n_folder = st.selectbox(
        "Folder", folders,
        index=folders.index(folder_default) if folder_default in folders else 0,
        key=f"nf_{sel}_{nonce}",
    )
    body = st.text_area("Note", value=cur["text"] if cur else "", height=240, key=f"nb_{sel}_{nonce}")

    c1, c2, _ = st.columns([1, 1, 5])
    if c1.button("Save note", type="primary", disabled=not (title.strip() and body.strip())):
        if sel and title.strip() != sel:  # renamed
            notes_store.delete_note(sel)
        try:
            with st.spinner("Saving and indexing..."):
                n = notes_store.save_note(title.strip(), body, n_folder)
        except Exception as e:
            st.error(f"Could not save the note: {err_text(e)}")
        else:
            st.session_state.flash = f"Saved '{title.strip()}' in {n_folder} ({n} chunks indexed)."
            st.session_state.note_nonce += 1
            st.rerun()
    if sel and c2.button("Delete"):
        notes_store.delete_note(sel)
        st.session_state.flash = f"Deleted '{sel}'."
        st.session_state.note_nonce += 1
        st.rerun()


# =====================================================================
# 📊 EVALUATION
# =====================================================================
else:
    st.subheader("📊 Evaluation: StudyRAG vs. plain Gemini")
    st.caption(
        "Add test questions about your documents, plus keywords the correct answer must contain "
        "(comma-separated, all must appear). The app measures retrieval quality and compares "
        "StudyRAG's answers with plain Gemini that has no access to your documents."
    )
    top_k, min_score, rel_cutoff = tuning_controls()
    ev_folders = st.multiselect("Limit to folders (empty = all)", folders, key="ev_folders")
    delay = st.slider("Pause between questions (seconds, avoids free-tier rate limits)", 0, 15, 4)

    if "eval_df" not in st.session_state:
        st.session_state.eval_df = pd.DataFrame(
            {
                "question": ["Example: How many internship hours are required?"],
                "expected_keywords": ["replace, with, words from the correct answer"],
            }
        )
    edited = st.data_editor(
        st.session_state.eval_df, num_rows="dynamic", key="eval_editor",
        column_config={
            "question": st.column_config.TextColumn("Question", width="large"),
            "expected_keywords": st.column_config.TextColumn("Expected keywords (comma-separated)", width="medium"),
        },
    )

    if st.button("▶ Run evaluation", type="primary"):
        rows = []
        for _, r in edited.iterrows():
            q = r.get("question")
            kws = parse_keywords(r.get("expected_keywords"))
            if isinstance(q, str) and q.strip() and kws:
                rows.append((q.strip(), kws))
        if not rows:
            st.warning("Add at least one question with expected keywords.")
        else:
            results, bar = [], st.progress(0.0, text="Starting...")
            for n, (q, kws) in enumerate(rows, start=1):
                bar.progress((n - 1) / len(rows), text=f"Question {n} of {len(rows)}")
                try:
                    hits = search(DOCS, q, k=top_k, min_score=min_score, rel_cutoff=rel_cutoff,
                                  folders=ev_folders or None)
                except Exception as e:
                    st.error(err_text(e))
                    break
                rag = answer_from_chunks(q, [dict(h, kind="document") for h in hits],
                                         model=chosen_model, kinds=("document",))
                plain = answer_plain(q, chosen_model)
                res = evaluate_question(q, kws, hits, rag["answer"], plain["answer"])
                res["error"] = rag["error"] or plain["error"]
                results.append(res)
                if delay and n < len(rows):
                    time.sleep(delay)
            bar.progress(1.0, text="Done")
            st.session_state.eval_results = results

    results = st.session_state.get("eval_results")
    if results:
        s = summarize(results)
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieval hit rate", f"{s['hit_at_k']:.0%}", help="Right chunk found among the retrieved ones")
        m2.metric("MRR", f"{s['mrr']:.2f}", help="Mean reciprocal rank of the first correct chunk")
        m3.metric("StudyRAG correct", f"{s['rag_acc']:.0%}")
        m4.metric("Plain Gemini correct", f"{s['plain_acc']:.0%}")
        if any(r["error"] for r in results):
            st.warning("Some answers failed (rate limit or model error), so they count as incorrect. Re-run with a longer pause.")

        table = pd.DataFrame(
            [{"Question": r["question"], "Retrieval hit": r["retrieval_hit"], "Rank": r["rank"],
              "StudyRAG correct": r["rag_correct"], "Plain correct": r["plain_correct"]}
             for r in results]
        )
        st.dataframe(table, hide_index=True)
        st.download_button("⬇ Download results (CSV)", table.to_csv(index=False),
                           file_name="evaluation_results.csv", mime="text/csv")
        with st.expander("See full answers"):
            for r in results:
                st.markdown(f"**{r['question']}**")
                st.markdown(f"StudyRAG: {r['rag_answer']}")
                st.markdown(f"Plain Gemini: {r['plain_answer']}")
                st.divider()
