# StudyRAG

AI study assistant: upload course files and notes, then ask questions, take quizzes and
practise flashcards, all grounded in your own material (with sources).

## Features
- Chat from Documents, Notes, Web (any combination), with citations
- Folders (tracks) to organise documents and notes; chat/quiz can be limited to a folder
- Quiz and flashcard generation (flashcards export to CSV for Anki)
- Evaluation page: retrieval hit rate, MRR, and StudyRAG vs plain Gemini accuracy
- Private workspace per visitor on the deployed version

## Run locally
```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements-local.txt    # includes the local embedding model
# create a .env file next to app.py containing:
# GEMINI_API_KEY=your_key
streamlit run app.py
```

