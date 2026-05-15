# AI Chatbot — Document-Only RAG

A FastAPI conversational AI service that answers questions strictly from documents the user uploads. No tool calling, no web search, no general-knowledge fallback.

* Real-time streaming chat (SSE)
* Session-based conversational memory
* Retrieval-Augmented Generation (RAG) over uploaded PDF/TXT files
* Document-grounded answers only — refuses to answer outside the uploaded context

---

# Core Features

## Conversational AI

* Real-time streaming responses using Server-Sent Events (SSE)
* Multi-turn conversation support
* Session-based memory management

## RAG (Retrieval-Augmented Generation)

* Upload PDF or TXT documents
* Automatic document chunking
* Embedding generation using local HuggingFace models
* Semantic search using ChromaDB
* Document-only question answering
* Hallucination prevention — the assistant says "I could not find this information in the uploaded documents." when retrieval is empty or unrelated

---

# Technology Stack

## Backend

* FastAPI
* Uvicorn
* Pydantic Settings

## LLM

* Local Ollama runtime for offline LLM inference (streaming chat completions)
* Default model: `llama3` (any locally-pulled Ollama model works)

## RAG Stack

* ChromaDB for vector storage
* HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
* `RecursiveCharacterTextSplitter`
* `PyPDFLoader`

---

# Architecture

```text
User Request
      ↓
FastAPI API Layer
      ↓
Chat Service
      ↓
RAG Retrieval (ChromaDB + HuggingFace embeddings)
      ↓
Local Ollama LLM (document-grounded prompt)
      ↓
Streaming AI Response
```

---

# RAG Pipeline

```text
Document Upload
      ↓
Text Extraction (PDF / TXT)
      ↓
Chunking
      ↓
Embedding Generation
      ↓
Store Embeddings in ChromaDB (per-session collection)
      ↓
Semantic Retrieval
      ↓
Inject Excerpts into LLM Prompt
      ↓
Generate Final Response
```

---

# Behaviour Rules

* When documents are uploaded for the session, the assistant answers only from the retrieved excerpts.
* When no documents have been uploaded, the assistant asks the user to upload a PDF or TXT first.
* When retrieval returns nothing relevant, the assistant says so in one or two sentences.
* The assistant never answers from general knowledge and never invents missing details.

---

# Environment Variables

Configuration is managed using `.env`. No API keys are required — the LLM runs locally via Ollama.

## Optional

| Variable            | Default                  |
| ------------------- | ------------------------ |
| `OLLAMA_BASE_URL`   | `http://localhost:11434` |
| `OLLAMA_MODEL`      | `llama3`                 |
| `CHUNK_SIZE`        | `1000`                   |
| `CHUNK_OVERLAP`     | `200`                    |
| `RAG_TOP_K`         | `4`                      |
| `UPLOAD_DIR`        | `uploads`                |
| `CHROMA_DB_DIR`     | `chroma_db`              |
| `EMBEDDING_MODEL`   | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_CACHE_DIR` | `models`               |

---

# Installation

## Prerequisites

1. Install [Ollama](https://ollama.com/download) and make sure the local server is running (default at `http://localhost:11434`).
2. Pull a chat model, for example:

```bash
ollama pull llama3
```

## Python environment

```bash
py -m venv venv
.\venv\Scripts\Activate           # Windows
# source venv/bin/activate        # Linux / macOS
pip install -r requirements.txt
```

## One-time bootstrap (downloads the embedding model into `./models/`)

```bash
python scripts/prefetch_embeddings.py
```

After this completes, the application runs fully offline — no network calls at runtime.

Create `.env` (optional — defaults work for a standard local Ollama install):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

## Run

```bash
py -m uvicorn app.main:app --reload
```

---

# API Endpoints

| Method | Endpoint                 | Description              |
| ------ | ------------------------ | ------------------------ |
| GET    | `/`                      | Health check             |
| POST   | `/api/chat`              | Streaming document chat  |
| POST   | `/api/upload`            | Upload PDF/TXT documents |
| DELETE | `/api/chat/{session_id}` | Clear chat history       |

---

# Example Chat Request

```json
{
  "message": "What does the contract say about termination?",
  "session_id": "default"
}
```

---

# Notes

* Embedding models download automatically during first run
* ChromaDB runs locally with persistent storage
* All embeddings are generated locally
* Uploaded documents remain session-isolated
