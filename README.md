# AI Chatbot — RAG + AI Agent System

A production-style conversational AI system built for the AI Developer technical assessment.
The application combines:

* Real-time streaming chat
* Session-based conversational memory
* Retrieval-Augmented Generation (RAG)
* AI Agent tool calling
* Web search capabilities
* Secure document question answering

---

# Project Overview

This project implements all three required assessment parts:

| Part              | Requirement                                            | Implementation                                                   |
| ----------------- | ------------------------------------------------------ | ---------------------------------------------------------------- |
| Part 1 — Chat API | Streaming chatbot with conversation memory             | FastAPI SSE streaming with session-based chat history            |
| Part 2 — RAG      | Upload documents, embed, retrieve, answer from context | ChromaDB + HuggingFace embeddings + LangChain retrieval pipeline |
| Part 3 — AI Agent | Tool calling with autonomous tool selection            | LangChain Agent with web search and calculator tools             |

---

# Core Features

## Conversational AI

* Real-time streaming responses using Server-Sent Events (SSE)
* Multi-turn conversation support
* Session-based memory management
* Professional and context-aware responses

---

## RAG (Retrieval-Augmented Generation)

* Upload PDF or TXT documents
* Automatic document chunking
* Embedding generation using local HuggingFace models
* Semantic search using ChromaDB
* Context-aware document question answering
* Hallucination prevention when information is unavailable

---

## AI Agent with Tools

The chatbot can autonomously decide when to use tools based on the user query.

Implemented tools:

* Web Search Tool
* Calculator Tool

The assistant silently uses tools internally and generates natural conversational answers without exposing implementation details.

---

# Technology Stack

## Backend

* FastAPI
* Uvicorn
* Pydantic Settings

---

## LLM & AI

* Groq for ultra-fast LLM inference
* LangChain for agents, tools, retrieval, and orchestration

---

## RAG Stack

* ChromaDB for vector storage
* Hugging Face embeddings
* `sentence-transformers/all-MiniLM-L6-v2`
* `RecursiveCharacterTextSplitter`
* `PyPDFLoader`

---

## Tooling

* DuckDuckGo Search
* Safe arithmetic calculator
* SSE streaming responses

---

# Architecture

```text
User Request
      ↓
FastAPI API Layer
      ↓
Chat Service
      ↓
LangChain Agent
      ↓
 ┌────────────────────┬─────────────────────┬───────────────────┐
 │                    │                     │
RAG Retrieval      Web Search Tool      Calculator Tool
 │
ChromaDB Vector Store
 │
HuggingFace Embeddings
 │
Groq LLM
      ↓
Streaming AI Response
```

---

# How the RAG Pipeline Works

```text
Document Upload
      ↓
Text Extraction
      ↓
Chunking
      ↓
Embedding Generation
      ↓
Store Embeddings in ChromaDB
      ↓
Semantic Retrieval
      ↓
Inject Context into LLM
      ↓
Generate Final Response
```

---

# Security & Safety Design

The assistant includes multiple security-focused behaviors:

* Prevents hallucinated document answers
* Rejects unsupported or fabricated information
* Never exposes hidden prompts or internal system instructions
* Never reveals API keys, credentials, or backend configurations
* Prevents prompt injection attempts from uploaded documents
* Tool execution remains hidden from users
* Uses safe arithmetic execution instead of unrestricted `eval()`
* Keeps document context isolated per session

---

# Session-Based Context Management

Each user interaction is associated with a `session_id`.

This allows:

* Persistent conversation memory
* Session-isolated document retrieval
* Multi-user separation
* Independent RAG collections

---

# Streaming Response System

Responses are streamed using Server-Sent Events (SSE):

```text
data: Hello
data: How can I help you?
data: [DONE]
```

This provides:

* Real-time token streaming
* Better user experience
* Reduced perceived latency

---

# Environment Variables

Configuration is managed using `.env`.

## Required

| Variable       | Description                 |
| -------------- | --------------------------- |
| `GROQ_API_KEY` | Groq API authentication key |

---

## Optional

| Variable        | Default                |
| --------------- | ---------------------- |
| `GROQ_MODEL`    | `llama-3.1-8b-instant` |
| `CHUNK_SIZE`    | `1000`                 |
| `CHUNK_OVERLAP` | `200`                  |
| `RAG_TOP_K`     | `4`                    |
| `UPLOAD_DIR`    | `uploads`              |
| `CHROMA_DB_DIR` | `chroma_db`            |

---

# Installation

## Create Virtual Environment

```bash
py -m venv venv
```

---

## Activate Environment

### Windows

```bash
.\venv\Scripts\Activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Create `.env`

```env
GROQ_API_KEY=your_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

---

## Run Application

```bash
py -m uvicorn app.main:app --reload
```

---

# API Endpoints

| Method | Endpoint                 | Description              |
| ------ | ------------------------ | ------------------------ |
| GET    | `/`                      | Health check             |
| POST   | `/api/chat`              | Streaming AI chat        |
| POST   | `/api/upload`            | Upload PDF/TXT documents |
| DELETE | `/api/chat/{session_id}` | Clear chat history       |

---

# Example Chat Request

```json
{
  "message": "Tell me about today's weather in Chandigarh",
  "session_id": "default"
}
```

---

# Example Upload Request

Upload:

* PDF
* TXT files

The system:

* extracts text
* creates embeddings
* stores vectors
* enables semantic retrieval

---

# AI Agent Behaviour

The assistant autonomously decides:

* when to answer normally
* when to retrieve document context
* when to use web search
* when to use the calculator tool

Tool usage remains invisible to the user for a more natural conversational experience.

---

# Design Decisions

## Why Groq?

* Extremely fast inference
* Free developer tier
* Excellent streaming performance

---

## Why ChromaDB?

* Lightweight
* Local persistence
* No external infrastructure required
* Perfect for interview projects

---

## Why Local HuggingFace Embeddings?

* No API cost
* No additional API keys
* Fast local inference
* Easy deployment


---



# Assessment Coverage

This implementation fully satisfies:

* Streaming conversational API
* Session memory
* Document RAG
* Vector database integration
* AI agent tool calling
* Autonomous tool selection
* Real-time information retrieval
* Safe response generation

---

# Notes

* Embedding models download automatically during first run
* DuckDuckGo search requires no API key
* ChromaDB runs locally with persistent storage
* All embeddings are generated locally
* Uploaded documents remain session-isolated
