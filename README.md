# CodeSage — AI Code Review & Q&A Assistant

> Ask natural language questions about any GitHub repository. Get answers grounded in the actual code, with citations.

![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20LangChain%20%7C%20ChromaDB%20%7C%20React-purple)

## What it does

1. **Paste a GitHub URL** → CodeSage clones the repo and indexes all code files
2. **Ask any question** → "What does the auth module do?" / "Are there any SQL injection risks?" / "Explain the main entry point"
3. **Get a cited answer** → Every answer shows which files it came from

## Architecture

```
GitHub Repo / Uploaded Files
        ↓
FastAPI /ingest → git clone → AST-aware chunking → OpenAI embeddings → ChromaDB
        
User Question
        ↓
FastAPI /chat → embed question → ChromaDB similarity search → GPT-4o with context → streamed answer
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o |
| Embeddings | text-embedding-3-small (1536 dims) |
| Vector DB | ChromaDB (local) |
| Orchestration | LangChain |
| Backend | FastAPI + Python 3.11 |
| Frontend | React 18 + TypeScript + Tailwind CSS |
| Deploy | Docker Compose |

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone this repo
```bash
git clone <this-repo>
cd CodeSage
```

### 2. Set up the backend
```bash
cd backend
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

### 3. Set up the frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### Or: run with Docker (one command)
```bash
# Set your API key first
export OPENAI_API_KEY=sk-...

docker compose up --build
```

## Project Structure

```
CodeSage/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── ingest.py       # POST /ingest/github, POST /ingest/files
│   │   │   └── chat.py         # POST /chat/stream, GET /chat/indexed-files
│   │   ├── core/
│   │   │   └── config.py       # Centralized settings (env vars)
│   │   └── services/
│   │       ├── ingestion_service.py   # Clone → chunk → embed → store
│   │       └── retrieval_service.py  # Retrieve → prompt → stream
│   ├── main.py                 # FastAPI app entry point
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── IngestPanel.tsx  # Left sidebar (repo URL input + file list)
│       │   ├── ChatWindow.tsx   # Main chat area
│       │   └── MessageBubble.tsx # Individual message with syntax highlighting
│       ├── hooks/
│       │   └── useChat.ts       # Chat state + streaming fetch logic
│       └── types/index.ts       # TypeScript interfaces
└── docker-compose.yml
```

## Key Engineering Decisions

**Why AST-aware chunking?**
Generic text splitting would cut a Python function in half. LangChain's `RecursiveCharacterTextSplitter.from_language()` splits on `class`/`def` boundaries first, preserving logical units.

**Why MMR retrieval instead of plain similarity?**
Plain cosine similarity often returns 5 nearly-identical chunks. Maximal Marginal Relevance (MMR) balances relevance + diversity, giving the LLM more varied context.

**Why streaming?**
Without streaming, users see a blank screen for 10-30 seconds. With streaming, the first token appears in ~300ms. The UX difference is enormous.

**Why ChromaDB locally?**
Zero signup, zero cost, runs in-process as a Python library. For production, swap to Pinecone with one line change in `config.py`.
