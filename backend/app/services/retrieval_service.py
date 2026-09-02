"""
retrieval_service.py — The "RA" in RAG (Retrieval-Augmented Generation).

Flow for each question:
  1. Embed the user's question (same embedding model as ingestion!)
  2. ChromaDB finds the top-K most similar chunks (cosine similarity)
  3. Build a prompt: system instructions + retrieved code + user question
  4. Stream GPT-4o's response token by token back to the client

WHY STREAMING MATTERS:
Without streaming, the user sees nothing for 10-30 seconds, then the whole
answer appears. With streaming, they see the first token in ~300ms and the
answer types out in real time. This is the difference between "broken" and "fast".
"""

import json
from functools import lru_cache
from typing import AsyncGenerator
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from langchain_core.documents import Document
from app.services.reranker import rerank
from app.services.llm_factory import get_chat_llm, get_embedding_fn
from app.services.hybrid_retriever import BM25Index, reciprocal_rank_fusion
from app.services.query_enhancer import extract_file_scope, expand_query, compact_chat_history
from app.services.token_counter import get_token_callback, increment_request

settings = get_settings()

# Module-level cached BM25 index and version tracker
_bm25_index_cache: BM25Index | None = None
_bm25_doc_count: int = -1


def _get_bm25_index(vectorstore: Chroma) -> BM25Index | None:
    """
    Builds or reuses an in-memory BM25 index over the documents in ChromaDB.
    """
    global _bm25_index_cache, _bm25_doc_count
    try:
        results = vectorstore._collection.get(include=["documents", "metadatas"])
        docs_raw = results.get("documents") or []
        metas_raw = results.get("metadatas") or []
        current_count = len(docs_raw)

        if _bm25_index_cache is not None and current_count == _bm25_doc_count:
            return _bm25_index_cache

        documents = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(docs_raw, metas_raw)
        ]
        _bm25_index_cache = BM25Index(documents)
        _bm25_doc_count = current_count
        return _bm25_index_cache
    except Exception:
        return None


# ── System Prompt ─────────────────────────────────────────────────────────────
# Uses a two-part structure instead of a single template string with {context}.
#
# WHY NOT .replace("{context}", context)?
# If any retrieved code chunk contains the literal string "{context}" — which is
# completely plausible in Python format strings, Jinja templates, or f-strings —
# the .replace() call substitutes it recursively, injecting garbage into the prompt.
# This is a silent prompt injection vector.
#
# Fix: split into SYSTEM_PREFIX (static instructions) + a separate user message
# that injects context as a plain string. The LLM receives context as data, not
# as part of the instruction template, so no substitution can happen.
SYSTEM_PREFIX = """You are CodeSage, an expert AI code reviewer and codebase assistant.
You have been given relevant code snippets from a repository to answer the user's question.

Rules you MUST follow:
1. ONLY answer based on the provided code context. Do not hallucinate code that isn't there.
2. Always cite the file name and location (e.g., "In `auth.py`, line ~45:") when referencing code.
3. If the context doesn't contain enough information, say so clearly — do not guess.
4. Format code examples in proper markdown code blocks with the language specified.
5. Be concise but thorough. If there's a bug, explain WHY it's a bug and HOW to fix it.
6. If asked for a code review, structure your response as:
   - 🐛 Bugs / Issues
   - ⚠️ Potential Problems
   - 💡 Suggestions
   - ✅ What looks good"""

@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """
    Module-level singleton for the ChromaDB client.

    WHY CACHE THIS?
    Creating a new ChromaDB PersistentClient per request opens a new SQLite
    connection (~50ms overhead). Under concurrent load this creates multiple
    unmanaged connections. lru_cache(maxsize=1) gives one warm connection
    reused across all requests.

    WHY IS THIS SAFE?
    ChromaDB's PersistentClient is thread-safe for reads. Writes (ingestion) are
    serialised by asyncio.to_thread, so there's no concurrent write conflict.
    The embedding function comes from llm_factory — correct for the active provider.
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_fn(),   # provider-aware: OpenAI or local MiniLM
        persist_directory=settings.chroma_persist_directory,
    )


async def stream_answer(
    question: str,
    chat_history: list[dict],
    active_repo_url: str | None = None,       # single-repo filter (legacy)
    active_repo_urls: list[str] | None = None, # multi-repo cross-search (new)
) -> AsyncGenerator[str, None]:
    """
    Core RAG function. Yields answer tokens one by one for streaming.

    WHY AN ASYNC GENERATOR?
    Instead of returning one big string, this function uses `yield` inside
    an async loop. FastAPI's StreamingResponse consumes these yields and
    pushes each token to the browser immediately via chunked HTTP.
    The browser's EventSource API reads them and appends to the chat bubble.
    Result: the typing effect you see in ChatGPT.

    active_repo_url / active_repo_urls:
    ChromaDB supports metadata filtering at query time.
    - Single repo (active_repo_url): exact match filter → {"repo_url": url}
    - Multi-repo (active_repo_urls): $in filter → {"repo_url": {"$in": [url1, url2, ...]}}
    When neither is set, retrieval spans all indexed repos.
    """

    # Resolve which repos to filter to
    # active_repo_urls (plural) takes priority; fall back to active_repo_url (singular)
    _repo_filter_urls: list[str] | None = None
    if active_repo_urls and len(active_repo_urls) > 0:
        _repo_filter_urls = active_repo_urls
    elif active_repo_url:
        _repo_filter_urls = [active_repo_url]

    # ── Step 1: Query Enhancement & Scoping ──────────────────────────────────
    # Extract @filename scope tag (e.g., "@auth.py where is verify_token defined?")
    search_query, file_scope = extract_file_scope(question)

    # ── Step 2: Hybrid Retrieval (Dense Vector MMR + Lexical BM25 + RRF) ─────
    vectorstore = _get_vectorstore()
    # Fetch 2× top_k candidates (was 3×). Halves cross-encoder scoring pairs
    # with negligible recall loss — the reranker's precision compensates.
    CANDIDATE_COUNT = settings.top_k_results * 2  # fetch 10, keep 5

    search_kwargs: dict = {
        "k": CANDIDATE_COUNT,
        "fetch_k": CANDIDATE_COUNT * 2,
    }
    if _repo_filter_urls:
        if len(_repo_filter_urls) == 1:
            # Single repo: simple equality filter (ChromaDB universal support)
            search_kwargs["filter"] = {"repo_url": _repo_filter_urls[0]}
        else:
            # Multi-repo: $in operator (supported in ChromaDB ≥ 0.4)
            search_kwargs["filter"] = {"repo_url": {"$in": _repo_filter_urls}}

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    # Branch A: Dense vector search
    dense_candidates = await retriever.ainvoke(search_query)

    # Branch B: BM25 lexical keyword search
    bm25_index = _get_bm25_index(vectorstore)
    bm25_candidates = (
        bm25_index.search(
            query=search_query,
            top_k=CANDIDATE_COUNT,
            repo_urls=_repo_filter_urls,  # supports both single and multi
            file_filter=file_scope,
        )
        if bm25_index
        else []
    )

    # Combine candidates using Reciprocal Rank Fusion (RRF)
    if bm25_candidates:
        fused_candidates = reciprocal_rank_fusion(
            [dense_candidates, bm25_candidates],
            k=60,
            top_n=CANDIDATE_COUNT,
        )
    else:
        fused_candidates = dense_candidates

    # If file scope was requested (@file), apply strict post-filtering
    if file_scope:
        scoped = [
            doc for doc in fused_candidates
            if file_scope.lower() in doc.metadata.get("source", "").lower()
            or file_scope.lower() in doc.metadata.get("file_name", "").lower()
        ]
        if scoped:
            fused_candidates = scoped

    # ── Step 3: Cross-Encoder Re-ranking ─────────────────────────────────────
    relevant_docs = await rerank(search_query, fused_candidates, top_n=settings.top_k_results)

    if not relevant_docs:
        yield "I couldn't find relevant code in the indexed repository for your question. Try rephrasing or make sure the repository was ingested first."
        return

    # ── Step 2: Build context string ─────────────────────────────────────────
    # We format each chunk with its metadata so the LLM knows WHERE the code is.
    # We deduplicate sources so the same file doesn't appear multiple times in
    # the citations panel (a file can have multiple relevant chunks).
    context_parts = []
    sources = []
    seen_sources: set[str] = set()

    for doc in relevant_docs:
        file_name = doc.metadata.get("file_name", "unknown")
        language = doc.metadata.get("language", "")
        source = doc.metadata.get("source", "")

        context_parts.append(
            f"### File: {file_name}\n```{language}\n{doc.page_content}\n```"
        )
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append({
                "file_name": file_name,
                "source": source,
                "language": language,
            })

    context = "\n\n".join(context_parts)

    # ── Step 4: Compact chat history ─────────────────────────────────────────
    history_str = compact_chat_history(chat_history, max_turns=6)

    # ── Step 4: Build the final prompt (injection-safe) ───────────────────────
    # Context and question are passed as a separate user message — NOT embedded
    # into the system prompt via string substitution. This prevents prompt
    # injection if the retrieved code contains strings like "{context}".
    user_message_content = (
        f"Here are the relevant code snippets from the repository:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Conversation so far:\n{history_str}\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    messages_to_send = [
        SystemMessage(content=SYSTEM_PREFIX),
        HumanMessage(content=user_message_content),
    ]

    # ── Step 5: Stream response ──────────────────────────────────────────────
    # get_chat_llm(streaming=True) returns the cached streaming LLM for the
    # active provider (OpenAI GPT-4o or Gemini 1.5 Flash).
    # Yield sources first so the UI renders the citations panel immediately.
    yield f"__SOURCES__{json.dumps(sources)}__SOURCES_END__\n"

    increment_request("chat")
    llm = get_chat_llm(streaming=True).with_config(callbacks=[get_token_callback()])
    async for chunk in llm.astream(messages_to_send):
        if chunk.content:
            yield chunk.content


async def get_indexed_files() -> list[dict]:
    """
    Returns a list of all unique files that have been indexed.
    Used by the frontend's 'Repo Map' feature to show what's been indexed.
    """
    vectorstore = _get_vectorstore()

    # ChromaDB lets us query metadata without a semantic search
    # We get all documents and extract unique file names
    collection = vectorstore._collection
    results = collection.get(include=["metadatas"])

    # BUG FIX: collection.get() returns {"metadatas": None} when collection is empty,
    # not {"metadatas": []}. Without this guard, iterating None raises TypeError.
    metadatas = results.get("metadatas") or []

    seen = set()
    files = []
    for metadata in metadatas:
        key = metadata.get("source", "")
        if key and key not in seen:
            seen.add(key)
            files.append({
                "file_name": metadata.get("file_name", ""),
                "language": metadata.get("language", ""),
                "repo_url": metadata.get("repo_url", ""),
                "source": key,
            })

    return sorted(files, key=lambda x: x["file_name"])
