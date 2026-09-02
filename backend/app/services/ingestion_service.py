"""
ingestion_service.py — Handles everything from "raw code" to "searchable vectors".

The pipeline:
  GitHub URL / uploaded files
       ↓
  Clone or save to temp directory
       ↓
  Walk files, filter by extension (only code files)
       ↓
  Split into chunks (language-aware)
       ↓
  Embed each chunk via OpenAI
       ↓
  Upsert into ChromaDB with metadata (file path, language, line numbers)
"""

import hashlib
import os
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import Callable, Optional

import git
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.core.config import get_settings
from app.services.llm_factory import get_embedding_fn
from app.services.ast_chunker import chunk_python_file

settings = get_settings()

# ── Language detection ────────────────────────────────────────────────────────
# Maps file extensions → LangChain Language enum
# Why? RecursiveCharacterTextSplitter has language-specific rules.
# For Python it splits on "class ", "def ", "\n\n" — respecting code structure.
# For plain text it just splits on newlines and spaces — wrong for code.
EXTENSION_TO_LANGUAGE = {
    ".py":   Language.PYTHON,
    ".js":   Language.JS,
    ".jsx":  Language.JS,
    ".ts":   Language.JS,     # TS shares JS splitter rules
    ".tsx":  Language.JS,
    ".go":   Language.GO,
    ".java": Language.JAVA,
    ".cpp":  Language.CPP,
    ".c":    Language.CPP,
    ".rs":   Language.RUST,
    ".rb":   Language.RUBY,
    ".md":   Language.MARKDOWN,
}

# Extensions we actually want to index — skip binaries, images, lock files
ALLOWED_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys()) | {".txt", ".json", ".yaml", ".yml", ".env.example"}

# Directories that are never worth indexing
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}


def _get_vectorstore() -> Chroma:
    """
    ChromaDB vector store for write operations (ingestion, clear, metadata queries).

    NOT cached — intentionally creates a fresh client for each write operation.
    Unlike retrieval_service._get_vectorstore() which is a read-only singleton,
    writes here need a fresh connection to ensure ChromaDB's WAL (write-ahead log)
    is flushed and visible to subsequent read clients.

    get_embedding_fn() is provider-aware: OpenAI embeddings or local MiniLM
    depending on LLM_PROVIDER. MUST match whatever was used at index time.
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_fn(),
        persist_directory=settings.chroma_persist_directory,
    )


def _collect_files(root: Path) -> list[Path]:
    """
    Walk the directory tree and return only files worth indexing.
    We skip huge dirs (node_modules) that would waste tokens and money.
    """
    collected = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Modify dirnames IN PLACE — this tells os.walk not to recurse into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix in ALLOWED_EXTENSIONS:
                # Skip files over 500KB — likely auto-generated or minified
                if fpath.stat().st_size < 500_000:
                    collected.append(fpath)

    return collected


def _load_and_split(files: list[Path], repo_url: str) -> list[Document]:
    """
    Load each file and split into chunks.

    Python files use AST-boundary chunking (one chunk per function/class) for
    better retrieval precision — see ast_chunker.py for the rationale.

    All other file types use RecursiveCharacterTextSplitter as before.

    chunk_overlap=200 means adjacent chunks share 200 characters.
    This prevents a function signature being in chunk N and its body in chunk N+1
    with no overlap — the model would see the body without knowing what function it's in.
    """
    documents = []

    for fpath in files:
        ext = fpath.suffix
        language = EXTENSION_TO_LANGUAGE.get(ext)

        try:
            # TextLoader reads the file as a string
            loader = TextLoader(str(fpath), encoding="utf-8", autodetect_encoding=True)
            raw_docs = loader.load()
        except Exception:
            # Some files have weird encodings — skip them
            continue

        # Compute content hash once per file — used for delta re-indexing.
        # SHA-256 of the raw file bytes is stable across runs for identical content.
        file_bytes = b"".join(d.page_content.encode() for d in raw_docs)
        content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]  # 16 hex chars is plenty

        source_text = raw_docs[0].page_content if raw_docs else ""

        # ── Python: AST-boundary chunking ─────────────────────────────────────
        # Produces one chunk per top-level function/class — far better retrieval
        # precision than character-based splitting for code Q&A.
        if ext == ".py" and source_text:
            ast_chunks = chunk_python_file(
                source=source_text,
                file_path=str(fpath),
                file_name=fpath.name,
                language="py",
                repo_url=repo_url,
                content_hash=content_hash,
            )
            if ast_chunks:
                documents.extend(ast_chunks)
                continue
            # AST parse failed (syntax error) — fall through to char splitter

        # ── Non-Python (or AST fallback): RecursiveCharacterTextSplitter ──────
        if language:
            # Language-aware splitter: knows to split on class/function boundaries
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        else:
            # Generic splitter for JSON, YAML, etc.
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

        chunks = splitter.split_documents(raw_docs)

        # Enrich metadata — this is what shows up in the "Sources" panel in the UI
        # Without metadata, you'd get answers but no way to cite where they came from
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source": str(fpath),           # absolute path on disk
                "file_name": fpath.name,         # just "auth.py"
                "language": ext.lstrip("."),     # "py", "js", etc.
                "repo_url": repo_url,
                "chunk_index": i,
                "content_hash": content_hash,   # for incremental delta re-indexing
            })

        documents.extend(chunks)

    return documents


async def ingest_github_repo(
    repo_url: str,
    branch: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Main ingestion entry point for GitHub repos.

    1. Clone to a temp directory (auto-deleted when done)
    2. Collect + split files
    3. Batch embed + store in ChromaDB
    Returns a summary dict for the API response.

    branch: when provided, clones that specific branch. When None, git uses
            the repo's default branch (whatever HEAD points to).
    """
    tmp_dir = tempfile.mkdtemp()

    try:
        # ── Step 1: Clone ─────────────────────────────────────────────────────
        clone_msg = f"Cloning {repo_url}" + (f" @ {branch}" if branch else "") + "..."
        if progress_callback:
            await progress_callback({"step": "cloning", "message": clone_msg})

        # Build kwargs: only add `branch` when explicitly requested.
        # git.Repo.clone_from accepts branch= as the ref to checkout.
        clone_kwargs: dict = {"depth": 1}  # shallow clone — latest commit only
        if branch:
            clone_kwargs["branch"] = branch

        # We run git.Repo.clone_from in a thread because it's blocking I/O
        # asyncio.to_thread prevents it from blocking the FastAPI event loop
        await asyncio.to_thread(
            git.Repo.clone_from,
            repo_url,
            tmp_dir,
            **clone_kwargs,
        )

        # ── Step 2: Collect files ─────────────────────────────────────────────
        if progress_callback:
            await progress_callback({"step": "scanning", "message": "Scanning files..."})

        files = _collect_files(Path(tmp_dir))

        if not files:
            return {"status": "error", "message": "No indexable code files found in this repo."}

        if progress_callback:
            await progress_callback({
                "step": "splitting",
                "message": f"Found {len(files)} files. Splitting into chunks...",
            })

        # ── Step 3: Split into chunks ─────────────────────────────────────────
        documents = await asyncio.to_thread(_load_and_split, files, repo_url)

        if progress_callback:
            await progress_callback({
                "step": "embedding",
                "message": f"Embedding {len(documents)} chunks (this may take a minute)...",
            })

        # ── Step 4: Incremental delta embed + store in ChromaDB ───────────────
        # Delta re-indexing: only embed files whose content has changed since
        # the last ingest. This makes re-indexing 10x faster for repos where
        # only a few files changed between runs.
        #
        # Algorithm:
        #   1. Fetch all existing (source → content_hash) pairs for this repo
        #   2. Compare with hashes of files collected this run
        #   3. Skip files whose hash hasn't changed (no re-embedding needed)
        #   4. Delete IDs for removed/changed files before adding new chunks
        #   5. Embed only the new/changed chunks
        vectorstore = _get_vectorstore()

        # Build a map: source_path → {hash, ids} from the current index
        indexed_map: dict[str, dict] = {}  # source → {"hash": str, "ids": [str]}
        try:
            existing = vectorstore._collection.get(
                where={"repo_url": repo_url},
                include=["metadatas"],
            )
            for eid, emeta in zip(
                existing.get("ids") or [],
                existing.get("metadatas") or [],
            ):
                src = emeta.get("source", "")
                chash = emeta.get("content_hash", "")
                if src not in indexed_map:
                    indexed_map[src] = {"hash": chash, "ids": []}
                indexed_map[src]["ids"].append(eid)
        except Exception:
            pass  # Collection may not exist yet on first run — fine

        # Partition documents into "need embedding" vs "unchanged"
        new_docs: list[Document] = []
        stale_ids: list[str] = []
        seen_sources: set[str] = set()

        for doc in documents:
            src = doc.metadata["source"]
            chash = doc.metadata["content_hash"]
            if src not in seen_sources:
                seen_sources.add(src)
                if src in indexed_map:
                    if indexed_map[src]["hash"] == chash:
                        # File unchanged — keep existing vectors, skip re-embedding
                        continue
                    else:
                        # File changed — mark old vectors for deletion
                        stale_ids.extend(indexed_map[src]["ids"])
            new_docs.append(doc)

        # Delete IDs for files that no longer exist in the repo (removed files)
        current_sources = {str(f) for f in files}
        for src, info in indexed_map.items():
            if src not in current_sources:
                stale_ids.extend(info["ids"])

        # Purge stale vectors (changed + removed files)
        if stale_ids:
            try:
                vectorstore._collection.delete(ids=stale_ids)
            except Exception:
                pass

        files_skipped = len(files) - len({d.metadata["source"] for d in new_docs})

        if progress_callback:
            msg = (
                f"Embedding {len(new_docs)} new/changed chunks"
                + (f" ({files_skipped} files unchanged, skipped)" if files_skipped else "")
                + "..."
            )
            await progress_callback({"step": "embedding", "message": msg})

        # Embed and insert in batches to provide incremental progress updates
        # and prevent reverse-proxy timeout (SSE keepalive).
        BATCH_SIZE = 100
        for i in range(0, len(new_docs), BATCH_SIZE):
            batch = new_docs[i:i + BATCH_SIZE]
            await asyncio.to_thread(
                vectorstore.add_documents,
                documents=batch,
            )
            if progress_callback:
                processed = min(i + BATCH_SIZE, len(new_docs))
                await progress_callback({
                    "step": "embedding",
                    "message": f"Embedding chunks: {processed}/{len(new_docs)}...",
                })

        # Invalidate the read-singleton in retrieval_service so the next query
        # opens a fresh ChromaDB connection that sees the newly written data.
        # Without this, the cached Chroma instance holds a handle to the old
        # SQLite WAL state and may return stale (or empty) results immediately
        # after ingestion completes.
        try:
            from app.services.retrieval_service import _get_vectorstore as _rv
            _rv.cache_clear()
        except Exception:
            pass  # non-fatal — next cold start will fix it

        chunks_added = len(new_docs)
        if progress_callback:
            skip_note = f" ({files_skipped} unchanged)" if files_skipped else ""
            await progress_callback({
                "step": "done",
                "message": f"✅ Indexed {chunks_added} chunks from {len(files) - files_skipped} files{skip_note}.",
            })

        return {
            "status": "success",
            "repo_url": repo_url,
            "files_indexed": len(files) - files_skipped,
            "files_skipped": files_skipped,
            "chunks_created": chunks_added,
        }

    finally:
        # Always clean up the temp clone — repos can be hundreds of MB
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def ingest_uploaded_files(files_content: list[tuple[str, bytes]]) -> dict:
    """
    Ingestion for direct file uploads (when user doesn't have a GitHub URL).
    files_content: list of (filename, raw_bytes) tuples
    """
    tmp_dir = tempfile.mkdtemp()

    try:
        # Write uploaded bytes to temp files so we can reuse the same pipeline
        paths = []
        for filename, content in files_content:
            fpath = Path(tmp_dir) / filename
            fpath.write_bytes(content)
            paths.append(fpath)

        documents = await asyncio.to_thread(_load_and_split, paths, "uploaded_files")

        # Guard: if every file failed to parse, documents will be empty.
        # Chroma.from_documents([]) raises a ValueError — handle it explicitly.
        if not documents:
            return {
                "status": "error",
                "message": "No content could be extracted from the uploaded files. "
                           "Check that the files are valid text/code files.",
            }

        await asyncio.to_thread(
            Chroma.from_documents,
            documents=documents,
            embedding=get_embedding_fn(),
            collection_name=settings.chroma_collection_name,
            persist_directory=settings.chroma_persist_directory,
        )

        # Invalidate read singleton — same reason as in ingest_github_repo
        try:
            from app.services.retrieval_service import _get_vectorstore as _rv
            _rv.cache_clear()
        except Exception:
            pass

        return {
            "status": "success",
            "files_indexed": len(paths),
            "chunks_created": len(documents),
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def clear_index(repo_url: str | None = None) -> dict:
    """
    Delete vectors from ChromaDB.

    repo_url=None  → wipe the entire collection (full reset)
    repo_url=<url> → delete only chunks that belong to that repo

    WHY NOT JUST DELETE THE CHROMA FOLDER?
    Deleting the folder works but is OS-dependent and not thread-safe if another
    request is reading. ChromaDB's own delete() API is the correct way — it handles
    locking and keeps the SQLite WAL consistent.
    """
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection

    def _bust_read_cache():
        """Invalidate the retrieval singleton so next query sees the mutations."""
        try:
            from app.services.retrieval_service import _get_vectorstore as _rv
            _rv.cache_clear()
        except Exception:
            pass

    if repo_url:
        # Targeted delete — only this repo's chunks
        try:
            existing = vectorstore.get(where={"repo_url": repo_url})
            ids = existing.get("ids") or []
            if ids:
                vectorstore.delete(ids=ids)
            _bust_read_cache()
            return {
                "status": "success",
                "message": f"Cleared {len(ids)} chunks for repo: {repo_url}",
                "deleted": len(ids),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    else:
        # Full reset — fetch all IDs then delete by ID.
        # WHY NOT collection.delete(where={})?
        # The behaviour of an empty `where` filter changed between ChromaDB versions:
        # in 0.4.x it deletes nothing; in 0.5.x it deletes everything.
        # Fetching IDs first and deleting by ID is correct in all versions.
        try:
            all_items = collection.get(include=[])  # IDs only — no embeddings fetched
            ids = all_items.get("ids") or []
            if ids:
                collection.delete(ids=ids)
            _bust_read_cache()
            return {
                "status": "success",
                "message": f"Cleared entire index ({len(ids)} chunks deleted).",
                "deleted": len(ids),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


async def get_indexed_repos() -> list[dict]:
    """
    Returns all unique repo URLs currently in the vector store, with chunk counts.
    Used by the frontend's active-repo selector.
    """
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    repo_counts: dict[str, int] = {}
    for m in metadatas:
        url = m.get("repo_url", "")
        if url:
            repo_counts[url] = repo_counts.get(url, 0) + 1

    return [
        {"repo_url": url, "chunk_count": count}
        for url, count in sorted(repo_counts.items())
    ]
