"""
review.py — API routes for the agentic code review feature.

Two endpoints:
  POST /review/file    — review a file from the indexed repo (by path)
  POST /review/paste   — review code pasted directly into the UI

Both return a streaming response. The stream has two types of chunks:
  __STATUS__...text...{"step": "..."}__STATUS_END__  →  progress update (tool running)
  everything else                                    →  actual review text tokens
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from app.limiter import limiter
from app.services.review_agent import stream_code_review
from app.services.multi_review_agent import stream_multi_review

router = APIRouter(prefix="/review", tags=["review"])


def _get_allowed_roots() -> list[Path]:
    """
    Returns the directories the review endpoint is permitted to read files from.
    Populated from settings so it stays in sync with the configured data directory.
    Temp dir covers GitHub repos cloned during ingestion.
    """
    from app.core.config import get_settings
    s = get_settings()
    return [
        Path(s.chroma_persist_directory).resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]


class ReviewFileRequest(BaseModel):
    file_path: str       # Absolute path from the indexed files list
    file_name: str
    language: str = ""

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """
        Prevent path traversal attacks.

        Without this, a crafted request body like {"file_path": "/etc/passwd"}
        would cause the server to read and return the contents of any file
        accessible to the process — a critical information disclosure vulnerability.

        Fix: resolve the path to its absolute canonical form (resolves .. and symlinks),
        then check it starts with one of the known safe roots (temp dir or data dir).

        We return the ORIGINAL (unresolved) path so it matches the `source` value
        stored in ChromaDB at index time. On macOS /tmp is a symlink to /private/tmp;
        resolving it would break the ChromaDB metadata lookup in the fallback.
        """
        resolved = Path(v).resolve()
        allowed_roots = _get_allowed_roots()
        if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
            raise ValueError(
                f"File path is outside the allowed directory. "
                f"Only files within the server's data or temp directories may be reviewed."
            )
        # Return the original value, not resolved — preserves the path as stored in ChromaDB
        return v


class ReviewPasteRequest(BaseModel):
    code: str            # Raw code content
    file_name: str = "snippet"
    language: str = ""


class ReviewMultiRequest(BaseModel):
    files: list[ReviewFileRequest]

    @field_validator("files")
    @classmethod
    def validate_files(cls, v: list) -> list:
        if not v:
            raise ValueError("At least one file must be provided.")
        if len(v) > 30:
            raise ValueError("Too many files. Maximum 30 files per multi-review request.")
        return v


class PRWebhookRequest(BaseModel):
    repo: str
    pr_number: int
    title: str = ""
    diff: str


@router.post("/file")
@limiter.limit("10/minute")   # reviews are expensive — 8 LLM calls each
async def review_indexed_file(request: Request, body: ReviewFileRequest):
    """
    Review a file that's already been indexed into the vector store.
    Attempts to read from disk; if cleaned up, reconstructs from ChromaDB chunks.
    """
    content = ""
    try:
        with open(body.file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (FileNotFoundError, OSError):
        # Fallback: recover content from ChromaDB chunks if file was in a temporary clone directory
        from app.services.ingestion_service import _get_vectorstore
        try:
            vs = _get_vectorstore()
            results = vs._collection.get(
                where={"source": body.file_path},
                include=["documents", "metadatas"],
            )
            docs = results.get("documents") or []
            metadatas = results.get("metadatas") or []
            if docs and metadatas:
                # Sort chunks by chunk_index to reconstruct sequence
                sorted_chunks = sorted(
                    zip(metadatas, docs),
                    key=lambda item: item[0].get("chunk_index", 0),
                )
                content = "\n".join(chunk[1] for chunk in sorted_chunks)
        except Exception:
            pass

        if not content:
            raise HTTPException(
                status_code=404,
                detail=f"File not found on disk or vector store: {body.file_path}. "
                       "Please re-index or use direct paste review.",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        stream_code_review(body.file_name, content, body.language),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/paste")
@limiter.limit("10/minute")
async def review_pasted_code(request: Request, body: ReviewPasteRequest):
    """
    Review code pasted directly — no indexing required.
    This is the "try it instantly" mode for users who don't have a GitHub URL.
    """
    if not body.code.strip():
        raise HTTPException(status_code=400, detail="No code provided.")

    if len(body.code) > 50_000:
        raise HTTPException(
            status_code=413,
            detail="Code too large (max 50,000 chars). Split into smaller files.",
        )

    return StreamingResponse(
        stream_code_review(body.file_name, body.code, body.language),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/multi")
@limiter.limit("3/minute")   # very expensive — each file runs the full ReAct loop
async def review_multiple_files(request: Request, body: ReviewMultiRequest):
    """
    Review multiple indexed files in a single request.
    Files are processed sequentially; results are streamed as a combined report
    with per-file sections followed by a cross-file summary.
    """
    file_dicts: list[dict] = []

    for item in body.files:
        content = ""
        try:
            with open(item.file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (FileNotFoundError, OSError):
            # Fallback: reconstruct from ChromaDB chunks (same pattern as /review/file)
            from app.services.ingestion_service import _get_vectorstore
            try:
                vs = _get_vectorstore()
                results = vs._collection.get(
                    where={"source": item.file_path},
                    include=["documents", "metadatas"],
                )
                docs = results.get("documents") or []
                metadatas = results.get("metadatas") or []
                if docs and metadatas:
                    sorted_chunks = sorted(
                        zip(metadatas, docs),
                        key=lambda pair: pair[0].get("chunk_index", 0),
                    )
                    content = "\n".join(chunk[1] for chunk in sorted_chunks)
            except Exception:
                pass

        if not content:
            # Skip unrecoverable files rather than aborting the whole batch.
            # The multi_review_agent will still process the remaining files.
            continue

        file_dicts.append({
            "file_path": item.file_path,
            "file_name": item.file_name,
            "language": item.language,
            "content": content,
        })

    if not file_dicts:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail="None of the requested files could be read from disk or vector store.",
        )

    return StreamingResponse(
        stream_multi_review(file_dicts),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/pr-webhook")
@limiter.limit("15/minute")
async def review_pr_webhook(request: Request, body: PRWebhookRequest):
    """
    Automated CI/CD GitHub PR Review webhook endpoint.
    Consumes a git diff, runs the ReAct agent review loop, and returns a structured JSON comment.
    """
    if not body.diff.strip():
        raise HTTPException(status_code=400, detail="PR diff cannot be empty.")

    review_chunks = []
    async for token in stream_code_review(
        file_name=f"PR #{body.pr_number}: {body.title}",
        file_content=body.diff[:45000],
        language="diff",
    ):
        # Filter out UI status telemetry markers from the stream
        if not token.startswith("__STATUS__"):
            review_chunks.append(token)

    full_review = "".join(review_chunks).strip()
    return {
        "status": "success",
        "repo": body.repo,
        "pr_number": body.pr_number,
        "review": full_review,
    }
