"""
multi_review_agent.py — Orchestrates code review across multiple files.

Strategy:
  1. Review each file sequentially using the existing stream_code_review().
  2. Stream each file's review with a clear section header so the frontend
     can render per-file cards.
  3. After all files, make a single summary LLM call that synthesises
     the cross-file findings into an overall repo health assessment.

WHY SEQUENTIAL, NOT PARALLEL?
Free-tier Gemini has a 15 req/min rate limit. Running N files in parallel
would immediately exhaust it for any repo with > 2 files.
Sequential also makes the streaming UX feel intentional — you watch the
agent work through each file one by one.

SECTION DELIMITER FORMAT:
  \n\n---FILE_SECTION---filename\n\n
The frontend splits on this marker to render collapsible per-file cards.
"""

import json
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm_factory import get_chat_llm
from app.services.review_agent import stream_code_review

FILE_SECTION_MARKER = "---FILE_SECTION---"
SUMMARY_SECTION_MARKER = "---REPO_SUMMARY---"


async def stream_multi_review(
    files: list[dict],  # list of {file_name, content, language}
) -> AsyncGenerator[str, None]:
    """
    Stream a combined review of multiple files.

    Each file produces:
      __SECTION_START__filename__SECTION_END__
      ... review tokens for that file ...

    Then a final repo-level summary section.
    """
    per_file_summaries: list[str] = []

    for idx, file_info in enumerate(files):
        file_name = file_info["file_name"]
        content = file_info["content"]
        language = file_info.get("language", "")

        # ── Section header ────────────────────────────────────────────────────
        yield f"__SECTION_START__{file_name}__SECTION_END__\n"

        # ── Status update ─────────────────────────────────────────────────────
        status_meta = json.dumps({"step": "file", "file": file_name, "index": idx + 1, "total": len(files)})
        yield f"__STATUS__Reviewing file {idx + 1}/{len(files)}: `{file_name}`...{status_meta}__STATUS_END__\n"

        # ── Stream the per-file review ────────────────────────────────────────
        file_review_tokens: list[str] = []
        try:
            async for token in stream_code_review(file_name, content, language):
                # Propagate ERROR markers directly; collect non-status tokens for summary
                yield token
                if not token.startswith("__STATUS__") and not token.startswith("__ERROR__"):
                    file_review_tokens.append(token)
        except Exception as e:
            err_msg = str(e)[:200]
            yield f"__ERROR__{err_msg}__ERROR_END__\n"
            # Continue to next file rather than aborting the whole batch

        # Collect a brief summary of this file's review for the repo summary
        full_review = "".join(file_review_tokens).strip()
        # Take just the first 400 chars — enough to capture the key findings
        per_file_summaries.append(
            f"**{file_name}**: {full_review[:400]}{'...' if len(full_review) > 400 else ''}"
        )

    # ── Repo-level summary ────────────────────────────────────────────────────
    if len(files) > 1:
        yield f"__SECTION_START__📊 Overall Repo Summary__SECTION_END__\n"
        summary_meta = json.dumps({"step": "summary"})
        yield f"__STATUS__Generating repo-wide summary...{summary_meta}__STATUS_END__\n"

        try:
            llm = get_chat_llm(streaming=True)
            summaries_text = "\n\n".join(per_file_summaries)

            messages = [
                SystemMessage(
                    content="You are CodeSage. You have just reviewed multiple files in a repository. "
                            "Write a concise overall health assessment of the codebase based on the per-file findings."
                ),
                HumanMessage(
                    content=f"Here are the per-file review summaries:\n\n{summaries_text}\n\n"
                            f"Write a **Overall Repo Health Assessment** with:\n"
                            f"- 🏥 Overall health score (1–10) with justification\n"
                            f"- 🔁 Cross-file patterns (repeated issues across files)\n"
                            f"- 🎯 Top 3 highest-priority fixes\n"
                            f"- ✅ Strengths across the codebase"
                ),
            ]

            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            err_msg = str(e)[:200]
            yield f"__ERROR__{err_msg}__ERROR_END__\n"
