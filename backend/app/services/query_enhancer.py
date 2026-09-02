"""
query_enhancer.py — Query expansion and conversation summarization/compaction.

1. File Tag Extraction:
   Detects `@filename` or `@path/to/file` in user queries (e.g., "@auth.py where is login implemented?")
   and extracts both the clean question and the target file filter.

2. Query Expansion (Multi-Query):
   Generates complementary technical variations of the query for better retrieval recall.

3. Context Compaction:
   Compacts long conversation turns into a succinct structured summary so token budgets
   are preserved for code snippets.
"""

import re
from typing import Tuple, Optional, List
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_factory import get_chat_llm


def extract_file_scope(question: str) -> Tuple[str, Optional[str]]:
    """
    Extracts `@filename` or `@path` tokens from the query.
    Example: "@auth.py how is JWT validated?" -> ("how is JWT validated?", "auth.py")
    """
    match = re.search(r"@([A-Za-z0-9_\-\./\\]+)", question)
    if match:
        file_filter = match.group(1).strip()
        cleaned_question = question.replace(match.group(0), "").strip()
        return cleaned_question or question, file_filter
    return question, None


async def expand_query(query: str) -> List[str]:
    """
    Expands a developer query into 2 alternative technical query representations
    to improve BM25 and dense vector recall (especially for slang or shorthand queries).
    """
    # Don't expand very short or purely symbol queries to avoid unnecessary LLM latency
    if len(query.split()) < 3:
        return [query]

    try:
        llm = get_chat_llm(streaming=False)
        prompt = [
            SystemMessage(
                content="You are a code search query assistant. Given a developer's question about a codebase, "
                "generate 2 distinct, concise technical search phrases or keywords that help find relevant code. "
                "Output each query on a new line. Do not include numbered bullets, preamble, or explanations."
            ),
            HumanMessage(content=f"Query: {query}"),
        ]
        res = await llm.ainvoke(prompt)
        text = res.content if hasattr(res, "content") else str(res)
        variations = [line.strip().lstrip("1234567890.- ") for line in text.strip().split("\n") if line.strip()]
        return [query] + variations[:2]
    except Exception:
        return [query]


def compact_chat_history(chat_history: list[dict], max_turns: int = 6) -> str:
    """
    Compacts chat history into a structured concise format preserving code context.
    """
    if not chat_history:
        return ""

    HISTORY_TRUNCATE = 250
    history_str = ""
    for msg in chat_history[-max_turns:]:
        role = "User" if msg.get("role") == "user" else "CodeSage"
        content = msg.get("content", "")
        if len(content) > HISTORY_TRUNCATE:
            content = content[:HISTORY_TRUNCATE] + "…[truncated]"
        history_str += f"{role}: {content}\n"
    return history_str
