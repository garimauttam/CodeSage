"""
review_agent.py — Agentic code review using LangChain tools + GPT-4o.

WHAT MAKES THIS "AGENTIC"?
A regular LLM call is: prompt → one answer.
An agent is: prompt → LLM decides what tools to call → calls them → sees results →
             decides next step → ... → synthesises final answer.

The LLM has AUTONOMY. It decides:
  - Which sections of code to examine
  - What patterns to search for (e.g., "find all SQL queries")
  - When it has enough information to write the review

WHY IS THIS INTERVIEW-IMPRESSIVE?
Most RAG demos are passive: ask → retrieve → answer.
Agents are active: they plan, investigate, and reason across multiple steps.
This is the direction the entire industry is moving (AutoGPT, Devin, Copilot Workspace).

ARCHITECTURE:
  User requests review of file.py
       ↓
  Agent receives: file contents + tool definitions
       ↓
  Agent calls tools in a reasoning loop (ReAct pattern):
    Thought: "I should check for security issues first"
    Action: search_pattern("SQL", file_content)
    Observation: [found raw SQL strings]
    Thought: "Found potential SQL injection. Now check error handling."
    Action: search_pattern("except:", file_content)
    ...
       ↓
  Agent writes structured review with all findings
"""

import asyncio
import json
import re
import time
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.core.config import get_settings
from app.services.llm_factory import get_chat_llm
from app.services.token_counter import get_token_callback, increment_request

settings = get_settings()


# ── Tool factory ──────────────────────────────────────────────────────────────
# WHY A FACTORY INSTEAD OF MODULE-LEVEL @tool FUNCTIONS?
#
# Original design had tools take `code: str` as a parameter, meaning the LLM
# had to re-send the entire file content on every tool call. For a 500-line file
# that's ~3000 tokens wasted PER tool call, and it also gave the LLM an
# opportunity to accidentally truncate or modify the code before passing it.
#
# Fix: capture `file_content` in a closure. Tools take only what they actually
# need from the LLM (e.g., just a `pattern: str`). The file content is already
# in scope — the LLM never touches it.
#
# This is the "context injection" pattern used in production tool-use systems.

def _make_tools(file_content: str):
    """Create tool instances bound to a specific file's content."""

    @tool
    def search_pattern(pattern: str) -> str:
        """
        Search for a regex pattern in the code. Use this to find specific constructs
        like SQL queries, hardcoded secrets, bare except clauses, print statements, etc.
        Returns all matching lines with their line numbers.
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex pattern '{pattern}': {e}"

        matches = []
        for i, line in enumerate(file_content.split("\n"), 1):
            if compiled.search(line):
                matches.append(f"Line {i}: {line.rstrip()}")
        if not matches:
            return f"No matches found for pattern: {pattern}"
        return "\n".join(matches[:30])  # Cap at 30 matches to avoid flooding the context

    @tool
    def get_function_list() -> str:
        """
        Extract all function, method, and class definitions from the code.
        Use this to understand the structure of the file before reviewing it.
        Works for Python, JavaScript, TypeScript, Go, Java, Rust, and Ruby.
        Returns definition names with their line numbers.
        """
        # Language-aware patterns — previously only matched Python syntax,
        # silently returning "nothing found" for JS/Go/Java files.
        # The agent then had no structural understanding of non-Python files.
        DEFINITION_PATTERNS = [
            # Python
            r"^\s*(async\s+def|def|class)\s+\w",
            # JavaScript / TypeScript: function declarations and arrow functions
            r"^\s*(export\s+)?(async\s+)?function\s+\w",
            r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s*)?\(",
            r"^\s*(export\s+)?(default\s+)?class\s+\w",
            # Go
            r"^\s*func\s+(\(\w+\s+\*?\w+\)\s+)?\w+\s*\(",
            # Java / C# / Kotlin
            r"^\s*(public|private|protected|static|override|abstract).*\s+\w+\s*\(",
            # Rust
            r"^\s*(pub(\(.*\))?\s+)?(async\s+)?fn\s+\w",
            # Ruby
            r"^\s*def\s+\w",
        ]
        compiled = [re.compile(p) for p in DEFINITION_PATTERNS]

        definitions = []
        for i, line in enumerate(file_content.split("\n"), 1):
            if any(pat.search(line) for pat in compiled):
                # Trim to first 80 chars so definitions don't flood the context
                definitions.append(f"Line {i}: {line.rstrip()[:80]}")
        if not definitions:
            return "No function or class definitions found."
        return "\n".join(definitions)

    @tool
    def count_complexity_indicators() -> str:
        """
        Count cyclomatic complexity indicators: nested loops, deeply nested conditions,
        long functions. Use this to flag overly complex code that should be refactored.
        Returns a summary of complexity metrics.
        """
        lines = file_content.split("\n")
        metrics = {
            "total_lines": len(lines),
            "nested_loops": 0,
            "bare_excepts": 0,
            "long_lines": 0,
            "todo_comments": 0,
            "magic_numbers": 0,
        }

        # Track nesting depth properly using indentation level, not a single boolean.
        # The old `in_loop = True` approach never reset, so every loop after the first
        # was counted as nested — even loops in completely separate functions.
        loop_indent_stack: list[int] = []
        for line in lines:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            # Pop any loops that are now out of scope (de-indented past them)
            while loop_indent_stack and indent <= loop_indent_stack[-1]:
                loop_indent_stack.pop()

            if re.match(r"^(for|while)\b", stripped):
                if loop_indent_stack:  # there's already an enclosing loop
                    metrics["nested_loops"] += 1
                loop_indent_stack.append(indent)

            if stripped == "except:":
                metrics["bare_excepts"] += 1
            if len(line) > 120:
                metrics["long_lines"] += 1
            if "TODO" in line.upper() or "FIXME" in line.upper() or "HACK" in line.upper():
                metrics["todo_comments"] += 1
            if re.search(r"(?<![=\w])\b[0-9]{2,}\b(?!\s*[=\w])", stripped):
                metrics["magic_numbers"] += 1

        return json.dumps(metrics, indent=2)

    return [search_pattern, get_function_list, count_complexity_indicators]


# ── Agent system prompt ───────────────────────────────────────────────────────
REVIEW_SYSTEM_PROMPT = """You are CodeSage, a senior software engineer performing a thorough code review.

You have access to these tools to investigate the code:
- search_pattern: find specific patterns (SQL, secrets, error handling, etc.)
- get_function_list: understand the file's structure
- count_complexity_indicators: measure code complexity metrics

REVIEW PROCESS:
1. First call get_function_list to understand the structure
2. Call count_complexity_indicators to get metrics
3. Search for common issues: hardcoded secrets, SQL injection risks, bare excepts,
   missing input validation, TODO/FIXME comments
4. Search for patterns relevant to what the code does

After your investigation, write a structured review with these sections:

## 📁 File Overview
Brief description of what the file does, based on its structure.

## 🐛 Bugs & Critical Issues
Concrete bugs found with line numbers. If none, say so.

## 🔒 Security Concerns
Any potential security vulnerabilities.

## ⚠️ Code Quality Issues
Complexity, duplication, poor naming, missing error handling.

## 💡 Suggestions for Improvement
Actionable refactoring suggestions with example code.

## ✅ What's Done Well
Positive observations — a good review is balanced.

## 📊 Summary Score
Rate the code 1-10 on: Correctness, Security, Readability, Maintainability.

Be specific. Cite line numbers. Show fixed code in code blocks.
"""


async def stream_code_review(
    file_name: str,
    file_content: str,
    language: str = "",
) -> AsyncGenerator[str, None]:
    """
    Run an agentic code review using a ReAct (Reason + Act) loop.

    HOW THE ReAct LOOP WORKS:
    1. We give the LLM the code + tool definitions
    2. The LLM responds with a "tool call" (it wants to run a tool)
    3. We execute the tool, get the result
    4. We feed the result back to the LLM
    5. LLM either calls another tool OR writes the final review
    6. Repeat until the LLM stops calling tools

    WHY NOT JUST SEND THE WHOLE FILE AND ASK FOR A REVIEW?
    We could. But the agent approach is better because:
    - The LLM actively investigates rather than passively reading
    - Tools constrain what the LLM can "hallucinate" — findings are real
    - It demonstrates agentic architecture, which is interview gold
    - The reasoning trace (Thought → Action → Observation) is visible
    """

    # Two LLM instances are needed:
    # 1. llm           — non-streaming, used for tool-calling phase (needs complete JSON)
    # 2. streaming_llm — streaming, used only for the final review write-up
    #
    # We call get_chat_llm() which returns the cached base instance for the
    # active provider (OpenAI or Gemini). bind_tools() is called below after
    # the closure-bound tools are created — we can't cache a tool-bound LLM
    # because tools are different for each file.
    increment_request("review")
    llm = get_chat_llm(streaming=False)
    streaming_llm = get_chat_llm(streaming=True)

    # Create tools bound to this file's content (closure pattern — no token waste)
    tools = _make_tools(file_content)

    # Build a tool lookup dict for fast dispatch
    tool_map = {t.name: t for t in tools}

    # ── Initial message ───────────────────────────────────────────────────────
    preview = file_content[:8000]
    truncation_note = f"\n[File truncated in preview — {len(file_content)} chars total.]" if len(file_content) > 8000 else ""
    initial_user_msg = f"""Please review this {language} file: `{file_name}`

```{language}
{preview}
```{truncation_note}

Investigate the code systematically and write a thorough, structured code review following your instructions."""

    messages = [
        SystemMessage(content=REVIEW_SYSTEM_PROMPT),
        HumanMessage(content=initial_user_msg),
    ]

    # ── Yield initial status ──────────────────────────────────────────────────
    status_meta = json.dumps({"step": "starting"})
    yield f"__STATUS__Analyzing `{file_name}`...{status_meta}__STATUS_END__\n"

    # For OpenAI or providers with native tool calling support:
    if settings.llm_provider != "gemini":
        llm_with_tools = llm.bind_tools(tools)
        max_iterations = 8
        max_seconds = 90
        loop_start = time.monotonic()
        iteration = 0

        try:
            while iteration < max_iterations:
                iteration += 1

                if time.monotonic() - loop_start > max_seconds:
                    yield f"\n\n*Review timed out after {max_seconds}s — partial investigation complete.*"
                    return

                response = await llm_with_tools.with_config(callbacks=[get_token_callback()]).ainvoke(messages)
                messages.append(response)

                if response.tool_calls:
                    for tc in response.tool_calls:
                        tool_name = tc["name"]
                        tool_args = tc["args"]

                        tool_meta = json.dumps({"step": "tool", "tool": tool_name})
                        yield f"__STATUS__Running tool: `{tool_name}`...{tool_meta}__STATUS_END__\n"

                        tool_fn = tool_map.get(tool_name)
                        if tool_fn:
                            try:
                                tool_result = await asyncio.to_thread(tool_fn.invoke, tool_args)
                            except Exception as e:
                                tool_result = f"Tool error: {str(e)}"
                        else:
                            tool_result = f"Unknown tool: {tool_name}"

                        messages.append(
                            ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
                        )
                else:
                    writing_meta = json.dumps({"step": "writing"})
                    yield f"__STATUS__Writing review...{writing_meta}__STATUS_END__\n"
                    messages.append(
                        HumanMessage(
                            content="Based on your investigation above, now write the full "
                                    "structured code review following the format in your instructions."
                        )
                    )
                    async for chunk in streaming_llm.with_config(callbacks=[get_token_callback()]).astream(messages):
                        if chunk.content:
                            yield chunk.content
                    return
        except Exception as e:
            err_msg = str(e)[:200]
            yield f"__ERROR__{err_msg}__ERROR_END__\n"
            return

    else:
        # For Gemini provider: run deterministic structural & complexity tools directly,
        # then stream the comprehensive review without multi-turn RPC thought_signature mismatch
        try:
            tool_meta = json.dumps({"step": "tool", "tool": "get_function_list"})
            yield f"__STATUS__Running tool: `get_function_list`...{tool_meta}__STATUS_END__\n"
            fn_list = tool_map["get_function_list"].invoke({})

            tool_meta2 = json.dumps({"step": "tool", "tool": "count_complexity_indicators"})
            yield f"__STATUS__Running tool: `count_complexity_indicators`...{tool_meta2}__STATUS_END__\n"
            complexity = tool_map["count_complexity_indicators"].invoke({})

            gemini_prompt = [
                SystemMessage(content=REVIEW_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""Please perform a thorough code review for `{file_name}` ({language}).

### Code:
```{language}
{preview}
```{truncation_note}

### Automated Static Inspection Results:
- Function & Class Definitions:
{fn_list}

- Complexity Metrics:
{complexity}

Now write a comprehensive, balanced review following all headings in your instructions."""
                )
            ]

            writing_meta = json.dumps({"step": "writing"})
            yield f"__STATUS__Writing review...{writing_meta}__STATUS_END__\n"

            async for chunk in streaming_llm.with_config(callbacks=[get_token_callback()]).astream(gemini_prompt):
                if chunk.content:
                    yield chunk.content
            return
        except Exception as e:
            err_msg = str(e)[:200]
            yield f"__ERROR__{err_msg}__ERROR_END__\n"
            return

    # Fallback: if we hit max_iterations without a final answer
    yield "\n\n*Review incomplete — agent reached maximum iterations. Try a smaller file.*"
