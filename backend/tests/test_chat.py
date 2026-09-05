"""
test_chat.py — Tests for the chat/stream API.

Tests verify:
  1. Empty question returns 422 (not a crash)
  2. Oversized question (>2000 chars) returns 422
  3. Valid question triggers the RAG pipeline and streams a response
  4. Response contains the __SOURCES__ marker (citation format)
"""

import pytest
from unittest.mock import patch


def test_chat_rejects_empty_question(client):
    """
    An empty question must return 422, not a 500 crash or an empty stream.
    The @field_validator on ChatRequest enforces this.
    """
    response = client.post(
        "/api/v1/chat/stream",
        json={"question": "", "chat_history": []},
    )
    assert response.status_code == 422


def test_chat_rejects_question_over_2000_chars(client):
    """
    Questions over 2000 characters are rejected with 422.

    WHY A CHARACTER LIMIT?
    A 5000-char question ≈ 1250 tokens — before any context is added.
    Combine that with 5 retrieved chunks (~2500 tokens) and the user is already
    at 3750 tokens before the LLM writes a single word. The limit prevents
    accidental (or malicious) token inflation.
    """
    long_question = "a" * 2001
    response = client.post(
        "/api/v1/chat/stream",
        json={"question": long_question, "chat_history": []},
    )
    assert response.status_code == 422


def test_chat_valid_question_streams_with_sources(client):
    """
    Happy path: a valid question returns a streaming response that:
      1. Has status 200
      2. Contains the __SOURCES__ marker (the frontend reads this to render citations)
      3. Contains actual answer text after the sources marker

    We mock stream_answer to avoid OpenAI + ChromaDB calls.
    The mock returns the exact format the real function would produce.
    """
    import json

    sources = [{"file_name": "auth.py", "source": "/tmp/auth.py", "language": "py"}]

    async def fake_stream(*args, **kwargs):
        # Exact format from retrieval_service.stream_answer
        yield f"__SOURCES__{json.dumps(sources)}__SOURCES_END__\n"
        yield "The auth module handles JWT token validation.\n"

    with patch(
        "app.api.chat.stream_answer",
        side_effect=fake_stream,
    ):
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "question": "What does the auth module do?",
                "chat_history": [],
            },
        )

    assert response.status_code == 200
    body = response.text
    assert "__SOURCES__" in body
    assert "__SOURCES_END__" in body
    assert "auth module" in body


def test_indexed_files_returns_valid_shape(client):
    """
    GET /chat/indexed-files always returns {"files": [...], "total": int}.
    Even with an empty collection, shape must be consistent — the frontend
    does files.length and will crash on undefined.
    """
    from unittest.mock import MagicMock

    mock_chroma = MagicMock()
    mock_chroma._collection.get.return_value = {"metadatas": None}

    with patch("app.services.retrieval_service._get_vectorstore", return_value=mock_chroma):
        response = client.get("/api/v1/chat/indexed-files")

    assert response.status_code == 200
    body = response.json()
    assert "files" in body
    assert "total" in body
    assert isinstance(body["files"], list)
