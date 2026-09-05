"""
test_health.py — Tests for the /health endpoint.

The health endpoint is the most important endpoint:
  - Railway uses it to decide if your instance is healthy
  - It's the first thing you check when a deploy fails
  - It validates that the configured LLM provider and ChromaDB are reachable

These tests verify that:
  1. A healthy system returns 200 with status "ok"
  2. A degraded system (bad API key) returns 503 with status "degraded"

WHY "llm" NOT "openai"?
After the LLM_PROVIDER refactor the health endpoint uses a provider-neutral
"llm" key so the same check works for both OpenAI and Gemini providers.
"""

from unittest.mock import AsyncMock, patch, MagicMock
import pytest


def test_health_returns_200_when_all_checks_pass(client):
    """
    Happy path: both the LLM provider (OpenAI by default) and ChromaDB respond.
    Expected: HTTP 200, status "ok", checks["llm"] starts with "ok".
    """
    mock_openai_client = MagicMock()
    mock_openai_client.models.list = AsyncMock(return_value=[])

    mock_chroma_client = MagicMock()
    mock_chroma_client.heartbeat.return_value = True

    with patch("openai.AsyncOpenAI", return_value=mock_openai_client), \
         patch("chromadb.PersistentClient", return_value=mock_chroma_client):

        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # Provider-neutral key — works for both openai and gemini providers
    assert body["checks"]["llm"].startswith("ok")
    assert body["checks"]["chromadb"] == "ok"


def test_health_returns_503_when_openai_fails(client):
    """
    Degraded path: OpenAI throws an AuthenticationError (wrong API key).
    Expected: HTTP 503, status "degraded", checks["llm"] contains "error".

    WHY THIS MATTERS:
    A shallow health check (return {"status": "ok"}) would return 200 here.
    Our deep check catches the bad API key and returns 503, so Railway
    won't route traffic to a misconfigured instance.
    """
    import openai
    from app.core.config import Settings

    fake_openai_settings = Settings(llm_provider="openai", openai_api_key="sk-invalid")

    mock_openai_client = MagicMock()
    mock_openai_client.models.list = AsyncMock(
        side_effect=openai.AuthenticationError(
            "Incorrect API key", response=MagicMock(), body={}
        )
    )

    mock_chroma_client = MagicMock()
    mock_chroma_client.heartbeat.return_value = True

    with patch("main.settings", fake_openai_settings), \
         patch("openai.AsyncOpenAI", return_value=mock_openai_client), \
         patch("chromadb.PersistentClient", return_value=mock_chroma_client):

        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    # Provider-neutral key — was "openai" before the LLM_PROVIDER refactor
    assert "error" in body["checks"]["llm"]
    assert body["checks"]["chromadb"] == "ok"
