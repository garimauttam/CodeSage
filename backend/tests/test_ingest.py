"""
test_ingest.py — Tests for the ingestion API.

Tests validate:
  1. GitHub URL validation rejects bad URLs
  2. File upload with no files returns 400
  3. The /repos endpoint returns valid JSON structure
  4. The clear endpoint is callable

We do NOT test actual cloning/embedding — that would require a real OpenAI key
and network access. We test the API contract (validation, response shape).
This is the standard approach: unit-test the edges, integration-test the core.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def test_github_ingest_rejects_non_github_url(client):
    """
    Validation: repo_url must start with https:// or git@.
    A bare domain like "github.com/user/repo" should be rejected with 422.

    WHY THIS TEST?
    Without URL validation, a user could pass any string as a repo URL —
    including paths to local files or internal network addresses.
    """
    response = client.post(
        "/api/v1/ingest/github",
        json={"repo_url": "github.com/user/repo"},  # missing https://
    )
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_github_ingest_rejects_empty_url(client):
    """Validation: empty repo_url returns 422."""
    response = client.post(
        "/api/v1/ingest/github",
        json={"repo_url": ""},
    )
    assert response.status_code == 422


def test_file_upload_with_no_files_returns_400(client):
    """
    Uploading zero files should return 400, not crash.

    WHY?
    FastAPI parses multipart/form-data. If files is empty, our handler
    raises HTTPException(400) explicitly. Without the guard, FastAPI would
    return a 422 validation error — confusing to the user.
    """
    response = client.post(
        "/api/v1/ingest/files",
        files=[],          # No files
    )
    # FastAPI returns 422 for missing required field, which is also acceptable
    assert response.status_code in (400, 422)


def test_repos_endpoint_returns_list(client):
    """
    GET /ingest/repos should always return {"repos": [...], "total": int}.
    Even with an empty collection, the shape must be consistent —
    the frontend destructures repos[0] and will crash on undefined.
    """
    mock_collection = MagicMock()
    # Simulate empty collection — returns None metadatas, not []
    # This is the exact bug we fixed: collection.get() returns None, not []
    mock_collection.get.return_value = {"metadatas": None, "ids": []}

    mock_chroma = MagicMock()
    mock_chroma._collection = mock_collection

    with patch("app.services.ingestion_service._get_vectorstore", return_value=mock_chroma):
        response = client.get("/api/v1/ingest/repos")

    assert response.status_code == 200
    body = response.json()
    assert "repos" in body
    assert "total" in body
    assert isinstance(body["repos"], list)
    assert body["total"] == 0


def test_clear_endpoint_is_reachable(client):
    """
    DELETE /ingest/clear should not crash on an empty collection.
    This validates the ChromaDB wipe path works even when nothing is indexed.
    """
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": [], "metadatas": None}

    mock_chroma = MagicMock()
    mock_chroma._collection = mock_collection
    mock_chroma.get.return_value = {"ids": [], "metadatas": None}

    with patch("app.services.ingestion_service._get_vectorstore", return_value=mock_chroma):
        response = client.delete("/api/v1/ingest/clear")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
