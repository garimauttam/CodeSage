"""
test_review.py — Tests for the code review API.

Covers the two most critical security + correctness properties:
  1. Path traversal guard — /etc/passwd must be rejected
  2. Empty paste returns 400, not 500
  3. Oversized paste returns 413
  4. Valid paste starts streaming (SSE format check)
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def test_review_file_rejects_path_traversal(client):
    """
    SECURITY: file_path=/etc/passwd must return 422 (validation error),
    not 200 (file contents) or 500 (crash).

    This is the path traversal vulnerability we explicitly fixed.
    Without the allowed-roots validator, this would return the contents
    of any file accessible to the server process.
    """
    response = client.post(
        "/api/v1/review/file",
        json={
            "file_path": "/etc/passwd",
            "file_name": "passwd",
            "language": "",
        },
    )
    # 422 = Pydantic validation rejected the path
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_review_paste_empty_code_returns_400(client):
    """
    Empty code paste returns 400 with a clear error message.
    Without this guard, stream_code_review would receive empty content
    and the LLM would write a review of nothing.
    """
    response = client.post(
        "/api/v1/review/paste",
        json={"code": "   ", "file_name": "test.py", "language": "python"},
    )
    assert response.status_code == 400
    assert "No code" in response.json()["detail"]


def test_review_paste_oversized_code_returns_413(client):
    """
    Pasting 50,001 characters should return 413 (Payload Too Large).

    WHY THIS LIMIT?
    50k chars ≈ 12k tokens. GPT-4o's context window is 128k tokens, but
    the review prompt + tool results easily consume another 8k. Setting a
    hard limit here prevents accidental 40k-token requests that cost $0.60 each.
    """
    oversized = "x" * 50_001
    response = client.post(
        "/api/v1/review/paste",
        json={"code": oversized, "file_name": "big.py", "language": "python"},
    )
    assert response.status_code == 413


def test_review_paste_valid_code_streams(client):
    """
    Happy path: valid code returns a streaming response.
    We don't check the exact content (that's the LLM's job) —
    we check that the response starts and contains our STATUS marker format.

    WHY MOCK stream_code_review?
    We're testing the API contract (does it stream? does it set the right headers?),
    not the review quality. Calling the real function requires an OpenAI key.
    """
    async def fake_stream(*args, **kwargs):
        yield '__STATUS__Analyzing `test.py`...{"step": "starting"}__STATUS_END__\n'
        yield "## File Overview\nThis is a test file.\n"

    with patch(
        "app.api.review.stream_code_review",
        side_effect=fake_stream,
    ):
        response = client.post(
            "/api/v1/review/paste",
            json={"code": "def hello(): pass", "file_name": "test.py", "language": "python"},
        )

    assert response.status_code == 200
    # Streaming response should have text/plain content type
    assert "text/plain" in response.headers.get("content-type", "")
    body = response.text
    assert "__STATUS__" in body
    assert "File Overview" in body
