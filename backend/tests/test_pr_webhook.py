"""
test_pr_webhook.py — Test automated GitHub PR webhook review endpoint.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


def test_pr_webhook_rejects_empty_diff(client):
    res = client.post(
        "/api/v1/review/pr-webhook",
        json={"repo": "owner/repo", "pr_number": 42, "title": "Test PR", "diff": "   "},
    )
    assert res.status_code == 400


def test_pr_webhook_valid_diff_returns_structured_review(client):
    mock_review_stream = ["## 📁 File Overview\n", "Looks good!\n"]

    async def fake_stream(*args, **kwargs):
        for token in mock_review_stream:
            yield token

    with patch("app.api.review.stream_code_review", side_effect=fake_stream):
        res = client.post(
            "/api/v1/review/pr-webhook",
            json={
                "repo": "owner/repo",
                "pr_number": 101,
                "title": "Add auth endpoint",
                "diff": "+ def login(): pass",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["pr_number"] == 101
        assert "Looks good!" in data["review"]
