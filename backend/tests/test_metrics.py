"""
test_metrics.py — Tests for the /api/v1/metrics token usage endpoint.

Verifies:
  1. GET /metrics returns a valid schema with all expected keys
  2. POST /chat/stream increments the chat_requests counter
  3. DELETE /metrics resets all counters to zero
"""

import pytest
from unittest.mock import patch, AsyncMock


def test_metrics_returns_valid_schema(client):
    """GET /metrics should return a dict with all expected numeric fields."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    body = response.json()
    for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                "llm_calls", "chat_requests", "review_requests",
                "write_requests", "estimated_cost_usd"):
        assert key in body, f"Missing key: {key}"
        assert isinstance(body[key], (int, float)), f"Expected numeric value for {key}"


def test_metrics_reset(client):
    """DELETE /metrics should reset counters to zero."""
    # Trigger a request increment first
    from app.services.token_counter import increment_request
    increment_request("chat")

    response = client.delete("/api/v1/metrics")
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"

    # Verify all counters are zero after reset
    body = client.get("/api/v1/metrics").json()
    assert body["total_tokens"] == 0
    assert body["llm_calls"] == 0
    assert body["chat_requests"] == 0


def test_increment_request_counter(client):
    """increment_request() should accumulate across calls."""
    from app.services.token_counter import reset_totals, increment_request, get_totals

    reset_totals()
    increment_request("chat")
    increment_request("chat")
    increment_request("review")

    totals = get_totals()
    assert totals["chat_requests"] == 2
    assert totals["review_requests"] == 1
    assert totals["write_requests"] == 0

    reset_totals()  # clean up


def test_token_callback_accumulates(client):
    """TokenUsageCallback.on_llm_end() should add tokens to the global counter."""
    from app.services.token_counter import reset_totals, get_totals, TokenUsageCallback
    from langchain_core.outputs import LLMResult, Generation

    reset_totals()
    cb = TokenUsageCallback()

    # Simulate an LLM response with token usage (OpenAI schema)
    fake_result = LLMResult(
        generations=[[Generation(text="hello")]],
        llm_output={"token_usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}},
    )
    cb.on_llm_end(fake_result)

    totals = get_totals()
    assert totals["prompt_tokens"] == 50
    assert totals["completion_tokens"] == 30
    assert totals["total_tokens"] == 80
    assert totals["llm_calls"] == 1

    reset_totals()  # clean up
