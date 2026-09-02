"""
metrics.py — API route for token usage and request metrics.

GET /api/v1/metrics  →  returns current session token counts + request breakdown.

WHY IS THIS USEFUL FOR A RESUME PROJECT?
It demonstrates awareness of LLM cost management — a real production concern.
Every token costs money; every serious AI product tracks usage.
The dashboard shows interviewers you think beyond "just make it work" to
"make it observable, cost-aware, and production-ready".
"""

from fastapi import APIRouter
from app.services.token_counter import get_totals, reset_totals

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def get_metrics():
    """
    Returns session-level token usage and request counts.
    Counters reset on server restart (in-memory, no persistence needed for dev).
    """
    return get_totals()


@router.delete("")
async def clear_metrics():
    """Reset all counters to zero. Useful for testing or starting a fresh session."""
    reset_totals()
    return {"status": "cleared"}
