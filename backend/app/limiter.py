"""
limiter.py — Single shared rate limiter instance.

WHY A SEPARATE MODULE?
main.py creates the FastAPI app and imports routers (chat, review).
Those routers need the limiter for @limiter.limit() decorators.
If chat.py imported from main.py it would be a circular import.

Solution: put the limiter in its own module that neither main.py nor
the routers depend on transitively. Both import from here instead.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# One instance, shared across all routes.
# Each Limiter has its own in-memory counter store — if you create two,
# they track requests independently and rate limits can be bypassed by
# alternating between endpoints.
limiter = Limiter(key_func=get_remote_address)
