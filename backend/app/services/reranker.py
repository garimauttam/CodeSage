"""
reranker.py — Cross-encoder re-ranking of retrieved chunks.

WHY TWO-STAGE RETRIEVAL?
Stage 1 (embedding search): Fast. Embeds query once → cosine similarity over all vectors.
  Problem: embeddings are "compressed summaries" — they lose nuance.
  A chunk about "password validation" and one about "token validation" look similar
  in embedding space even if only one answers your question.

Stage 2 (cross-encoder re-ranking): Accurate. Takes the question AND each chunk together
  as a single input, producing a relevance score. It reads both simultaneously,
  so it understands the relationship between them — much more precise.
  Problem: slow. Can't do this over 100,000 chunks. But over 15 chunks? ~50ms. Fine.

The pipeline is: fetch 15 candidates (MMR) → re-rank → keep top 5.
This gives you the speed of embedding search with the accuracy of cross-encoders.

INTERVIEW TALKING POINT:
"I implemented a two-stage retrieval pipeline: first MMR for diversity, then a
local cross-encoder (ms-marco-MiniLM) for precision. This improved answer quality
noticeably on ambiguous questions without adding any API cost."
"""

import asyncio
from functools import lru_cache
from langchain_core.documents import Document

# We import lazily inside the function to avoid loading the 80MB model
# at import time (which would slow down every cold start, even for requests
# that don't need re-ranking)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """
    Load the cross-encoder model once and cache it.
    lru_cache(maxsize=1) = singleton pattern for the model.
    The model is loaded from HuggingFace Hub on first call, then cached on disk.

    ms-marco-MiniLM-L-6-v2:
    - Trained on MS MARCO (Microsoft MAchine Reading COmprehension) — a massive
      passage-retrieval dataset. Perfect for code Q&A.
    - 6 transformer layers → fast enough for real-time re-ranking
    - ~80MB — small enough to include in a Docker image
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


async def rerank(
    query: str,
    documents: list[Document],
    top_n: int = 5,
) -> list[Document]:
    """
    Re-rank a list of documents by relevance to the query.
    Returns the top_n most relevant documents, sorted by score descending.

    We run it in a thread because CrossEncoder.predict() is synchronous CPU work.
    Running it directly in the async event loop would block all other requests.
    asyncio.to_thread() moves it to a thread pool — event loop stays free.
    """
    if len(documents) <= top_n:
        # Not enough docs to bother re-ranking — return as-is
        return documents

    def _score():
        cross_encoder = _get_cross_encoder()
        # CrossEncoder expects a list of (query, passage) pairs
        pairs = [(query, doc.page_content) for doc in documents]
        scores = cross_encoder.predict(pairs)
        # Zip scores with docs, sort by score descending, return top_n docs
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]

    return await asyncio.to_thread(_score)
