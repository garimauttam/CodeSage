"""
llm_factory.py — Builds LLM and embedding instances based on LLM_PROVIDER.

WHY A FACTORY?
Every service (retrieval, review agent) needs an LLM. Without this, switching
providers means finding every ChatOpenAI() call across multiple files.
With this factory, the entire provider swap is one env var + this one file.

SUPPORTED PROVIDERS:
  openai  → ChatOpenAI (GPT-4o) + OpenAIEmbeddings (text-embedding-3-small)
            Requires: OPENAI_API_KEY
            Cost: ~$0.01–0.05 per question

  gemini  → ChatGoogleGenerativeAI (Gemini 1.5 Flash) + local HuggingFaceEmbeddings
            Requires: GEMINI_API_KEY (free at aistudio.google.com — no credit card)
            Cost: FREE (15 req/min, 1M tokens/day on free tier)
            Embeddings: all-MiniLM-L6-v2 runs on CPU, no API, no cost

ADDING A NEW PROVIDER (e.g. Ollama for fully local):
  1. Add "ollama" to the Literal type in config.py
  2. Add an elif branch here
  3. No other files need to change
"""

from functools import lru_cache
from typing import Any

from app.core.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_chat_llm(streaming: bool = False) -> Any:
    """
    Returns a cached chat LLM instance for the configured provider.

    streaming=True  → used by retrieval_service for token-by-token output
    streaming=False → used by review_agent during the tool-calling ReAct loop
                      (needs complete JSON, not a stream)

    WHY lru_cache?
    LLM clients are stateless — safe to reuse across requests.
    Caching avoids rebuilding the HTTP client + re-validating the API key
    on every user question. The cache key includes `streaming` so we get
    two separate cached instances (one for each mode).
    """
    if settings.llm_provider == "gemini":
        _require_key("GEMINI_API_KEY", settings.gemini_api_key)
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
            streaming=streaming,
            max_retries=3,
            # convert_system_message_to_human is deprecated in newer langchain-google-genai.
            # The new way is to pass a system instruction string via the `model_kwargs` or
            # simply remove the flag — recent versions handle SystemMessage natively.
        )
    else:
        # Default: OpenAI
        _require_key("OPENAI_API_KEY", settings.openai_api_key)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
            streaming=streaming,
        )


@lru_cache(maxsize=1)
def get_embedding_fn() -> Any:
    """
    Returns a cached embedding function for the configured provider.

    IMPORTANT: Whatever model indexes the data MUST be the same model used
    to query it. Mixing embedding models = garbage retrieval results.
    If you switch providers, you MUST re-index (clear + re-ingest).

    gemini provider uses local HuggingFace embeddings (all-MiniLM-L6-v2):
    - 384 dimensions (vs 1536 for OpenAI) — smaller, faster, still very good
    - Runs on CPU — no GPU needed, no API call, no cost
    - sentence-transformers is already in requirements.txt (for the reranker)
    - Downloads ~90MB once, cached at ~/.cache/huggingface/
    """
    if settings.llm_provider == "gemini":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        # all-MiniLM-L6-v2: trained on 1B sentence pairs, excellent semantic similarity
        # Good enough for code retrieval — the reranker compensates for any precision loss
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            # batch_size=1: query encoding is always a single sentence.
            # Default batch_size=32 pads a 1-item batch to 32 slots — wasted CPU.
            encode_kwargs={"normalize_embeddings": True, "batch_size": 1},
        )
    else:
        _require_key("OPENAI_API_KEY", settings.openai_api_key)
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )


def get_provider_name() -> str:
    """Human-readable provider name for logging and health checks."""
    if settings.llm_provider == "gemini":
        return f"Gemini ({settings.gemini_chat_model}) + local MiniLM embeddings"
    return f"OpenAI ({settings.openai_chat_model}) + {settings.openai_embedding_model}"


def _require_key(name: str, value) -> None:
    """Fail fast at startup if a required API key is missing."""
    if not value:
        raise ValueError(
            f"{name} is required when LLM_PROVIDER={settings.llm_provider}. "
            f"Set it in your .env file."
        )
