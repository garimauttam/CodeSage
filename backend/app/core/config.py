"""
config.py — Single source of truth for all environment variables.

Why Pydantic BaseSettings?
- Reads from .env file automatically
- Validates types at startup (fail fast, not at 3am in production)
- Gives you autocomplete in your IDE

PROVIDER SYSTEM:
  LLM_PROVIDER=openai  → GPT-4o for chat, text-embedding-3-small for embeddings (paid)
  LLM_PROVIDER=gemini  → Gemini 1.5 Flash for chat (free tier), local all-MiniLM-L6-v2 for
                          embeddings (completely free, runs on CPU, no API key needed)

  Switch with a single env var. No code changes required.
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


def _load_json_configs() -> dict:
    """Load configuration from configs.json if present in workspace root or backend dir."""
    possible_paths = [
        Path("configs.json"),
        Path(__file__).resolve().parent.parent.parent / "configs.json",
        Path(__file__).resolve().parent.parent.parent.parent / "configs.json",
    ]
    for p in possible_paths:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


class Settings(BaseSettings):
    # --- LLM Provider ---
    # "openai"  → GPT-4o + text-embedding-3-small  (best quality, paid)
    # "gemini"  → Gemini 1.5 Flash + local MiniLM  (free, good quality for code)
    llm_provider: Literal["openai", "gemini"] = "openai"

    # --- OpenAI (used when llm_provider=openai) ---
    openai_api_key: Optional[str] = None        # Required when llm_provider=openai
    openai_chat_model: str = "gpt-4o"           # Which OpenAI LLM to use
    openai_embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model

    # --- Gemini (used when llm_provider=gemini) ---
    # Free tier: 15 req/min, 1M tokens/day — plenty for dev + demos.
    # Get key at: https://aistudio.google.com/app/apikey  (no credit card)
    gemini_api_key: Optional[str] = None        # Required when llm_provider=gemini
    gemini_chat_model: str = "gemini-3.5-flash-lite" # Lowest-latency free-tier model

    # --- ChromaDB ---
    # We store ChromaDB data on disk so it survives server restarts
    chroma_persist_directory: str = "./chroma_data"
    chroma_collection_name: str = "codesage"

    # --- Chunking strategy ---
    # chunk_size: how many characters per chunk
    # 700 chars fits within the 256 token limit of local MiniLM embeddings without silent truncation,
    # while still providing complete syntax blocks for GPT-4o / Gemini 1.5 Flash.
    chunk_size: int = 700
    chunk_overlap: int = 150   # Overlap prevents cutting a function definition in half

    # --- Retrieval ---
    # How many chunks to pull from ChromaDB per question
    # More = more context for LLM, but also more tokens = more cost
    top_k_results: int = 5

    # --- LangSmith Observability ---
    # LangSmith traces every LangChain call automatically when these vars are set.
    # Without them, nothing changes — tracing is purely opt-in.
    # Set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY to enable.
    # Free tier at smith.langchain.com — no credit card needed.
    langchain_tracing_v2: Optional[str] = None       # "true" to enable
    langchain_api_key: Optional[str] = None           # from smith.langchain.com
    langchain_project: str = "codesage"               # project name in LangSmith UI

    # --- App ---
    # PORT: Railway injects the PORT env var and routes external traffic to it.
    # We read it here so uvicorn can bind to the right port in production.
    # Default 8000 keeps local dev working without any .env change.
    port: int = 8000

    # Accept Union of str or List[str] so pydantic_settings doesn't treat it as complex JSON decode
    cors_origins: str | List[str] = "http://localhost:3000,http://localhost:5173"

    @field_validator("cors_origins", mode="after")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(o).rstrip("/") for o in parsed]
                except Exception:
                    pass
            return [origin.strip().rstrip("/") for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return [origin.rstrip("/") if isinstance(origin, str) else origin for origin in v]
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# lru_cache means this is only created once — the same Settings object
# is reused across all requests (singleton pattern)
@lru_cache()
def get_settings() -> Settings:
    json_config = _load_json_configs()
    if json_config:
        return Settings(**json_config)
    return Settings()
