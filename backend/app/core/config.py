"""
config.py — Single source of truth for all environment variables.

Why Pydantic BaseSettings?
- Reads from .env file automatically
- Validates types at startup (fail fast, not at 3am in production)
- Gives you autocomplete in your IDE
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- OpenAI ---
    openai_api_key: str                          # Required — no default, will crash if missing
    openai_chat_model: str = "gpt-4o"           # Which LLM to use for answers
    openai_embedding_model: str = "text-embedding-3-small"  # Cheaper + fast, 1536 dims

    # --- ChromaDB ---
    # We store ChromaDB data on disk so it survives server restarts
    chroma_persist_directory: str = "./chroma_data"
    chroma_collection_name: str = "codesage"

    # --- Chunking strategy ---
    # chunk_size: how many characters per chunk
    # Too large → LLM gets too much noise, hits token limits
    # Too small → chunks lose context (a function split across 3 chunks makes no sense)
    chunk_size: int = 1000
    chunk_overlap: int = 200   # Overlap prevents cutting a function definition in half

    # --- Retrieval ---
    # How many chunks to pull from ChromaDB per question
    # More = more context for LLM, but also more tokens = more cost
    top_k_results: int = 5

    # --- App ---
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# lru_cache means this is only created once — the same Settings object
# is reused across all requests (singleton pattern)
@lru_cache()
def get_settings() -> Settings:
    return Settings()
