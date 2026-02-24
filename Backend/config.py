"""
Central configuration for the Parrot Agentic RAG system.
All settings are read from environment variables with sensible defaults.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    db_path: str = str(BASE_DIR / "store_social_data.db")

    # --- Neo4j graph + vector store ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    neo4j_vector_index: str = "post_embeddings"
    neo4j_embedding_dim: int = 384          # all-MiniLM-L6-v2 output dim

    # --- Embedding model ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_cache_size: int = 512          # LRU cache entries

    # --- Ollama / LLM ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"           # model tag pulled in Ollama
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- Retrieval ---
    default_top_k: int = 5
    advanced_top_k: int = 15
    similarity_threshold: float = 0.30       # cosine similarity floor

    # --- Query cache (SQLite table) ---
    cache_ttl_seconds: int = 3600


settings = Settings()
