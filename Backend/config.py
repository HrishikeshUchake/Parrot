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

    # --- Neo4j graph + vector store ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    neo4j_vector_index: str = "post_embeddings"
    neo4j_embedding_dim: int = 1024          # intfloat/e5-large-v2 output dim
    neo4j_message_vector_index: str = "message_embeddings"
    neo4j_comment_vector_index: str = "comment_embeddings"

    # --- Embedding model ---
    embedding_model: str = "intfloat/e5-large-v2"
    embedding_batch_size: int = 64
    embedding_cache_size: int = 512          # LRU cache entries

    # --- Ollama / LLM ---
    llm_backend: str = "ollama"              # ollama or openai
    openai_api_key: str = ""
    openai_base_url: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"           # model tag pulled in Ollama
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- Retrieval ---
    default_top_k: int = 5
    advanced_top_k: int = 15
    similarity_threshold: float = 0.40       # cosine similarity floor

    # --- Query cache (SQLite table) ---
    cache_ttl_seconds: int = 3600

    # --- Mastodon ingestion ---
    mastodon_instance_url: str = "https://mastodon.social"
    # optional; leave empty for public timeline
    mastodon_access_token: str = ""
    mastodon_fetch_limit: int = 200          # total statuses to ingest per run
    # max per API page (Mastodon cap = 40)
    mastodon_page_size: int = 40
    # True = only statuses from that instance
    mastodon_local_only: bool = False

    # --- User data ingestion ---
    user_data_import_batch_size: int = 64


settings = Settings()
