"""
Central configuration for the Parrot Agentic RAG system.
All settings are read from environment variables with sensible defaults.
"""
import logging
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Suppress Neo4j driver schema warnings for missing nodes/properties natively handled gracefully
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

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
    neo4j_thread_chunk_vector_index: str = "thread_chunk_embeddings"

    # --- Embedding model ---
    embedding_model: str = "intfloat/e5-large-v2"
    embedding_batch_size: int = 64
    embedding_cache_size: int = 512          # LRU cache entries

    # --- Ollama / LLM ---
    llm_backend: str = "ollama"              # ollama or openai
    analyzer_llm_backend: str = "ollama"     # query analyzer node backend
    router_llm_backend: str = "ollama"       # router node backend
    synthesis_llm_backend: str = "ollama"    # synthesis node backend
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"           # model tag pulled in Ollama

    # --- OpenRouter / Remote LLM ---
    openrouter_api_key: str = ""             # Set OPENROUTER_API_KEY env var
    # openrouter_model: str = "deepseek/deepseek-chat-v3-0324"  # or other models
    openrouter_model: str = "gtp-4o-mini"  # or other models

    # --- OpenAI / Remote LLM ---
    openai_api_key: str = ""                 # Set OPENAI_API_KEY env var
    openai_model: str = "gpt-4o-mini"       # OpenAI model name

    # --- LLM Common Settings ---
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # --- LLM Provider Selection ---
    llm_provider: str = "openrouter"         # "ollama" or "openrouter"

    # --- Privacy & Anonymization ---
    privacy_enabled: bool = True
    privacy_anonymizer: str = "presidio"     # "presidio" or "noop"
    default_top_k: int = 5
    advanced_top_k: int = 15
    similarity_threshold: float = 0.40       # cosine similarity floor
    analytics_default_trend_days: int = 90
    analytics_bucket_days: int = 7
    analytics_top_entities: int = 10
    
    # --- Session retrieval cache ---
    session_cache_enabled: bool = True
    session_cache_max_entries: int = 50
    session_cache_semantic_threshold: float = 0.92

    # --- User data ingestion ---
    user_data_import_batch_size: int = 64
    personal_assistant_url: str = "http://localhost:5002"
    graphrag_username: str = ""

    # --- Hugging Face ---
    hf_token: str = ""


settings = Settings()

# Set HF_TOKEN in environment for libraries like sentence-transformers/huggingface_hub
if settings.hf_token:
    os.environ["HF_TOKEN"] = settings.hf_token
