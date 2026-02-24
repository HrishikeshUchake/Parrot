"""Pydantic models for database entities."""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class Post(BaseModel):
    """Represents a single Mastodon status (toot)."""

    id: str                              # Mastodon status ID (large integer as string)
    content: str = ""                   # HTML-stripped status text
    created_at: str = ""                # ISO-8601 timestamp
    account_id: str = ""                # author's Mastodon account ID
    account_username: str = ""          # e.g. "alice"
    account_display_name: str = ""      # e.g. "Alice Smith"
    account_acct: str = ""              # e.g. "alice@mastodon.social"
    tags: list[str] = Field(default_factory=list)   # hashtag names
    reblogs_count: int = 0
    favourites_count: int = 0
    replies_count: int = 0
    url: str = ""
    visibility: str = "public"
    language: str = ""

    @property
    def full_text(self) -> str:
        return self.content.strip()


class SearchResult(BaseModel):
    post: Post
    score: float
    source: str = "vector"   # "vector" | "keyword" | "hybrid"


class QueryCache(BaseModel):
    query_hash: str
    results_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentResponse(BaseModel):
    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
    route_taken: str = ""
    reasoning: str = ""
