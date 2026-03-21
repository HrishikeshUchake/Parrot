"""Pydantic models for database entities."""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal


class Post(BaseModel):
    """Represents a single Mastodon status (toot)."""

    # Mastodon status ID (large integer as string)
    id: str
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


class UserMessage(BaseModel):
    """Represents a user-to-user message in the social graph."""

    id: str
    sender_name: str
    receiver_name: str
    text: str
    date: str = ""
    time_ms: int = 0
    source: str = "individual_chat"
    user_context_username: str = ""


class UserComment(BaseModel):
    """Represents a comment attached to a social post."""

    id: str
    post_id: str
    commenter_name: str
    content: str
    time: str = ""
    source: str = ""
    user_context_username: str = ""


class SearchResult(BaseModel):
    # For legacy post-only paths, `post` remains populated.
    post: Post | None = None
    result_type: Literal["post", "comment", "message"] = "post"
    item_id: str = ""
    content: str = ""
    metadata: dict = Field(default_factory=dict)
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
