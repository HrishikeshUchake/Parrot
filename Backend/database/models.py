"""Pydantic models for database entities."""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal


class Post(BaseModel):
    """Represents a social post or activity."""

    id: str
    content: str = ""
    created_at: str = ""
    account_id: str = ""
    account_username: str = ""
    account_display_name: str = ""
    account_acct: str = ""
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


class ConversationThread(BaseModel):
    """Represents a full conversation thread (e.g. parent post + replies, or group chat)."""
    id: str
    source_type: Literal["post", "message"]
    participants: list[str] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)  # structured messages
    summary: str = ""  # AI-generated summary of the thread
    created_at: str = ""
    updated_at: str = ""


class ThreadChunk(BaseModel):
    """Represents a smaller chunk of a conversation thread, indexed for retrieval."""
    id: str
    thread_id: str
    content: str  # Either a chunk of raw messages or the thread summary
    chunk_index: int = 0
    is_summary: bool = False


class SearchResult(BaseModel):
    # For legacy post-only paths, `post` remains populated.
    post: Post | None = None
    thread: ConversationThread | None = None
    result_type: Literal["post", "comment", "message", "thread"] = "post"
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
