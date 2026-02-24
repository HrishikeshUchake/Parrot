"""Pydantic models for database entities."""
from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class Post(BaseModel):
    id: int
    title: str = ""
    body: str = ""
    tags: list[str] = Field(default_factory=list)
    reactions: dict[str, Any] = Field(default_factory=dict)
    views: int = 0
    user_id: int = 0

    @property
    def full_text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


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
