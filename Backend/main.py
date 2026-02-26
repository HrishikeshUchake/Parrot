"""
Parrot – Agentic RAG System for Social Media Analysis.

Interactive CLI:  python -m Backend.main
FastAPI server:   uvicorn Backend.main:app --reload --port 8000
"""
from __future__ import annotations
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── FastAPI application ───────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    from .agents.graph import rag_graph
    from .llm.ollama_client import OllamaClient

    app = FastAPI(title="Parrot RAG API", version="0.1.0")

    class QueryRequest(BaseModel):
        query: str

    class QueryResponse(BaseModel):
        answer: str
        route: str
        num_sources: int
        reasoning: str

    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest) -> QueryResponse:
        result = await rag_graph.ainvoke({"query": req.query, "search_results": []})
        return QueryResponse(
            answer=result.get("answer", ""),
            route=result.get("route", ""),
            num_sources=len(result.get("search_results", [])),
            reasoning=result.get("reasoning", ""),
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "ollama": OllamaClient().is_available(),
        }

except ImportError as e:
    logger.warning("FastAPI not available (%s) API server disabled.", e)
    app = None  # type: ignore


# ── Interactive CLI ────────────────────────────────────────────────────────────
async def interactive_loop() -> None:
    from .agents.graph import rag_graph

    print("\n Parrot Agentic RAG type your query (Ctrl-C to exit)\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        result = await rag_graph.ainvoke({"query": query, "search_results": []})
        print(f"\n--- Answer ---")
        print(result.get("answer", "<no answer>"))
        print(
            f"\n[Route: {result.get('route', '?')} | "
            f"Sources: {len(result.get('search_results', []))}]\n"
        )


if __name__ == "__main__":
    asyncio.run(interactive_loop())
