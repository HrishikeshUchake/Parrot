# Parrot: Agentic Graph-RAG for Social Data Analysis

Parrot is a backend-first system for retrieving and synthesizing answers from social-style data (posts, comments, and messages) using a graph-enhanced Retrieval-Augmented Generation (RAG) pipeline.

The system combines:

- LangGraph orchestration for multi-step agent flows
- Neo4j as graph store plus native vector indexes
- SentenceTransformer embeddings (E5 family)
- Node-specific LLM routing (analyzer/router/synthesis can use different backends)
- Ollama (local LLM) with optional OpenAI-compatible synthesis path
- FastAPI for API serving and a built-in CLI for local workflows

## Current Project Status

This repository is in active prototype-to-product hardening.

What is production-relevant today:

- End-to-end query flow from analysis to routing to retrieval to synthesis
- Mixed-source retrieval over posts, comments, and messages
- User-scoped import pipeline from JSONL datasets
- Neo4j graph-neighbor enrichment in advanced retrieval

What is still maturing:

- Frontend implementation (currently scaffold only)
- Automated tests and CI workflows
- Expanded operational hardening and security guardrails

## Repository Layout

```text
Backend/
  agents/       # LangGraph nodes: analyzer, router, retrieval, synthesis
  database/     # Data models + Neo4j/SQLite repositories
  llm/          # Provider abstraction, Ollama client, prompts
  scripts/      # Import, inspection, and embedding migration tools
  services/     # Embeddings, hybrid search, vector store
  config.py     # Runtime configuration via pydantic-settings
  main.py       # FastAPI app + CLI entrypoint

Frontend/       # UI scaffold (placeholder)
Documents/      # Documentation scaffold (placeholder)

activities.jsonl
feed.jsonl
messages.jsonl
```

## Architecture Overview

### Query Lifecycle

1. Input enters via API (`POST /query`) or interactive CLI.
2. `query_analyzer` classifies intent and extracts filters and sub-queries.
3. `router` chooses simple or advanced retrieval.
4. Retrieval executes hybrid vector plus keyword logic, with optional user scoping.
5. Advanced path expands via graph-neighbor traversal.
6. `synthesis` composes final answer grounded in retrieved evidence.

### Graph Topology

```text
query_analyzer -> router -> {simple_retrieval | advanced_retrieval} -> synthesis -> END
```

### Retrieval Strategy

Simple retrieval:

- Single-query hybrid search
- Optional user-scoped semantic matches from messages and comments
- Fast path for straightforward factual lookups

Advanced retrieval:

- Multi-query retrieval from decomposed sub-queries
- Deduplicate and rerank by score
- Graph-neighbor expansion over related posts
- Better for trend, comparison, and synthesis-style questions

## Core Components

- API and CLI entrypoint: `Backend/main.py`
- Workflow graph: `Backend/agents/graph.py`
- Query analyzer: `Backend/agents/query_analyzer.py`
- Router: `Backend/agents/router.py`
- Retrieval nodes: `Backend/agents/retrieval.py`
- Final synthesis: `Backend/agents/synthesis.py`
- Vector store and graph operations: `Backend/services/vector_store.py`
- Hybrid search engine: `Backend/services/search_engine.py`
- Embedding service: `Backend/services/embedding_service.py`
- Config and defaults: `Backend/config.py`

## Technology Stack

- Python 3.11+
- LangGraph and LangChain
- Neo4j (graph + vector indexes)
- sentence-transformers and torch
- FastAPI and Uvicorn
- Ollama (primary local inference)
- Optional OpenAI-compatible backend
- SQLite (query cache sidecar)

## Prerequisites

1. Python environment (virtualenv recommended)
2. Running Neo4j instance reachable at your configured URI
3. Running Ollama instance with desired model pulled (default `llama3.2`)

## Quick Start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install backend dependencies

```bash
pip install -r Backend/requirements.txt
```

### 3) Configure environment

The backend reads settings from environment variables and `Backend/.env`.

Minimum recommended values:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OPENAI_MODEL=gpt-4o-mini

EMBEDDING_MODEL=intfloat/e5-large-v2
```

### Optional: Synthesis-only remote reasoning

If you want query analysis and routing to stay local while only the final synthesis answer uses a remote model, configure:

```env
ANALYZER_LLM_BACKEND=ollama
ROUTER_LLM_BACKEND=ollama
SYNTHESIS_LLM_BACKEND=openai

OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
# Optional for OpenAI-compatible gateways:
# OPENAI_BASE_URL=https://your-endpoint/v1
```

Behavior summary:

- `query_analyzer` uses `ANALYZER_LLM_BACKEND`
- `router` uses `ROUTER_LLM_BACKEND`
- `synthesis` uses `SYNTHESIS_LLM_BACKEND`
- If remote synthesis fails, the provider falls back to local Ollama

### 4) Run API server

```bash
uvicorn Backend.main:app --reload --port 8000
```

### 5) Run CLI mode

```bash
python -m Backend.main
```

## API Usage

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "ollama": true
}
```

### Query Endpoint

```http
POST /query
Content-Type: application/json
```

Request body:

```json
{
  "query": "What topics did @alice discuss last week?",
  "user_context_username": "alice"
}
```

Response body:

```json
{
  "answer": "...",
  "route": "advanced",
  "num_sources": 8,
  "reasoning": "Route: advanced | Results: 8"
}
```

## Data Ingestion Workflows

Parrot supports user-centric import from local JSONL files.

### Import user data through the main entrypoint

```bash
python -m Backend.main \
  --load-user-data \
  --username albert336 \
  --activities activities.jsonl \
  --feed feed.jsonl \
  --messages messages.jsonl
```

### Standalone import script

```bash
python -m Backend.scripts.fetch_user_data \
  --username albert336 \
  --activities activities.jsonl \
  --feed feed.jsonl \
  --messages messages.jsonl
```

### Inspect Neo4j contents

```bash
python -m Backend.scripts.inspect_neo4j
```

### Recompute embeddings for existing nodes

```bash
python -m Backend.scripts.migrate_embeddings --batch-size 128
```

## Configuration Reference

The following areas are configured in `Backend/config.py`:

- Neo4j connection and index names
- Embedding model and batch/cache behavior
- Global and node-specific LLM backend selection and generation settings
- Retrieval defaults (`top_k`, thresholds)
- Import batch sizes and mastodon-source options

Commonly tuned settings:

- `DEFAULT_TOP_K`
- `ADVANCED_TOP_K`
- `SIMILARITY_THRESHOLD`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `ANALYZER_LLM_BACKEND`
- `ROUTER_LLM_BACKEND`
- `SYNTHESIS_LLM_BACKEND`
- `OPENAI_MODEL`
- `USER_DATA_IMPORT_BATCH_SIZE`

## Operational Notes

- Startup may be heavy on first run because embedding model initialization and Neo4j index checks occur during component initialization.
- Advanced retrieval quality depends strongly on embedding quality and graph completeness.
- User scoping is supported through `user_context_username` and metadata filters.

## Security and Privacy Considerations

Current implementation notes:

- Data is persisted in Neo4j for retrieval.
- User-scoped relationships (`CAN_SEE`) are used for contextual scoping.
- You should enforce strict environment-based secret management and avoid default credentials in any shared environment.

Recommended hardening for deployment:

- Move all credentials to secure secret stores
- Add request validation and rate limiting at API boundary
- Add structured audit logging for retrieval and synthesis decisions
- Add test coverage for privacy scoping rules

## Development and Contribution

### Local development checklist

1. Start Neo4j and Ollama
2. Activate virtual environment
3. Install dependencies
4. Ingest seed and user data
5. Run API or CLI and iterate on prompts and agents

### Suggested quality gates

- Unit tests for analyzer, router, and retrieval logic
- Integration tests for API plus Neo4j plus Ollama contract
- Regression tests for user-scoped retrieval filters

## Known Gaps and Roadmap

Near-term priorities:

- Implement frontend experience in `Frontend/`
- Add robust automated tests
- Add CI workflows under `.github/`
- Improve observability and error surfaces
- Expand project documentation in `Documents/`

## License

No explicit license file is currently present in this repository. Add one before external distribution.

## Support

Use repository issues and discussions for bug reports, support questions, and feature requests.
