# Parrot: Agentic Graph-RAG System for Social Data Analysis

https://docs.google.com/presentation/d/1D73lFssUi_TNW2H9O_EDqh-aH-wN6g1L8_holvc_OPE/edit?usp=sharing

Parrot is a production-ready, backend-first agentic system for intelligent retrieval and synthesis of answers from social-style data (posts, comments, and messages) using a graph-enhanced Retrieval-Augmented Generation (RAG) pipeline with privacy-first data handling.

## Overview

Parrot combines multiple specialized agent nodes orchestrated via LangGraph to deliver a sophisticated multi-stage reasoning pipeline:

- **Query Analysis**: Intent classification, entity extraction, and complexity assessment
- **Intelligent Routing**: Dynamic selection between simple, analytical, and advanced retrieval paths
- **Multi-Source Retrieval**: Hybrid vector + keyword search with optional graph-neighbor traversal
- **Privacy-First Synthesis**: Anonymization of PII before remote LLM calls, with automatic restoration of original values
- **Flexible LLM Backends**: Local inference via Ollama with fallback to remote providers (OpenAI, OpenRouter)

## Key Features

- **Graph-Enhanced RAG**: Neo4j as knowledge store with native vector indexes for semantic search
- **Privacy by Design**: Microsoft Presidio integration for automatic PII detection and anonymization
- **Multi-Backend LLM Support**: Per-node LLM routing (analyzer, router, synthesis can use different backends)
- **Advanced Retrieval Strategies**: Graph-neighbor expansion, reranking, and semantic deduplication
- **Production-Grade API**: FastAPI with health checks, error handling, and configurable endpoints
- **CLI and Programmatic Access**: Interactive CLI mode for development, RESTful API for integration
- **Flexible Data Ingestion**: Support for JSONL imports with user-scoped relationship tracking

## Project Status

**Production-Ready Modules:**
- End-to-end query flow from analysis through routing, retrieval, and synthesis
- Mixed-source retrieval over posts, comments, and messages with user scoping
- Privacy layer with Presidio-based PII anonymization
- Neo4j graph enrichment in advanced retrieval mode
- Remote and local LLM synthesis with fallback behavior

**In Development:**
- Frontend implementation (UI scaffold in place)
- Comprehensive test coverage and CI workflows
- Expanded observability and audit logging

For a detailed view of the system architecture and diagrams, please see [SYSTEM_ARCHITECTURE.md](Documents/SYSTEM_ARCHITECTURE.md).

## Repository Structure

```
Backend/
  agents/              # LangGraph agent nodes
    ├── query_analyzer.py     # Intent classification and entity extraction
    ├── router.py             # Route decision logic (simple/analytics/advanced)
    ├── retrieval.py          # Retrieval strategies
    ├── synthesis.py          # Answer generation with synthesis routing
    ├── synthesis_privacy.py  # Privacy-aware synthesis
    ├── graph.py              # Graph topology and orchestration
    └── state.py              # Shared agent state schema

  database/
    ├── models.py             # Data structures (Post, Message, Comment, etc.)
    └── repository.py         # Neo4j and SQLite access layer

  services/
    ├── embedding_service.py  # SentenceTransformer embeddings
    ├── vector_store.py       # Neo4j vector index operations
    ├── search_engine.py      # Hybrid search implementation
    ├── privacy.py            # Presidio-based PII handling
    ├── session_cache.py      # Query result caching
    └── chunking_service.py   # Text chunking for embeddings

  llm/
    ├── ollama_client.py      # Local Ollama inference
    ├── openrouter_client.py  # OpenRouter API client
    ├── llm_provider.py       # Provider abstraction and routing
    ├── prompts.py            # Local LLM prompts
    └── remote_prompts.py     # Remote LLM prompts

  scripts/
    ├── fetch_user_data.py    # Data ingestion from JSONL
    ├── inspect_neo4j.py      # Database inspection utilities
    ├── migrate_embeddings.py # Batch embedding computation
    └── run_social_user_queries.py # Benchmarking utilities

  config.py                    # Unified configuration via pydantic-settings
  main.py                      # FastAPI application and CLI entrypoint
  requirements.txt             # Python dependencies

Frontend/                       # UI implementation (in progress)
Documents/                      # Extended documentation & system diagrams
PRIVACY_ARCHITECTURE.md         # Detailed privacy layer documentation
docker-compose.yml              # Local development environment (Backend + Neo4j)
Dockerfile                      # Docker image definition for backend
```

## Getting Started (Docker)

The easiest way to spin up the Parrot backend and Neo4j database is using Docker Compose. This ensures a unified environment and eliminates local dependency issues.

1. **Build and start the containers**:
   ```bash
   docker compose up --build
   ```
2. **Access the services**:
   - **FastAPI Backend**: `http://localhost:8000`
   - **Neo4j Browser**: `http://localhost:7474` (Credentials: `neo4j` / `password`)

*Note: The `docker-compose.yml` mounts the `./Backend` directory as a volume, enabling hot-reloading for local development. It is also configured to access local Ollama instances via `host.docker.internal`.*

## Architecture Overview

### Query Lifecycle

The system processes user queries through a sophisticated multi-stage pipeline:

```
User Query Input
    ↓
Query Analyzer (Intent Classification & Entity Extraction)
    ├─ Classifies intent (search, analytics, trend, comparison)
    ├─ Extracts entities and filters
    ├─ Decomposes into sub-queries
    └─ Assesses complexity
    ↓
Router (Route Decision)
    ├─ Simple Retrieval: Fast, single-query semantic search
    ├─ Analytics: Aggregate or trend analysis over messages
    └─ Advanced: Multi-query with graph-neighbor expansion
    ↓
Retrieval (Hybrid Search & Graph Enrichment)
    ├─ Hybrid vector + keyword search
    ├─ User-scoped filtering (CAN_SEE relationships)
    ├─ Optional graph-neighbor expansion
    └─ Deduplication and reranking
    ↓
Privacy Layer (Conditional Anonymization)
    ├─ Detect PII entities (names, emails, phones, etc.)
    ├─ Replace with consistent tokens [ENTITY_N]
    └─ Store mappings for restoration
    ↓
Synthesis (Answer Generation)
    ├─ Route to local (Ollama) or remote (OpenAI, OpenRouter)
    ├─ Generate grounded, cited response
    └─ Restore original values from PII mappings
    ↓
Final Answer Output (with Reasoning & Sources)
```

### Agent Graph Topology

```
query_analyzer
    ↓
router
    ├─ simple_retrieval
    ├─ analytics_retrieval
    └─ advanced_retrieval
         ↓
     synthesis_mode (Local or Remote decision)
         ├─ synthesis_local
         └─ synthesis_remote (with fallback to local)
             ↓
           END
```

### Retrieval Strategies

**Simple Retrieval** (Fast Path)
- Single-query hybrid search combining vector and BM25 similarity
- Optional user-scoped filtering of messages and comments
- Ideal for straightforward factual queries

**Analytics Retrieval** (Aggregation Path)
- Supports analytical operations: top topics, top partners, trend analysis
- Precomputed aggregations with optional temporal binning
- Efficient for high-level questions about activity patterns

**Advanced Retrieval** (Semantic-Rich Path)
- Multi-query decomposition and execution
- Deduplication and reranking by relevance score
- Graph-neighbor expansion to surface related entities
- Superior for complex questions requiring multiple evidence sources

### Privacy Layer Architecture

The privacy layer implements **consistent token-based anonymization**:

```
Original: "Contact Sarah at sarah@example.com or 555-1234567"
    ↓ (Presidio Detection)
Tokens: "[PERSON_1]" "[EMAIL_ADDRESS_1]" "[PHONE_NUMBER_1]"
Mapping: {
  "[PERSON_1]": "Sarah",
  "[EMAIL_ADDRESS_1]": "sarah@example.com",
  "[PHONE_NUMBER_1]": "555-1234567"
}
    ↓ (Send to LLM with anonymized context)
    ↓ (LLM returns tokens in response)
    ↓ (Restore from mappings)
Restored: "Contact Sarah at sarah@example.com or 555-1234567"
```

**Key Properties:**
- Tokens remain consistent within a session for the same PII value
- Multiple entity types supported: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, LOCATION, IP_ADDRESS, MEDICAL_LICENSE, and more
- Stateful per-request for request isolation in stateless APIs
- Zero PII data transmitted to remote LLM services when enabled

## Technology Stack

- **Language & Frameworks**: Python 3.11+, LangGraph, LangChain
- **Vector Database & Knowledge Graph**: Neo4j 5.x with native vector indexes
- **Embeddings**: Sentence-transformers (intfloat/e5-large-v2)
- **Local Inference**: Ollama with model flexibility
- **Remote Inference**: OpenAI API, OpenRouter
- **Privacy**: Microsoft Presidio for PII detection
- **Web Framework**: FastAPI with Uvicorn
- **NLP**: spaCy for language understanding
- **Caching**: SQLite for query result caching, LRU in-memory for embeddings

## Prerequisites

- **Python 3.11+** with virtualenv or conda for environment management
- **Neo4j 5.x** instance (easily provisioned via Docker)
- **Ollama** with at least one model pulled (default: `llama3.2`)
- **4 GB+ RAM** recommended for embeddings and local LLM inference
- **GPU acceleration** optional but recommended for embedding computation

## Installation and Setup

### Step 1: Set up the Environment

```bash
# Clone the repository
git clone <repository-url>
cd parrot

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Start Infrastructure Services

Using Docker (recommended):

```bash
# Start Neo4j and configure basic authentication
docker-compose up -d

# Neo4j Browser: http://localhost:7474
# Default credentials: neo4j / password
```

Standalone Ollama:

```bash
# Download and run Ollama (https://ollama.ai)
ollama pull llama3.2  # or preferred model
# Server runs at http://localhost:11434
```

### Step 3: Install Dependencies

```bash
# Install backend dependencies
pip install -r Backend/requirements.txt

# Download spaCy language model for PII detection
python -m spacy download en_core_web_lg
```

### Step 4: Configure Environment

Copy and customize the environment file:

```bash
cd Backend
cp .env_example .env
```

Edit `Backend/.env` with your configuration:

```env
# ─── Neo4j Configuration ───
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# ─── LLM Configuration ───
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# ─── Embedding Model ───
EMBEDDING_MODEL=intfloat/e5-large-v2
EMBEDDING_BATCH_SIZE=64

# ─── Privacy Settings (NEW) ───
PRIVACY_ENABLED=true
PRIVACY_ANONYMIZER=presidio

# ─── Optional: Remote Synthesis ───
# SYNTHESIS_LLM_BACKEND=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
```

### Step 5: Run the Application

**API Server Mode** (FastAPI):

```bash
uvicorn Backend.main:app --reload --port 8000
# API available at http://localhost:8000
# API docs (Swagger UI) at http://localhost:8000/docs
```

**Interactive CLI Mode** (for local testing):

```bash
python -m Backend.main --username <your-username>
```

### Step 6: Ingest Data (Optional)

```bash
# Import social data from JSONL files
python -m Backend.main \
  --load-user-data \
  --username alice \
  --activities activities.jsonl \
  --feed feed.jsonl \
  --messages messages.jsonl

# Inspect imported data
python -m Backend.scripts.inspect_neo4j
```


## Security and Privacy

### Privacy Layer (Presidio Integration)

The system automatically anonymizes PII before sending context to remote LLMs:

**Enabled by default when:**
- `PRIVACY_ENABLED=true`
- `PRIVACY_ANONYMIZER=presidio`
- Using remote synthesis backends (OpenAI, OpenRouter)

**Automatically detected and anonymized:**
- Person names (PERSON)
- Email addresses (EMAIL_ADDRESS)
- Phone numbers (PHONE_NUMBER)
- Credit card numbers (CREDIT_CARD)
- IP addresses (IP_ADDRESS)
- Locations (LOCATION)
- Medical/license information
- Custom patterns (configurable)

**Key Privacy Guarantees:**
- No raw PII transmitted to remote services
- Consistent token mapping within request context
- Automatic restoration in final response
- Audit trail via `privacy_debug` field

See [PRIVACY_ARCHITECTURE.md](./PRIVACY_ARCHITECTURE.md) for detailed architecture and examples.

### Data Security Best Practices

**Credentials Management:**
- Never commit `.env` files or credentials to version control
- Use environment variable injection in production
- Rotate API keys regularly
- Store secrets in dedicated secret management systems (AWS Secrets Manager, HashiCorp Vault, etc.)

**Network Security:**
- Run Neo4j on localhost or private networks in development
- Enforce encryption in transit (TLS) for production deployments
- Use strong passwords for Neo4j and database access
- Implement API rate limiting and authentication

**Data Retention:**
- Establish clear data retention policies
- Regularly audit stored data in Neo4j
- Implement data deletion workflows for compliance (GDPR, CCPA, etc.)
- Consider per-user data isolation for multi-tenant scenarios

**Recommended Hardening for Production:**

1. **Authentication & Authorization**
   - Add JWT or OAuth2 authentication to FastAPI
   - Implement role-based access control (RBAC)
   - Add per-user data scoping

2. **Monitoring & Audit**
   - Enable structured logging for all queries
   - Track LLM API usage and costs
   - Log all data access with timestamps
   - Set up alerting for unusual patterns

3. **Network Isolation**
   - Run behind API gateway with rate limiting
   - Implement CORS policies
   - Use VPC/private subnets for backend services
   - Enable request validation and sanitization

4. **Data Protection**
   - Encrypt data at rest in Neo4j
   - Use encrypted connections between services
   - Consider field-level encryption for sensitive data
   - Implement data anonymization for backups



## Known Limitations

- Embeddings are computed at indexing time; dynamic queries use pre-computed vectors
- Privacy layer adds 10-20% latency overhead for PII detection
- Neo4j vector indexes are best for semantic search; keyword filters are separate
- Ollama performance varies significantly by model size and hardware


## Acknowledgments

Built with:
- [LangGraph](https://python.langchain.com/docs/langgraph) by LangChain
- [Neo4j](https://neo4j.com/) for graph database
- [Presidio](https://microsoft.github.io/presidio/) by Microsoft for PII detection
- [Ollama](https://ollama.ai/) for local LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for API framework

---

## Quick Reference

| Task | Command |
|------|---------|
| Start services | `docker-compose up -d` |
| Install deps | `pip install -r Backend/requirements.txt` |
| Run API | `uvicorn Backend.main:app --reload` |
| Run CLI | `python -m Backend.main --username alice` |
| Import data | `python -m Backend.main --load-user-data --username alice --activities activities.jsonl` |
| Inspect DB | `python -m Backend.scripts.inspect_neo4j` |
| Recompute embeddings | `python -m Backend.scripts.migrate_embeddings` |
| View API docs | http://localhost:8000/docs |
| Neo4j browser | http://localhost:7474 |
