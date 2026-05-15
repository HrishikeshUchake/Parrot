# System Architecture

## Overview
Parrot is composed of a backend processing engine, a graph database, and an LLM service.

## High-Level Architecture
```mermaid
graph TD
    Client["Frontend/Client"] -->|HTTP / API| Backend["Backend Application (Python/FastAPI)"]
    
    subgraph "Backend System"
        BackendAPI["Main API (main.py)"]
        Agents["Agentic Workflow (LangGraph)"]
        Services["Core Services (Chuking, Embeddings, Privacy)"]
        Database["Database Client (Neo4j)"]
        LLMClients["LLM Clients (Ollama, OpenRouter)"]
        
        BackendAPI --> Agents
        Agents --> Services
        Agents --> Database
        Agents --> LLMClients
    end
    
    Database -->|Bolt Protocol| Neo4j["Neo4j Graph Database"]
    LLMClients -->|HTTP| ExternalLLM["Local Ollama or External API"]

```

## Agent Workflow
```mermaid
graph TD
    Start["User Query"] --> Routing["Router Node"]
    Routing --> QA["Query Analyzer Node"]
    QA --> Retrieval["Retrieval Node"]
    Retrieval --> Synthesis["Synthesis Node"]
    Synthesis --> End["Final Response"]
```

## Data Privacy & Anonymization
```mermaid
sequenceDiagram
    participant User
    participant Backend
    participant PrivacyService
    participant LLM
    
    User->>Backend: Sensitive Query
    Backend->>PrivacyService: Anonymize(Query)
    PrivacyService-->>Backend: Anonymized Query
    Backend->>LLM: Process(Anonymized Query)
    LLM-->>Backend: LLM Response
    Backend->>PrivacyService: Deanonymize(Response)
    PrivacyService-->>Backend: Original Entities Restored
    Backend-->>User: Final Response
```
