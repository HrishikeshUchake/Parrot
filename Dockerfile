FROM python:3.11-slim

WORKDIR /app

# Install build dependencies required by some Python packages (e.g., hnswlib, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY Backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download a spaCy model which is commonly required by presidio-analyzer
RUN python -m spacy download en_core_web_lg || python -m spacy download en_core_web_sm || true

# Copy backend source code
COPY Backend/ ./Backend/

# Make sure Python can find the top-level Backend module
ENV PYTHONPATH=/app

# Default environment variables for Neo4j (overridden by docker-compose)
ENV NEO4J_URI=bolt://neo4j:7687
ENV NEO4J_USER=neo4j
ENV NEO4J_PASSWORD=password

EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
