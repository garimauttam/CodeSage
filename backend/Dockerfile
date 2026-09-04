# Multi-stage build for the backend
# Stage 1: Install dependencies (cached layer — only rebuilds when requirements.txt changes)
FROM python:3.11-slim AS deps

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Copy app code (separate layer — fast rebuilds when only code changes)
FROM deps AS runtime

WORKDIR /app
COPY . .

# Create the ChromaDB data directory
RUN mkdir -p /app/chroma_data

# EXPOSE 8000 is a hint for documentation/tooling only.
# Railway injects a PORT env var; we read it via the shell form CMD below.
EXPOSE 8000

# Shell form (not exec form) so that $PORT is expanded at container start time.
# Railway sets PORT automatically; fallback to 8000 for local docker builds.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
