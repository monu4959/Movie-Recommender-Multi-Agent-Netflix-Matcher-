# Multi-service image: Postgres + pgvector + FastAPI app in one container.
# This is an anti-pattern in production but required by the assessment's
# two-command spec (docker build / docker run). In prod we'd split these.

FROM pgvector/pgvector:pg16

# Install Python 3.11 and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python dep management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy Python dep manifests first for better layer caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install Python deps into a venv
RUN uv sync --no-dev

# Copy source
COPY app/ ./app/
COPY agents/ ./agents/
COPY data/ ./data/
COPY ui/ ./ui/
COPY eval/ ./eval/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh

# FastAPI on port 80 inside the container (maps to host 8080 via -p 8080:80)
EXPOSE 80

ENV PGDATA=/var/lib/postgresql/data
ENV POSTGRES_DB=movies
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_HOST=localhost

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]