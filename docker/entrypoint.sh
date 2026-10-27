#!/bin/bash
set -e

echo "[entrypoint] Starting Postgres..."

# Initialize Postgres data dir if empty (first run only)
if [ ! -s "$PGDATA/PG_VERSION" ]; then
    echo "[entrypoint] Initializing Postgres data directory..."
    mkdir -p "$PGDATA"
    chown -R postgres:postgres "$PGDATA"
    su postgres -c "/usr/lib/postgresql/16/bin/initdb -D $PGDATA --auth-local=trust --auth-host=trust"
fi

# Start Postgres in the background
su postgres -c "/usr/lib/postgresql/16/bin/pg_ctl -D $PGDATA -l /tmp/pg.log -w start"

# Wait for Postgres to accept connections
echo "[entrypoint] Waiting for Postgres..."
for i in {1..30}; do
    if su postgres -c "psql -U postgres -c 'SELECT 1'" > /dev/null 2>&1; then
        echo "[entrypoint] Postgres is ready"
        break
    fi
    sleep 1
done

# Create the movies database if it doesn't exist
su postgres -c "psql -U postgres -tc \"SELECT 1 FROM pg_database WHERE datname = 'movies'\" | grep -q 1" \
    || su postgres -c "psql -U postgres -c 'CREATE DATABASE movies'"

# Apply schema (idempotent — CREATE TABLE IF NOT EXISTS)
echo "[entrypoint] Applying schema..."
su postgres -c "psql -U postgres -d movies -f /app/data/schema.sql"

# Check if the index is already built
ROW_COUNT=$(su postgres -c "psql -U postgres -d movies -tAc 'SELECT COUNT(*) FROM movies'" 2>/dev/null || echo "0")
ROW_COUNT=$(echo "$ROW_COUNT" | tr -d '[:space:]')

if [ "$ROW_COUNT" -lt "8000" ]; then
    echo "[entrypoint] Building index (this takes ~3 minutes on first run)..."
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "[entrypoint] ERROR: OPENAI_API_KEY is not set."
        echo "[entrypoint] Pass it with: docker run -p 8080:80 -e OPENAI_API_KEY=sk-... movie-recommender"
        exit 1
    fi
    cd /app && /app/.venv/bin/python data/build_index.py
else
    echo "[entrypoint] Index already built ($ROW_COUNT rows), skipping build"
fi

# Start FastAPI in the foreground — this is the container's main process
echo "[entrypoint] Starting FastAPI on port 80..."
cd /app && exec /app/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 80