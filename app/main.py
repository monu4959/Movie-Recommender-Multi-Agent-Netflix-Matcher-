"""
FastAPI entry point. Exposes the orchestrator as an HTTP API and serves
the minimal HTML UI at the root path.

Run: uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.db import close_pool, get_pool
from app.models import ExplainedResult, ParsedQuery, RecommendRequest
from app.orchestrator import run_turn


UI_INDEX = Path(__file__).parent.parent / "ui" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the DB connection pool on startup so the first request isn't slow
    await get_pool()
    yield
    await close_pool()


app = FastAPI(
    title="Movie Recommender",
    description="Multi-agent Netflix-style recommender (TeloHive assessment)",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — dev-friendly, lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- API endpoints ----------

@app.post("/recommend")
async def recommend(req: RecommendRequest) -> dict:
    """
    Main endpoint. Takes a session_id and user query, runs the agent graph,
    returns either recommendations (new_search/refine/more_like) or a chat
    answer (follow_up).
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = await run_turn(req.session_id, req.query)

    # Serialize Pydantic models to dicts for the JSON response
    return {
        "session_id": result["session_id"],
        "results": [r.model_dump() for r in result["results"]],
        "chat_answer": result["chat_answer"],
        "parsed_query": result["parsed_query"].model_dump()
            if result["parsed_query"] else None,
        "latency_ms": result["latency_ms"],
    }


@app.get("/health")
async def health() -> dict:
    """Liveness check for the server + DB."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM movies")
    return {"status": "ok", "movies_indexed": count}


# ---------- UI ----------

@app.get("/")
async def root():
    """Serve the single-page UI."""
    if not UI_INDEX.exists():
        raise HTTPException(status_code=500, detail="UI not built")
    return FileResponse(UI_INDEX)