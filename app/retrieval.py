"""
Hybrid retrieval: pgvector semantic search + tsvector keyword search,
combined with Reciprocal Rank Fusion. Pure code, no LLM.

Takes a ParsedQuery (from the Understanding agent) and returns ~30 Candidates.
"""

from __future__ import annotations

from openai import OpenAI

from app.config import settings
from app.db import get_pool
from app.models import Candidate, ParsedQuery

# Number of candidates each sub-search returns before fusion
SEMANTIC_POOL = 60
KEYWORD_POOL = 40
FINAL_POOL = 30

# RRF constant. 60 is the standard value from the original RRF paper.
RRF_K = 60

# Minimum results before we consider retrieval a "success"
MIN_ACCEPTABLE = 5


_openai_client: OpenAI | None = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def embed_query(text: str) -> list[float]:
    """Embed a query string using the same model as the index."""
    resp = _get_openai().embeddings.create(
        model=settings.openai_embed_model,
        input=text,
    )
    return resp.data[0].embedding


def _build_where_clause(pq: ParsedQuery, relax: int = 0) -> tuple[str, list]:
    """
    Build a WHERE clause and parameter list from the ParsedQuery.
    `relax` controls how many filter levels to drop (for retry-on-empty):
        0 = all filters, 1 = drop era, 2 = drop genres too, 3 = drop type too.
    """
    conditions: list[str] = []
    params: list = []
    p = 1  # parameter index for $1, $2, ...

    # Type filter (Movie / TV Show)
    if pq.type and relax < 3:
        conditions.append(f"type = ${p}")
        params.append(pq.type)
        p += 1

    # Genre filter — row's genres array must overlap with requested genres
    if pq.genres and relax < 2:
        conditions.append(f"genres && ${p}")
        params.append(pq.genres)
        p += 1

    # Era filter
    if pq.era_start and relax < 1:
        conditions.append(f"release_year >= ${p}")
        params.append(pq.era_start)
        p += 1
    if pq.era_end and relax < 1:
        conditions.append(f"release_year <= ${p}")
        params.append(pq.era_end)
        p += 1

    # Exclusions — we always honor these regardless of relaxation
    if pq.exclude_keywords:
        # Exclude if genres array overlaps with excluded terms
        conditions.append(f"NOT (genres && ${p})")
        params.append(pq.exclude_keywords)
        p += 1

    where_sql = " AND ".join(conditions) if conditions else "TRUE"
    return where_sql, params


async def _semantic_search(
    pq: ParsedQuery, query_vec: list[float], relax: int
) -> list[dict]:
    """Top-K by cosine similarity on embedding, with filters applied."""
    where_sql, params = _build_where_clause(pq, relax)
    # Vector param comes AFTER the where-clause params
    vec_param_idx = len(params) + 1
    limit_param_idx = vec_param_idx + 1

    sql = f"""
        SELECT
            show_id, type, title, release_year, genres, description,
            director, cast_members, duration_value, duration_unit, rating,
            1 - (embedding <=> ${vec_param_idx}) AS similarity
        FROM movies
        WHERE {where_sql}
        ORDER BY embedding <=> ${vec_param_idx}
        LIMIT ${limit_param_idx}
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params, query_vec, SEMANTIC_POOL)
    return [dict(r) for r in rows]


async def _keyword_search(
    pq: ParsedQuery, raw_query: str, relax: int
) -> list[dict]:
    """Top-K by ts_rank on the weighted tsvector, with filters applied."""
    where_sql, params = _build_where_clause(pq, relax)
    q_param_idx = len(params) + 1
    limit_param_idx = q_param_idx + 1

    # plainto_tsquery handles user text safely (no operator-parsing issues)
    sql = f"""
        SELECT
            show_id, type, title, release_year, genres, description,
            director, cast_members, duration_value, duration_unit, rating,
            ts_rank(search_tsv, plainto_tsquery('english', ${q_param_idx})) AS kw_score
        FROM movies
        WHERE {where_sql}
          AND search_tsv @@ plainto_tsquery('english', ${q_param_idx})
        ORDER BY kw_score DESC
        LIMIT ${limit_param_idx}
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params, raw_query, KEYWORD_POOL)
    return [dict(r) for r in rows]


def _rrf_fuse(
    semantic: list[dict], keyword: list[dict], k: int = RRF_K
) -> list[Candidate]:
    """
    Reciprocal Rank Fusion:
      score(doc) = sum over lists of 1 / (k + rank_in_list)
    Docs in only one list still get a partial score.
    When a doc appears in both lists we merge scores so the final Candidate
    reflects both semantic similarity and keyword rank.
    """
    scores: dict[str, float] = {}
    semantic_rank: dict[str, int] = {}
    keyword_rank: dict[str, int] = {}
    row_by_id: dict[str, dict] = {}

    for rank, row in enumerate(semantic, start=1):
        sid = row["show_id"]
        scores[sid] = scores.get(sid, 0) + 1 / (k + rank)
        semantic_rank[sid] = rank
        row_by_id[sid] = row

    for rank, row in enumerate(keyword, start=1):
        sid = row["show_id"]
        scores[sid] = scores.get(sid, 0) + 1 / (k + rank)
        keyword_rank[sid] = rank
        if sid in row_by_id:
            # Row was also in semantic results — merge keyword score into it
            row_by_id[sid]["kw_score"] = row.get("kw_score", 0.0)
        else:
            row_by_id[sid] = row

    # Sort by fused score, take top FINAL_POOL
    ranked_ids = sorted(scores, key=lambda s: scores[s], reverse=True)[:FINAL_POOL]

    candidates: list[Candidate] = []
    for sid in ranked_ids:
        row = row_by_id[sid]
        candidates.append(
            Candidate(
                show_id=row["show_id"],
                type=row["type"],
                title=row["title"],
                release_year=row.get("release_year"),
                genres=list(row.get("genres") or []),
                description=row.get("description"),
                director=list(row.get("director") or []),
                cast_members=list(row.get("cast_members") or []),
                duration_value=row.get("duration_value"),
                duration_unit=row.get("duration_unit"),
                rating=row.get("rating"),
                semantic_score=float(row.get("similarity") or 0.0),
                keyword_score=float(row.get("kw_score") or 0.0),
                fused_score=scores[sid],
            )
        )
    return candidates


async def retrieve(pq: ParsedQuery) -> list[Candidate]:
    """
    Full hybrid retrieval pipeline.
    Runs semantic + keyword in sequence, fuses with RRF, retries with relaxed
    filters if we get fewer than MIN_ACCEPTABLE results.
    """
    # The raw query for keyword search is the semantic_query string plus
    # any distinctive mood keywords / reference titles the user mentioned.
    keyword_query = " ".join(
        [pq.semantic_query]
        + pq.mood_keywords
        + pq.reference_titles
    ).strip()

    query_vec = embed_query(pq.semantic_query)

    for relax in range(4):  # 0..3 progressively drops filters
        semantic = await _semantic_search(pq, query_vec, relax)
        keyword = await _keyword_search(pq, keyword_query, relax) if keyword_query else []
        candidates = _rrf_fuse(semantic, keyword)
        if len(candidates) >= MIN_ACCEPTABLE:
            return candidates

    # Final fallback: pure semantic, no filters whatsoever
    semantic = await _semantic_search(
        ParsedQuery(semantic_query=pq.semantic_query, intent="new_search"),
        query_vec,
        relax=3,
    )
    return _rrf_fuse(semantic, [])