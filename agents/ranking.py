"""
Ranking Agent.

Takes a user brief + ~30 retrieved candidates and returns the top N ranked
by actual fit, each with a 0-100 score and a one-line reason.

Retrieval optimizes for recall. Ranking optimizes for precision.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from openai import OpenAI

from app.config import settings
from app.models import Candidate, ChatMessage, ParsedQuery, RankedMovie, RankingResult


_openai_client: OpenAI | None = None

# How many ranked results to return to the user
DEFAULT_TOP_N = 8

# How many candidates we send to the LLM (should match retrieval's FINAL_POOL)
MAX_CANDIDATES_IN_PROMPT = 30


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


# ---------- System prompt ----------
#
# Design notes:
#  1. Aggressive score calibration — without this, models rate everything 80+
#     and scores become meaningless.
#  2. Tell the model to ignore list order explicitly (defense against the
#     position bias we also mitigate by shuffling).
#  3. Require the model to REFUSE to include poor fits rather than pad the
#     list. Better to return 4 great picks than 8 mediocre ones.
#  4. Keep `brief_reason` short — the Explanation agent will write the polished
#     version later. This reason is just for the ranker's own bookkeeping.

SYSTEM_PROMPT = """You are the Ranking module for a Netflix-style recommender. \
You receive a user's brief and a numbered list of candidate titles retrieved \
by an upstream search. Your job is to select and rank the titles that best \
fit the brief.

# Scoring scale (IMPORTANT)

- 90-100: exceptional fit, exactly what the user asked for
- 70-89:  strong fit, clearly matches the brief
- 50-69:  partial fit, matches some aspects but not all
- 30-49:  weak fit, tangentially related
- 0-29:   do not include this title in your output

Calibrate hard. If most candidates are only loosely related to the brief, \
your top score might be 75, not 95. Be honest about fit. Better to return 4 \
strong picks than 10 mediocre ones.


# How to judge fit

Read the user's brief carefully. What are they actually asking for? Consider:
- Mood, vibe, and theme — cozy, dark, cerebral, etc.
- Reference titles — if they said "like Inception," weight titles that share \
Inception's specific qualities (cerebral, layered, sci-fi), not just any \
thriller
- The OVERALL shape of what they want, not individual words

IMPORTANT: Hard filters (type, genre, era, exclusions) have already been \
applied by the retrieval layer. Every candidate in the pool already satisfies \
those constraints. Do NOT penalize candidates for "not being recent enough" \
or "not being the right type" — those filters were enforced upstream. Focus \
on qualitative fit: mood, theme, and quality of match.

For refinement queries ("more recent", "shorter", "darker"), treat the \
refinement as signal that retrieval has already narrowed the pool. Your job \
is to pick the BEST of what came back, not to re-verify constraints.

# Rules

1. Return at most {top_n} titles, fewer if the candidate pool doesn't have \
{top_n} genuine fits.
2. Order by score descending.
3. The list order is random — do not use position as a signal. Re-evaluate \
every candidate against the brief.
4. Include EVERY selected title's `show_id` exactly as given. Do not invent \
shows or modify IDs.
5. `brief_reason`: one concise phrase (5-15 words) explaining the match. \
This is internal bookkeeping — don't polish it for the user.
6. For follow-up queries (when the user asked about a previous result rather \
than requesting new recommendations), return an empty list.
"""


# ---------- JSON schema for structured output ----------

def _build_response_schema() -> dict[str, Any]:
    schema = RankingResult.model_json_schema()

    # Strip Pydantic's internal $defs so OpenAI's strict mode accepts it
    defs = schema.pop("$defs", {})

    def resolve_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                return resolve_refs(defs[ref_name])
            return {k: resolve_refs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve_refs(v) for v in obj]
        return obj

    schema = resolve_refs(schema)

    # OpenAI strict mode: additionalProperties=false and required=all keys
    def enforce_strict(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object" and "properties" in obj:
                obj["additionalProperties"] = False
                obj["required"] = list(obj["properties"].keys())
            for v in obj.values():
                enforce_strict(v)
        elif isinstance(obj, list):
            for v in obj:
                enforce_strict(v)

    enforce_strict(schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ranking_result",
            "schema": schema,
            "strict": True,
        },
    }


_RESPONSE_FORMAT = _build_response_schema()


# ---------- Candidate formatting ----------

def _format_candidate(idx: int, c: Candidate) -> str:
    """Compact, LLM-friendly representation of one candidate."""
    genres = ", ".join(c.genres) if c.genres else "(none)"
    desc = (c.description or "").strip()
    if len(desc) > 300:
        desc = desc[:300] + "..."
    year = c.release_year if c.release_year else "unknown"
    return (
        f"[{idx}] id={c.show_id} | {c.title} ({c.type}, {year})\n"
        f"    Genres: {genres}\n"
        f"    {desc}"
    )


def _format_history(history: list[ChatMessage], max_turns: int = 4) -> str:
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for msg in recent:
        role = "User" if msg.role == "user" else "Assistant"
        content = msg.content.strip()
        if msg.role == "assistant" and len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------- Main entry point ----------

async def rank(
    user_query: str,
    candidates: list[Candidate],
    parsed: ParsedQuery,
    history: list[ChatMessage] | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[RankedMovie]:
    """
    Rank candidates against the brief. Returns at most `top_n` RankedMovie
    objects, ordered by score descending.
    """
    # Short-circuit: follow-up queries don't need ranking
    if parsed.intent == "follow_up":
        return []

    # Nothing to rank
    if not candidates:
        return []

    # Cap candidates defensively — retrieval already limits, but belt + braces
    pool = candidates[:MAX_CANDIDATES_IN_PROMPT]

    # Shuffle to neutralize position bias from retrieval's ordering
    shuffled = list(pool)
    random.shuffle(shuffled)

    # Build the candidates block
    candidate_block = "\n\n".join(
        _format_candidate(i + 1, c) for i, c in enumerate(shuffled)
    )

    history_block = _format_history(history or [])
    history_section = f"\n# Recent conversation\n{history_block}\n" if history_block else ""

    user_prompt = f"""# User brief
{user_query}
{history_section}
# Candidates (random order — DO NOT use position as a signal)
{candidate_block}

Select the best-fitting titles and return a ranked list of at most {top_n} \
results. Use each candidate's `show_id` verbatim. Follow the scoring scale \
strictly."""

    client = _get_openai()
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(top_n=top_n)},
            {"role": "user", "content": user_prompt},
        ],
        response_format=_RESPONSE_FORMAT,
        temperature=0.3,
    )

    raw_json = resp.choices[0].message.content
    data = json.loads(raw_json)
    result = RankingResult(**data)

    # Defensive: drop any show_ids the model invented (not in our candidate pool)
    valid_ids = {c.show_id for c in pool}
    filtered = [r for r in result.ranked if r.show_id in valid_ids]

    # Already ordered by score from the model, but enforce it server-side too
    filtered.sort(key=lambda r: r.score, reverse=True)
    return filtered[:top_n]