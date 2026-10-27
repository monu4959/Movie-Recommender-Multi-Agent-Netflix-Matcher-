"""
Query Understanding Agent.

Takes a raw user message + recent conversation history and produces a
structured ParsedQuery that downstream retrieval/ranking can act on.

This is the most important agent in the pipeline — everything below it
depends on it parsing intent correctly.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from datetime import datetime
import asyncio



from app.config import settings
from app.models import ChatMessage, ParsedQuery


_openai_client: OpenAI | None = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


# ---------- System prompt ----------
#
# Design notes for anyone reading this later (including me):
#  1. We use STRUCTURED OUTPUT via response_format=json_schema. GPT-4o is forced
#     to produce valid JSON matching ParsedQuery's schema — no regex parsing,
#     no string cleanup, no "the model forgot a comma" errors.
#  2. The prompt is defensive. LLMs love to over-fill fields. We tell it
#     explicitly: if the user didn't specify, leave it empty/None.
#  3. We give examples covering the four intents and common edge cases.
#  4. Conversation history is injected as a compact transcript. The model
#     uses it to resolve references like "more modern" or "the third one."

SYSTEM_PROMPT = """You are the Query Understanding module for a Netflix-style \
movie and TV-show recommender. Your job is to read a user's message plus the \
recent conversation history and output a STRUCTURED query object that the \
retrieval engine can act on.

# Fields you produce

- `semantic_query`: a cleaned, vibe-focused version of the user's brief, used \
for semantic search. Strip out filter-able specifics (like "Movie", "90s", \
"not horror") but KEEP mood/theme/plot descriptors. If the user references a \
title ("like Inception"), include that title here as an anchor.

- `intent`: one of:
    * "new_search"  — user wants fresh recommendations, ignore history
    * "refine"      — user is narrowing/modifying the previous search \
("more modern", "shorter", "darker")
    * "more_like"   — user liked specific previous results and wants similar
    * "follow_up"   — user is asking about a specific previous result \
("tell me more about the second one", "who directed it"). Retrieval is skipped \
for follow-ups.

- `type`: "Movie" or "TV Show" if the user was specific, else null. \
"show" / "series" / "episodes" → "TV Show". "film" / "movie" → "Movie". \
Vague terms ("something to watch") → null.

- `genres`: Netflix genre labels the user mentioned. Only use labels from this \
approved list (case-sensitive):
    Action & Adventure, Anime Features, Anime Series, British TV Shows, \
Children & Family Movies, Classic Movies, Classic & Cult TV, Comedies, \
Crime TV Shows, Cult Movies, Documentaries, Docuseries, Dramas, Faith & \
Spirituality, Horror Movies, Independent Movies, International Movies, \
International TV Shows, Kids' TV, Korean TV Shows, LGBTQ Movies, Movies, \
Music & Musicals, Reality TV, Romantic Movies, Romantic TV Shows, Sci-Fi & \
Fantasy, Science & Nature TV, Spanish-Language TV Shows, Sports Movies, \
Stand-Up Comedy, Stand-Up Comedy & Talk Shows, TV Action & Adventure, \
TV Comedies, TV Dramas, TV Horror, TV Mysteries, TV Sci-Fi & Fantasy, \
TV Shows, TV Thrillers, Teen TV Shows, Thrillers

Leave empty if the user didn't name a genre. DO NOT invent genres from mood.

Genre/type consistency: the approved list contains BOTH movie genres (e.g., \
"Romantic Movies", "Comedies", "Horror Movies") AND TV genres (e.g., \
"Romantic TV Shows", "TV Comedies", "TV Horror"). Pick the right variant based \
on `type`. If `type` is "TV Show", use TV-flavored genres. If `type` is \
"Movie", use Movie-flavored genres. NEVER mix both in the same query.

If the user's phrasing implies a type (e.g., "rom-com series" → TV, "rom-com \
film" → Movie, "Korean rom-com" → TV because K-rom-coms are dominantly a TV \
form), set `type` accordingly and pick matching genres. If genuinely \
ambiguous, leave `type` null AND leave `genres` empty rather than mix TV + \
Movie labels.

- `mood_keywords`: descriptive words the user used that aren't formal genres \
("cozy", "dark", "uplifting", "gritty", "campy"). These guide semantic search \
but aren't filters.

- `era_start` / `era_end`: year range, if the user gave one. "90s" → 1990/1999. \
"recent" or "new" → current year minus 5, null end. "classic" → null/1999. \
Leave both null if the user didn't reference time.

- `exclude_keywords`: genres to exclude (from the approved list above). \
"no horror" → ["Horror Movies", "TV Horror"]. Only exclude what the user \
EXPLICITLY ruled out.

- `reference_titles`: exact show/movie titles the user mentioned as anchors \
("something like Inception" → ["Inception"]).

- `follow_up_target`: if intent is "follow_up", the 1-indexed position of the \
result they're asking about. "the third one" → 3. Otherwise null.

# Rules

1. Do not hallucinate specifics. If the user said "something cozy," leave \
`genres` empty — don't guess "Comedies". Put "cozy" in mood_keywords.
2. Named titles go in `reference_titles` AND `semantic_query`, never as \
genre filters.
3. For refinements, MERGE new constraints with context from the prior parsed \
query. PRESERVE `type`, `genres`, and `exclude_keywords` from the prior turn \
unless the user explicitly changes them. If the prior turn asked for "Korean \
romantic comedies" and the user says "something more recent", the refined \
query should keep type="TV Show" and genres=["Korean TV Shows", "Romantic TV \
Shows"] and add era_start=<current year - 5>. Do NOT silently drop filters \
that the user is still implicitly carrying forward.
4. For follow-ups that aren't asking for new recommendations (e.g., "tell me \
more about the first one"), set intent="follow_up", populate \
`follow_up_target`, and leave most other fields minimal.
5. Always produce a non-empty `semantic_query` — even for follow-ups, echo the \
user's question into it."""


# ---------- JSON schema for structured output ----------
# We derive this from Pydantic to stay in sync if ParsedQuery changes.

def _build_response_schema() -> dict[str, Any]:
    """Build the JSON schema OpenAI structured outputs expects."""
    schema = ParsedQuery.model_json_schema()
    # OpenAI requires additionalProperties: false
    schema["additionalProperties"] = False
    # All fields listed in `required` — OpenAI structured output needs every
    # property in required, even optional ones (they just accept null).
    schema["required"] = list(schema["properties"].keys())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "parsed_query",
            "schema": schema,
            "strict": True,
        },
    }


# Cache the schema so we don't rebuild it on every call
_RESPONSE_FORMAT = _build_response_schema()


# ---------- History formatting ----------

def _format_history(history: list[ChatMessage], max_turns: int = 6) -> str:
    """Compact the recent conversation into a readable transcript."""
    if not history:
        return "(no previous conversation)"
    recent = history[-max_turns:]
    lines = []
    for msg in recent:
        role = "User" if msg.role == "user" else "Assistant"
        # Keep assistant turns short — we don't need the full recommendation list
        content = msg.content.strip()
        if msg.role == "assistant" and len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------- Main entry point ----------

async def understand(query: str, history: list[ChatMessage] | None = None) -> ParsedQuery:
    """
    Parse a user message into a structured ParsedQuery.
    Takes optional conversation history for multi-turn refinement.
    """
    history = history or []
    transcript = _format_history(history)

    current_year = datetime.now().year
    user_prompt = f"""# Context
Today's date: {datetime.now().strftime("%Y-%m-%d")}
Current year: {current_year}
"Recent" = {current_year - 5} onward. "New" = {current_year - 2} onward. \
"Classic" = before 2000.

# Conversation history
{transcript}


# Current user message
{query}

Parse the current message into a ParsedQuery. Use the history to resolve \
references like "more modern", "the second one", or "something similar". \
For a brand-new topic, set intent=new_search and ignore history."""

    client = _get_openai()
    # Run the sync OpenAI client in a thread so we don't block the event loop.
    # For this assessment the latency cost of sync-in-thread is fine; if we
    # were optimizing we'd use the AsyncOpenAI client.
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=_RESPONSE_FORMAT,
        temperature=0.2,  # low but not zero — small amount of slack for edge cases
    )

    raw_json = resp.choices[0].message.content
    data = json.loads(raw_json)
    return ParsedQuery(**data)