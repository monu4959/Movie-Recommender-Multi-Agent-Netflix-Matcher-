"""
Explanation Agent.

Takes the user's brief + ranked results and produces a polished one-line
"why this matches your vibe" for each pick. This is the voice-of-the-product
layer — what the user actually reads.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.models import Candidate, ExplainedResult, RankedMovie


_openai_client: OpenAI | None = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


# ---------- Internal LLM-facing schema ----------
# We need a minimal shape for the LLM to fill — just show_id + explanation.
# We'll merge this back with the full RankedMovie data afterward.

class _ExplanationItem(BaseModel):
    show_id: str
    explanation: str = Field(
        description="One sentence, 15-35 words, second-person voice, "
                    "references specifics from the user's brief."
    )


class _ExplanationBatch(BaseModel):
    items: list[_ExplanationItem]


# ---------- System prompt ----------

SYSTEM_PROMPT = """You write short, punchy recommendation blurbs for a \
Netflix-style recommender. For each title in the ranked list, produce ONE \
sentence explaining why it matches the user's brief.

# Style rules

- One sentence, 15-35 words. No longer.
- Second-person voice ("you asked for…", "you'll like…", "fits your…"). \
Not third-person.
- Reference SPECIFICS from the brief — the mood, era, exclusion, or reference \
title the user mentioned. Don't write generic blurbs like "a great show with \
great acting."
- Reference SPECIFICS from the title too — a plot element, a stylistic quality, \
a tonal note. Not just the genre.
- Confident but not salesy. No "you'll love it!!" No "a must-watch." No \
hyperbole. Imagine a friend with taste telling you why this one in particular.
- Don't quote the brief back verbatim. Synthesize.
- Do NOT mention the title in your sentence at all — not at the start, not in \
the middle, not in quotes. The UI displays the title above your sentence, so \
repeating it wastes words. Refer to the title as "it", "this one", or describe \
it by its qualities.
- Use second-person voice throughout ("you'll find…", "fits your…", "shares \
Mindhunter's…"). Avoid imperative openers like "Explore…", "Dive into…", \
"Discover…" — these sound like tour-brochure copy, not personal recommendation.

# Bad examples (do not imitate)

- "A thrilling ride from start to finish." (generic)
- "You'll love this amazing film!" (hyperbolic)
- "The Matrix is a science fiction film about a hacker…" (summary, not a match)
- "This is a great cozy comedy that you asked for." (parrots the brief)

# Good examples

Brief: "Cozy TV show for a rainy Sunday"
Title: Schitt's Creek
→ "Hits that cozy register you're after — a formerly rich family learning \
small-town rhythms, warm without being saccharine, and easy to dip in and out of."

Brief: "90s action movies with car chases, no horror"
Title: Bad Boys
→ "Pure 90s action energy — Miami buddy-cop chaos, multiple extended car \
chases, and exactly zero horror anywhere near it."

Brief: "Dark psychological thriller like Mindhunter"
Title: True Detective
→ "Shares Mindhunter's slow investigator-and-case structure, with a similarly \
dark, procedural obsession — the first season especially fits that cerebral \
thriller space."

# Output

Return an `items` list with one entry per ranked title you received. Use each \
title's `show_id` verbatim. Write every explanation from scratch; do not reuse \
the brief_reason you were given."""


# ---------- JSON schema (same pattern as ranking.py) ----------

def _build_response_schema() -> dict[str, Any]:
    schema = _ExplanationBatch.model_json_schema()
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
            "name": "explanation_batch",
            "schema": schema,
            "strict": True,
        },
    }


_RESPONSE_FORMAT = _build_response_schema()


# ---------- Candidate lookup helper ----------

def _build_candidate_lookup(candidates: list[Candidate]) -> dict[str, Candidate]:
    return {c.show_id: c for c in candidates}


def _format_item_for_prompt(rank_idx: int, ranked: RankedMovie, cand: Candidate) -> str:
    genres = ", ".join(cand.genres) if cand.genres else "(none)"
    desc = (cand.description or "").strip()
    if len(desc) > 250:
        desc = desc[:250] + "..."
    year = cand.release_year or "unknown"
    return (
        f"[{rank_idx}] id={ranked.show_id} | {ranked.title} ({cand.type}, {year})\n"
        f"    Genres: {genres}\n"
        f"    Ranker score: {ranked.score}\n"
        f"    Ranker's brief_reason: {ranked.brief_reason}\n"
        f"    Description: {desc}"
    )


# ---------- Main entry point ----------

async def explain(
    user_query: str,
    ranked: list[RankedMovie],
    candidates: list[Candidate],
) -> list[ExplainedResult]:
    """
    Produce polished per-result explanations and return ExplainedResult list
    (ready for API response).
    """
    if not ranked:
        return []

    lookup = _build_candidate_lookup(candidates)

    # Build the prompt items
    prompt_items = []
    pairs: list[tuple[RankedMovie, Candidate]] = []
    for i, r in enumerate(ranked, start=1):
        cand = lookup.get(r.show_id)
        if cand is None:
            # Shouldn't happen (ranker filters to valid ids) but be safe
            continue
        prompt_items.append(_format_item_for_prompt(i, r, cand))
        pairs.append((r, cand))

    if not pairs:
        return []
    
    items_block = "\n\n".join(prompt_items)
    user_prompt = f"""# User's brief
{user_query}

# Ranked titles to write blurbs for
{items_block}

Write one polished sentence per title, following the style rules. Every \
`show_id` in your output must match one of the ids above exactly."""

    client = _get_openai()
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=_RESPONSE_FORMAT,
        temperature=0.6,  # higher — this is writing, we want some voice
    )

    raw_json = resp.choices[0].message.content
    data = json.loads(raw_json)
    batch = _ExplanationBatch(**data)

    # Merge explanations with the full ranked data
    expl_by_id = {item.show_id: item.explanation for item in batch.items}

    results: list[ExplainedResult] = []
    for r, cand in pairs:
        explanation = expl_by_id.get(
            r.show_id,
            r.brief_reason,  # fallback if the model somehow missed one
        )
        results.append(
            ExplainedResult(
                show_id=r.show_id,
                title=r.title,
                type=cand.type,
                release_year=cand.release_year,
                genres=list(cand.genres),
                score=r.score,
                explanation=explanation,
            )
        )
    return results