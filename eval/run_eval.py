"""
LLM-as-judge evaluation harness.

Runs the full pipeline over a test set of briefs, then uses GPT-4o as a judge
to score each result on four dimensions. Writes a JSON + readable Markdown
report.

Usage:
    uv run python eval\run_eval.py

This intentionally uses temperature=0 everywhere for reproducibility,
overriding the agents' default temperatures.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from app.config import settings
from app.db import close_pool
from app.orchestrator import run_turn


TEST_SET_PATH = Path(__file__).parent / "test_briefs.json"
RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_MODEL = "gpt-4o"

# Number of top results to judge per brief
JUDGE_TOP_N = 5


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a movie recommendation \
system. For each recommendation, you score how well it matches the user's \
brief on four dimensions.

# Scoring dimensions (each 0-5)

1. `relevance` — How well does the title match the user's explicit \
constraints (type, era, genre, exclusions)?
   - 5: fully satisfies all explicit constraints
   - 3: satisfies most, violates a minor one
   - 0: violates hard constraints (wrong type, wrong era, excluded category)

2. `vibe_match` — How well does the title match the user's mood / vibe / \
reference anchors?
   - 5: captures exactly what the user meant (e.g., "cerebral sci-fi like \
Inception" → a title that's genuinely cerebral sci-fi)
   - 3: adjacent vibe but misses the nuance
   - 0: wrong vibe entirely

3. `explanation_quality` — How good is the one-line explanation shown to the \
user?
   - 5: specific to this title AND references the brief, human-sounding, \
under 35 words
   - 3: correct but generic or slightly off-tone
   - 0: wrong, hallucinated, or copy-paste boilerplate

4. `confidence` — Your overall confidence this is a result the user would be \
happy to see.
   - 5: would strongly recommend
   - 3: acceptable
   - 0: bad pick

Return JSON with scores for each recommendation plus a short `rationale` per \
title (1 sentence) and an overall `brief_comment` (1 sentence about the list \
as a whole — diversity, overall quality, failure modes)."""


def _build_judge_schema(num_items: int) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_result",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["items", "brief_comment"],
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "position", "title",
                                "relevance", "vibe_match",
                                "explanation_quality", "confidence",
                                "rationale",
                            ],
                            "properties": {
                                "position": {"type": "integer"},
                                "title": {"type": "string"},
                                "relevance": {"type": "integer"},
                                "vibe_match": {"type": "integer"},
                                "explanation_quality": {"type": "integer"},
                                "confidence": {"type": "integer"},
                                "rationale": {"type": "string"},
                            },
                        },
                    },
                    "brief_comment": {"type": "string"},
                },
            },
            "strict": True,
        },
    }


async def judge_results(brief: dict, results: list[dict]) -> dict:
    """Run the LLM judge over one brief's results."""
    if not results:
        return {
            "items": [],
            "brief_comment": "System returned no results — cannot evaluate.",
        }

    client = OpenAI(api_key=settings.openai_api_key)

    items_text = "\n\n".join(
        f"[{i + 1}] {r['title']} ({r.get('type', '?')}, "
        f"{r.get('release_year', '?')})\n"
        f"    Genres: {', '.join(r.get('genres', []))}\n"
        f"    Explanation shown to user: {r.get('explanation', '')}"
        for i, r in enumerate(results)
    )

    user_prompt = f"""# User brief
{brief['brief']}

# Evaluation criteria for this brief
{json.dumps(brief['criteria'], indent=2)}

# System's top {len(results)} recommendations
{items_text}

Score each recommendation and add a brief_comment about the list overall."""

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=_build_judge_schema(len(results)),
        temperature=0,
    )

    return json.loads(resp.choices[0].message.content)


async def run_one(brief: dict) -> dict:
    """Run the pipeline on one brief + judge the output."""
    session_id = f"eval-{uuid.uuid4()}"
    t0 = time.monotonic()
    turn = await run_turn(session_id, brief["brief"])
    latency_ms = int((time.monotonic() - t0) * 1000)

    results = [r.model_dump() for r in turn["results"]][:JUDGE_TOP_N]
    judgment = await judge_results(brief, results)

    # Aggregate scores
    scores = {"relevance": 0.0, "vibe_match": 0.0,
              "explanation_quality": 0.0, "confidence": 0.0}
    n = len(judgment["items"])
    if n > 0:
        for item in judgment["items"]:
            for k in scores:
                scores[k] += item[k]
        for k in scores:
            scores[k] /= n

    return {
        "brief_id": brief["id"],
        "brief": brief["brief"],
        "latency_ms": latency_ms,
        "num_results": len(results),
        "parsed_intent": turn["parsed_query"].intent
            if turn["parsed_query"] else None,
        "avg_scores": scores,
        "judge_items": judgment["items"],
        "brief_comment": judgment["brief_comment"],
        "results": results,  # keep for the markdown report
    }


def write_markdown_report(runs: list[dict], out_path: Path) -> None:
    """Human-readable report. Useful for debugging and for the README."""
    lines = [
        "# Evaluation Report",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Aggregate scores (across all briefs)",
        "",
    ]

    # Overall aggregates
    totals = {"relevance": 0.0, "vibe_match": 0.0,
              "explanation_quality": 0.0, "confidence": 0.0}
    count = 0
    for r in runs:
        if r["num_results"] > 0:
            count += 1
            for k in totals:
                totals[k] += r["avg_scores"][k]
    if count:
        for k in totals:
            totals[k] /= count

    lines.append("| Metric | Score (/5) |")
    lines.append("|---|---|")
    for k in ["relevance", "vibe_match", "explanation_quality", "confidence"]:
        lines.append(f"| {k.replace('_', ' ').title()} | {totals[k]:.2f} |")
    lines.append("")

    # Per-brief details
    lines.append("## Per-brief results")
    for r in runs:
        lines.append(f"\n### {r['brief_id']}: _{r['brief']}_")
        lines.append(
            f"- Intent: `{r['parsed_intent']}`  | "
            f"Latency: {r['latency_ms']} ms  | "
            f"Results: {r['num_results']}"
        )
        if r["num_results"] > 0:
            s = r["avg_scores"]
            lines.append(
                f"- Avg scores — relevance {s['relevance']:.1f}, "
                f"vibe {s['vibe_match']:.1f}, "
                f"explanation {s['explanation_quality']:.1f}, "
                f"confidence {s['confidence']:.1f}"
            )
        lines.append(f"- Judge says: _{r['brief_comment']}_")
        lines.append("")
        for i, (item, rec) in enumerate(zip(r["judge_items"], r["results"]), 1):
            lines.append(
                f"  {i}. **{rec['title']}** "
                f"(R={item['relevance']}, V={item['vibe_match']}, "
                f"E={item['explanation_quality']}, C={item['confidence']})"
            )
            lines.append(f"     - {item['rationale']}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


async def main():
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)

    briefs = test_set["briefs"]
    print(f"Running eval on {len(briefs)} briefs...\n")

    runs = []
    for i, brief in enumerate(briefs, 1):
        print(f"[{i}/{len(briefs)}] {brief['id']}: {brief['brief']}")
        try:
            run = await run_one(brief)
            if run["num_results"] > 0:
                s = run["avg_scores"]
                print(
                    f"    → R={s['relevance']:.1f} V={s['vibe_match']:.1f} "
                    f"E={s['explanation_quality']:.1f} C={s['confidence']:.1f} "
                    f"({run['latency_ms']} ms)"
                )
            else:
                print(f"    → NO RESULTS")
            runs.append(run)
        except Exception as e:
            print(f"    ! FAILED: {type(e).__name__}: {e}")
            runs.append({
                "brief_id": brief["id"],
                "brief": brief["brief"],
                "error": f"{type(e).__name__}: {e}",
                "num_results": 0,
            })

    # Write reports
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"eval_{timestamp}.json"
    md_path = RESULTS_DIR / f"eval_{timestamp}.md"
    latest_md = RESULTS_DIR / "latest.md"

    json_path.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    write_markdown_report(runs, md_path)
    write_markdown_report(runs, latest_md)

    print(f"\nWrote:")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print(f"  {latest_md}")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())