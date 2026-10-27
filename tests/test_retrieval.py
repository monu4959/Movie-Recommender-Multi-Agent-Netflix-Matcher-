"""
Standalone test for retrieval.py — bypasses the Understanding agent
and feeds hand-crafted ParsedQuery objects directly.

Run: uv run python tests\test_retrieval.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import close_pool
from app.models import ParsedQuery
from app.retrieval import retrieve


async def show_results(label: str, pq: ParsedQuery):
    print(f"\n=== {label} ===")
    print(f"  Query: {pq.semantic_query!r}")
    print(f"  Filters: type={pq.type}, genres={pq.genres}, era=({pq.era_start},{pq.era_end})")
    results = await retrieve(pq)
    print(f"  Got {len(results)} candidates")
    for i, c in enumerate(results[:10], 1):
        print(
            f"   {i:2d}. {c.title[:50]:50s} "
            f"[{c.type}, {c.release_year}] "
            f"fused={c.fused_score:.4f} sem={c.semantic_score:.3f} kw={c.keyword_score:.3f}"
        )


async def main():
    # 1. Pure vibe query
    await show_results(
        "Mind-bending sci-fi thrillers",
        ParsedQuery(
            semantic_query="mind-bending cerebral sci-fi thriller with layered plot",
            intent="new_search",
            genres=["Sci-Fi & Fantasy", "Thrillers"],
        ),
    )

    # 2. Specific title (keyword should dominate)
    await show_results(
        "Exact title: Breaking Bad",
        ParsedQuery(
            semantic_query="Breaking Bad",
            intent="new_search",
        ),
    )

    # 3. Type + era filter
    await show_results(
        "90s action movies",
        ParsedQuery(
            semantic_query="action adventure explosions",
            intent="new_search",
            type="Movie",
            era_start=1990,
            era_end=1999,
        ),
    )

    # 4. Mood + exclusion
    await show_results(
        "Cozy TV, no horror",
        ParsedQuery(
            semantic_query="feel-good comfort show warm lighthearted",
            intent="new_search",
            type="TV Show",
            mood_keywords=["cozy", "warm"],
            exclude_keywords=["Horror Movies", "TV Horror"],
        ),
    )

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())