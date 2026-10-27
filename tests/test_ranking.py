"""
End-to-end test for the Ranking agent.
Runs Understanding + Retrieval + Ranking on a few briefs.

Run: uv run python tests\test_ranking.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ranking import rank
from agents.understanding import understand
from app.db import close_pool
from app.retrieval import retrieve


async def run_pipeline(brief: str):
    print(f"\n{'='*70}")
    print(f"BRIEF: {brief}")
    print('='*70)

    parsed = await understand(brief)
    print(f"Parsed intent: {parsed.intent}, type={parsed.type}, "
          f"genres={parsed.genres}, era=({parsed.era_start},{parsed.era_end})")

    candidates = await retrieve(parsed)
    print(f"Retrieved {len(candidates)} candidates")

    ranked = await rank(brief, candidates, parsed)
    print(f"\nRanked top {len(ranked)}:")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. [{r.score:3d}] {r.title}")
        print(f"       {r.brief_reason}")


async def main():
    await run_pipeline("Something like Inception but less confusing")
    await run_pipeline("Cozy TV show for a rainy Sunday afternoon")
    await run_pipeline("90s action movies with car chases, nothing horror-y")
    await run_pipeline("A Korean romantic comedy series")
    await run_pipeline("Dark psychological thriller like Mindhunter")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())