"""
Full pipeline test: Understanding -> Retrieval -> Ranking -> Explanation.

Run: uv run python tests\test_explanation.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.explanation import explain
from agents.ranking import rank
from agents.understanding import understand
from app.db import close_pool
from app.retrieval import retrieve


async def run_pipeline(brief: str):
    print(f"\n{'='*72}")
    print(f"BRIEF: {brief}")
    print('='*72)

    parsed = await understand(brief)
    candidates = await retrieve(parsed)
    ranked = await rank(brief, candidates, parsed)
    explained = await explain(brief, ranked, candidates)

    for i, r in enumerate(explained, 1):
        print(f"\n  {i}. {r.title} ({r.type}, {r.release_year})  [score: {r.score}]")
        print(f"     {r.explanation}")


async def main():
    await run_pipeline("Something like Inception but less confusing")
    await run_pipeline("Cozy TV show for a rainy Sunday afternoon")
    await run_pipeline("90s action movies with car chases, nothing horror-y")
    await run_pipeline("A Korean romantic comedy series")
    await run_pipeline("Dark psychological thriller like Mindhunter")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())