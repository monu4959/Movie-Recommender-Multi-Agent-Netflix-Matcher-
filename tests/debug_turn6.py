"""Trace Turn 6 through the full graph to find where results disappear."""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import close_pool
from app.models import ChatMessage, ParsedQuery
from app.retrieval import retrieve
from agents.ranking import rank
from agents.explanation import explain


async def main():
    # Simulate the parsed state that Turn 6 ends up with
    pq = ParsedQuery(
        semantic_query="recent Korean romantic comedy",
        intent="refine",
        type=None,
        genres=['Korean TV Shows', 'Romantic TV Shows'],
        mood_keywords=[],
        era_start=2021,
        era_end=None,
        exclude_keywords=[],
        reference_titles=[],
    )

    history = [
        ChatMessage(role="user", content="Actually forget that. I want a feel-good Korean rom-com."),
        ChatMessage(role="assistant", content="Top picks: 1. Romance is a bonus book, 2. When the Camellia Blooms, 3. Touch Your Heart, 4. My Secret Romance, 5. Nevertheless, 6. Something in the Rain, 7. 100 Days My Prince, 8. Lovestruck in the City"),
        ChatMessage(role="user", content="Something more recent please"),
    ]

    print("=== Retrieval ===")
    cands = await retrieve(pq)
    print(f"  Got {len(cands)} candidates")
    if cands:
        print(f"  First 3: {[c.title for c in cands[:3]]}")

    print("\n=== Ranking ===")
    try:
        ranked = await rank(
            user_query="Something more recent please",
            candidates=cands,
            parsed=pq,
            history=history,
        )
        print(f"  Got {len(ranked)} ranked")
        for r in ranked:
            print(f"    [{r.score}] {r.title}")
    except Exception as e:
        print(f"  RANKING FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await close_pool()
        return

    print("\n=== Explanation ===")
    try:
        results = await explain(
            user_query="Something more recent please",
            ranked=ranked,
            candidates=cands,
        )
        print(f"  Got {len(results)} explained")
        for r in results:
            print(f"    [{r.score}] {r.title}: {r.explanation[:80]}")
    except Exception as e:
        print(f"  EXPLAIN FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())