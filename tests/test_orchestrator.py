"""
End-to-end orchestrator test — simulates a multi-turn conversation.

Run: uv run python tests\test_orchestrator.py
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import close_pool
from app.orchestrator import run_turn


def print_turn(turn_label: str, user_msg: str, result: dict):
    print(f"\n{'='*72}")
    print(f"[{turn_label}] USER: {user_msg}")
    print('='*72)
    pq = result["parsed_query"]
    print(f"Intent: {pq.intent}  |  type={pq.type}  genres={pq.genres}  "
          f"era=({pq.era_start},{pq.era_end})  follow_up_target={pq.follow_up_target}")
    print(f"Latency: {result['latency_ms']} ms")
    print(f"chat_answer: {result['chat_answer']!r}")
    print(f"results count: {len(result['results'])}")

    if result["chat_answer"]:
        print(f"\nASSISTANT (chat): {result['chat_answer']}")
    if result["results"]:
        print("\nASSISTANT recommendations:")
        for i, r in enumerate(result["results"], 1):
            print(f"  {i}. {r.title} ({r.type}, {r.release_year})  [{r.score}]")
            print(f"     {r.explanation}")
    if not result["chat_answer"] and not result["results"]:
        print("\n[!] BOTH chat_answer AND results are empty — something went wrong")


async def main():
    session = str(uuid.uuid4())
    print(f"Session: {session}")

    r = await run_turn(session, "A dark psychological thriller like Mindhunter")
    print_turn("Turn 1", "A dark psychological thriller like Mindhunter", r)

    r = await run_turn(session, "Any TV shows in that list? Only shows.")
    print_turn("Turn 2 (refine)", "Any TV shows in that list? Only shows.", r)

    r = await run_turn(session, "Tell me more about the first one")
    print_turn("Turn 3 (follow-up)", "Tell me more about the first one", r)

    r = await run_turn(session, "Which of those would you start with tonight?")
    print_turn("Turn 4 (follow-up meta)", "Which of those would you start with tonight?", r)

    r = await run_turn(session, "Actually forget that. I want a feel-good Korean rom-com.")
    print_turn("Turn 5 (new search)", "Actually forget that. I want a feel-good Korean rom-com.", r)

    r = await run_turn(session, "Something more recent please")
    print_turn("Turn 6 (refine)", "Something more recent please", r)

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())