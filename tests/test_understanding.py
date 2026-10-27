"""
Standalone test for the Understanding agent.
Runs hand-crafted briefs through and prints the parsed output.

Run: uv run python tests\test_understanding.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.understanding import understand
from app.models import ChatMessage


def print_parsed(label: str, parsed):
    print(f"\n=== {label} ===")
    print(f"  intent:           {parsed.intent}")
    print(f"  semantic_query:   {parsed.semantic_query!r}")
    print(f"  type:             {parsed.type}")
    print(f"  genres:           {parsed.genres}")
    print(f"  mood_keywords:    {parsed.mood_keywords}")
    print(f"  era:              ({parsed.era_start}, {parsed.era_end})")
    print(f"  exclude_keywords: {parsed.exclude_keywords}")
    print(f"  reference_titles: {parsed.reference_titles}")
    print(f"  follow_up_target: {parsed.follow_up_target}")


async def main():
    # --- Single-turn tests ---

    p = await understand("Something like Inception but less confusing")
    print_parsed("Like Inception but less confusing", p)

    p = await understand("I want a cozy comedy for a rainy Sunday")
    print_parsed("Cozy comedy rainy Sunday", p)

    p = await understand("90s action movies with car chases, no horror")
    print_parsed("90s action, no horror", p)

    p = await understand("Recommend me a Korean TV show")
    print_parsed("Korean TV show", p)

    p = await understand("Something to watch")  # vague — should stay null
    print_parsed("Vague: something to watch", p)

    p = await understand("A recent thriller series like Mindhunter")
    print_parsed("Recent thriller like Mindhunter", p)

    # --- Multi-turn tests ---

    history1 = [
        ChatMessage(role="user", content="Recommend me a cozy comedy"),
        ChatMessage(role="assistant", content="Here are some cozy comedies: Schitt's Creek, Ted Lasso, The Good Place, Parks and Recreation, Brooklyn Nine-Nine"),
    ]
    p = await understand("More modern ones", history=history1)
    print_parsed("Refine: more modern (cozy comedy context)", p)

    history2 = [
        ChatMessage(role="user", content="Show me some sci-fi thrillers"),
        ChatMessage(role="assistant", content="Top picks: 1. Inception, 2. Tenet, 3. Ex Machina, 4. Annihilation, 5. Arrival"),
    ]
    p = await understand("Tell me more about the third one", history=history2)
    print_parsed("Follow-up: third one (sci-fi context)", p)

    p = await understand("More like the second one", history=history2)
    print_parsed("More like: the second one (sci-fi context)", p)

    p = await understand("Actually, forget all that. I want a documentary.", history=history2)
    print_parsed("Topic switch: new search", p)


if __name__ == "__main__":
    asyncio.run(main())