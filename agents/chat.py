"""
Chat Agent — handles follow-up questions about previously-shown results.

Called when the Understanding agent classifies intent as "follow_up"
(e.g., "tell me more about the second one", "who's in it?").

No retrieval, no ranking — just answer from the stored context.
"""

from __future__ import annotations

import asyncio
import json

from openai import OpenAI

from app.config import settings
from app.models import ChatMessage, ExplainedResult, ParsedQuery


_openai_client: OpenAI | None = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


SYSTEM_PROMPT = """You are the Chat module for a Netflix-style recommender. \
The user previously received a list of ranked recommendations and is now asking \
a follow-up question about one of them (or about the list in general).

# What you have access to

- The list of previously-recommended titles, with their metadata (type, year, \
genres, score) and the one-line explanation that was shown to the user.
- The recent conversation history.
- The user's current question.

# Your job

Answer the user's question in 1-3 short sentences. Be specific and helpful. \
Address them in second person. Don't restate the entire list; focus on what \
they actually asked about.

# Rules

1. Only use information from the provided context. If the user asks about a \
detail you don't have (e.g., "who directed it?" and you only have description \
+ genres), say so briefly and offer what you do know.
2. If the user's question isn't really a follow-up but a new search disguised \
as one, still answer conversationally and suggest they rephrase as a fresh \
request.
3. No hyperbole, no "I'd be happy to help!", no restating the question.
4. Match the tone of the explanations — knowledgeable friend, not sales copy.

# Examples

User: "Tell me more about the third one."
Context: Third result was "Shutter Island (Movie, 2010), Thrillers".
→ "Shutter Island is Scorsese's psychological thriller where a U.S. marshal \
investigates a disappearance at a remote mental institution — the reality-\
questioning structure is what makes it sit closest to the Mindhunter vibe you \
asked for."

User: "How long is the second one?"
Context: Second result is "Breaking Bad (TV Show, 2013)", no duration info in \
context beyond genre.
→ "Breaking Bad ran for 5 seasons total. I don't have episode-count details \
here, but it's one of the most complete arcs on the platform — worth the \
commitment if you liked the premise."

User: "Which one is funniest?"
Context: List of 8 titles with genres and explanations.
→ "Schitt's Creek is the clearest comedy on the list — the others lean \
dramatic-cozy rather than laugh-out-loud. Ted Lasso would be second."
"""


def _format_previous_results(results: list[ExplainedResult]) -> str:
    if not results:
        return "(no previous results)"
    lines = []
    for i, r in enumerate(results, 1):
        genres = ", ".join(r.genres) if r.genres else "(no genres)"
        lines.append(
            f"{i}. {r.title} ({r.type}, {r.release_year}) — score {r.score}\n"
            f"     Genres: {genres}\n"
            f"     Shown to user: {r.explanation}"
        )
    return "\n".join(lines)


def _format_history(history: list[ChatMessage], max_turns: int = 6) -> str:
    if not history:
        return "(no prior conversation)"
    recent = history[-max_turns:]
    lines = []
    for msg in recent:
        role = "User" if msg.role == "user" else "Assistant"
        content = msg.content.strip()
        if msg.role == "assistant" and len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def chat(
    user_query: str,
    previous_results: list[ExplainedResult],
    parsed: ParsedQuery,
    history: list[ChatMessage] | None = None,
) -> str:
    """
    Answer a follow-up question about previously-shown recommendations.
    Returns a plain string (no structured output needed — it's conversational).
    """
    history = history or []

    # If the user referenced a specific position and we have it, surface it
    target_hint = ""
    if parsed.follow_up_target and 1 <= parsed.follow_up_target <= len(previous_results):
        target = previous_results[parsed.follow_up_target - 1]
        target_hint = (
            f"\n# Target the user is asking about (position {parsed.follow_up_target})\n"
            f"{target.title} ({target.type}, {target.release_year})\n"
            f"Genres: {', '.join(target.genres)}\n"
            f"Shown to user: {target.explanation}\n"
        )

    user_prompt = f"""# Previous recommendations
{_format_previous_results(previous_results)}
{target_hint}
# Recent conversation
{_format_history(history)}

# User's follow-up question
{user_query}

Answer in 1-3 short sentences."""

    client = _get_openai()
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

    return resp.choices[0].message.content.strip()