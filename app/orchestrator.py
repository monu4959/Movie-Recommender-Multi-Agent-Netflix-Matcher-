"""
LangGraph orchestrator — wires the four agents into a single pipeline.

Flow:
    START
      │
      ▼
   understand   (always runs first — produces ParsedQuery)
      │
      ├── intent == "follow_up" ──► chat ──► END
      │
      ▼
   retrieve
      │
      ▼
   rank
      │
      ▼
   explain
      │
      ▼
    END
"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.chat import chat
from agents.explanation import explain
from agents.ranking import rank
from agents.understanding import understand
from app.models import (
    Candidate,
    ChatMessage,
    ExplainedResult,
    ParsedQuery,
    RankedMovie,
    SessionState,
)
from app.retrieval import retrieve


# ---------- Session store (in-memory) ----------

_sessions: dict[str, SessionState] = {}


def get_or_create_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState(session_id=session_id)
    return _sessions[session_id]


# ---------- Graph state ----------
#
# IMPORTANT: Every key that flows between nodes MUST be declared here.
# LangGraph drops keys that aren't in this TypedDict — silently. That's why
# we include `ranked` explicitly (was previously `_ranked` and got lost).

class GraphState(TypedDict, total=False):
    # Inputs
    session_id: str
    user_query: str
    history: list[ChatMessage]
    previous_results: list[ExplainedResult]

    # Intermediate agent outputs
    parsed: ParsedQuery
    candidates: list[Candidate]
    ranked: list[RankedMovie]

    # Final outputs
    results: list[ExplainedResult]
    chat_answer: str


# ---------- Nodes ----------
#
# Each node returns a partial dict of JUST the keys it updates.
# LangGraph merges these into the running state — we don't need to spread the
# whole state forward ourselves.

async def _understand_node(state: GraphState) -> dict:
    parsed = await understand(state["user_query"], state.get("history", []))
    return {"parsed": parsed}


async def _retrieve_node(state: GraphState) -> dict:
    candidates = await retrieve(state["parsed"])
    return {"candidates": candidates}


async def _rank_node(state: GraphState) -> dict:
    # Only pass history when the user is genuinely building on prior context.
    # For a fresh search, conversation history is noise for the ranker.
    intent = state["parsed"].intent
    history_for_rank = (
        state.get("history", []) if intent in ("refine", "more_like") else []
    )
    ranked = await rank(
        user_query=state["user_query"],
        candidates=state["candidates"],
        parsed=state["parsed"],
        history=history_for_rank,
    )
    return {"ranked": ranked}

async def _explain_node(state: GraphState) -> dict:
    results = await explain(
        user_query=state["user_query"],
        ranked=state["ranked"],
        candidates=state["candidates"],
    )
    return {"results": results}


async def _chat_node(state: GraphState) -> dict:
    answer = await chat(
        user_query=state["user_query"],
        previous_results=state.get("previous_results", []),
        parsed=state["parsed"],
        history=state.get("history", []),
    )
    return {"chat_answer": answer, "results": []}


# ---------- Routing ----------

def _route_after_understand(state: GraphState) -> str:
    """Branch based on the intent classified by the Understanding agent."""
    intent = state["parsed"].intent
    has_previous = bool(state.get("previous_results"))

    # Follow-up only makes sense if there are previous results to follow up on
    if intent == "follow_up" and has_previous:
        return "chat"
    return "retrieve"


# ---------- Graph assembly ----------

def _build_graph():
    g = StateGraph(GraphState)
    g.add_node("understand", _understand_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("rank", _rank_node)
    g.add_node("explain", _explain_node)
    g.add_node("chat", _chat_node)

    g.add_edge(START, "understand")
    g.add_conditional_edges(
        "understand",
        _route_after_understand,
        {"retrieve": "retrieve", "chat": "chat"},
    )
    g.add_edge("retrieve", "rank")
    g.add_edge("rank", "explain")
    g.add_edge("explain", END)
    g.add_edge("chat", END)

    return g.compile()


_graph = _build_graph()


# ---------- Public entry point ----------

async def run_turn(session_id: str, user_query: str) -> dict:
    """
    Execute one turn of conversation. Updates session state as a side effect.
    """
    t0 = time.monotonic()
    session = get_or_create_session(session_id)

    initial_state: GraphState = {
        "session_id": session_id,
        "user_query": user_query,
        "history": list(session.history),
        "previous_results": list(session.last_results),
        "candidates": [],
        "ranked": [],
        "results": [],
        "chat_answer": "",
    }

    final_state = await _graph.ainvoke(initial_state)

    # Update session state
    session.history.append(ChatMessage(role="user", content=user_query))
    session.last_parsed = final_state["parsed"]

    if final_state.get("chat_answer"):
        session.history.append(
            ChatMessage(role="assistant", content=final_state["chat_answer"])
        )
    elif final_state.get("results"):
        # Compact summary of results in history; full prose stays in `results`
        titles = ", ".join(
            f"{i+1}. {r.title}" for i, r in enumerate(final_state["results"][:8])
        )
        session.history.append(
            ChatMessage(role="assistant", content=f"Top picks: {titles}")
        )
        session.last_results = final_state["results"]

    # Keep history bounded
    if len(session.history) > 20:
        session.history = session.history[-20:]

    latency_ms = int((time.monotonic() - t0) * 1000)

    return {
        "session_id": session_id,
        "results": final_state.get("results", []),
        "chat_answer": final_state.get("chat_answer", ""),
        "parsed_query": final_state["parsed"],
        "latency_ms": latency_ms,
    }