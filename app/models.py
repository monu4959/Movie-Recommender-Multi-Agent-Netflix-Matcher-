from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------- Query Understanding output ----------

class ChatMessage(BaseModel):
    """A single turn in the conversation history."""
    role: Literal["user", "assistant"]
    content: str

class ParsedQuery(BaseModel):
    """Structured representation of a user brief, produced by the Understanding agent."""

    semantic_query: str = Field(
        description="Cleaned query string for semantic retrieval. "
                    "Captures vibe/theme without filter-able specifics."
    )
    intent: Literal["new_search", "refine", "more_like", "follow_up"] = Field(
        description="What the user is trying to do this turn."
    )
    type: Optional[Literal["Movie", "TV Show"]] = None
    genres: list[str] = Field(default_factory=list)
    mood_keywords: list[str] = Field(default_factory=list)
    era_start: Optional[int] = None
    era_end: Optional[int] = None
    exclude_keywords: list[str] = Field(default_factory=list)
    reference_titles: list[str] = Field(
        default_factory=list,
        description="Titles the user mentioned as anchors ('like Inception')."
    )
    follow_up_target: Optional[int] = Field(
        default=None,
        description="If intent=follow_up, the 1-indexed position from last results."
    )


# ---------- Retrieval output ----------

class Candidate(BaseModel):
    show_id: str
    type: str
    title: str
    release_year: Optional[int]
    genres: list[str]
    description: Optional[str]
    director: list[str] = Field(default_factory=list)
    cast_members: list[str] = Field(default_factory=list)
    duration_value: Optional[int] = None
    duration_unit: Optional[str] = None
    rating: Optional[str] = None
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0


# ---------- Ranking output ----------

class RankedMovie(BaseModel):
    show_id: str
    title: str
    score: int = Field(ge=0, le=100)
    brief_reason: str


class RankingResult(BaseModel):
    ranked: list[RankedMovie]


# ---------- Explanation output ----------

class ExplainedResult(BaseModel):
    show_id: str
    title: str
    type: str
    release_year: Optional[int]
    genres: list[str]
    score: int
    explanation: str


# ---------- API contract ----------

class RecommendRequest(BaseModel):
    session_id: str
    query: str


class RecommendResponse(BaseModel):
    session_id: str
    results: list[ExplainedResult]
    parsed_query: ParsedQuery  # for debug/transparency; strip in prod
    latency_ms: int

# ---------- Session state ----------

class SessionState(BaseModel):
    """Per-user conversation state, held in memory by the server."""
    session_id: str
    history: list[ChatMessage] = Field(default_factory=list)
    last_results: list[ExplainedResult] = Field(default_factory=list)
    last_parsed: Optional[ParsedQuery] = None


# ---------- Chat agent output (for follow-ups) ----------

class ChatAnswer(BaseModel):
    """Output of the Chat agent when answering a follow-up question."""
    answer: str