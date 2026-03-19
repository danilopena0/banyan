"""
LangGraph state schema using Pydantic v2.
All state transitions are validated at each node.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field
from schemas.briefing import DailyBriefing


class ResearchState(BaseModel):
    """
    Typed state passed through the LangGraph graph.

    Pattern: Using Pydantic for state ensures type safety and makes
    debugging easier — you always know exactly what's in each field.
    """
    # ReAct tool-calling loop messages (list of LangChain message objects)
    messages: list[Any] = Field(default_factory=list)

    # AI news fetched before the ReAct loop via curated Tavily queries
    web_news: list[str] = Field(default_factory=list)

    # Raw data collected by tools
    raw_papers: list[dict] = Field(default_factory=list)

    # After deduplication against ChromaDB
    new_content_ids: list[str] = Field(default_factory=list)

    # After semantic RAG retrieval — these chunks go to synthesis
    retrieved_context: list[str] = Field(default_factory=list)

    # Final structured output
    briefing: Optional[DailyBriefing] = None

    # Metadata
    date: str = Field(default="")
    errors: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
