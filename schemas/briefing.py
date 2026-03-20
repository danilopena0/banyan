"""Pydantic model for the final structured briefing output."""
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
from schemas.paper import PaperSummary


class ConceptExplanation(BaseModel):
    """A foundational DS/ML concept explained in plain English."""
    name: str = Field(description="Name of the concept (e.g., 'Attention Mechanism')")
    plain_english: str = Field(
        description="2-3 sentence plain-English explanation grounded in mathematical intuition — what is being optimized, computed, or approximated and why"
    )
    example: str = Field(
        description="A concrete, intuitive example or analogy that also touches on the underlying mechanics (e.g. what the vectors, gradients, or distributions are doing)"
    )
    why_it_matters: str = Field(
        description="Why this concept matters in practice, including key theoretical properties such as convergence guarantees, complexity, or approximation tradeoffs"
    )
    connected_to_today: str = Field(
        description="How this concept appears in today's papers — cite specific equations, objectives, or architectural choices from the research"
    )
    learn_more_url: str = Field(
        default="",
        description="URL to a beginner-friendly resource for this concept"
    )


class DailyBriefing(BaseModel):
    """Structured daily AI/ML research briefing."""
    date: str = Field(description="Date of the briefing in YYYY-MM-DD format")

    most_discussed: list[PaperSummary] = Field(
        description="Top papers worth highlighting from today's research",
        default_factory=list
    )
    notable_papers: list[PaperSummary] = Field(
        description="Other significant papers worth highlighting",
        default_factory=list
    )
    emerging_themes: str = Field(
        description=(
            "2-3 paragraph analysis of emerging trends across all sources. "
            "Anchor observations in specific techniques, objectives, or theoretical results — "
            "use inline math notation (e.g. KL divergence, attention formulas) where it sharpens the insight."
        )
    )
    web_insights: list[str] = Field(
        description="Key findings from web search about current AI developments",
        default_factory=list
    )

    @field_validator("web_insights", mode="before")
    @classmethod
    def coerce_web_insights(cls, v: Any) -> list[str]:
        """LLMs sometimes return a plain string instead of a list — wrap it."""
        if isinstance(v, str):
            return [s.strip() for s in v.split("\n") if s.strip()]
        return v

    concept_of_the_day: Optional[ConceptExplanation] = Field(
        description="A foundational DS/ML concept drawn from today's papers",
        default=None
    )
    total_papers_analyzed: int = Field(default=0)
    errors: list[str] = Field(
        description="Non-fatal errors encountered during research",
        default_factory=list
    )
