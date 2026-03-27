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


class FoundationalConcept(BaseModel):
    """A mathematically rigorous explanation of a core DS/ML concept for MS-level practitioners."""
    name: str = Field(description="Name of the concept (e.g., 'Bias-Variance Decomposition')")
    plain_english: str = Field(
        description=(
            "Formal mathematical definition and derivation targeting an MS-level reader with 5+ years of experience. "
            "State the concept precisely using notation (e.g. E[X], sigma^2, argmin). "
            "Include the key theorem, decomposition, or identity and explain the intuition behind it. "
            "LaTeX math is encouraged: use $...$ for inline and $$...$$ for display equations."
        )
    )
    example: str = Field(
        description=(
            "A worked example or non-trivial application that reveals the mathematical structure — "
            "e.g. a derivation step, a connection to a loss function, or a counterexample to a common misconception. "
            "Suitable for someone who already knows calculus, linear algebra, and probability theory. "
            "LaTeX math is encouraged."
        )
    )
    why_it_matters: str = Field(
        description=(
            "Why this concept is load-bearing in modern ML — include key theoretical properties "
            "(convergence guarantees, approximation tradeoffs, complexity bounds) and connect to "
            "practical consequences in model design, training, or evaluation. LaTeX math is encouraged."
        )
    )
    learn_more_url: str = Field(
        default="",
        description="URL to a resource for this concept"
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
            "Anchor observations in specific techniques, objectives, or theoretical results. "
            "Write equations in plain ASCII (e.g. 'L = -log P(y|x)'). "
            "NEVER use LaTeX delimiters ($, $$) or backslash commands (\\sigma, \\cdot, etc.)."
        )
    )
    concepts_of_the_day: list[ConceptExplanation] = Field(
        description="1-3 foundational DS/ML concepts drawn from today's papers — more when multiple distinct concepts appear",
        default_factory=list
    )
    foundational_concepts: list[FoundationalConcept] = Field(
        description="Exactly 2 beginner-friendly foundational DS/ML concepts, independent of today's papers",
        default_factory=list
    )

    @field_validator("concepts_of_the_day", mode="before")
    @classmethod
    def coerce_concepts(cls, v: Any) -> list:
        """LLMs may return a single object instead of a list — wrap it."""
        if isinstance(v, dict):
            return [v]
        if v is None:
            return []
        return v
    total_papers_analyzed: int = Field(default=0)
    errors: list[str] = Field(
        description="Non-fatal errors encountered during research",
        default_factory=list
    )
