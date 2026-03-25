"""Pydantic models for arXiv papers."""
from pydantic import BaseModel, Field


class PaperSummary(BaseModel):
    """LLM-structured summary of an arXiv paper."""
    title: str
    authors: list[str] = Field(description="List of author names")
    plain_english_summary: str = Field(
        description="2-3 sentence summary understandable by non-experts"
    )
    methods: str = Field(
        description=(
            "The core methods, architectures, or techniques used. Describe key ideas and loss functions in plain text — "
            "e.g. 'optimizes a contrastive objective where positive pairs are pulled together and negatives pushed apart'. "
            "Use ASCII-style math (L = ...) for equations. NEVER use LaTeX delimiters ($, $$, \\sigma, \\cdot) or backslash commands."
        )
    )
    significance: str = Field(
        description="Why this paper matters to the AI/ML field"
    )
    key_contribution: str = Field(
        description="The single most important technical contribution"
    )
    url: str = Field(default="")
    categories: list[str] = Field(default_factory=list)
