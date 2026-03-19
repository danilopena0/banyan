"""Pydantic models for arXiv papers."""
from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """Raw paper data fetched from arXiv."""
    id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: str
    categories: list[str]


class PaperSummary(BaseModel):
    """LLM-structured summary of an arXiv paper."""
    title: str
    authors: list[str] = Field(description="List of author names")
    plain_english_summary: str = Field(
        description="2-3 sentence summary understandable by non-experts"
    )
    methods: str = Field(
        description="The core methods, architectures, or techniques used (e.g. 'fine-tunes LLaMA-3 with DPO on a synthetic preference dataset')"
    )
    significance: str = Field(
        description="Why this paper matters to the AI/ML field"
    )
    key_contribution: str = Field(
        description="The single most important technical contribution"
    )
    url: str = Field(default="")
    categories: list[str] = Field(default_factory=list)
