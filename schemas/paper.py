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
            "The core methods, architectures, or techniques used. Include conceptual and mathematical detail where relevant — "
            "e.g. loss functions, objective formulations, architectural choices, training procedures, or key algorithmic steps. "
            "Example: 'Fine-tunes LLaMA-3 using DPO (Direct Preference Optimization), which optimizes a Bradley-Terry preference "
            "model directly without a separate reward model. The training objective is: L = -E[log σ(β log π_θ(y_w|x)/π_ref(y_w|x) "
            "- β log π_θ(y_l|x)/π_ref(y_l|x))]'"
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
