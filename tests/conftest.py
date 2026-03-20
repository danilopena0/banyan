"""
Shared fixtures for the banyan test suite.
"""
import pytest
from schemas.briefing import DailyBriefing, ConceptExplanation
from schemas.paper import PaperSummary


@pytest.fixture
def sample_paper_dict() -> dict:
    """Minimal valid paper dict as returned by search_arxiv."""
    return {
        "id": "http://arxiv.org/abs/2401.00001v1",
        "title": "Attention Is All You Need Again",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "We propose a new transformer architecture that improves upon the original.",
        "url": "http://arxiv.org/pdf/2401.00001v1",
        "published": "2024-01-01T00:00:00+00:00",
        "categories": ["cs.LG", "cs.AI"],
    }


@pytest.fixture
def sample_paper_summary() -> PaperSummary:
    """A fully populated PaperSummary instance."""
    return PaperSummary(
        title="Attention Is All You Need Again",
        authors=["Alice Smith", "Bob Jones"],
        plain_english_summary="A new transformer architecture that beats the baseline.",
        methods="Uses scaled dot-product attention: A = softmax(QK^T / sqrt(d_k)) V.",
        significance="Advances state-of-the-art on five benchmarks.",
        key_contribution="A novel positional encoding scheme that generalises to long sequences.",
        url="http://arxiv.org/abs/2401.00001v1",
        categories=["cs.LG"],
    )


@pytest.fixture
def sample_concept() -> ConceptExplanation:
    """A fully populated ConceptExplanation instance."""
    return ConceptExplanation(
        name="Attention Mechanism",
        plain_english="Attention allows a model to focus on relevant parts of the input.",
        example="Like a search engine that scores every word for relevance to a query.",
        why_it_matters="Enables parallelisation unlike recurrence; O(n^2) complexity.",
        connected_to_today="Used in every paper today via self-attention layers.",
        learn_more_url="https://distill.pub/2016/augmented-rnns/",
    )


@pytest.fixture
def minimal_briefing() -> DailyBriefing:
    """Minimal valid DailyBriefing with only required fields filled."""
    return DailyBriefing(
        date="2024-01-01",
        emerging_themes="Transformers continue to dominate the field.",
    )


@pytest.fixture
def full_briefing(sample_paper_summary, sample_concept) -> DailyBriefing:
    """DailyBriefing with all optional sections populated."""
    return DailyBriefing(
        date="2024-01-01",
        most_discussed=[sample_paper_summary],
        notable_papers=[sample_paper_summary],
        emerging_themes="Transformers continue to dominate. Diffusion models are rising.",
        web_insights=["OpenAI released GPT-5.", "Google announced Gemini Ultra 2."],
        concept_of_the_day=sample_concept,
        total_papers_analyzed=42,
        errors=["arXiv rate limit hit for query 'diffusion'"],
    )
