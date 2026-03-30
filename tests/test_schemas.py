"""
Tests for schemas/briefing.py and schemas/paper.py.

These tests exercise Pydantic v2 validation directly — no mocking needed.
"""
import pytest
from pydantic import ValidationError

from schemas.briefing import DailyBriefing, ConceptExplanation
from schemas.paper import PaperSummary


# ---------------------------------------------------------------------------
# PaperSummary
# ---------------------------------------------------------------------------

class TestPaperSummary:
    def test_paper_summary_valid_all_required_fields(self):
        # Arrange / Act
        paper = PaperSummary(
            title="Test Paper",
            authors=["Alice"],
            plain_english_summary="A test.",
            methods="We used a neural net.",
            significance="It matters.",
            key_contribution="A new loss function.",
        )

        # Assert
        assert paper.title == "Test Paper"
        assert paper.authors == ["Alice"]
        assert paper.plain_english_summary == "A test."
        assert paper.methods == "We used a neural net."
        assert paper.significance == "It matters."
        assert paper.key_contribution == "A new loss function."

    def test_paper_summary_defaults_for_optional_fields(self):
        # Arrange / Act
        paper = PaperSummary(
            title="Test Paper",
            authors=["Alice"],
            plain_english_summary="A test.",
            methods="Methods here.",
            significance="Significance here.",
            key_contribution="Contribution here.",
        )

        # Assert — optional fields have defaults
        assert paper.url == ""
        assert paper.categories == []

    def test_paper_summary_optional_fields_accept_values(self):
        # Arrange / Act
        paper = PaperSummary(
            title="Test Paper",
            authors=["Alice"],
            plain_english_summary="A test.",
            methods="Methods here.",
            significance="Significance here.",
            key_contribution="Contribution here.",
            url="https://arxiv.org/abs/2401.00001",
            categories=["cs.LG", "cs.AI"],
        )

        # Assert
        assert paper.url == "https://arxiv.org/abs/2401.00001"
        assert paper.categories == ["cs.LG", "cs.AI"]

    def test_paper_summary_missing_title_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PaperSummary(
                authors=["Alice"],
                plain_english_summary="A test.",
                methods="Methods here.",
                significance="Significance here.",
                key_contribution="Contribution here.",
            )
        assert "title" in str(exc_info.value)

    def test_paper_summary_missing_authors_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PaperSummary(
                title="Test Paper",
                plain_english_summary="A test.",
                methods="Methods here.",
                significance="Significance here.",
                key_contribution="Contribution here.",
            )
        assert "authors" in str(exc_info.value)

    def test_paper_summary_missing_plain_english_summary_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PaperSummary(
                title="Test Paper",
                authors=["Alice"],
                methods="Methods here.",
                significance="Significance here.",
                key_contribution="Contribution here.",
            )
        assert "plain_english_summary" in str(exc_info.value)

    def test_paper_summary_missing_methods_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PaperSummary(
                title="Test Paper",
                authors=["Alice"],
                plain_english_summary="A test.",
                significance="Significance here.",
                key_contribution="Contribution here.",
            )
        assert "methods" in str(exc_info.value)

    def test_paper_summary_missing_significance_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PaperSummary(
                title="Test Paper",
                authors=["Alice"],
                plain_english_summary="A test.",
                methods="Methods here.",
                key_contribution="Contribution here.",
            )
        assert "significance" in str(exc_info.value)

    def test_paper_summary_missing_key_contribution_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PaperSummary(
                title="Test Paper",
                authors=["Alice"],
                plain_english_summary="A test.",
                methods="Methods here.",
                significance="Significance here.",
            )
        assert "key_contribution" in str(exc_info.value)

    def test_paper_summary_multiple_authors(self):
        # Arrange / Act
        paper = PaperSummary(
            title="Test Paper",
            authors=["Alice", "Bob", "Carol", "Dave", "Eve"],
            plain_english_summary="A test.",
            methods="Methods here.",
            significance="Significance here.",
            key_contribution="Contribution here.",
        )

        # Assert
        assert len(paper.authors) == 5
        assert "Carol" in paper.authors

    def test_paper_summary_empty_authors_list_is_valid(self):
        # Pydantic allows empty list; the field has no min_length constraint
        paper = PaperSummary(
            title="Test Paper",
            authors=[],
            plain_english_summary="A test.",
            methods="Methods here.",
            significance="Significance here.",
            key_contribution="Contribution here.",
        )
        assert paper.authors == []


# ---------------------------------------------------------------------------
# ConceptExplanation
# ---------------------------------------------------------------------------

class TestConceptExplanation:
    def test_concept_explanation_valid_required_fields(self):
        # Arrange / Act
        concept = ConceptExplanation(
            name="Gradient Descent",
            plain_english="An optimisation algorithm that minimises a loss function.",
            example="Rolling a ball downhill to the lowest point in a valley.",
            why_it_matters="Underpins almost all of modern deep learning.",
            connected_to_today="Every paper today trains with some form of gradient descent.",
        )

        # Assert
        assert concept.name == "Gradient Descent"
        assert concept.plain_english == "An optimisation algorithm that minimises a loss function."
        assert concept.example == "Rolling a ball downhill to the lowest point in a valley."
        assert concept.why_it_matters == "Underpins almost all of modern deep learning."
        assert concept.connected_to_today == "Every paper today trains with some form of gradient descent."

    def test_concept_explanation_learn_more_url_defaults_to_empty_string(self):
        # Arrange / Act
        concept = ConceptExplanation(
            name="Gradient Descent",
            plain_english="An optimisation algorithm.",
            example="Rolling a ball downhill.",
            why_it_matters="Underpins deep learning.",
            connected_to_today="Used in all papers today.",
        )

        # Assert
        assert concept.learn_more_url == ""

    def test_concept_explanation_learn_more_url_accepts_value(self):
        # Arrange / Act
        concept = ConceptExplanation(
            name="Gradient Descent",
            plain_english="An optimisation algorithm.",
            example="Rolling a ball downhill.",
            why_it_matters="Underpins deep learning.",
            connected_to_today="Used in all papers today.",
            learn_more_url="https://distill.pub/2016/augmented-rnns/",
        )

        # Assert
        assert concept.learn_more_url == "https://distill.pub/2016/augmented-rnns/"

    def test_concept_explanation_missing_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ConceptExplanation(
                plain_english="An optimisation algorithm.",
                example="Rolling a ball downhill.",
                why_it_matters="Underpins deep learning.",
                connected_to_today="Used in all papers today.",
            )

    def test_concept_explanation_all_five_required_fields_present(self):
        """Verifies the schema has exactly the expected required fields."""
        required = {
            name
            for name, field in ConceptExplanation.model_fields.items()
            if field.is_required()
        }
        assert required == {
            "name",
            "plain_english",
            "example",
            "why_it_matters",
            "connected_to_today",
        }


# ---------------------------------------------------------------------------
# DailyBriefing — coerce_concepts validator
# ---------------------------------------------------------------------------

class TestDailyBriefingCoerceConcepts:
    def test_coerce_concepts_dict_wraps_in_list(self, sample_concept):
        # Arrange — LLMs sometimes return a single object instead of a list
        raw = sample_concept.model_dump()

        # Act
        briefing = DailyBriefing(
            date="2024-01-01",
            emerging_themes="Some themes.",
            concepts_of_the_day=raw,
        )

        # Assert
        assert isinstance(briefing.concepts_of_the_day, list)
        assert len(briefing.concepts_of_the_day) == 1
        assert briefing.concepts_of_the_day[0].name == sample_concept.name

    def test_coerce_concepts_none_returns_empty_list(self):
        # Act
        briefing = DailyBriefing(
            date="2024-01-01",
            emerging_themes="Some themes.",
            concepts_of_the_day=None,
        )

        # Assert
        assert briefing.concepts_of_the_day == []

    def test_coerce_concepts_list_passes_through_unchanged(self, sample_concept):
        # Act
        briefing = DailyBriefing(
            date="2024-01-01",
            emerging_themes="Some themes.",
            concepts_of_the_day=[sample_concept],
        )

        # Assert
        assert len(briefing.concepts_of_the_day) == 1

    def test_coerce_concepts_empty_list_passes_through(self):
        # Act
        briefing = DailyBriefing(
            date="2024-01-01",
            emerging_themes="Some themes.",
            concepts_of_the_day=[],
        )

        # Assert
        assert briefing.concepts_of_the_day == []


# ---------------------------------------------------------------------------
# DailyBriefing — defaults and optional fields
# ---------------------------------------------------------------------------

class TestDailyBriefingDefaults:
    def test_daily_briefing_minimal_construction_succeeds(self):
        # Arrange / Act — only required fields provided
        briefing = DailyBriefing(
            date="2024-01-01",
            emerging_themes="Some themes.",
        )

        # Assert
        assert briefing.date == "2024-01-01"
        assert briefing.emerging_themes == "Some themes."

    def test_daily_briefing_most_discussed_defaults_to_empty_list(self):
        briefing = DailyBriefing(date="2024-01-01", emerging_themes="Themes.")
        assert briefing.most_discussed == []

    def test_daily_briefing_notable_papers_defaults_to_empty_list(self):
        briefing = DailyBriefing(date="2024-01-01", emerging_themes="Themes.")
        assert briefing.notable_papers == []

    def test_daily_briefing_concepts_of_the_day_defaults_to_empty_list(self):
        briefing = DailyBriefing(date="2024-01-01", emerging_themes="Themes.")
        assert briefing.concepts_of_the_day == []

    def test_daily_briefing_foundational_concepts_defaults_to_empty_list(self):
        briefing = DailyBriefing(date="2024-01-01", emerging_themes="Themes.")
        assert briefing.foundational_concepts == []

    def test_daily_briefing_total_papers_analyzed_defaults_to_zero(self):
        briefing = DailyBriefing(date="2024-01-01", emerging_themes="Themes.")
        assert briefing.total_papers_analyzed == 0

    def test_daily_briefing_errors_defaults_to_empty_list(self):
        briefing = DailyBriefing(date="2024-01-01", emerging_themes="Themes.")
        assert briefing.errors == []

    def test_daily_briefing_missing_date_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            DailyBriefing(emerging_themes="Themes.")
        assert "date" in str(exc_info.value)

    def test_daily_briefing_missing_emerging_themes_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            DailyBriefing(date="2024-01-01")
        assert "emerging_themes" in str(exc_info.value)

    def test_daily_briefing_accepts_paper_summaries_in_most_discussed(
        self, sample_paper_summary
    ):
        # Arrange / Act
        briefing = DailyBriefing(
            date="2024-01-01",
            emerging_themes="Themes.",
            most_discussed=[sample_paper_summary],
        )

        # Assert
        assert len(briefing.most_discussed) == 1
        assert briefing.most_discussed[0].title == "Attention Is All You Need Again"

    def test_daily_briefing_accepts_concepts_of_the_day(self, sample_concept):
        # Arrange / Act
        briefing = DailyBriefing(
            date="2024-01-01",
            emerging_themes="Themes.",
            concepts_of_the_day=[sample_concept],
        )

        # Assert
        assert len(briefing.concepts_of_the_day) == 1
        assert briefing.concepts_of_the_day[0].name == "Attention Mechanism"

    def test_daily_briefing_model_validate_round_trips(self, full_briefing):
        # Arrange — serialise to dict then re-validate
        data = full_briefing.model_dump()

        # Act
        restored = DailyBriefing.model_validate(data)

        # Assert
        assert restored.date == full_briefing.date
        assert restored.total_papers_analyzed == full_briefing.total_papers_analyzed
        assert len(restored.most_discussed) == len(full_briefing.most_discussed)
        assert len(restored.concepts_of_the_day) == len(full_briefing.concepts_of_the_day)
