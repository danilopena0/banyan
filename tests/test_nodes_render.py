"""
Tests for render_briefing_markdown in agent/nodes.py.

render_briefing_markdown converts a DailyBriefing Pydantic model into a
markdown string. Tests verify the structure and content of the output
without coupling to exact whitespace or line positions.
"""
import pytest

from schemas.briefing import DailyBriefing, ConceptExplanation, FoundationalConcept
from schemas.paper import PaperSummary
from agent.renderer import render_briefing_markdown, _strip_latex, _render_paper_lines, _render_concept_lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_briefing(**overrides) -> DailyBriefing:
    defaults = dict(date="2024-01-15", emerging_themes="Transformers dominate.")
    defaults.update(overrides)
    return DailyBriefing(**defaults)


def _paper(**overrides) -> PaperSummary:
    defaults = dict(
        title="Sample Paper Title",
        authors=["Author A", "Author B"],
        plain_english_summary="A short summary.",
        methods="Transformer with cross-attention.",
        significance="Advances the state-of-the-art.",
        key_contribution="New positional encoding.",
        url="https://arxiv.org/abs/2401.00001",
    )
    defaults.update(overrides)
    return PaperSummary(**defaults)


# ---------------------------------------------------------------------------
# Header / top-level structure
# ---------------------------------------------------------------------------

class TestRenderBriefingMarkdownHeader:
    def test_render_produces_string(self):
        # Act
        output = render_briefing_markdown(_minimal_briefing())
        assert isinstance(output, str)

    def test_render_includes_main_title(self):
        output = render_briefing_markdown(_minimal_briefing())
        assert "# Daily AI Research Briefing" in output

    def test_render_includes_briefing_date(self):
        output = render_briefing_markdown(_minimal_briefing(date="2024-03-19"))
        assert "2024-03-19" in output

    def test_render_includes_total_papers_analyzed(self):
        briefing = _minimal_briefing(total_papers_analyzed=57)
        output = render_briefing_markdown(briefing)
        assert "57" in output

    def test_render_includes_emerging_themes_section(self):
        briefing = _minimal_briefing(emerging_themes="Diffusion models are rising fast.")
        output = render_briefing_markdown(briefing)
        assert "# Emerging Themes" in output
        assert "Diffusion models are rising fast." in output


# ---------------------------------------------------------------------------
# most_discussed papers
# ---------------------------------------------------------------------------

class TestRenderBriefingMarkdownMostDiscussed:
    def test_render_includes_most_discussed_heading(self):
        briefing = _minimal_briefing(most_discussed=[_paper()])
        output = render_briefing_markdown(briefing)
        assert "# Most Discussed" in output

    def test_render_includes_paper_title(self):
        briefing = _minimal_briefing(most_discussed=[_paper(title="Attention Is Key")])
        output = render_briefing_markdown(briefing)
        assert "Attention Is Key" in output

    def test_render_includes_paper_authors(self):
        briefing = _minimal_briefing(
            most_discussed=[_paper(authors=["Dr. Alice", "Prof. Bob"])]
        )
        output = render_briefing_markdown(briefing)
        assert "Dr. Alice" in output
        assert "Prof. Bob" in output

    def test_render_includes_paper_summary(self):
        briefing = _minimal_briefing(
            most_discussed=[_paper(plain_english_summary="This paper proposes a novel approach.")]
        )
        output = render_briefing_markdown(briefing)
        assert "This paper proposes a novel approach." in output

    def test_render_includes_paper_methods(self):
        briefing = _minimal_briefing(
            most_discussed=[_paper(methods="Uses DPO with Bradley-Terry preference model.")]
        )
        output = render_briefing_markdown(briefing)
        assert "Uses DPO with Bradley-Terry preference model." in output

    def test_render_includes_paper_key_contribution(self):
        briefing = _minimal_briefing(
            most_discussed=[_paper(key_contribution="Novel loss function for RLHF.")]
        )
        output = render_briefing_markdown(briefing)
        assert "Novel loss function for RLHF." in output

    def test_render_includes_paper_significance(self):
        briefing = _minimal_briefing(
            most_discussed=[_paper(significance="First to surpass human parity on benchmark.")]
        )
        output = render_briefing_markdown(briefing)
        assert "First to surpass human parity on benchmark." in output

    def test_render_includes_paper_url_as_link(self):
        briefing = _minimal_briefing(
            most_discussed=[_paper(url="https://arxiv.org/abs/2401.99999")]
        )
        output = render_briefing_markdown(briefing)
        assert "https://arxiv.org/abs/2401.99999" in output

    def test_render_skips_read_paper_link_when_url_empty(self):
        briefing = _minimal_briefing(most_discussed=[_paper(url="")])
        output = render_briefing_markdown(briefing)
        # The "[Read paper](...)" link should not appear for empty url
        assert "[Read paper]()" not in output

    def test_render_no_most_discussed_section_when_empty(self):
        briefing = _minimal_briefing(most_discussed=[])
        output = render_briefing_markdown(briefing)
        assert "# Most Discussed" not in output

    def test_render_multiple_most_discussed_papers_all_appear(self):
        briefing = _minimal_briefing(
            most_discussed=[
                _paper(title="Paper Alpha"),
                _paper(title="Paper Beta"),
                _paper(title="Paper Gamma"),
            ]
        )
        output = render_briefing_markdown(briefing)
        assert "Paper Alpha" in output
        assert "Paper Beta" in output
        assert "Paper Gamma" in output


# ---------------------------------------------------------------------------
# notable_papers
# ---------------------------------------------------------------------------

class TestRenderBriefingMarkdownNotablePapers:
    def test_render_includes_notable_papers_heading(self):
        briefing = _minimal_briefing(notable_papers=[_paper()])
        output = render_briefing_markdown(briefing)
        assert "# Notable Papers" in output

    def test_render_no_notable_papers_section_when_empty(self):
        briefing = _minimal_briefing(notable_papers=[])
        output = render_briefing_markdown(briefing)
        assert "# Notable Papers" not in output

    def test_render_notable_paper_title_appears(self):
        briefing = _minimal_briefing(notable_papers=[_paper(title="Notable Research Work")])
        output = render_briefing_markdown(briefing)
        assert "Notable Research Work" in output


# ---------------------------------------------------------------------------
# concepts_of_the_day
# ---------------------------------------------------------------------------

class TestRenderBriefingMarkdownConceptsOfTheDay:
    def _concept(self, **overrides):
        defaults = dict(
            name="Gradient Descent",
            plain_english="Minimises a loss by following the negative gradient.",
            example="Rolling a ball downhill in a loss landscape.",
            why_it_matters="Underpins all of deep learning training.",
            connected_to_today="Every paper today uses some form of gradient-based optimisation.",
            learn_more_url="https://distill.pub/2017/momentum/",
        )
        defaults.update(overrides)
        return ConceptExplanation(**defaults)

    def test_render_includes_concepts_heading(self):
        briefing = _minimal_briefing(concepts_of_the_day=[self._concept()])
        output = render_briefing_markdown(briefing)
        assert "# Concepts of the Day" in output

    def test_render_includes_concept_name(self):
        briefing = _minimal_briefing(concepts_of_the_day=[self._concept(name="KL Divergence")])
        output = render_briefing_markdown(briefing)
        assert "KL Divergence" in output

    def test_render_includes_concept_plain_english(self):
        briefing = _minimal_briefing(
            concepts_of_the_day=[self._concept(
                plain_english="Measures how one probability distribution differs from another."
            )]
        )
        output = render_briefing_markdown(briefing)
        assert "Measures how one probability distribution differs from another." in output

    def test_render_includes_concept_example(self):
        briefing = _minimal_briefing(
            concepts_of_the_day=[self._concept(example="Like comparing two histograms.")]
        )
        output = render_briefing_markdown(briefing)
        assert "Like comparing two histograms." in output

    def test_render_includes_concept_why_it_matters(self):
        briefing = _minimal_briefing(
            concepts_of_the_day=[self._concept(why_it_matters="Used in VAEs and RL policy optimisation.")]
        )
        output = render_briefing_markdown(briefing)
        assert "Used in VAEs and RL policy optimisation." in output

    def test_render_includes_concept_connected_to_today(self):
        briefing = _minimal_briefing(
            concepts_of_the_day=[self._concept(connected_to_today="Paper A uses KL in its loss.")]
        )
        output = render_briefing_markdown(briefing)
        assert "Paper A uses KL in its loss." in output

    def test_render_includes_learn_more_url_as_link(self):
        briefing = _minimal_briefing(
            concepts_of_the_day=[self._concept(learn_more_url="https://distill.pub/2017/momentum/")]
        )
        output = render_briefing_markdown(briefing)
        assert "https://distill.pub/2017/momentum/" in output

    def test_render_skips_learn_more_link_when_url_empty(self):
        briefing = _minimal_briefing(
            concepts_of_the_day=[self._concept(learn_more_url="")]
        )
        output = render_briefing_markdown(briefing)
        assert "[Learn more]()" not in output

    def test_render_multiple_concepts_all_appear(self):
        briefing = _minimal_briefing(
            concepts_of_the_day=[
                self._concept(name="Concept Alpha"),
                self._concept(name="Concept Beta"),
            ]
        )
        output = render_briefing_markdown(briefing)
        assert "Concept Alpha" in output
        assert "Concept Beta" in output

    def test_render_no_concepts_section_when_empty(self):
        briefing = _minimal_briefing(concepts_of_the_day=[])
        output = render_briefing_markdown(briefing)
        assert "# Concepts of the Day" not in output


# ---------------------------------------------------------------------------
# errors section
# ---------------------------------------------------------------------------

class TestRenderBriefingMarkdownErrors:
    def test_render_includes_errors_section_heading(self):
        briefing = _minimal_briefing(errors=["arXiv API timed out."])
        output = render_briefing_markdown(briefing)
        assert "# Errors" in output

    def test_render_each_error_as_bullet(self):
        briefing = _minimal_briefing(
            errors=["arXiv API timed out.", "ChromaDB unavailable."]
        )
        output = render_briefing_markdown(briefing)
        assert "- arXiv API timed out." in output
        assert "- ChromaDB unavailable." in output

    def test_render_no_errors_section_when_empty(self):
        briefing = _minimal_briefing(errors=[])
        output = render_briefing_markdown(briefing)
        assert "# Errors" not in output

    def test_render_single_error_appears(self):
        briefing = _minimal_briefing(errors=["One transient failure."])
        output = render_briefing_markdown(briefing)
        assert "One transient failure." in output


# ---------------------------------------------------------------------------
# Full integration: minimal briefing smoke test
# ---------------------------------------------------------------------------

class TestRenderBriefingMarkdownMinimal:
    def test_render_minimal_briefing_contains_no_optional_sections(self):
        # Arrange
        briefing = _minimal_briefing()

        # Act
        output = render_briefing_markdown(briefing)

        # Assert — optional sections absent
        assert "# Most Discussed" not in output
        assert "# Notable Papers" not in output
        assert "# Web & Industry News" not in output
        assert "# Concept of the Day" not in output
        assert "# Errors" not in output

    def test_render_minimal_briefing_always_contains_core_sections(self):
        # Arrange
        briefing = _minimal_briefing()

        # Act
        output = render_briefing_markdown(briefing)

        # Assert — core sections always present
        assert "# Daily AI Research Briefing" in output
        assert "# Emerging Themes" in output


# ---------------------------------------------------------------------------
# _strip_latex
# ---------------------------------------------------------------------------

class TestStripLatex:
    def test_plain_text_unchanged(self):
        assert _strip_latex("Hello world") == "Hello world"

    def test_inline_math_delimiters_stripped(self):
        assert _strip_latex("$x^2$") == "x^2"

    def test_display_math_delimiters_stripped(self):
        assert _strip_latex("$$x^2$$") == "x^2"

    def test_paren_delimiters_stripped(self):
        assert _strip_latex("\\\\(x\\\\)") == "x"

    def test_sigma_replaced_with_unicode(self):
        assert _strip_latex("\\sigma") == "σ"

    def test_nabla_replaced_with_unicode(self):
        assert _strip_latex("\\nabla") == "∇"

    def test_multiple_symbols_in_one_string(self):
        result = _strip_latex("\\alpha and \\beta")
        assert "α" in result
        assert "β" in result

    def test_text_wrapper_stripped(self):
        assert _strip_latex("\\text{foo}") == "foo"

    def test_non_string_returned_as_is(self):
        assert _strip_latex(42) == 42

    def test_none_returned_as_is(self):
        assert _strip_latex(None) is None


# ---------------------------------------------------------------------------
# _render_paper_lines
# ---------------------------------------------------------------------------

class TestRenderPaperLines:
    def _paper(self, **overrides):
        defaults = dict(
            title="Test Paper",
            authors=["Author A"],
            plain_english_summary="A short summary.",
            methods="Transformer attention.",
            significance="Advances SOTA.",
            key_contribution="Novel encoding.",
            url="https://arxiv.org/abs/2401.00001",
        )
        defaults.update(overrides)
        return PaperSummary(**defaults)

    def test_includes_key_contribution_when_flag_true(self):
        lines = _render_paper_lines(self._paper(), include_key_contribution=True)
        combined = "\n".join(lines)
        assert "Key contribution:" in combined
        assert "Novel encoding." in combined

    def test_omits_key_contribution_when_flag_false(self):
        lines = _render_paper_lines(self._paper(), include_key_contribution=False)
        combined = "\n".join(lines)
        assert "Key contribution:" not in combined

    def test_url_appears_as_link(self):
        lines = _render_paper_lines(self._paper(url="https://arxiv.org/abs/1234"))
        combined = "\n".join(lines)
        assert "https://arxiv.org/abs/1234" in combined

    def test_empty_url_produces_no_link(self):
        lines = _render_paper_lines(self._paper(url=""))
        combined = "\n".join(lines)
        assert "[Read paper]" not in combined

    def test_returns_list_of_strings(self):
        lines = _render_paper_lines(self._paper())
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)


# ---------------------------------------------------------------------------
# _render_concept_lines
# ---------------------------------------------------------------------------

class TestRenderConceptLines:
    def _concept(self, **overrides):
        defaults = dict(
            name="Attention",
            plain_english="Focuses on relevant tokens.",
            example="Like a spotlight.",
            why_it_matters="Enables parallelization.",
            connected_to_today="Used in every paper today.",
            learn_more_url="https://distill.pub/",
        )
        defaults.update(overrides)
        return ConceptExplanation(**defaults)

    def _foundational_concept(self, **overrides):
        defaults = dict(
            name="Bias-Variance",
            plain_english="The fundamental tradeoff.",
            example="Fitting noise vs. signal.",
            why_it_matters="Guides model selection.",
            learn_more_url="",
        )
        defaults.update(overrides)
        return FoundationalConcept(**defaults)

    def test_includes_connected_to_today_when_flag_true(self):
        lines = _render_concept_lines(self._concept(), include_connected_to_today=True)
        combined = "\n".join(lines)
        assert "In today's research:" in combined
        assert "Used in every paper today." in combined

    def test_omits_connected_to_today_when_flag_false(self):
        lines = _render_concept_lines(self._foundational_concept(), include_connected_to_today=False)
        combined = "\n".join(lines)
        assert "In today's research:" not in combined

    def test_learn_more_url_appears_when_non_empty(self):
        lines = _render_concept_lines(self._concept(learn_more_url="https://distill.pub/"), include_connected_to_today=True)
        combined = "\n".join(lines)
        assert "https://distill.pub/" in combined

    def test_empty_url_produces_no_learn_more_link(self):
        lines = _render_concept_lines(self._concept(learn_more_url=""), include_connected_to_today=True)
        combined = "\n".join(lines)
        assert "[Learn more]" not in combined

    def test_returns_list_of_strings(self):
        lines = _render_concept_lines(self._concept(), include_connected_to_today=True)
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)
