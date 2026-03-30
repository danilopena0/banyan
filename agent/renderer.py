"""
Markdown rendering for DailyBriefing.

Converts a DailyBriefing Pydantic model into a structured markdown string,
applying LaTeX post-processing to ensure readable output even when the LLM
ignores the no-LaTeX instruction.
"""
import re
from typing import Any

from schemas.briefing import DailyBriefing

_LATEX_SYMBOLS = {
    r"\sigma": "σ", r"\theta": "θ", r"\alpha": "α", r"\beta": "β",
    r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε", r"\lambda": "λ",
    r"\mu": "μ", r"\pi": "π", r"\tau": "τ", r"\phi": "φ", r"\psi": "ψ",
    r"\omega": "ω", r"\nabla": "∇", r"\partial": "∂", r"\infty": "∞",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\rightarrow": "→", r"\leftarrow": "←", r"\cdot": "·",
    r"\times": "×", r"\in": "∈", r"\sum": "Σ", r"\prod": "Π",
    r"\log": "log", r"\exp": "exp", r"\max": "max", r"\min": "min",
}


def _strip_latex(text: Any) -> str:
    """
    Remove LaTeX math delimiters and replace common commands with unicode.

    Safety-net post-processor: the prompt forbids LaTeX, but if the LLM
    produces it anyway this ensures the markdown output stays readable.
    """
    if not isinstance(text, str):
        return text

    # Replace display math blocks first, then inline
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", r"\1", text)
    # \\( ... \\) and \\[ ... \\] delimiters
    text = re.sub(r"\\\\\((.+?)\\\\\)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\\\[(.+?)\\\\\]", r"\1", text, flags=re.DOTALL)

    # Replace known LaTeX commands with unicode
    for cmd, symbol in _LATEX_SYMBOLS.items():
        text = text.replace(cmd, symbol)

    # Strip remaining \text{...} wrappers, keeping content
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    # Remove stray braces left from grouping (e.g. _{...} → _...)
    text = re.sub(r"\{([^{}]*)\}", r"\1", text)

    return text


def _render_paper_lines(paper, include_key_contribution: bool = False) -> list[str]:
    """Render a single PaperSummary to a list of markdown lines."""
    lines = [
        f"### {_strip_latex(paper.title)}",
        f"*{', '.join(paper.authors)}*",
        "",
        f"**Summary:** {_strip_latex(paper.plain_english_summary)}",
        "",
        f"**Methods:** {_strip_latex(paper.methods)}",
        "",
    ]
    if include_key_contribution:
        lines += [f"**Key contribution:** {_strip_latex(paper.key_contribution)}", ""]
    lines += [
        f"**Why it matters:** {_strip_latex(paper.significance)}",
        "",
        f"> [Read paper]({paper.url})" if paper.url else "",
        "",
        "---",
        "",
    ]
    return lines


def _render_concept_lines(c, include_connected_to_today: bool = False) -> list[str]:
    """Render a single concept (ConceptExplanation or FoundationalConcept) to markdown lines."""
    lines = [
        f"## {c.name}",
        "",
        _strip_latex(c.plain_english) if include_connected_to_today else c.plain_english,
        "",
        f"**Example:** {_strip_latex(c.example) if include_connected_to_today else c.example}",
        "",
        f"**Why it matters:** {_strip_latex(c.why_it_matters) if include_connected_to_today else c.why_it_matters}",
        "",
    ]
    if include_connected_to_today:
        lines += [f"**In today's research:** {_strip_latex(c.connected_to_today)}", ""]
    lines += [
        f"> [Learn more]({c.learn_more_url})" if c.learn_more_url else "",
        "",
        "---",
        "",
    ]
    return lines


def render_briefing_markdown(briefing: DailyBriefing) -> str:
    """Render a DailyBriefing Pydantic model to a markdown string."""
    lines = [
        "# Daily AI Research Briefing",
        f"## {briefing.date}",
        "",
        f"> Analyzed **{briefing.total_papers_analyzed} papers**",
        "",
        "---",
        "",
    ]

    if briefing.most_discussed:
        lines += ["# Most Discussed", ""]
        for paper in briefing.most_discussed:
            lines += _render_paper_lines(paper, include_key_contribution=True)

    if briefing.notable_papers:
        lines += ["# Notable Papers", ""]
        for paper in briefing.notable_papers:
            lines += _render_paper_lines(paper, include_key_contribution=False)

    lines += ["# Emerging Themes", "", _strip_latex(briefing.emerging_themes), "", "---", ""]

    if briefing.concepts_of_the_day:
        lines += ["# Concepts of the Day", ""]
        for c in briefing.concepts_of_the_day:
            lines += _render_concept_lines(c, include_connected_to_today=True)

    if briefing.foundational_concepts:
        lines += ["# Foundational Concepts", ""]
        for c in briefing.foundational_concepts:
            lines += _render_concept_lines(c, include_connected_to_today=False)

    if briefing.errors:
        lines += ["# Errors (non-fatal)", ""]
        for err in briefing.errors:
            lines.append(f"- {err}")

    return "\n".join(lines)
