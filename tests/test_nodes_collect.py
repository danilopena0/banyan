"""
Tests for collect_tool_results in agent/nodes.py.

collect_tool_results scans the messages list for ToolMessage objects,
attempts to JSON-parse their content, and extracts paper dicts whose
payloads contain an 'abstract' key.
"""
import json
import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.nodes import collect_tool_results
from agent.state import ResearchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper_json(
    paper_id="http://arxiv.org/abs/2401.00001v1",
    title="Fake Paper",
    abstract="This is a fake abstract.",
    authors=None,
) -> str:
    """Return JSON string mimicking search_arxiv tool output."""
    return json.dumps([
        {
            "id": paper_id,
            "title": title,
            "authors": authors or ["Alice Smith"],
            "abstract": abstract,
            "url": f"http://arxiv.org/pdf/{paper_id.split('/')[-1]}",
            "published": "2024-01-01T00:00:00+00:00",
            "categories": ["cs.LG"],
        }
    ])


def _make_state(messages: list) -> ResearchState:
    return ResearchState(messages=messages, date="2024-01-01")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollectToolResults:
    def test_collect_single_paper_tool_message_extracts_paper(self):
        # Arrange
        state = _make_state([
            ToolMessage(content=_make_paper_json(), tool_call_id="call_001"),
        ])

        # Act
        result = collect_tool_results(state)

        # Assert
        assert "raw_papers" in result
        assert len(result["raw_papers"]) == 1
        assert result["raw_papers"][0]["title"] == "Fake Paper"

    def test_collect_multiple_papers_in_one_tool_message(self):
        # Arrange
        papers = [
            {
                "id": f"http://arxiv.org/abs/240{i}.00001v1",
                "title": f"Paper {i}",
                "abstract": "Some abstract.",
                "authors": ["Author A"],
                "url": "http://arxiv.org/pdf/2401.00001v1",
                "published": "2024-01-01T00:00:00+00:00",
                "categories": ["cs.LG"],
            }
            for i in range(3)
        ]
        state = _make_state([
            ToolMessage(content=json.dumps(papers), tool_call_id="call_001"),
        ])

        # Act
        result = collect_tool_results(state)

        # Assert
        assert len(result["raw_papers"]) == 3

    def test_collect_multiple_tool_messages_aggregates_papers(self):
        # Arrange
        state = _make_state([
            ToolMessage(content=_make_paper_json("http://arxiv.org/abs/2401.00001v1", "Paper A"), tool_call_id="call_001"),
            ToolMessage(content=_make_paper_json("http://arxiv.org/abs/2401.00002v1", "Paper B"), tool_call_id="call_002"),
        ])

        # Act
        result = collect_tool_results(state)

        # Assert
        assert len(result["raw_papers"]) == 2
        titles = {p["title"] for p in result["raw_papers"]}
        assert titles == {"Paper A", "Paper B"}

    def test_collect_plain_text_tool_message_is_skipped(self):
        # Arrange — web_search returns plain text, not JSON
        state = _make_state([
            ToolMessage(
                content="OpenAI released a new model. It scores well on benchmarks.",
                tool_call_id="call_web_001",
            ),
        ])

        # Act
        result = collect_tool_results(state)

        # Assert — no papers extracted, no crash
        assert result["raw_papers"] == []

    def test_collect_invalid_json_is_skipped(self):
        # Arrange
        state = _make_state([
            ToolMessage(content="{not valid json}", tool_call_id="call_001"),
        ])

        # Act — must not raise
        result = collect_tool_results(state)

        # Assert
        assert result["raw_papers"] == []

    def test_collect_json_list_without_abstract_key_is_skipped(self):
        # Arrange — valid JSON list but items don't look like arXiv papers
        reddit_data = json.dumps([
            {"post_id": "abc", "title": "Discussion post", "score": 100},
        ])
        state = _make_state([
            ToolMessage(content=reddit_data, tool_call_id="call_reddit_001"),
        ])

        # Act
        result = collect_tool_results(state)

        # Assert — no papers, because 'abstract' key is absent
        assert result["raw_papers"] == []

    def test_collect_non_tool_messages_are_ignored(self):
        # Arrange — mix of message types
        state = _make_state([
            HumanMessage(content="Research today's AI papers."),
            AIMessage(content="I will search arXiv now."),
            ToolMessage(content=_make_paper_json(), tool_call_id="call_001"),
        ])

        # Act
        result = collect_tool_results(state)

        # Assert — only the ToolMessage contributed papers
        assert len(result["raw_papers"]) == 1

    def test_collect_empty_messages_list_returns_empty_papers(self):
        # Arrange
        state = _make_state([])

        # Act
        result = collect_tool_results(state)

        # Assert
        assert result["raw_papers"] == []

    def test_collect_empty_json_list_in_tool_message_returns_no_papers(self):
        # Arrange — tool returned an empty list (no papers found)
        state = _make_state([
            ToolMessage(content="[]", tool_call_id="call_001"),
        ])

        # Act
        result = collect_tool_results(state)

        # Assert
        assert result["raw_papers"] == []

    def test_collect_mixed_valid_and_invalid_messages_only_extracts_valid(self):
        # Arrange
        state = _make_state([
            ToolMessage(content="plain text result", tool_call_id="call_web"),
            ToolMessage(content="{bad json}", tool_call_id="call_bad"),
            ToolMessage(content=_make_paper_json(title="Good Paper"), tool_call_id="call_good"),
        ])

        # Act
        result = collect_tool_results(state)

        # Assert — only the valid paper message contributed
        assert len(result["raw_papers"]) == 1
        assert result["raw_papers"][0]["title"] == "Good Paper"

    def test_collect_preserves_paper_fields_from_tool_message(self):
        # Arrange
        state = _make_state([
            ToolMessage(
                content=_make_paper_json(
                    paper_id="http://arxiv.org/abs/2401.99999v1",
                    title="Precision Test Paper",
                    abstract="Detailed abstract content here.",
                    authors=["First Author", "Second Author"],
                ),
                tool_call_id="call_001",
            )
        ])

        # Act
        result = collect_tool_results(state)

        # Assert
        paper = result["raw_papers"][0]
        assert paper["id"] == "http://arxiv.org/abs/2401.99999v1"
        assert paper["title"] == "Precision Test Paper"
        assert paper["abstract"] == "Detailed abstract content here."
        assert paper["authors"] == ["First Author", "Second Author"]
