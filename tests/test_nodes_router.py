"""
Tests for should_continue in agent/nodes.py.

should_continue is the ReAct loop router. It inspects the last message
in state.messages and returns "tools" if the LLM made tool calls, or
"process" if it is done researching.
"""
import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.nodes import should_continue
from agent.state import ResearchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(messages: list) -> ResearchState:
    return ResearchState(messages=messages, date="2024-01-01")


def _ai_message_with_tool_calls() -> AIMessage:
    """Return an AIMessage that contains at least one tool call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_abc123",
                "name": "search_arxiv",
                "args": {"query": "large language models"},
            }
        ],
    )


def _ai_message_without_tool_calls() -> AIMessage:
    """Return an AIMessage with no pending tool calls (research complete)."""
    return AIMessage(content="I have finished researching.")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def test_should_continue_ai_message_with_tool_calls_returns_tools(self):
        # Arrange
        state = _make_state([_ai_message_with_tool_calls()])

        # Act
        result = should_continue(state)

        # Assert
        assert result == "tools"

    def test_should_continue_ai_message_without_tool_calls_returns_process(self):
        # Arrange
        state = _make_state([_ai_message_without_tool_calls()])

        # Act
        result = should_continue(state)

        # Assert
        assert result == "process"

    def test_should_continue_empty_messages_returns_process(self):
        # Arrange
        state = _make_state([])

        # Act
        result = should_continue(state)

        # Assert — no crash, defaults to process
        assert result == "process"

    def test_should_continue_checks_last_message_not_first(self):
        # Arrange — first message has tool calls, last does not
        state = _make_state([
            _ai_message_with_tool_calls(),
            ToolMessage(content="[result]", tool_call_id="call_abc123"),
            _ai_message_without_tool_calls(),
        ])

        # Act
        result = should_continue(state)

        # Assert — should look at the LAST message
        assert result == "process"

    def test_should_continue_last_message_has_tool_calls_returns_tools(self):
        # Arrange — last message requests more tool calls
        state = _make_state([
            _ai_message_without_tool_calls(),
            ToolMessage(content="[]", tool_call_id="call_001"),
            _ai_message_with_tool_calls(),
        ])

        # Act
        result = should_continue(state)

        # Assert
        assert result == "tools"

    def test_should_continue_human_message_as_last_returns_process(self):
        # Arrange — HumanMessage has no tool_calls attribute in the same sense
        state = _make_state([
            HumanMessage(content="Research today's AI papers."),
        ])

        # Act — must not raise AttributeError
        result = should_continue(state)

        # Assert
        assert result == "process"

    def test_should_continue_tool_message_as_last_returns_process(self):
        # Arrange — ToolMessage has no tool_calls
        state = _make_state([
            ToolMessage(content='[{"id": "1", "title": "Paper"}]', tool_call_id="call_001"),
        ])

        # Act
        result = should_continue(state)

        # Assert
        assert result == "process"

    def test_should_continue_multiple_tool_calls_in_last_message_returns_tools(self):
        # Arrange — LLM batched two tool calls
        msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "search_arxiv", "args": {"query": "diffusion models"}},
                {"id": "call_2", "name": "web_search", "args": {"query": "AI news"}},
            ],
        )
        state = _make_state([msg])

        # Act
        result = should_continue(state)

        # Assert
        assert result == "tools"

    def test_should_continue_returns_only_valid_routing_values(self):
        # Assumption: the function only ever returns "tools" or "process"
        valid_returns = {"tools", "process"}

        state_with_calls = _make_state([_ai_message_with_tool_calls()])
        state_without_calls = _make_state([_ai_message_without_tool_calls()])
        state_empty = _make_state([])

        assert should_continue(state_with_calls) in valid_returns
        assert should_continue(state_without_calls) in valid_returns
        assert should_continue(state_empty) in valid_returns
