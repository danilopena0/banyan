"""
Tests for agent/tavily.py.

Tests the tavily_search function which is the single boundary with the Tavily API.
Since TavilyClient is instantiated once at module import time, tests patch the
module-level _client directly rather than patching TavilyClient construction.
"""
from unittest.mock import MagicMock, patch

from agent.tavily import tavily_search


class TestTavilySearchNoApiKey:
    def test_tavily_search_no_client_returns_empty_dict(self):
        # Arrange — simulate no API key by patching _client to None
        with patch("agent.tavily._client", None):
            result = tavily_search("large language models")

        assert result == {}

    def test_tavily_search_no_client_does_not_call_search(self):
        mock_client = MagicMock()
        with patch("agent.tavily._client", None):
            tavily_search("any query")

        mock_client.search.assert_not_called()


class TestTavilySearchWithApiKey:
    def test_tavily_search_with_key_returns_response_dict(self):
        # Arrange
        fake_response = {
            "results": [
                {"title": "Test Result", "url": "https://example.com", "content": "Some content"},
            ],
            "answer": "A synthesised answer.",
        }
        mock_client = MagicMock()
        mock_client.search.return_value = fake_response

        with patch("agent.tavily._client", mock_client):
            result = tavily_search("LLM benchmarks", max_results=3, include_answer=True)

        assert result == fake_response
        assert "results" in result
        assert result["answer"] == "A synthesised answer."

    def test_tavily_search_passes_correct_arguments_to_client(self):
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        with patch("agent.tavily._client", mock_client):
            tavily_search("diffusion models", max_results=7, include_answer=False)

        mock_client.search.assert_called_once_with(
            query="diffusion models",
            search_depth="basic",
            max_results=7,
            include_answer=False,
        )

    def test_tavily_search_with_key_and_exception_returns_empty_dict(self):
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Connection timeout")

        with patch("agent.tavily._client", mock_client):
            result = tavily_search("reinforcement learning")

        assert result == {}

    def test_tavily_search_default_parameters(self):
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        with patch("agent.tavily._client", mock_client):
            tavily_search("test query")

        mock_client.search.assert_called_once_with(
            query="test query",
            search_depth="basic",
            max_results=5,
            include_answer=False,
        )

    def test_tavily_search_empty_results_list_still_returns_dict(self):
        fake_response = {"results": [], "query": "nothing found"}
        mock_client = MagicMock()
        mock_client.search.return_value = fake_response

        with patch("agent.tavily._client", mock_client):
            result = tavily_search("obscure query with no results")

        assert result == fake_response
        assert result["results"] == []
