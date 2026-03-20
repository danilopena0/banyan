"""
Tests for agent/tavily.py.

Tests the tavily_search function which is the single boundary with the Tavily API.
All external calls are mocked at the TavilyClient boundary.
"""
import pytest
from unittest.mock import MagicMock, patch

from agent.tavily import tavily_search


class TestTavilySearchNoApiKey:
    def test_tavily_search_no_key_returns_empty_dict(self, monkeypatch):
        # Arrange — ensure the env var is absent
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        # Act
        result = tavily_search("large language models")

        # Assert
        assert result == {}

    def test_tavily_search_no_key_does_not_call_tavily_client(self, monkeypatch):
        # Arrange
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        with patch("agent.tavily.TavilyClient") as mock_client_cls:
            # Act
            tavily_search("any query")

            # Assert — client was never instantiated
            mock_client_cls.assert_not_called()


class TestTavilySearchWithApiKey:
    def test_tavily_search_with_key_returns_response_dict(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key-123")
        fake_response = {
            "results": [
                {"title": "Test Result", "url": "https://example.com", "content": "Some content"},
            ],
            "answer": "A synthesised answer.",
        }

        with patch("agent.tavily.TavilyClient") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.search.return_value = fake_response
            mock_client_cls.return_value = mock_instance

            # Act
            result = tavily_search("LLM benchmarks", max_results=3, include_answer=True)

        # Assert
        assert result == fake_response
        assert "results" in result
        assert result["answer"] == "A synthesised answer."

    def test_tavily_search_passes_correct_arguments_to_client(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key-123")

        with patch("agent.tavily.TavilyClient") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.search.return_value = {"results": []}
            mock_client_cls.return_value = mock_instance

            # Act
            tavily_search("diffusion models", max_results=7, include_answer=False)

            # Assert — TavilyClient instantiated with the API key
            mock_client_cls.assert_called_once_with(api_key="test-key-123")
            # Assert — search called with correct parameters
            mock_instance.search.assert_called_once_with(
                query="diffusion models",
                search_depth="basic",
                max_results=7,
                include_answer=False,
            )

    def test_tavily_search_with_key_and_exception_returns_empty_dict(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key-123")

        with patch("agent.tavily.TavilyClient") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.search.side_effect = Exception("Connection timeout")
            mock_client_cls.return_value = mock_instance

            # Act
            result = tavily_search("reinforcement learning")

        # Assert — exception swallowed, empty dict returned
        assert result == {}

    def test_tavily_search_default_parameters(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key-xyz")

        with patch("agent.tavily.TavilyClient") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.search.return_value = {"results": []}
            mock_client_cls.return_value = mock_instance

            # Act
            tavily_search("test query")

            # Assert — defaults are max_results=5, include_answer=False
            mock_instance.search.assert_called_once_with(
                query="test query",
                search_depth="basic",
                max_results=5,
                include_answer=False,
            )

    def test_tavily_search_empty_results_list_still_returns_dict(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key-123")
        fake_response = {"results": [], "query": "nothing found"}

        with patch("agent.tavily.TavilyClient") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.search.return_value = fake_response
            mock_client_cls.return_value = mock_instance

            # Act
            result = tavily_search("obscure query with no results")

        # Assert — caller gets the dict as-is (they handle empty results)
        assert result == fake_response
        assert result["results"] == []
