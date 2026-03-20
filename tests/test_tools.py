"""
Tests for agent/tools.py.

search_arxiv and web_search are @tool decorated functions. We call the
underlying Python function via .func to bypass LangChain's tool wrapper,
which makes unit testing simpler and avoids coupling to the LangChain
tool invocation machinery.
"""
import datetime
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from agent.tools import search_arxiv, web_search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_arxiv_result(
    entry_id="http://arxiv.org/abs/2401.00001v1",
    title="Fake Paper Title",
    authors=("Alice Smith", "Bob Jones"),
    summary="A fake abstract about transformers." * 5,
    pdf_url="http://arxiv.org/pdf/2401.00001v1",
    published=None,
    categories=None,
):
    """Build a mock arxiv.Result object."""
    result = MagicMock()
    result.entry_id = entry_id
    result.title = title
    result.authors = [MagicMock(__str__=lambda self, a=a: a) for a in authors]
    result.summary = summary
    result.pdf_url = pdf_url
    result.published = published or datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    result.categories = categories or ["cs.LG", "cs.AI"]
    return result


# ---------------------------------------------------------------------------
# search_arxiv
# ---------------------------------------------------------------------------

class TestSearchArxiv:
    def test_search_arxiv_returns_list_of_dicts(self):
        # Arrange
        fake_result = _make_fake_arxiv_result()

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([fake_result])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("large language models")

        # Assert — returns a list
        assert isinstance(results, list)
        assert len(results) == 1

    def test_search_arxiv_result_has_required_keys(self):
        # Arrange
        fake_result = _make_fake_arxiv_result()

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([fake_result])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("transformers")

        # Assert — each result has the expected keys
        paper = results[0]
        assert "id" in paper
        assert "title" in paper
        assert "authors" in paper
        assert "abstract" in paper
        assert "url" in paper
        assert "published" in paper
        assert "categories" in paper

    def test_search_arxiv_maps_entry_id_to_id(self):
        # Arrange
        fake_result = _make_fake_arxiv_result(entry_id="http://arxiv.org/abs/2401.00099v2")

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([fake_result])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("test query")

        # Assert
        assert results[0]["id"] == "http://arxiv.org/abs/2401.00099v2"

    def test_search_arxiv_maps_summary_to_abstract(self):
        # Arrange
        long_summary = "X" * 2000  # longer than the 1000-char limit
        fake_result = _make_fake_arxiv_result(summary=long_summary)

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([fake_result])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("test query")

        # Assert — abstract is truncated to 1000 chars
        assert len(results[0]["abstract"]) == 1000

    def test_search_arxiv_authors_are_strings(self):
        # Arrange
        fake_result = _make_fake_arxiv_result(authors=("Alice Smith", "Bob Jones", "Carol King"))

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([fake_result])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("test query")

        # Assert — authors are plain strings
        for author in results[0]["authors"]:
            assert isinstance(author, str)

    def test_search_arxiv_truncates_authors_to_five(self):
        # Arrange — 8 authors, should keep at most 5
        many_authors = tuple(f"Author {i}" for i in range(8))
        fake_result = _make_fake_arxiv_result(authors=many_authors)

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([fake_result])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("test query")

        # Assert
        assert len(results[0]["authors"]) == 5

    def test_search_arxiv_uses_pdf_url_when_available(self):
        # Arrange
        fake_result = _make_fake_arxiv_result(
            pdf_url="http://arxiv.org/pdf/2401.00001v1",
            entry_id="http://arxiv.org/abs/2401.00001v1",
        )

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([fake_result])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("test query")

        # Assert — prefers pdf_url
        assert results[0]["url"] == "http://arxiv.org/pdf/2401.00001v1"

    def test_search_arxiv_falls_back_to_entry_id_when_no_pdf_url(self):
        # Arrange
        fake_result = _make_fake_arxiv_result(
            pdf_url=None,
            entry_id="http://arxiv.org/abs/2401.00001v1",
        )

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([fake_result])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("test query")

        # Assert — falls back to entry_id
        assert results[0]["url"] == "http://arxiv.org/abs/2401.00001v1"

    def test_search_arxiv_published_is_iso_format_string(self):
        # Arrange
        pub_date = datetime.datetime(2024, 3, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
        fake_result = _make_fake_arxiv_result(published=pub_date)

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([fake_result])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("test query")

        # Assert — published is an ISO format string
        assert isinstance(results[0]["published"], str)
        assert "2024-03-15" in results[0]["published"]

    def test_search_arxiv_returns_multiple_results(self):
        # Arrange
        fake_results = [
            _make_fake_arxiv_result(entry_id=f"http://arxiv.org/abs/240{i}.00001v1", title=f"Paper {i}")
            for i in range(3)
        ]

        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter(fake_results)
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("test query", max_results=3)

        # Assert
        assert len(results) == 3

    def test_search_arxiv_exception_returns_empty_list(self):
        # Arrange
        with patch("agent.tools.arxiv.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.results.side_effect = Exception("Network error")
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("test query")

        # Assert — error path returns []
        assert results == []

    def test_search_arxiv_empty_results_returns_empty_list(self):
        # Arrange
        with patch("agent.tools.arxiv.Client") as mock_client_cls, \
             patch("agent.tools.arxiv.Search"):
            mock_client = MagicMock()
            mock_client.results.return_value = iter([])
            mock_client_cls.return_value = mock_client

            # Act
            results = search_arxiv.func("no papers for this query")

        # Assert
        assert results == []


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_web_search_no_api_key_returns_unavailable_message(self, monkeypatch):
        # Arrange
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        # Act
        result = web_search.func("OpenAI news")

        # Assert
        assert "unavailable" in result.lower() or "TAVILY_API_KEY" in result

    def test_web_search_with_key_and_answer_returns_formatted_string(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        fake_response = {
            "answer": "OpenAI released GPT-5 last week.",
            "results": [
                {
                    "title": "GPT-5 Release",
                    "url": "https://openai.com/blog/gpt5",
                    "content": "OpenAI announced GPT-5 with significant improvements.",
                }
            ],
        }

        with patch("agent.tools.tavily_search", return_value=fake_response):
            # Act
            result = web_search.func("OpenAI GPT-5 release")

        # Assert — result is a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0
        assert "OpenAI released GPT-5" in result

    def test_web_search_with_key_formats_results_as_bullet_list(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        fake_response = {
            "results": [
                {
                    "title": "Some AI News",
                    "url": "https://example.com/ai-news",
                    "content": "New AI model sets benchmark record.",
                }
            ],
        }

        with patch("agent.tools.tavily_search", return_value=fake_response):
            # Act
            result = web_search.func("AI news")

        # Assert — results are formatted as a list item
        assert "- [Some AI News]" in result or "Some AI News" in result

    def test_web_search_with_key_and_empty_response_returns_failure_message(
        self, monkeypatch
    ):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        with patch("agent.tools.tavily_search", return_value={}):
            # Act
            result = web_search.func("some query")

        # Assert — communicates failure gracefully
        assert isinstance(result, str)
        assert "failed" in result.lower() or "not set" in result.lower()

    def test_web_search_truncates_content_to_300_chars(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        long_content = "X" * 600
        fake_response = {
            "results": [
                {
                    "title": "Long Article",
                    "url": "https://example.com",
                    "content": long_content,
                }
            ],
        }

        with patch("agent.tools.tavily_search", return_value=fake_response):
            # Act
            result = web_search.func("test")

        # Assert — content is capped at 300 chars before being included
        assert long_content not in result
        # The truncated version (300 Xs) should appear
        assert "X" * 300 in result
        assert "X" * 301 not in result

    def test_web_search_handles_missing_title_gracefully(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        fake_response = {
            "results": [
                {
                    "url": "https://example.com",
                    "content": "Some content without a title.",
                }
            ],
        }

        with patch("agent.tools.tavily_search", return_value=fake_response):
            # Act — should not raise KeyError
            result = web_search.func("test query")

        assert isinstance(result, str)
        assert "No title" in result

    def test_web_search_calls_tavily_with_max_results_5(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        with patch("agent.tools.tavily_search", return_value={"results": []}) as mock_ts:
            # Act
            web_search.func("AI news")

            # Assert — always requests 5 results with include_answer=True
            mock_ts.assert_called_once_with("AI news", max_results=5, include_answer=True)
