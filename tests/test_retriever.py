"""
Tests for rag/retriever.py.

Mocks get_vector_store to avoid ChromaDB disk I/O and embedding model loading.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.retriever import retrieve_relevant_context, retrieve_across_dates


def _make_docs(n: int = 2) -> list[Document]:
    return [Document(page_content=f"Content {i}", metadata={"date": "2024-01-01"}) for i in range(n)]


class TestRetrieveRelevantContext:
    def test_returns_documents_from_vector_store(self):
        docs = _make_docs(3)
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = docs

        with patch("rag.retriever.get_vector_store", return_value=mock_vs):
            result = retrieve_relevant_context("transformers and attention")

        assert result == docs
        assert len(result) == 3

    def test_passes_k_to_similarity_search(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        with patch("rag.retriever.get_vector_store", return_value=mock_vs):
            retrieve_relevant_context("query", k=7)

        call_kwargs = mock_vs.similarity_search.call_args
        assert call_kwargs[1].get("k") == 7 or call_kwargs[0][1] == 7

    def test_passes_filter_metadata_to_similarity_search(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        flt = {"date": "2024-01-15"}

        with patch("rag.retriever.get_vector_store", return_value=mock_vs):
            retrieve_relevant_context("query", filter_metadata=flt)

        call_kwargs = mock_vs.similarity_search.call_args
        assert call_kwargs[1].get("filter") == flt

    def test_no_filter_metadata_omits_filter_kwarg(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        with patch("rag.retriever.get_vector_store", return_value=mock_vs):
            retrieve_relevant_context("query", filter_metadata=None)

        call_kwargs = mock_vs.similarity_search.call_args
        assert "filter" not in call_kwargs[1]

    def test_vector_store_exception_returns_empty_list(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.side_effect = Exception("ChromaDB unavailable")

        with patch("rag.retriever.get_vector_store", return_value=mock_vs):
            result = retrieve_relevant_context("query")

        assert result == []

    def test_get_vector_store_exception_returns_empty_list(self):
        with patch("rag.retriever.get_vector_store", side_effect=Exception("disk error")):
            result = retrieve_relevant_context("query")

        assert result == []

    def test_returns_list_type(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = _make_docs(1)

        with patch("rag.retriever.get_vector_store", return_value=mock_vs):
            result = retrieve_relevant_context("query")

        assert isinstance(result, list)


class TestRetrieveAcrossDates:
    def test_returns_documents(self):
        docs = _make_docs(2)
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = docs

        with patch("rag.retriever.get_vector_store", return_value=mock_vs):
            result = retrieve_across_dates("trend analysis")

        assert result == docs

    def test_calls_without_date_filter(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        with patch("rag.retriever.get_vector_store", return_value=mock_vs):
            retrieve_across_dates("trends")

        call_kwargs = mock_vs.similarity_search.call_args
        assert "filter" not in call_kwargs[1]

    def test_passes_k_parameter(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        with patch("rag.retriever.get_vector_store", return_value=mock_vs):
            retrieve_across_dates("trends", k=20)

        call_kwargs = mock_vs.similarity_search.call_args
        assert call_kwargs[1].get("k") == 20 or call_kwargs[0][1] == 20
