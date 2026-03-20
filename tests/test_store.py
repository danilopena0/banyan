"""
Tests for rag/store.py.

We mock _get_vector_store and _get_chroma_client at the module level to
avoid real ChromaDB disk I/O and avoid loading the embedding model.
These tests verify the logic inside embed_and_store and get_seen_ids
without relying on any infrastructure.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from langchain_core.documents import Document

from rag.store import embed_and_store, get_seen_ids, COLLECTION_NAME


# ---------------------------------------------------------------------------
# embed_and_store
# ---------------------------------------------------------------------------

class TestEmbedAndStore:
    def _make_paper(self, suffix="001") -> dict:
        return {
            "id": f"http://arxiv.org/abs/2401.{suffix}v1",
            "title": f"Paper {suffix}",
            "authors": ["Alice Smith", "Bob Jones"],
            "abstract": "An abstract about neural networks and transformers.",
            "url": f"http://arxiv.org/pdf/2401.{suffix}v1",
            "published": "2024-01-01T00:00:00+00:00",
            "categories": ["cs.LG", "cs.AI"],
        }

    def test_embed_and_store_empty_papers_returns_empty_list(self):
        # Arrange / Act — no papers, no vector store interaction needed
        with patch("rag.store._get_vector_store") as mock_get_vs:
            result = embed_and_store(papers=[], date="2024-01-01")

        # Assert
        assert result == []
        mock_get_vs.assert_not_called()

    def test_embed_and_store_single_paper_calls_add_documents(self):
        # Arrange
        paper = self._make_paper("001")
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act
            result = embed_and_store(papers=[paper], date="2024-01-01")

        # Assert
        mock_vs.add_documents.assert_called_once()

    def test_embed_and_store_returns_list_of_ids(self):
        # Arrange
        paper = self._make_paper("001")
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act
            result = embed_and_store(papers=[paper], date="2024-01-01")

        # Assert — returns a list of string IDs
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(id_, str) for id_ in result)

    def test_embed_and_store_id_derived_from_paper_id(self):
        # Arrange — paper id: .../2401.00001v1 → doc id: arxiv_2401.00001v1
        paper = self._make_paper("00001")
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act
            result = embed_and_store(papers=[paper], date="2024-01-01")

        # Assert
        assert result[0] == "arxiv_2401.00001v1"

    def test_embed_and_store_document_content_includes_title(self):
        # Arrange
        paper = self._make_paper("001")
        paper["title"] = "Unique Title For Testing"
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act
            embed_and_store(papers=[paper], date="2024-01-01")

        # Assert — inspect the Document passed to add_documents
        call_kwargs = mock_vs.add_documents.call_args
        documents = call_kwargs[1].get("documents") or call_kwargs[0][0]
        assert any("Unique Title For Testing" in doc.page_content for doc in documents)

    def test_embed_and_store_document_content_includes_abstract(self):
        # Arrange
        paper = self._make_paper("001")
        paper["abstract"] = "This is a distinctive abstract phrase."
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act
            embed_and_store(papers=[paper], date="2024-01-01")

        # Assert
        call_kwargs = mock_vs.add_documents.call_args
        documents = call_kwargs[1].get("documents") or call_kwargs[0][0]
        assert any("This is a distinctive abstract phrase." in doc.page_content for doc in documents)

    def test_embed_and_store_document_content_includes_authors(self):
        # Arrange
        paper = self._make_paper("001")
        paper["authors"] = ["Dr. Unique Author"]
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act
            embed_and_store(papers=[paper], date="2024-01-01")

        # Assert
        call_kwargs = mock_vs.add_documents.call_args
        documents = call_kwargs[1].get("documents") or call_kwargs[0][0]
        assert any("Dr. Unique Author" in doc.page_content for doc in documents)

    def test_embed_and_store_document_metadata_contains_required_keys(self):
        # Arrange
        paper = self._make_paper("001")
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act
            embed_and_store(papers=[paper], date="2024-01-15")

        # Assert — metadata fields
        call_kwargs = mock_vs.add_documents.call_args
        documents = call_kwargs[1].get("documents") or call_kwargs[0][0]
        meta = documents[0].metadata

        assert meta["source"] == "arxiv"
        assert meta["source_id"] == paper["id"]
        assert meta["date"] == "2024-01-15"
        assert meta["url"] == paper["url"]
        assert meta["title"] == paper["title"]

    def test_embed_and_store_categories_joined_as_comma_string(self):
        # Arrange
        paper = self._make_paper("001")
        paper["categories"] = ["cs.LG", "cs.AI", "stat.ML"]
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            embed_and_store(papers=[paper], date="2024-01-01")

        call_kwargs = mock_vs.add_documents.call_args
        documents = call_kwargs[1].get("documents") or call_kwargs[0][0]
        assert documents[0].metadata["categories"] == "cs.LG,cs.AI,stat.ML"

    def test_embed_and_store_multiple_papers_returns_correct_count(self):
        # Arrange
        papers = [self._make_paper(f"00{i}") for i in range(1, 4)]
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act
            result = embed_and_store(papers=papers, date="2024-01-01")

        # Assert
        assert len(result) == 3

    def test_embed_and_store_passes_ids_to_add_documents(self):
        # Arrange
        papers = [self._make_paper("111"), self._make_paper("222")]
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act
            result = embed_and_store(papers=papers, date="2024-01-01")

        # Assert — ids kwarg passed to add_documents matches returned ids
        call_kwargs = mock_vs.add_documents.call_args
        passed_ids = call_kwargs[1].get("ids") or call_kwargs[0][1]
        assert set(passed_ids) == set(result)

    def test_embed_and_store_vector_store_exception_returns_empty_list(self):
        # Arrange
        paper = self._make_paper("001")
        mock_vs = MagicMock()
        mock_vs.add_documents.side_effect = Exception("ChromaDB connection failed")

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act — should not raise
            result = embed_and_store(papers=[paper], date="2024-01-01")

        # Assert
        assert result == []

    def test_embed_and_store_paper_without_categories_uses_empty_string(self):
        # Arrange — paper has no categories key
        paper = {
            "id": "http://arxiv.org/abs/2401.00001v1",
            "title": "No Categories Paper",
            "authors": ["Author A"],
            "abstract": "Abstract.",
            "url": "http://arxiv.org/pdf/2401.00001v1",
        }
        mock_vs = MagicMock()

        with patch("rag.store._get_vector_store", return_value=mock_vs):
            # Act — must not raise KeyError
            result = embed_and_store(papers=[paper], date="2024-01-01")

        # Assert — stored successfully
        assert len(result) == 1
        call_kwargs = mock_vs.add_documents.call_args
        documents = call_kwargs[1].get("documents") or call_kwargs[0][0]
        assert documents[0].metadata["categories"] == ""


# ---------------------------------------------------------------------------
# get_seen_ids
# ---------------------------------------------------------------------------

class TestGetSeenIds:
    def _make_mock_collection(self, metadatas: list[dict | None]) -> MagicMock:
        """Build a mock ChromaDB collection with given metadatas list."""
        collection = MagicMock()
        collection.get.return_value = {"metadatas": metadatas}
        return collection

    def test_get_seen_ids_empty_collection_returns_empty_set(self):
        # Arrange
        mock_collection = self._make_mock_collection([])
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("rag.store._get_chroma_client", return_value=mock_client):
            # Act
            result = get_seen_ids()

        # Assert
        assert result == set()

    def test_get_seen_ids_single_metadata_with_source_id(self):
        # Arrange
        mock_collection = self._make_mock_collection([
            {"source_id": "http://arxiv.org/abs/2401.00001v1", "date": "2024-01-01"}
        ])
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("rag.store._get_chroma_client", return_value=mock_client):
            # Act
            result = get_seen_ids()

        # Assert
        assert result == {"http://arxiv.org/abs/2401.00001v1"}

    def test_get_seen_ids_multiple_metadatas_returns_all_ids(self):
        # Arrange
        mock_collection = self._make_mock_collection([
            {"source_id": "http://arxiv.org/abs/2401.00001v1"},
            {"source_id": "http://arxiv.org/abs/2401.00002v1"},
            {"source_id": "http://arxiv.org/abs/2401.00003v1"},
        ])
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("rag.store._get_chroma_client", return_value=mock_client):
            # Act
            result = get_seen_ids()

        # Assert
        assert len(result) == 3
        assert "http://arxiv.org/abs/2401.00001v1" in result
        assert "http://arxiv.org/abs/2401.00003v1" in result

    def test_get_seen_ids_skips_metadata_without_source_id(self):
        # Arrange — some entries lack source_id (e.g. from a different ingestion)
        mock_collection = self._make_mock_collection([
            {"source_id": "http://arxiv.org/abs/2401.00001v1"},
            {"title": "No source_id here", "date": "2024-01-01"},
        ])
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("rag.store._get_chroma_client", return_value=mock_client):
            # Act
            result = get_seen_ids()

        # Assert — only the entry with source_id is included
        assert result == {"http://arxiv.org/abs/2401.00001v1"}

    def test_get_seen_ids_skips_none_metadata_entries(self):
        # Arrange — ChromaDB can return None for metadata if not set
        mock_collection = self._make_mock_collection([
            None,
            {"source_id": "http://arxiv.org/abs/2401.00001v1"},
            None,
        ])
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("rag.store._get_chroma_client", return_value=mock_client):
            # Act — must not raise TypeError on None
            result = get_seen_ids()

        # Assert
        assert result == {"http://arxiv.org/abs/2401.00001v1"}

    def test_get_seen_ids_uses_correct_collection_name(self):
        # Arrange
        mock_collection = self._make_mock_collection([])
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("rag.store._get_chroma_client", return_value=mock_client):
            # Act
            get_seen_ids()

        # Assert — correct collection name used
        mock_client.get_or_create_collection.assert_called_once_with(COLLECTION_NAME)

    def test_get_seen_ids_collection_returns_set_type(self):
        # Arrange
        mock_collection = self._make_mock_collection([
            {"source_id": "id_1"},
            {"source_id": "id_2"},
        ])
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("rag.store._get_chroma_client", return_value=mock_client):
            # Act
            result = get_seen_ids()

        # Assert — return type is always a set
        assert isinstance(result, set)

    def test_get_seen_ids_deduplicates_repeated_source_ids(self):
        # Arrange — same source_id appears twice (shouldn't happen but be safe)
        mock_collection = self._make_mock_collection([
            {"source_id": "http://arxiv.org/abs/2401.00001v1"},
            {"source_id": "http://arxiv.org/abs/2401.00001v1"},
        ])
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("rag.store._get_chroma_client", return_value=mock_client):
            # Act
            result = get_seen_ids()

        # Assert — set semantics deduplicates
        assert result == {"http://arxiv.org/abs/2401.00001v1"}

    def test_get_seen_ids_client_exception_returns_empty_set(self):
        # Arrange
        with patch("rag.store._get_chroma_client", side_effect=Exception("Disk read error")):
            # Act — must not raise
            result = get_seen_ids()

        # Assert — graceful degradation
        assert result == set()
