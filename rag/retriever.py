"""
Semantic retrieval from ChromaDB.

Pattern (RAG - retrieve): The retriever converts the synthesis query into
a vector, then finds the most semantically similar chunks in ChromaDB.
This is more effective than just passing all raw content because:
1. It respects token limits
2. It surfaces the most relevant content first
3. It works across past briefings too (for the MCP tools)
"""
import logging
from typing import Optional

from langchain_core.documents import Document

from rag.store import _get_vector_store

logger = logging.getLogger(__name__)


def retrieve_relevant_context(
    query: str,
    k: int = 10,
    filter_metadata: Optional[dict] = None,
) -> list[Document]:
    """
    Semantic retrieval of most relevant content chunks.

    Args:
        query: Natural language query to embed and search against
        k: Number of top results to return
        filter_metadata: Optional ChromaDB metadata filter
                         e.g. {"date": "2024-01-15"} or {"source": "arxiv"}

    Returns:
        List of Document objects, sorted by semantic relevance
    """
    try:
        vector_store = _get_vector_store()

        search_kwargs: dict = {"k": k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} chunks for query: '{query[:60]}...'")
        return docs

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []


def retrieve_across_dates(query: str, k: int = 15) -> list[Document]:
    """
    Retrieve relevant content across ALL stored dates (for trend analysis).
    Used by the MCP server's get_trending_topics tool.
    """
    return retrieve_relevant_context(query=query, k=k, filter_metadata=None)
