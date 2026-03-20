"""
ChromaDB read/write operations.

Pattern (RAG - store): ChromaDB is embedded (no server needed), persists
to disk, and supports metadata filtering. We use it to:
1. Deduplicate content across runs (get_seen_ids)
2. Store embeddings for semantic retrieval (embed_and_store)
"""
import os
import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ai_research"
_CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def _get_chroma_client():
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=_CHROMA_PERSIST_DIR)


def _get_vector_store() -> Chroma:
    """Get LangChain Chroma vector store wrapper."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=_CHROMA_PERSIST_DIR,
    )


def get_seen_ids() -> set[str]:
    """
    Get all content IDs already stored in ChromaDB.
    Used for deduplication — we don't re-embed content we've already seen.
    """
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(COLLECTION_NAME)
        results = collection.get(include=["metadatas"])

        seen = set()
        for metadata in results.get("metadatas", []):
            if metadata and "source_id" in metadata:
                seen.add(metadata["source_id"])

        logger.info(f"ChromaDB has {len(seen)} known content IDs")
        return seen
    except Exception as e:
        logger.error(f"Failed to get seen IDs: {e}")
        return set()


def embed_and_store(papers: list[dict], date: str) -> list[str]:
    """
    Embed new papers, store in ChromaDB with metadata.

    Each document includes metadata for filtering:
    - source: 'arxiv'
    - date: YYYY-MM-DD
    - source_id: original ID for deduplication
    - url: link to original content

    Returns list of stored document IDs.
    """
    documents = []
    ids = []

    # Embed papers: title + abstract
    for paper in papers:
        doc_id = f"arxiv_{paper['id'].split('/')[-1]}"
        text = (
            f"PAPER: {paper['title']}\n\n"
            f"Authors: {', '.join(paper.get('authors', []))}\n\n"
            f"Abstract: {paper.get('abstract', '')}"
        )

        documents.append(Document(
            page_content=text,
            metadata={
                "source": "arxiv",
                "source_id": paper["id"],
                "date": date,
                "url": paper.get("url", ""),
                "title": paper.get("title", ""),
                "categories": ",".join(paper.get("categories", [])),
            }
        ))
        ids.append(doc_id)

    if not documents:
        logger.info("No new documents to embed")
        return []

    try:
        vector_store = _get_vector_store()
        vector_store.add_documents(documents=documents, ids=ids)
        logger.info(f"Embedded and stored {len(documents)} new documents")
        return ids
    except Exception as e:
        logger.error(f"Failed to store documents: {e}")
        return []
