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


def _get_chroma_client():
    """Get persistent ChromaDB client."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    return chromadb.PersistentClient(path=persist_dir)


def _get_vector_store() -> Chroma:
    """Get LangChain Chroma vector store wrapper."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
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


def embed_and_store(papers: list[dict], posts: list[dict], date: str) -> list[str]:
    """
    Embed new papers and posts, store in ChromaDB with metadata.

    Each document includes metadata for filtering:
    - source: 'arxiv' or 'reddit'
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

    # Embed Reddit posts: title + body
    for post in posts:
        doc_id = f"reddit_{post['id']}"
        text = (
            f"REDDIT [{post.get('subreddit', '')}]: {post['title']}\n\n"
            f"{post.get('body', '')}"
        )

        documents.append(Document(
            page_content=text,
            metadata={
                "source": "reddit",
                "source_id": post["id"],
                "date": date,
                "url": post.get("url", ""),
                "title": post.get("title", ""),
                "subreddit": post.get("subreddit", ""),
                "score": post.get("score", 0),
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
