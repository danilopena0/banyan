"""
Embedding logic using sentence-transformers (runs locally, completely free).

Pattern (RAG - embeddings): Using a local embedding model means zero API costs
for embeddings. all-MiniLM-L6-v2 is fast on CPU and produces good embeddings
for semantic similarity tasks.
"""
import os
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get the embedding model (cached — only loaded once per process).

    Uses sentence-transformers/all-MiniLM-L6-v2:
    - 22M parameters, runs fast on CPU
    - 384-dimensional embeddings
    - Great for semantic similarity
    """
    model_name = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
