"""
Shared Tavily search helper.

Both nodes.py (curated news + enrichment) and tools.py (ReAct web_search tool)
use TavilyClient with the same pattern. Centralizing here ensures any change
to the API key guard or call parameters only needs to happen in one place.
"""
import logging
import os

from tavily import TavilyClient

logger = logging.getLogger(__name__)


def tavily_search(query: str, max_results: int = 5, include_answer: bool = False) -> dict:
    """
    Run a single Tavily search. Returns the raw response dict or {} on failure.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        include_answer: Whether to request a synthesized answer from Tavily

    Returns:
        Tavily response dict with 'results' key, or {} if key is missing or call fails.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {}
    try:
        client = TavilyClient(api_key=api_key)
        return client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=include_answer,
        )
    except Exception as e:
        logger.warning(f"Tavily search failed for '{query}': {e}")
        return {}
