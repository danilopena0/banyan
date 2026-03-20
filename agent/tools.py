"""
LangChain tools with full schemas for arXiv, Reddit, and web search.

Pattern: Defining tools as @tool decorated functions lets LangGraph's
ToolNode automatically parse and execute the LLM's tool calls. The
docstrings become the tool descriptions the LLM uses to decide when
to call each tool.
"""
import logging

import arxiv
from langchain_core.tools import tool

from agent.tavily import tavily_search

logger = logging.getLogger(__name__)


@tool
def search_arxiv(query: str, max_results: int = 20) -> list[dict]:
    """
    Search arXiv for recent AI/ML papers.

    Use this tool to find the latest research papers on specific AI/ML topics.
    Results are sorted by submission date (most recent first).
    Good queries: 'large language models', 'diffusion models', 'reinforcement learning from human feedback',
    'multimodal models', 'AI agents', 'neural architecture search', 'model alignment'

    Args:
        query: Search query string — do NOT include years (e.g., use 'large language models' not 'large language models 2024')
        max_results: Maximum number of papers to return (default: 20)

    Returns:
        List of paper dicts with title, authors, abstract, url, published date
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        for result in client.results(search):
            papers.append({
                "id": result.entry_id,
                "title": result.title,
                "authors": [str(a) for a in result.authors[:5]],
                "abstract": result.summary[:1000],
                "url": result.pdf_url or result.entry_id,
                "published": result.published.isoformat(),
                "categories": result.categories,
            })

        logger.info(f"arXiv search '{query}' returned {len(papers)} papers")
        return papers

    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return []



@tool
def web_search(query: str) -> str:
    """
    Search the web for current AI news, product releases, and industry developments.

    Use this for finding news that wouldn't be on arXiv or Reddit yet, like:
    - New model releases from OpenAI, Anthropic, Google, Meta, etc.
    - AI product launches and startup news
    - Industry announcements and policy developments
    - Benchmark results and model comparisons

    Args:
        query: Search query (e.g., 'OpenAI new model release 2024', 'AI regulation news')

    Returns:
        String with search results and snippets
    """
    import os
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search unavailable: TAVILY_API_KEY not set"

    response = tavily_search(query, max_results=5, include_answer=True)
    if not response:
        return "Web search failed or TAVILY_API_KEY not set"

    results = []
    if response.get("answer"):
        results.append(f"Summary: {response['answer']}\n")

    for r in response.get("results", []):
        results.append(
            f"- [{r.get('title', 'No title')}]({r.get('url', '')}): "
            f"{r.get('content', '')[:300]}"
        )

    return "\n".join(results) if results else "No results found"


# All tools for binding to the LLM
ALL_TOOLS = [search_arxiv, web_search]
