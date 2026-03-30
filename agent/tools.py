"""
LangChain tools with full schemas for arXiv, HuggingFace Papers, and web search.

Pattern: Defining tools as @tool decorated functions lets LangGraph's
ToolNode automatically parse and execute the LLM's tool calls. The
docstrings become the tool descriptions the LLM uses to decide when
to call each tool.
"""
import logging
import os

import arxiv
import requests
from langchain_core.tools import tool

from agent.tavily import tavily_search

logger = logging.getLogger(__name__)

_HF_PAPERS_API = "https://huggingface.co/api/papers"


@tool
def search_arxiv(query: str, max_results: int = 20) -> list[dict]:
    """
    Search arXiv for recent AI/ML papers sorted by submission date. Use as a fallback when search_hf_papers gives thin results.

    Good for niche or very recent topics not yet indexed on HuggingFace Papers.

    Args:
        query: Keyword search string — no years (e.g. 'neural architecture search' not 'NAS 2024')
        max_results: Number of papers to return (default: 20)

    Returns:
        List of dicts with title, authors, abstract, url, published, categories.
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
def search_hf_papers(query: str, max_results: int = 20) -> list[dict]:
    """
    Search HuggingFace Papers using semantic search. Returns papers with community upvote counts and GitHub repo links.

    Use this as your primary paper search — it ranks results by relevance rather than date alone.
    Good for: 'mixture of experts', 'speculative decoding', 'vision language models', 'test-time compute'.

    Args:
        query: Concept or technique to search for (no year needed)
        max_results: Number of papers to return (default: 20)

    Returns:
        List of dicts with title, authors, abstract, url, published, hf_upvotes, github_repo, project_page.
    """
    try:
        response = requests.get(
            _HF_PAPERS_API,
            params={"q": query, "limit": max_results},
            timeout=15,
        )
        response.raise_for_status()
        items = response.json()

        papers = []
        for item in items:
            arxiv_id = item.get("id", "")
            # Use the arXiv ID as the deduplication key (matches embed_and_store logic).
            # Note: search_arxiv uses full arXiv URLs as IDs — cross-tool deduplication
            # within a run is handled by seen_this_run in deduplicate_and_embed_node.
            papers.append({
                "id": arxiv_id,
                "title": item.get("title", ""),
                "authors": [a["name"] for a in item.get("authors", [])[:5]],
                "abstract": item.get("summary", "") or item.get("ai_summary", ""),
                "url": f"https://huggingface.co/papers/{arxiv_id}",
                "published": item.get("publishedAt", ""),
                "hf_upvotes": item.get("upvotes", 0),
                "github_repo": item.get("githubRepo") or "",
                "project_page": item.get("projectPage") or "",
                "source": "hf_papers",
            })

        logger.info(f"HF Papers search '{query}' returned {len(papers)} papers")
        return papers

    except Exception as e:
        logger.error(f"HF Papers search failed: {e}")
        return []


@tool
def web_search(query: str) -> str:
    """
    Search the web for current AI news, product releases, and industry developments.

    Use this for finding news that wouldn't be in research papers yet, like:
    - New model releases from OpenAI, Anthropic, Google, Meta, etc.
    - AI product launches and startup news
    - Industry announcements and policy developments
    - Benchmark results and model comparisons

    Args:
        query: Search query (e.g., 'OpenAI new model release 2024', 'AI regulation news')

    Returns:
        String with search results and snippets
    """
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
ALL_TOOLS = [search_arxiv, search_hf_papers, web_search]
