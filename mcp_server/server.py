"""
MCP (Model Context Protocol) server exposing the agent as a tool.

Pattern (MCP): By exposing our agent as an MCP server, it can be called
from Claude Desktop, Cursor, or any MCP-compatible AI assistant. This turns
our daily briefing agent into a reusable tool in the AI ecosystem.

Tools exposed:
1. run_daily_briefing    — triggers a full agent run
2. get_latest_briefing   — returns most recent saved briefing
3. search_past_briefings — semantic search over past briefings
4. get_trending_topics   — analyzes recurring themes across days
"""
import glob
import logging
import os
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from agent.config import OUTPUT_DIR
from rag.retriever import retrieve_across_dates

logger = logging.getLogger(__name__)

mcp = FastMCP("ai-research-briefing")


@mcp.tool()
async def run_daily_briefing() -> str:
    """Trigger a full AI research briefing agent run.
    Fetches from arXiv and web, then synthesizes a markdown briefing."""
    try:
        from agent.graph import build_graph
        from agent.state import ResearchState

        date = datetime.now().strftime("%Y-%m-%d")
        graph = build_graph()
        graph.invoke(
            ResearchState(date=date),
            config={"recursion_limit": 50},
        )

        filepath = f"{OUTPUT_DIR}/{date}.md"
        if os.path.exists(filepath):
            with open(filepath) as f:
                return f.read()
        return "Briefing generated but file not found."
    except Exception as e:
        return f"Error running briefing: {e}"


@mcp.tool()
async def get_latest_briefing() -> str:
    """Return the most recent saved daily AI briefing as markdown text."""
    files = sorted(glob.glob(f"{OUTPUT_DIR}/*.md"), reverse=True)
    if not files:
        return "No briefings found. Run run_daily_briefing first."
    with open(files[0]) as f:
        return f.read()


@mcp.tool()
async def search_past_briefings(query: str, k: int = 10) -> str:
    """Semantic search over all past briefings stored in ChromaDB.
    Find content related to a specific topic across multiple days.

    Args:
        query: Search query (e.g., 'diffusion models', 'AI safety')
        k: Number of results to return (default: 10)
    """
    docs = retrieve_across_dates(query=query, k=k)
    if not docs:
        return "No results found."

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        results.append(
            f"**Result {i}** (source: {meta.get('source', 'unknown')}, "
            f"date: {meta.get('date', 'unknown')})\n"
            f"{doc.page_content[:500]}\n"
        )
    return "\n---\n".join(results)


@mcp.tool()
async def get_trending_topics(days: int = 7) -> str:
    """Analyze recurring themes and topics across the last N days of briefings.

    Args:
        days: Number of past days to analyze (default: 7)
    """
    query = "trending topics recurring themes important developments AI machine learning"
    docs = retrieve_across_dates(query=query, k=days * 5)

    if not docs:
        return "Not enough historical data for trend analysis."

    by_date: dict[str, list[str]] = {}
    for doc in docs:
        date = doc.metadata.get("date", "unknown")
        by_date.setdefault(date, []).append(doc.page_content[:200])

    lines = [f"## Trending Topics (last {days} days)\n"]
    for date in sorted(by_date.keys(), reverse=True)[:days]:
        lines.append(f"### {date}")
        for chunk in by_date[date][:3]:
            lines.append(f"- {chunk[:150]}")
        lines.append("")

    return "\n".join(lines)


def create_mcp_server():
    """Return the configured MCP server app."""
    return mcp
