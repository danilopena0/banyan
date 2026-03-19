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

from mcp.server import Server
from mcp import types

from rag.retriever import retrieve_across_dates

logger = logging.getLogger(__name__)

app = Server("ai-research-briefing")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all tools this MCP server exposes."""
    return [
        types.Tool(
            name="run_daily_briefing",
            description=(
                "Trigger a full AI research briefing agent run. "
                "Fetches from arXiv and web, then synthesizes a markdown briefing."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="get_latest_briefing",
            description="Return the most recent saved daily AI briefing as markdown text.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="search_past_briefings",
            description=(
                "Semantic search over all past briefings stored in ChromaDB. "
                "Find content related to a specific topic across multiple days."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'diffusion models', 'AI safety')",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_trending_topics",
            description="Analyze recurring themes and topics across the last N days of briefings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of past days to analyze (default: 7)",
                        "default": 7,
                    }
                },
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls from MCP clients."""
    if name == "run_daily_briefing":
        return await _run_daily_briefing()
    elif name == "get_latest_briefing":
        return await _get_latest_briefing()
    elif name == "search_past_briefings":
        return await _search_past_briefings(
            query=arguments.get("query", ""),
            k=arguments.get("k", 10),
        )
    elif name == "get_trending_topics":
        return await _get_trending_topics(days=arguments.get("days", 7))
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def _run_daily_briefing() -> list[types.TextContent]:
    """Trigger a full agent run."""
    try:
        from agent.graph import build_graph
        from agent.state import ResearchState

        date = datetime.now().strftime("%Y-%m-%d")
        graph = build_graph()
        final_state = graph.invoke(
            ResearchState(date=date),
            config={"recursion_limit": 25},
        )

        filepath = f"output/{date}.md"
        if os.path.exists(filepath):
            with open(filepath) as f:
                content = f.read()
            return [types.TextContent(type="text", text=content)]
        return [types.TextContent(type="text", text="Briefing generated but file not found.")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error running briefing: {e}")]


async def _get_latest_briefing() -> list[types.TextContent]:
    """Return most recent saved briefing."""
    files = sorted(glob.glob("output/*.md"), reverse=True)
    if not files:
        return [types.TextContent(
            type="text",
            text="No briefings found. Run run_daily_briefing first."
        )]
    with open(files[0]) as f:
        content = f.read()
    return [types.TextContent(type="text", text=content)]


async def _search_past_briefings(query: str, k: int) -> list[types.TextContent]:
    """Semantic search over ChromaDB."""
    docs = retrieve_across_dates(query=query, k=k)
    if not docs:
        return [types.TextContent(type="text", text="No results found.")]

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        results.append(
            f"**Result {i}** (source: {meta.get('source', 'unknown')}, "
            f"date: {meta.get('date', 'unknown')})\n"
            f"{doc.page_content[:500]}\n"
        )
    return [types.TextContent(type="text", text="\n---\n".join(results))]


async def _get_trending_topics(days: int) -> list[types.TextContent]:
    """Analyze trends across N days."""
    query = "trending topics recurring themes important developments AI machine learning"
    docs = retrieve_across_dates(query=query, k=days * 5)

    if not docs:
        return [types.TextContent(
            type="text",
            text="Not enough historical data for trend analysis."
        )]

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

    return [types.TextContent(type="text", text="\n".join(lines))]


def create_mcp_server():
    """Return the configured MCP server app."""
    return app
