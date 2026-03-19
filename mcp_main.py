"""
Entrypoint for MCP server mode.

Connect to Claude Desktop by adding to claude_desktop_config.json:

macOS:  ~/Library/Application Support/Claude/claude_desktop_config.json
Windows: %APPDATA%\\Claude\\claude_desktop_config.json

{
  "mcpServers": {
    "ai-research-briefing": {
      "command": "python",
      "args": ["/absolute/path/to/banyan/mcp_main.py"],
      "env": {
        "HUGGINGFACEHUB_API_TOKEN": "hf_your_token",
        "REDDIT_CLIENT_ID": "...",
        "REDDIT_CLIENT_SECRET": "...",
        "REDDIT_USER_AGENT": "AIBriefingAgent/1.0",
        "TAVILY_API_KEY": "tvly_..."
      }
    }
  }
}
"""
import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()

from mcp.server.stdio import stdio_server
from mcp_server.server import app


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
