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
        "GROQ_API_KEY": "gsk_...",
        "TAVILY_API_KEY": "tvly_..."
      }
    }
  }
}
"""
from dotenv import load_dotenv

load_dotenv()

from mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run()
