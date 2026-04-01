import asyncio
from dotenv import load_dotenv

load_dotenv()

from mcp_server.server import get_latest_briefing, search_past_briefings


async def test():
    print("=== Latest Briefing ===")
    print(await get_latest_briefing())

    print("\n=== Search: diffusion models ===")
    print(await search_past_briefings("diffusion models", k=3))


asyncio.run(test())
