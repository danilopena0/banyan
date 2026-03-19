"""
Main entrypoint for the Daily AI Research Briefing Agent.

Run with: python main.py
"""
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Load .env before any other imports
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Run the full research briefing pipeline."""
    logger.info("=" * 60)
    logger.info("Daily AI Research Briefing Agent")
    logger.info("=" * 60)

    # Validate required env vars
    required = ["HUGGINGFACEHUB_API_TOKEN"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Copy .env.example to .env and fill in your API keys")
        sys.exit(1)

    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not set: web search will be skipped")

    from agent.graph import build_graph
    from agent.state import ResearchState

    date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Generating briefing for {date}")

    graph = build_graph()
    initial_state = ResearchState(date=date)

    try:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 25},  # Safety limit for ReAct loop
        )

        logger.info("=" * 60)
        briefing = final_state.get("briefing") if isinstance(final_state, dict) else getattr(final_state, "briefing", None)
        if briefing:
            logger.info("Briefing complete!")
            logger.info(f"Papers analyzed: {briefing.total_papers_analyzed}")
            logger.info(f"Output saved to: output/{date}.md")

        errors = final_state.get("errors") if isinstance(final_state, dict) else getattr(final_state, "errors", [])
        if errors:
            logger.warning(f"Non-fatal errors: {errors}")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Agent run failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
