"""
LangGraph node functions.

Each node performs one focused step in the research pipeline.
The ReAct research loop (research_agent_node + tool_node) handles
dynamic tool calling; the remaining nodes handle processing and synthesis.
"""
import json
import logging
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from tavily import TavilyClient

from agent.state import ResearchState
from agent.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
)
from agent.tools import ALL_TOOLS
from rag.store import embed_and_store, get_seen_ids
from rag.retriever import retrieve_relevant_context
from schemas.briefing import DailyBriefing

logger = logging.getLogger(__name__)


def _build_llm(model_id: str, fallback_model_id: str = None):
    """
    Build a ChatHuggingFace LLM with automatic fallback.

    Pattern: Wrapping HuggingFaceEndpoint in ChatHuggingFace gives us
    the standard ChatModel interface (bind_tools, with_structured_output, etc.)
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")

    try:
        endpoint = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=token,
            task="text-generation",
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
        )
        return ChatHuggingFace(llm=endpoint)
    except Exception as e:
        if fallback_model_id:
            logger.warning(
                f"Primary model {model_id} failed ({e}), "
                f"trying fallback {fallback_model_id}"
            )
            endpoint = HuggingFaceEndpoint(
                repo_id=fallback_model_id,
                huggingfacehub_api_token=token,
                task="text-generation",
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
            )
            return ChatHuggingFace(llm=endpoint)
        raise


def _tavily_search(query: str, max_results: int = 5) -> dict:
    """Run a single Tavily search. Returns raw response or empty dict on failure."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {}
    try:
        client = TavilyClient(api_key=api_key)
        return client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
        )
    except Exception as e:
        logger.warning(f"Tavily search failed for '{query}': {e}")
        return {}


def fetch_ai_news_node(state: ResearchState) -> dict:
    """
    Fetch curated AI/ML news via Tavily before the ReAct loop.

    Fires targeted queries for model releases, benchmarks, and industry
    news so the research agent and synthesis have fresh web context.
    """
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not set — skipping AI news fetch")
        return {"web_news": []}

    queries = [
        f"AI machine learning model release announcement {state.date[:7]}",
        "large language model benchmark results latest",
        "AI research breakthrough paper publication",
    ]

    news_items = []
    for query in queries:
        response = _tavily_search(query, max_results=4)
        for r in response.get("results", []):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("content", "")[:200]
            if title:
                news_items.append(f"[{title}]({url}): {snippet}")

    logger.info(f"Fetched {len(news_items)} AI news items via Tavily")
    return {"web_news": news_items}


def enrich_papers_node(state: ResearchState) -> dict:
    """
    Score each paper by web presence using Tavily.

    Searches for each paper title and attaches web_mentions (result count)
    and web_context (snippets) to the paper dict. Papers with web_mentions > 0
    are being discussed online and will be ranked as most_discussed in synthesis.
    """
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not set — skipping paper enrichment")
        return {"raw_papers": state.raw_papers}

    enriched = []
    for paper in state.raw_papers[:20]:  # cap at 20 to stay within rate limits
        title = paper.get("title", "")
        if not title:
            enriched.append(paper)
            continue

        response = _tavily_search(f'"{title}" research paper', max_results=3)
        results = response.get("results", [])
        snippets = [r.get("content", "")[:150] for r in results if r.get("content")]

        enriched.append({
            **paper,
            "web_mentions": len(results),
            "web_context": " | ".join(snippets),
        })

    # Preserve any papers beyond the cap without enrichment
    for paper in state.raw_papers[20:]:
        enriched.append({**paper, "web_mentions": 0, "web_context": ""})

    discussed = sum(1 for p in enriched if p.get("web_mentions", 0) > 0)
    logger.info(f"Enriched {len(enriched)} papers — {discussed} have web presence")
    return {"raw_papers": enriched}


def enrich_concept_node(state: ResearchState) -> dict:
    """
    Find a beginner-friendly resource URL for the concept of the day via Tavily.
    """
    if not state.briefing or not state.briefing.concept_of_the_day:
        return {}
    if not os.getenv("TAVILY_API_KEY"):
        return {}

    concept_name = state.briefing.concept_of_the_day.name
    response = _tavily_search(
        f"{concept_name} explained machine learning beginner guide",
        max_results=3,
    )

    results = response.get("results", [])
    # Prefer results from known educational sites
    preferred_domains = ("distill.pub", "colah.github", "lilianweng", "arxiv", "explained.ai")
    url = ""
    for r in results:
        r_url = r.get("url", "")
        if any(d in r_url for d in preferred_domains):
            url = r_url
            break
    if not url and results:
        url = results[0].get("url", "")

    if url:
        briefing = state.briefing
        briefing.concept_of_the_day.learn_more_url = url
        logger.info(f"Concept '{concept_name}' learn_more_url: {url}")
        return {"briefing": briefing}

    return {}


def research_agent_node(state: ResearchState) -> dict:
    """
    ReAct research node: the LLM decides which tools to call.

    Pattern (ReAct): The LLM is given tool schemas and decides autonomously
    which tools to call with what arguments. LangGraph's ToolNode executes
    the actual tool calls, then loops back here until the LLM stops calling tools.
    """
    primary_model = os.getenv("PRIMARY_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    fallback_model = os.getenv("FALLBACK_MODEL", "microsoft/Phi-3.5-mini-instruct")

    try:
        llm = _build_llm(primary_model, fallback_model)
        # Bind tools so the LLM knows what's available and can call them
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
    except Exception as e:
        logger.error(f"Failed to build LLM: {e}")
        return {"errors": state.errors + [f"LLM initialization failed: {e}"]}

    # Initialize conversation with system prompt if first call
    messages = state.messages
    if not messages:
        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Today is {state.date}. Please research today's most important "
                    f"AI/ML developments. Search arXiv for recent papers, check key "
                    f"AI subreddits, and do web searches for major news."
                )
            ),
        ]

    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": messages + [response]}
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        return {
            "messages": messages,
            "errors": state.errors + [f"Research LLM call failed: {e}"],
        }


def should_continue(state: ResearchState) -> str:
    """
    Router: decides whether to keep looping (more tool calls) or move on.

    This is the key control flow for the ReAct loop. If the last message
    has tool_calls, we route to ToolNode. Otherwise, research is done.
    """
    last_message = state.messages[-1] if state.messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "process"


def collect_tool_results(state: ResearchState) -> dict:
    """
    After the ReAct loop ends, extract raw data from tool call results.

    Tool results are stored as ToolMessage objects in the messages list.
    We parse them back into structured dicts for the RAG pipeline.
    """
    papers = []

    for msg in state.messages:
        if not isinstance(msg, ToolMessage):
            continue

        try:
            content = msg.content
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                    if isinstance(data, list) and data:
                        first = data[0]
                        if isinstance(first, dict) and "abstract" in first:
                            papers.extend(data)
                except json.JSONDecodeError:
                    pass  # web_search returns plain text — skip
        except Exception as e:
            logger.warning(f"Failed to parse tool result: {e}")

    logger.info(f"Collected {len(papers)} papers from tool calls")
    return {"raw_papers": papers}


def deduplicate_and_embed_node(state: ResearchState) -> dict:
    """
    Filter content already in ChromaDB, then embed and store new content.

    Pattern (RAG - ingest): We only embed truly new content to avoid
    polluting the vector store with duplicates. The seen_ids check
    makes each daily run incremental.
    """
    try:
        seen_ids = get_seen_ids()

        # Deduplicate within this run first (LLM may call the same tool multiple times)
        seen_this_run: set[str] = set()
        unique_papers = []
        for p in state.raw_papers:
            pid = p.get("id")
            if pid and pid not in seen_this_run:
                seen_this_run.add(pid)
                unique_papers.append(p)

        new_papers = [p for p in unique_papers if p.get("id") not in seen_ids]

        logger.info(
            f"Deduplication: {len(new_papers)}/{len(state.raw_papers)} new papers"
        )

        new_ids = []
        if new_papers:
            new_ids = embed_and_store(
                papers=new_papers, posts=[], date=state.date
            )

        return {"new_content_ids": new_ids}

    except Exception as e:
        logger.error(f"Deduplication/embedding failed: {e}")
        return {"errors": state.errors + [f"Embedding failed: {e}"]}


def retrieve_context_node(state: ResearchState) -> dict:
    """
    Semantic RAG retrieval: get the most relevant content for synthesis.

    Pattern (RAG - retrieve): Instead of passing ALL raw content to the LLM
    (expensive, hits context limits), we do semantic search to get only the
    most relevant chunks. This is the core RAG pattern.
    """
    try:
        synthesis_query = (
            f"Most important AI and machine learning research developments, "
            f"breakthrough papers, community discussion, and emerging trends "
            f"for {state.date}"
        )

        docs = retrieve_relevant_context(
            query=synthesis_query,
            k=20,
            filter_metadata={"date": state.date},
        )

        context_chunks = [doc.page_content for doc in docs]
        logger.info(f"Retrieved {len(context_chunks)} context chunks for synthesis")
        return {"retrieved_context": context_chunks}

    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        # Fall back to raw content
        fallback = []
        for p in state.raw_papers[:10]:
            fallback.append(f"PAPER: {p.get('title', '')}\n{p.get('abstract', '')}")
        for post in state.raw_posts[:10]:
            fallback.append(f"REDDIT: {post.get('title', '')}\n{post.get('body', '')}")

        return {
            "retrieved_context": fallback,
            "errors": state.errors + [f"RAG retrieval failed, using raw content: {e}"],
        }


def synthesize_node(state: ResearchState) -> dict:
    """
    Generate structured DailyBriefing using with_structured_output().

    Pattern (Structured Outputs): Using .with_structured_output(DailyBriefing)
    ensures the LLM response is automatically parsed and validated against
    our Pydantic schema. No manual JSON parsing needed.
    """
    primary_model = os.getenv("PRIMARY_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    fallback_model = os.getenv("FALLBACK_MODEL", "microsoft/Phi-3.5-mini-instruct")

    try:
        llm = _build_llm(primary_model, fallback_model)
    except Exception as e:
        logger.error(f"Failed to build synthesis LLM: {e}")
        return {
            "errors": state.errors + [f"Synthesis LLM failed: {e}"],
            "briefing": _build_fallback_briefing(state),
        }

    context = "\n\n---\n\n".join(state.retrieved_context) if state.retrieved_context else "No context retrieved"

    # Annotate context chunks with web_mentions so LLM can rank by discussion
    papers_with_mentions = [
        f"[web_mentions={p.get('web_mentions', 0)}] {p.get('title', '')}"
        for p in state.raw_papers if p.get("title")
    ]
    if papers_with_mentions:
        context = "Paper web presence scores:\n" + "\n".join(papers_with_mentions) + "\n\n---\n\n" + context

    web_news = "\n".join(f"- {item}" for item in state.web_news) if state.web_news else "No web news fetched"

    user_prompt = SYNTHESIS_USER_TEMPLATE.format(
        date=state.date,
        context=context[:7000],
        web_news=web_news[:1000],
        total_papers=len(state.raw_papers),
    )

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)

        # Strip markdown code fences if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        json_str = json_match.group(1) if json_match else raw.strip()

        data = json.loads(json_str)
        briefing = DailyBriefing.model_validate(data)
        briefing.date = state.date
        briefing.total_papers_analyzed = len(state.raw_papers)
        briefing.errors = state.errors
        return {"briefing": briefing}
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {
            "errors": state.errors + [f"Synthesis failed: {e}"],
            "briefing": _build_fallback_briefing(state),
        }


def save_report_node(state: ResearchState) -> dict:
    """
    Render DailyBriefing Pydantic model to markdown and save to output/.
    """
    if not state.briefing:
        logger.error("No briefing to save")
        return {}

    briefing = state.briefing
    lines = [
        f"# Daily AI Research Briefing",
        f"## {briefing.date}",
        "",
        f"> Analyzed **{briefing.total_papers_analyzed} papers**",
        "",
        "---",
        "",
    ]

    if briefing.most_discussed:
        lines += ["# Most Discussed", ""]
        for paper in briefing.most_discussed:
            lines += [
                f"### {paper.title}",
                f"*{', '.join(paper.authors)}*",
                "",
                f"**Summary:** {paper.plain_english_summary}",
                "",
                f"**Methods:** {paper.methods}",
                "",
                f"**Key contribution:** {paper.key_contribution}",
                "",
                f"**Why it matters:** {paper.significance}",
                "",
                f"> [Read paper]({paper.url})" if paper.url else "",
                "",
                "---",
                "",
            ]

    if briefing.notable_papers:
        lines += ["# Notable Papers", ""]
        for paper in briefing.notable_papers:
            lines += [
                f"### {paper.title}",
                f"*{', '.join(paper.authors)}*",
                "",
                f"**Summary:** {paper.plain_english_summary}",
                "",
                f"**Methods:** {paper.methods}",
                "",
                f"**Why it matters:** {paper.significance}",
                "",
                f"> [Read paper]({paper.url})" if paper.url else "",
                "",
                "---",
                "",
            ]

    if briefing.web_insights:
        lines += ["# Web & Industry News", ""]
        for insight in briefing.web_insights:
            lines.append(f"- {insight}")
        lines += ["", "---", ""]

    lines += ["# Emerging Themes", "", briefing.emerging_themes, "", "---", ""]

    if briefing.concept_of_the_day:
        c = briefing.concept_of_the_day
        lines += [
            "# Concept of the Day",
            f"## {c.name}",
            "",
            c.plain_english,
            "",
            f"**Example:** {c.example}",
            "",
            f"**Why it matters:** {c.why_it_matters}",
            "",
            f"**In today's research:** {c.connected_to_today}",
            "",
            f"> [Learn more]({c.learn_more_url})" if c.learn_more_url else "",
            "",
            "---",
            "",
        ]

    if briefing.errors:
        lines += ["# Errors (non-fatal)", ""]
        for err in briefing.errors:
            lines.append(f"- {err}")

    markdown = "\n".join(lines)

    os.makedirs("output", exist_ok=True)
    filepath = f"output/{briefing.date}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown)

    logger.info(f"Briefing saved to {filepath}")
    return {}


def _build_fallback_briefing(state: ResearchState) -> DailyBriefing:
    """Build a minimal briefing from raw data when LLM synthesis fails."""
    return DailyBriefing(
        date=state.date,
        emerging_themes="Synthesis failed. See errors field for details.",
        total_papers_analyzed=len(state.raw_papers),
        errors=state.errors,
    )
