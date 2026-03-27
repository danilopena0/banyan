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
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from agent.state import ResearchState
from agent.tavily import tavily_search
from agent.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
)
from agent.concepts import CORE_DS_CONCEPTS, FOUNDATIONAL_DS_CONCEPTS
from agent.tools import ALL_TOOLS
from rag.store import embed_and_store, get_seen_ids
from rag.retriever import retrieve_relevant_context
from schemas.briefing import DailyBriefing

logger = logging.getLogger(__name__)

# --- Module-level configuration constants ---
_PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "llama-3.3-70b-versatile")
_FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama-3.1-8b-instant")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# LLM generation settings
_MAX_NEW_TOKENS = 2048
_TEMPERATURE = 0.1

# Enrichment limits
_ENRICH_MAX_PAPERS = 20
_ENRICH_TAVILY_RESULTS_PER_PAPER = 3
_ENRICH_CONCEPT_RESULTS = 3
_ENRICH_SNIPPET_CHARS = 200

# RAG retrieval
_RAG_K = 20

# Synthesis token budgets (character limits before sending to LLM)
_SYNTHESIS_CONTEXT_CHARS = 7000

def _build_llm(model_id: str, fallback_model_id: str = None):
    """
    Build a ChatGroq LLM with automatic fallback.

    Pattern: ChatGroq gives us the standard ChatModel interface
    (bind_tools, with_structured_output, etc.) backed by Groq's free tier.
    Fallback triggers on construction failure (e.g. invalid model name).
    API errors (rate limits, auth) surface at invoke() time.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    try:
        return ChatGroq(
            model=model_id,
            groq_api_key=api_key,
            max_tokens=_MAX_NEW_TOKENS,
            temperature=_TEMPERATURE,
        )
    except Exception as e:
        if fallback_model_id:
            logger.warning(
                f"Primary model {model_id} failed ({e}), "
                f"trying fallback {fallback_model_id}"
            )
            return ChatGroq(
                model=fallback_model_id,
                groq_api_key=api_key,
                max_tokens=_MAX_NEW_TOKENS,
                temperature=_TEMPERATURE,
            )
        raise



_TOOL_RESULT_TRUNCATE_CHARS = 300


def _window_messages(messages: list, keep_recent: int = 6) -> list:
    """
    Trim message history and truncate tool results to stay within TPM limits.

    Two-pronged approach:
    1. Window: keep system + human messages, then only the last `keep_recent`
       messages from the ReAct loop.
    2. Truncate: ToolMessage content is capped at _TOOL_RESULT_TRUNCATE_CHARS —
       the LLM only needs to know what was found, not re-read every abstract.
       Full content is preserved in state.messages for collect_tool_results.
    """
    windowed = messages[:2] + messages[-(keep_recent):] if len(messages) > keep_recent + 2 else messages

    truncated = []
    for msg in windowed:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and len(msg.content) > _TOOL_RESULT_TRUNCATE_CHARS:
            msg = ToolMessage(
                content=msg.content[:_TOOL_RESULT_TRUNCATE_CHARS] + "… [truncated]",
                tool_call_id=msg.tool_call_id,
            )
        truncated.append(msg)
    return truncated


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
    for paper in state.raw_papers[:_ENRICH_MAX_PAPERS]:
        title = paper.get("title", "")
        if not title:
            enriched.append(paper)
            continue

        # HF Papers already include upvotes — skip Tavily call and derive web_mentions
        # from the social signal so we don't burn API quota unnecessarily.
        if paper.get("hf_upvotes") is not None:
            enriched.append({
                **paper,
                "web_mentions": 1 if paper["hf_upvotes"] > 0 else 0,
                "web_context": f"HF upvotes: {paper['hf_upvotes']}",
            })
            continue

        response = tavily_search(f'"{title}" research paper', max_results=_ENRICH_TAVILY_RESULTS_PER_PAPER)
        results = response.get("results", [])
        snippets = [r.get("content", "")[:150] for r in results if r.get("content")]

        enriched.append({
            **paper,
            "web_mentions": len(results),
            "web_context": " | ".join(snippets),
        })

    # Preserve any papers beyond the cap without enrichment
    for paper in state.raw_papers[_ENRICH_MAX_PAPERS:]:
        enriched.append({**paper, "web_mentions": 0, "web_context": ""})

    discussed = sum(1 for p in enriched if p.get("web_mentions", 0) > 0)
    logger.info(f"Enriched {len(enriched)} papers — {discussed} have web presence")
    return {"raw_papers": enriched}


def enrich_concept_node(state: ResearchState) -> dict:
    """
    Find a beginner-friendly resource URL for each concept via Tavily.
    Covers both concepts_of_the_day and foundational_concepts.
    """
    if not state.briefing:
        return {}
    if not os.getenv("TAVILY_API_KEY"):
        return {}

    all_concepts = list(state.briefing.concepts_of_the_day) + list(state.briefing.foundational_concepts)
    if not all_concepts:
        return {}

    preferred_domains = ("distill.pub", "colah.github", "lilianweng", "arxiv", "explained.ai")
    briefing = state.briefing

    for concept in all_concepts:
        if concept.learn_more_url:
            continue  # already set

        response = tavily_search(
            f"{concept.name} explained machine learning beginner guide",
            max_results=_ENRICH_CONCEPT_RESULTS,
        )
        results = response.get("results", [])
        url = ""
        for r in results:
            r_url = r.get("url", "")
            if any(d in r_url for d in preferred_domains):
                url = r_url
                break
        if not url and results:
            url = results[0].get("url", "")

        if url:
            concept.learn_more_url = url
            logger.info(f"Concept '{concept.name}' learn_more_url: {url}")

    return {"briefing": briefing}


def research_agent_node(state: ResearchState) -> dict:
    """
    ReAct research node: the LLM decides which tools to call.

    Pattern (ReAct): The LLM is given tool schemas and decides autonomously
    which tools to call with what arguments. LangGraph's ToolNode executes
    the actual tool calls, then loops back here until the LLM stops calling tools.
    """
    try:
        llm = _build_llm(_PRIMARY_MODEL, _FALLBACK_MODEL)
        # tool_choice="auto" explicitly tells Groq to use JSON function-call format,
        # preventing the model from falling back to <function=...> XML-style generation.
        llm_with_tools = llm.bind_tools(ALL_TOOLS, tool_choice="auto")
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
                    f"AI/ML developments. Use search_hf_papers for broad topic coverage "
                    f"and search_arxiv for specific topics."
                )
            ),
        ]

    # Window messages to stay within TPM limits: always keep system + human,
    # then only the most recent exchanges to avoid 413s on long ReAct loops.
    windowed = _window_messages(messages, keep_recent=6)

    try:
        response = llm_with_tools.invoke(windowed)
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
            new_ids = embed_and_store(papers=new_papers, date=state.date)

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
            k=_RAG_K,
            filter_metadata={"date": state.date},
        )

        context_chunks = [doc.page_content for doc in docs]
        logger.info(f"Retrieved {len(context_chunks)} context chunks for synthesis")
        return {"retrieved_context": context_chunks}

    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        # Fall back to raw paper content
        fallback = []
        for p in state.raw_papers[:10]:
            fallback.append(f"PAPER: {p.get('title', '')}\n{p.get('abstract', '')}")

        return {
            "retrieved_context": fallback,
            "errors": state.errors + [f"RAG retrieval failed, using raw content: {e}"],
        }


def synthesize_node(state: ResearchState) -> dict:
    """
    Generate structured DailyBriefing from retrieved context.

    Note: HuggingFace Inference API does not reliably support function-calling,
    so we cannot use .with_structured_output(). Instead, we prompt the LLM to
    return JSON directly (see SYNTHESIS_USER_TEMPLATE for the embedded schema),
    then strip markdown fences and call DailyBriefing.model_validate() ourselves.
    Falls back to _build_fallback_briefing() if parsing fails.
    """
    try:
        llm = _build_llm(_PRIMARY_MODEL, _FALLBACK_MODEL)
    except Exception as e:
        logger.error(f"Failed to build synthesis LLM: {e}")
        return {
            "errors": state.errors + [f"Synthesis LLM failed: {e}"],
            "briefing": _build_fallback_briefing(state),
        }

    context = "\n\n---\n\n".join(state.retrieved_context) if state.retrieved_context else "No context retrieved"

    # Annotate context chunks with relevance signals so LLM can rank by discussion.
    # For HF Papers, include upvotes directly; for arXiv, use web_mentions from Tavily.
    papers_with_mentions = []
    for p in state.raw_papers:
        if not p.get("title"):
            continue
        upvotes = p.get("hf_upvotes")
        if upvotes is not None:
            papers_with_mentions.append(f"[hf_upvotes={upvotes}] {p['title']}")
        else:
            papers_with_mentions.append(f"[web_mentions={p.get('web_mentions', 0)}] {p['title']}")
    if papers_with_mentions:
        context = "Paper web presence scores:\n" + "\n".join(papers_with_mentions) + "\n\n---\n\n" + context

    concepts_list = "\n".join(f"- {c}" for c in CORE_DS_CONCEPTS)
    foundational_list = "\n".join(f"- {c}" for c in FOUNDATIONAL_DS_CONCEPTS)

    user_prompt = SYNTHESIS_USER_TEMPLATE.format(
        date=state.date,
        context=context[:_SYNTHESIS_CONTEXT_CHARS],
        total_papers=len(state.raw_papers),
        concepts=concepts_list,
        foundational_concepts=foundational_list,
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

        # Fix invalid JSON escape sequences (e.g. \s, \e) that LLMs sometimes emit
        json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

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


_LATEX_SYMBOLS = {
    r"\sigma": "σ", r"\theta": "θ", r"\alpha": "α", r"\beta": "β",
    r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε", r"\lambda": "λ",
    r"\mu": "μ", r"\pi": "π", r"\tau": "τ", r"\phi": "φ", r"\psi": "ψ",
    r"\omega": "ω", r"\nabla": "∇", r"\partial": "∂", r"\infty": "∞",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\rightarrow": "→", r"\leftarrow": "←", r"\cdot": "·",
    r"\times": "×", r"\in": "∈", r"\sum": "Σ", r"\prod": "Π",
    r"\log": "log", r"\exp": "exp", r"\max": "max", r"\min": "min",
}


def _strip_latex(text: Any) -> str:
    """
    Remove LaTeX math delimiters and replace common commands with unicode.

    Safety-net post-processor: the prompt forbids LaTeX, but if the LLM
    produces it anyway this ensures the markdown output stays readable.
    """
    if not isinstance(text, str):
        return text

    # Replace display math blocks first, then inline
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", r"\1", text)
    # \\( ... \\) and \\[ ... \\] delimiters
    text = re.sub(r"\\\\\((.+?)\\\\\)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\\\[(.+?)\\\\\]", r"\1", text, flags=re.DOTALL)

    # Replace known LaTeX commands with unicode
    for cmd, symbol in _LATEX_SYMBOLS.items():
        text = text.replace(cmd, symbol)

    # Strip remaining \text{...} wrappers, keeping content
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    # Remove stray braces left from grouping (e.g. _{...} → _...)
    text = re.sub(r"\{([^{}]*)\}", r"\1", text)

    return text


def render_briefing_markdown(briefing: DailyBriefing) -> str:
    """Render a DailyBriefing to a markdown string."""
    lines = [
        "# Daily AI Research Briefing",
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
                f"### {_strip_latex(paper.title)}",
                f"*{', '.join(paper.authors)}*",
                "",
                f"**Summary:** {_strip_latex(paper.plain_english_summary)}",
                "",
                f"**Methods:** {_strip_latex(paper.methods)}",
                "",
                f"**Key contribution:** {_strip_latex(paper.key_contribution)}",
                "",
                f"**Why it matters:** {_strip_latex(paper.significance)}",
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
                f"### {_strip_latex(paper.title)}",
                f"*{', '.join(paper.authors)}*",
                "",
                f"**Summary:** {_strip_latex(paper.plain_english_summary)}",
                "",
                f"**Methods:** {_strip_latex(paper.methods)}",
                "",
                f"**Why it matters:** {_strip_latex(paper.significance)}",
                "",
                f"> [Read paper]({paper.url})" if paper.url else "",
                "",
                "---",
                "",
            ]

    lines += ["# Emerging Themes", "", _strip_latex(briefing.emerging_themes), "", "---", ""]

    if briefing.concepts_of_the_day:
        lines += ["# Concepts of the Day", ""]
        for c in briefing.concepts_of_the_day:
            lines += [
                f"## {c.name}",
                "",
                _strip_latex(c.plain_english),
                "",
                f"**Example:** {_strip_latex(c.example)}",
                "",
                f"**Why it matters:** {_strip_latex(c.why_it_matters)}",
                "",
                f"**In today's research:** {_strip_latex(c.connected_to_today)}",
                "",
                f"> [Learn more]({c.learn_more_url})" if c.learn_more_url else "",
                "",
                "---",
                "",
            ]

    if briefing.foundational_concepts:
        lines += ["# Foundational Concepts", ""]
        for c in briefing.foundational_concepts:
            lines += [
                f"## {c.name}",
                "",
                c.plain_english,
                "",
                f"**Example:** {c.example}",
                "",
                f"**Why it matters:** {c.why_it_matters}",
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

    return "\n".join(lines)


def save_report_node(state: ResearchState) -> dict:
    """
    Render DailyBriefing Pydantic model to markdown and save to OUTPUT_DIR.
    """
    if not state.briefing:
        logger.error("No briefing to save")
        return {}

    markdown = render_briefing_markdown(state.briefing)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = f"{OUTPUT_DIR}/{state.briefing.date}.md"
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
