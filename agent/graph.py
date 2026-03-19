"""
LangGraph graph definition.

Pattern (ToolNode): LangGraph's ToolNode automatically handles the
routing between the LLM and tool execution. Combined with a conditional
edge, this creates the ReAct loop without manual orchestration code.

Graph flow:
START → fetch_ai_news → research_agent → [tools loop] → collect_results
      → enrich_papers → deduplicate_embed → retrieve_context
      → synthesize → enrich_concept → save → END
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.state import ResearchState
from agent.nodes import (
    fetch_ai_news_node,
    research_agent_node,
    should_continue,
    collect_tool_results,
    enrich_papers_node,
    deduplicate_and_embed_node,
    retrieve_context_node,
    synthesize_node,
    enrich_concept_node,
    save_report_node,
)
from agent.tools import ALL_TOOLS


def build_graph():
    """
    Build and compile the LangGraph research pipeline.

    The graph has two sections:
    1. ReAct loop: research_agent ↔ tools (until LLM has enough data)
    2. Processing pipeline: collect → enrich → dedupe/embed → retrieve → synthesize → enrich_concept → save
    """
    graph = StateGraph(ResearchState)

    # --- Nodes ---

    graph.add_node("fetch_ai_news", fetch_ai_news_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("collect_results", collect_tool_results)
    graph.add_node("enrich_papers", enrich_papers_node)
    graph.add_node("deduplicate_embed", deduplicate_and_embed_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("enrich_concept", enrich_concept_node)
    graph.add_node("save_report", save_report_node)

    # --- Edges ---

    graph.set_entry_point("fetch_ai_news")
    graph.add_edge("fetch_ai_news", "research_agent")

    graph.add_conditional_edges(
        "research_agent",
        should_continue,
        {
            "tools": "tools",
            "process": "collect_results",
        },
    )

    graph.add_edge("tools", "research_agent")
    graph.add_edge("collect_results", "enrich_papers")
    graph.add_edge("enrich_papers", "deduplicate_embed")
    graph.add_edge("deduplicate_embed", "retrieve_context")
    graph.add_edge("retrieve_context", "synthesize")
    graph.add_edge("synthesize", "enrich_concept")
    graph.add_edge("enrich_concept", "save_report")
    graph.add_edge("save_report", END)

    return graph.compile()
