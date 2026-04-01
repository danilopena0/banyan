# CLAUDE.md — Project Intelligence

> This file provides context and constraints for AI-assisted development.
> It persists across sessions and compensates for context window limitations.

## Project Overview

**Name:** Daily AI Research Briefing Agent (banyan)
**Type:** AI agent / agentic pipeline
**Status:** active-development
**One-liner:** Autonomous LangGraph agent that fetches HuggingFace Papers + arXiv + web search, embeds them in ChromaDB, and synthesizes a structured daily markdown briefing using Groq LLMs — all free tier.

## Architecture Summary

```
main.py → agent/graph.py (LangGraph compiled graph)
              │
              ├── research_agent_node   [Groq LLM via ChatGroq]
              │     └── ToolNode        [search_hf_papers | search_arxiv | web_search]
              │     (ReAct loop — capped at _MAX_TOOL_ROUNDS=8, exits when LLM stops calling tools)
              │
              ├── collect_tool_results  [parse ToolMessage objects → dicts]
              ├── deduplicate_embed     [filter seen IDs → embed new → ChromaDB]
              ├── retrieve_context      [semantic search top-k chunks]
              ├── synthesize_node       [with_structured_output → DailyBriefing]
              └── save_report_node      [render Pydantic → output/YYYY-MM-DD.md]

MCP server: mcp_main.py → mcp_server/server.py (4 tools for Claude Desktop)
```

### Key Components

| Path | What it does |
|------|-------------|
| `agent/graph.py` | LangGraph graph definition — nodes + edges + ReAct loop |
| `agent/nodes.py` | All node functions; also `_build_llm()` with fallback logic |
| `agent/tools.py` | `@tool` decorated: `search_hf_papers`, `search_arxiv`, `web_search` |
| `agent/state.py` | `ResearchState` Pydantic model — typed state for the graph |
| `agent/prompts.py` | All LLM prompts centralized here |
| `rag/embeddings.py` | `get_embeddings()` — lru_cached HuggingFaceEmbeddings (local CPU) |
| `rag/store.py` | `get_seen_ids()`, `embed_and_store()` — ChromaDB read/write |
| `rag/retriever.py` | `retrieve_relevant_context()`, `retrieve_across_dates()` |
| `schemas/` | Pydantic v2 models: `PaperSummary`, `DailyBriefing`, `ConceptExplanation` |
| `mcp_server/server.py` | MCP server with 4 tools |
| `output/` | Generated markdown briefings (auto-committed by GitHub Actions) |
| `chroma_db/` | ChromaDB persistent storage (gitignored) |

### Important Decisions

1. **Groq for LLMs, HuggingFace for embeddings**: Groq free tier for fast inference; local sentence-transformers for embeddings (zero cost, no latency)
2. **Local embeddings (sentence-transformers)**: Zero cost, no API latency for embeddings
3. **Pydantic state in LangGraph**: `ResearchState` is a Pydantic BaseModel — enables type safety across all nodes
4. **Deduplication before embedding**: `get_seen_ids()` prevents re-embedding content across daily runs — keeps ChromaDB clean
5. **RAG retrieval before synthesis**: Only top-k semantically relevant chunks go to LLM — respects context limits, reduces tokens
6. **Errors are non-fatal**: All errors appended to `state.errors`, included in briefing footer — agent never crashes on partial failures

## Development Principles

### Non-Negotiable Constraints

- **All secrets via env vars**: Never hardcode tokens. `.env` is gitignored. `.env.example` shows shape only.
- **Single responsibility per node**: Each LangGraph node does exactly one thing
- **Error resilience**: Every external call (HF Papers, arXiv, Groq, ChromaDB) wrapped in try/except — append to `state.errors`, continue
- **No re-embedding duplicates**: Always check `get_seen_ids()` before calling `embed_and_store()`
- **Structured outputs**: Synthesis prompts the LLM to return raw JSON, strips markdown fences, then validates with `DailyBriefing.model_validate()` — Groq doesn't reliably support `.with_structured_output()`

### Code Style

- Type hints on all function signatures
- Docstrings on public functions explaining the **pattern** being demonstrated (not just what the code does)
- Max function length ~40 lines — extract helpers if longer
- Prefer early returns over deep nesting
- No bare `except:` — always catch specific exceptions or `Exception as e`

## Common Commands

```bash
# Run locally
python main.py

# Run MCP server
python mcp_main.py

# Install dependencies
pip install -r requirements.txt

# Lint + format (if ruff installed)
ruff format .
ruff check --fix .

# Check imports are clean
python -c "from agent.graph import build_graph; print('OK')"
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | YES | Free at console.groq.com |
| `TAVILY_API_KEY` | No | tavily.com free tier |
| `PRIMARY_MODEL` | No | Default: `llama-3.3-70b-versatile` |
| `FALLBACK_MODEL` | No | Default: `llama-3.1-8b-instant` |
| `EMBEDDING_MODEL` | No | Default: `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | No | Default: `./chroma_db` |

## Testing Strategy

```bash
# No test suite yet — this is a priority to add
# When adding tests, use pytest:
pytest tests/ -v

# Priority test targets:
# - tools.py: mock HF Papers/arXiv/Tavily, verify return shape
# - store.py: embed + retrieve roundtrip
# - nodes.py: mock LLM calls, test state transitions
```

## Known Gotchas

1. **HuggingFace rate limits**: Free tier is ~1 req/min for some models. If synthesis fails, check for 429 errors and consider adding sleep or switching models.
2. **ChromaDB + LangChain**: Use `langchain-chroma` not `langchain_community.vectorstores.Chroma` — different package.
3. **LangGraph state with Pydantic**: `ResearchState` uses `arbitrary_types_allowed = True` because messages list contains LangChain objects.
4. **ToolMessage parsing**: Tool results come back as JSON strings in `ToolMessage.content`. Use `json.loads()` carefully — web_search returns plain text, not JSON.
5. **`should_continue` router**: Checks `msg.tool_calls` — this attribute only exists on `AIMessage`, not all message types. Guard with `hasattr()`. Also enforces `_MAX_TOOL_ROUNDS` cap by counting `ToolMessage`s in state — prevents `GraphRecursionError`.
6. **`tool_choice` on Groq**: Do NOT pass `tool_choice="auto"` — Groq interprets this as "must call a tool every time", which prevents the LLM from ever exiting the ReAct loop naturally. Omit it to let the model decide when to stop.
7. **Embedding model cache**: `get_embeddings()` is `@lru_cache` — it's loaded once per process. Don't call with different model names expecting different instances.

## Agent Toolkit

This project uses the Promptly agent toolkit. Available agents in `.claude/agents/`:

| Agent | When to use |
|-------|-------------|
| `orchestrator` | Planning multi-step changes |
| `explorer` | Understanding how existing code works |
| `architect` | Designing new features before implementation |
| `implementer` | Writing new code from a plan |
| `tester` | Writing tests |
| `reviewer` | Code review before merging |
| `debugger` | Investigating failures |
| `security-reviewer` | Security audit (especially API key handling) |
| `mcp-reviewer` | Reviewing the MCP server implementation |
| `maintainability` | Refactoring existing code |
| `evaluator` | Quality gate on completed work |

Available skills (slash commands): `/fix-ci`, `/review-pr`, `/scaffold`, `/test-coverage`, `/security-scan`, `/scaffold-mcp`, `/update-docs`, `/perf-audit`

## Changelog

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-03-09 | Initial build | Replaced parenting Q&A app with AI research briefing agent |
| 2026-03-09 | Added Promptly agents + skills | AI-assisted development toolkit |
| 2026-03-31 | Fixed ReAct loop infinite recursion | Removed `tool_choice="auto"` (forced tool calls on Groq); added `_MAX_TOOL_ROUNDS=8` cap in `should_continue`; bumped MCP server recursion_limit to 50 |
