# Daily AI Research Briefing Agent

An autonomous AI agent that researches and synthesizes daily AI/ML developments using production-grade patterns: **ReAct tool calling**, **RAG**, **structured outputs**, and **MCP server** exposure. Runs entirely on free tiers using HuggingFace + Tavily.

## Quick Start

```bash
git clone <your-repo-url> && cd banyan
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # add HUGGINGFACEHUB_API_TOKEN and TAVILY_API_KEY
python main.py         # briefing saved to output/YYYY-MM-DD.md
```

## What It Produces

A daily markdown briefing with:

- **Most Discussed** — papers ranked by actual web presence (Tavily-scored)
- **Notable Papers** — other significant research with methods + significance
- **Web & Industry News** — model releases, benchmarks, announcements
- **Emerging Themes** — trend analysis across all sources
- **Concept of the Day** — a foundational DS/ML concept drawn from today's papers, explained from first principles with a resource link

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **Agent orchestration** | LangGraph | ReAct loop with ToolNode, typed state |
| **Primary LLM** | HuggingFace Inference API (Qwen2.5-7B-Instruct) | Free tier, chat completions |
| **Fallback LLM** | HuggingFace Inference API (Phi-3.5-mini-instruct) | Automatic failover |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Local, CPU-fast, zero cost |
| **Vector store** | ChromaDB | Embedded, persistent, metadata filtering |
| **Paper source** | arXiv Python library | Always free |
| **Web search + ranking** | Tavily | ~25 calls/run, 1,000 free/month |
| **Structured output** | Pydantic v2 + JSON parsing | Type-safe LLM responses |
| **Tool protocol** | MCP SDK | Claude Desktop + Cursor integration |
| **Scheduling** | GitHub Actions | Free for public repos |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          LangGraph Agent Graph                            │
│                                                                           │
│  START                                                                    │
│    │                                                                      │
│    ▼                                                                      │
│  ┌─────────────────┐                                                      │
│  │  fetch_ai_news  │  3 curated Tavily queries (releases, benchmarks,     │
│  │                 │  industry news) → stored in state.web_news           │
│  └─────────────────┘                                                      │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────────┐  tool_calls?  ┌──────────────────────────────────┐  │
│  │ research_agent  │ ── YES ──────▶│            ToolNode              │  │
│  │  (ReAct loop)   │ ◀────────────│  search_arxiv                    │  │
│  │  Qwen2.5-7B     │              │  web_search (Tavily)             │  │
│  └─────────────────┘              └──────────────────────────────────┘  │
│         │ NO (done)                                                       │
│         ▼                                                                 │
│  ┌─────────────────┐                                                      │
│  │ collect_results │  Parse papers from ToolMessage objects               │
│  └─────────────────┘                                                      │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────────┐                                                      │
│  │ enrich_papers   │  Tavily search per paper title → web_mentions score  │
│  │                 │  Papers with web_mentions > 0 → most_discussed       │
│  └─────────────────┘                                                      │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────────┐            ┌──────────────────────────────────────┐  │
│  │deduplicate_embed│ ─ store ──▶│             ChromaDB                 │  │
│  └─────────────────┘           │  all-MiniLM-L6-v2 embeds             │  │
│         │                      │  persistent local storage             │  │
│         ▼                      └──────────────────────────────────────┘  │
│  ┌─────────────────┐                          ▲                           │
│  │retrieve_context │ ── semantic search ───────┘                          │
│  └─────────────────┘    (top-k relevant chunks)                           │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────────┐                                                      │
│  │    synthesize   │  JSON prompt → DailyBriefing (Pydantic validated)    │
│  └─────────────────┘                                                      │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────────┐                                                      │
│  │ enrich_concept  │  Tavily search → beginner resource URL for concept   │
│  └─────────────────┘                                                      │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────────┐                                                      │
│  │   save_report   │  Render to markdown → output/YYYY-MM-DD.md           │
│  └─────────────────┘                                                      │
│         │                                                                 │
│        END                                                                │
└──────────────────────────────────────────────────────────────────────────┘
```

## Tavily Usage

Tavily is used in three distinct places per run (~25 API calls total):

| Stage | Queries | Purpose |
|-------|---------|---------|
| `fetch_ai_news_node` | 3 curated | Model releases, benchmarks, industry news |
| `enrich_papers_node` | 1 per paper (max 20) | Score papers by web presence |
| `web_search` tool | LLM-driven, ad-hoc | Research agent's open-ended searches |
| `enrich_concept_node` | 1 | Find a beginner resource for concept of the day |

This means `most_discussed` is determined by actual online discussion, not LLM guesswork.

## Free Tier Cost Breakdown

| Service | Free Tier | Usage per run |
|---------|-----------|---------------|
| HuggingFace Inference API | Free (rate limited) | ~5 LLM calls |
| sentence-transformers | Local, always free | Embeddings |
| arXiv API | Always free | Paper data |
| Tavily | 1,000 searches/month | ~25 calls |
| ChromaDB | Local, always free | Vector store |
| GitHub Actions | Free for public repos | Daily scheduling |

**Total cost: $0**

## AI Engineering Patterns Demonstrated

### 1. ReAct Tool Calling with ToolNode

The LLM is given tool schemas and autonomously decides when and how to call them. LangGraph's `ToolNode` handles execution and routes results back to the LLM. The loop continues until the LLM determines it has enough information.

```python
# agent/graph.py — conditional edge creates the ReAct loop
graph.add_conditional_edges(
    "research_agent",
    should_continue,
    {"tools": "tools", "process": "collect_results"},
)
graph.add_edge("tools", "research_agent")  # loop back
```

### 2. Multi-Stage Tavily Integration

Rather than a single web search, Tavily is used strategically at three pipeline stages — pre-research news gathering, post-collect paper scoring, and post-synthesis concept enrichment.

```python
# enrich_papers_node: score every paper by web presence
response = _tavily_search(f'"{title}" research paper', max_results=3)
paper["web_mentions"] = len(response.get("results", []))
```

### 3. RAG Pipeline

Two-phase RAG — ingest then retrieve. Deduplication across runs keeps ChromaDB clean.

```python
# Phase 1: embed and persist (only new content)
seen_ids = get_seen_ids()
new_papers = [p for p in papers if p["id"] not in seen_ids]
embed_and_store(papers=new_papers, date=today)

# Phase 2: semantic retrieval for synthesis
docs = retrieve_relevant_context(query=synthesis_query, k=20)
```

### 4. Structured Outputs via JSON Prompt

HuggingFace's free API doesn't support JSON schema mode, so structured outputs are achieved by prompting for raw JSON and validating with Pydantic.

```python
# agent/nodes.py
response = llm.invoke(messages)
json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response.content)
data = json.loads(json_match.group(1) if json_match else response.content)
briefing = DailyBriefing.model_validate(data)
```

### 5. MCP Server

The agent is exposed as an MCP server so it can be called from Claude Desktop, Cursor, or any MCP-compatible client.

```python
# mcp_server/server.py
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "run_daily_briefing": ...
    elif name == "get_latest_briefing": ...
    elif name == "search_past_briefings": ...
    elif name == "get_trending_topics": ...
```

## Sample Output

```markdown
# Daily AI Research Briefing
## 2026-03-18

> Analyzed **34 papers**

---

# Most Discussed

### Scaling Laws for Reward Model Overoptimization
*Paul F. Christiano, Jan Leike, Tom Brown*

**Summary:** Researchers found that as you optimize against a reward model's score,
performance initially improves but then degrades — the model learns to game the signal.

**Methods:** Empirical study fine-tuning policies with PPO against a fixed reward model
at varying KL budgets across model sizes from 3B to 70B parameters.

**Key contribution:** Scaling laws quantifying the relationship between KL divergence
and gold reward as a function of reward model capacity.

**Why it matters:** Foundational result for anyone training LLMs with RLHF — explains
reward hacking and sets theoretical limits on over-optimization.

> [Read paper](https://arxiv.org/abs/2210.10760)

---

# Concept of the Day
## Direct Preference Optimization (DPO)

DPO is an alternative to RLHF that trains language models directly on human preference
data without needing a separate reward model or reinforcement learning loop...

**Example:** Given two responses to the same prompt, DPO adjusts model weights to make
the preferred response more likely, using a clever mathematical reformulation.

**Why it matters:** Simpler, more stable training than PPO-based RLHF with competitive results.

**In today's research:** Three of today's papers use DPO variants for alignment tasks.

> [Learn more](https://arxiv.org/abs/2305.18290)
```

## Project Structure

```
banyan/
├── agent/
│   ├── graph.py      # LangGraph graph: nodes, edges, ReAct loop
│   ├── nodes.py      # All node functions including Tavily enrichment
│   ├── tools.py      # @tool functions: search_arxiv, web_search
│   ├── state.py      # Pydantic ResearchState (includes web_news)
│   └── prompts.py    # All LLM prompts in one place
├── rag/
│   ├── embeddings.py # HuggingFaceEmbeddings (local, free)
│   ├── store.py      # ChromaDB read/write + deduplication
│   └── retriever.py  # Semantic retrieval (similarity search)
├── mcp_server/
│   └── server.py     # MCP server exposing 4 tools
├── schemas/
│   ├── paper.py      # ArxivPaper, PaperSummary Pydantic models
│   └── briefing.py   # DailyBriefing, ConceptExplanation schemas
├── output/           # Generated markdown briefings
├── main.py           # CLI entrypoint
├── mcp_main.py       # MCP server entrypoint
├── requirements.txt
├── .env.example
└── .github/
    └── workflows/
        └── daily_briefing.yml  # Runs at 7am UTC, commits output
```

## Setup

### 1. Get API Keys

**HuggingFace** (required)
- Go to https://huggingface.co/settings/tokens
- Create a token with "Read" access
- Set both `HUGGINGFACEHUB_API_TOKEN` and `HF_TOKEN` to the same value

**Tavily** (strongly recommended — powers paper ranking + news)
- Go to https://tavily.com
- Sign up for free tier (1,000 searches/month)

### 2. Local Setup

```bash
git clone <your-repo-url>
cd banyan

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — add HUGGINGFACEHUB_API_TOKEN, HF_TOKEN, and TAVILY_API_KEY

python main.py
```

### 3. GitHub Actions Deployment

1. Push to GitHub (public repo = free Actions minutes)
2. Go to **Settings → Secrets and variables → Actions**
3. Add secrets:

| Secret | Where to get it |
|--------|----------------|
| `HUGGINGFACEHUB_API_TOKEN` | https://huggingface.co/settings/tokens |
| `HF_TOKEN` | Same value as above |
| `TAVILY_API_KEY` | https://tavily.com |

4. The workflow runs at **7am UTC daily** and commits briefings to `output/`

To trigger manually: **Actions → Daily AI Research Briefing → Run workflow**

### 4. Connect to Claude Desktop (MCP)

Add to your Claude Desktop config:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ai-research-briefing": {
      "command": "python",
      "args": ["/absolute/path/to/banyan/mcp_main.py"],
      "env": {
        "HUGGINGFACEHUB_API_TOKEN": "hf_your_token",
        "HF_TOKEN": "hf_your_token",
        "TAVILY_API_KEY": "tvly_your_key"
      }
    }
  }
}
```

Restart Claude Desktop. You can then say:
- *"Run today's AI briefing"*
- *"What was the most discussed AI paper this week?"*
- *"Search past briefings for diffusion models"*
- *"What topics have been trending in AI over the last 7 days?"*

## Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace API token **(required)** | — |
| `HF_TOKEN` | HuggingFace token (new name, set same value) **(required)** | — |
| `TAVILY_API_KEY` | Tavily search API key **(strongly recommended)** | — |
| `PRIMARY_MODEL` | Primary HuggingFace model | `Qwen/Qwen2.5-7B-Instruct` |
| `FALLBACK_MODEL` | Fallback HuggingFace model | `microsoft/Phi-3.5-mini-instruct` |
| `EMBEDDING_MODEL` | Local sentence-transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./chroma_db` |

## Error Resilience

The agent degrades gracefully at every stage:
- If **Tavily** key is missing → skips news fetch, paper enrichment, and concept link (all non-fatal)
- If **arXiv** is unreachable → continues with web search only
- If **primary LLM** fails → automatically retries with fallback model
- If **synthesis JSON** is malformed → falls back to a minimal briefing with raw data
- All non-fatal errors are appended to `state.errors` and shown in the briefing footer

## Disclaimer

This agent fetches public data from arXiv and the web. Respect rate limits and API terms of service. Briefings are AI-generated summaries, not expert analysis.
