# Daily AI Research Briefing Agent

An autonomous AI agent that researches and synthesizes daily AI/ML developments using production-grade patterns: **ReAct tool calling**, **RAG**, **structured outputs**, and **MCP server** exposure. Runs entirely on free tiers using Groq + Tavily.

## Quick Start

```bash
git clone <your-repo-url> && cd banyan
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY and TAVILY_API_KEY
python main.py         # briefing saved to output/YYYY-MM-DD.md
```

## What It Produces

A daily markdown briefing with:

- **Most Discussed** — papers ranked by HuggingFace upvotes and web presence
- **Notable Papers** — other significant research with methods + significance
- **Web & Industry News** — model releases, benchmarks, announcements
- **Emerging Themes** — trend analysis across all sources
- **Concepts of the Day** — 1-3 foundational DS/ML concepts drawn from today's papers, explained from first principles with resource links. Selected from a curated list of 252 concepts to ensure consistent quality and broad coverage over time.

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **Agent orchestration** | LangGraph | ReAct loop with ToolNode, typed state |
| **Primary LLM** | Groq (llama-3.3-70b-versatile) | Free tier, fast inference, tool calling |
| **Fallback LLM** | Groq (llama-3.1-8b-instant) | Automatic failover, higher daily quota |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Local, CPU-fast, zero cost |
| **Vector store** | ChromaDB | Embedded, persistent, metadata filtering |
| **Paper source (primary)** | HuggingFace Papers API | Semantic search + community upvote signals |
| **Paper source (fallback)** | arXiv Python library | Always free, keyword search |
| **Web search + ranking** | Tavily | ~25 calls/run, 1,000 free/month |
| **Structured output** | Pydantic v2 + JSON parsing | Type-safe LLM responses |
| **Tool protocol** | MCP SDK | Claude Desktop + Cursor integration |
| **Scheduling** | GitHub Actions | Free for public repos |

## Architecture

```
START
  │
  ▼
fetch_ai_news          3 Tavily queries (model releases, benchmarks, industry news)
  │                    → state.web_news
  ▼
research_agent ◄─────┐  Groq llama-3.3-70b decides which tools to call
  │                  │  → appends AIMessage to state.messages
  │                  │
  ├─[has tool_calls]─┤
  │                  │
  ▼                  │
tools (ToolNode)     │  Executes tool calls chosen by the LLM:
  │                  │    • search_hf_papers → semantic search, returns upvotes + GitHub links
  │                  │    • search_arxiv     → keyword search, sorted by submission date
  │                  └─   • web_search      → plain text snippets (Tavily)
  │                        loops back until LLM stops calling tools
  ├─[no tool_calls]
  │
  ▼
collect_results        Scans ALL ToolMessages in state.messages
  │                    Extracts paper dicts (JSON with "abstract" key)
  │                    → state.raw_papers
  ▼
enrich_papers          Per-paper enrichment:
  │                    • HF Papers: uses hf_upvotes directly (no extra API call)
  │                    • arXiv papers: Tavily search to score web presence
  │                    Papers with upvotes/mentions → flagged as most_discussed
  ▼
deduplicate_embed      Checks ChromaDB for already-seen IDs (incremental runs)
  │                    Embeds only new papers via local sentence-transformers
  │                    → persisted to ChromaDB
  ▼
retrieve_context       Semantic search in ChromaDB (top-20 chunks)
  │                    Query: "most important AI/ML developments for {date}"
  │                    → state.retrieved_context
  ▼
synthesize             Groq llama-3.3-70b generates structured JSON briefing
  │                    Input: retrieved chunks + web news + paper social scores
  │                    + 252-concept list for concept selection
  │                    Output: DailyBriefing validated by Pydantic
  │                    Post-processing: LaTeX sanitizer strips any math delimiters
  ▼
enrich_concept         Tavily → beginner resource URL for each concept of the day
  │                    Prefers distill.pub, colah.github, lilianweng, arxiv
  ▼
save_report            Renders DailyBriefing → output/YYYY-MM-DD.md
  │
END
```

**Groq is called twice per run** — once in the ReAct loop (tool calling) and once for synthesis. Message history is windowed before each LLM call to stay within the 12K TPM free-tier limit.

**Tavily is called ~25 times per run** across three stages (see [Tavily Usage](#tavily-usage) below). HF Papers results skip the per-paper Tavily enrichment since they already carry upvote counts.

**Embeddings never leave your machine** — sentence-transformers runs locally on CPU.

## Tavily Usage

Tavily is used in three distinct places per run (~25 API calls total):

| Stage | Queries | Purpose |
|-------|---------|---------|
| `fetch_ai_news_node` | 3 curated | Model releases, benchmarks, industry news |
| `enrich_papers_node` | 1 per arXiv paper (max 20) | Score papers by web presence — skipped for HF Papers |
| `web_search` tool | LLM-driven, ad-hoc | Research agent's open-ended searches |
| `enrich_concept_node` | 1 per concept (1-3) | Find beginner resources for each concept of the day |

## Free Tier Cost Breakdown

| Service | Free Tier | Usage per run |
|---------|-----------|---------------|
| Groq API | 1,000 RPD (70B), 14,400 RPD (8B) | ~5 LLM calls |
| sentence-transformers | Local, always free | Embeddings |
| HuggingFace Papers API | Always free, no key needed | Primary paper source |
| arXiv API | Always free | Fallback paper source |
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

### 2. Semantic Paper Search via HuggingFace Papers API

The primary paper source uses HF's hybrid semantic search, which returns community upvotes alongside paper metadata. This replaces keyword-only arXiv search for most queries.

```python
# agent/tools.py
response = requests.get(
    "https://huggingface.co/api/papers",
    params={"q": query, "limit": max_results},
)
# Returns: title, abstract, upvotes, github_repo, project_page
```

### 3. Multi-Stage Tavily Integration

Rather than a single web search, Tavily is used strategically at three pipeline stages — pre-research news gathering, post-collect paper scoring (arXiv papers only), and post-synthesis concept enrichment.

```python
# enrich_papers_node: HF Papers use upvotes; arXiv papers use Tavily
if paper.get("hf_upvotes") is not None:
    paper["web_mentions"] = 1 if paper["hf_upvotes"] > 0 else 0
else:
    response = tavily_search(f'"{title}" research paper', max_results=3)
    paper["web_mentions"] = len(response.get("results", []))
```

### 4. RAG Pipeline

Two-phase RAG — ingest then retrieve. Deduplication across runs keeps ChromaDB clean.

```python
# Phase 1: embed and persist (only new content)
seen_ids = get_seen_ids()
new_papers = [p for p in papers if p["id"] not in seen_ids]
embed_and_store(papers=new_papers, date=today)

# Phase 2: semantic retrieval for synthesis
docs = retrieve_relevant_context(query=synthesis_query, k=20)
```

### 5. Structured Outputs via JSON Prompt

Groq's free tier doesn't guarantee JSON schema mode for all models, so structured outputs are achieved by prompting for raw JSON, sanitizing invalid escape sequences, and validating with Pydantic.

```python
# agent/nodes.py
response = llm.invoke(messages)
json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response.content)
json_str = json_match.group(1) if json_match else response.content.strip()
briefing = DailyBriefing.model_validate(json.loads(json_str))
```

### 6. LaTeX Sanitization

The synthesis prompt forbids LaTeX, but a post-processing sanitizer strips any that slips through, replacing commands with unicode and removing math delimiters.

```python
# agent/nodes.py — applied to every text field before writing markdown
def _strip_latex(text: str) -> str:
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", r"\1", text)
    for cmd, symbol in _LATEX_SYMBOLS.items():   # \sigma → σ, \cdot → ·, etc.
        text = text.replace(cmd, symbol)
    return text
```

### 7. Curated Concept Selection

Rather than letting the LLM free-generate concept names (which produces hallucinated or trivially obvious choices), each run selects from a fixed list of 252 curated DS/ML concepts. The LLM picks 1-3 most relevant to the day's papers.

```python
# agent/concepts.py — 252 entries across 17 categories
CORE_DS_CONCEPTS = [
    "Gradient Descent", "KL Divergence", "FlashAttention",
    "Mixture of Experts (MoE)", "Direct Preference Optimization (DPO)",
    # ... 247 more
]
```

### 8. MCP Server

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
## 2026-03-24

> Analyzed **48 papers**

---

# Most Discussed

### Scaling Laws for Reward Model Overoptimization
*Paul F. Christiano, Jan Leike, Tom Brown*

**Summary:** Researchers found that as you optimize against a reward model's score,
performance initially improves but then degrades — the model learns to game the signal.

**Methods:** Empirical study fine-tuning policies with PPO against a fixed reward model
at varying KL budgets. The training objective balances a task reward R against a KL
penalty: J = E[R(y)] - β · KL(π_θ || π_ref).

**Key contribution:** Scaling laws quantifying the relationship between KL divergence
and gold reward as a function of reward model capacity.

**Why it matters:** Foundational result for anyone training LLMs with RLHF — explains
reward hacking and sets theoretical limits on over-optimization.

> [Read paper](https://huggingface.co/papers/2210.10760)

---

# Concepts of the Day

## Direct Preference Optimization (DPO)

DPO trains language models directly on human preference data without needing
a separate reward model or reinforcement learning loop...

**Example:** Given two responses to the same prompt, DPO adjusts model weights
to make the preferred response more likely using a closed-form objective.

**Why it matters:** Simpler, more stable training than PPO-based RLHF with competitive results.

**In today's research:** Three of today's papers use DPO variants for alignment tasks.

> [Learn more](https://arxiv.org/abs/2305.18290)
```

## Project Structure

```
banyan/
├── agent/
│   ├── graph.py      # LangGraph graph: nodes, edges, ReAct loop
│   ├── nodes.py      # All node functions including enrichment + LaTeX sanitizer
│   ├── tools.py      # @tool functions: search_hf_papers, search_arxiv, web_search
│   ├── state.py      # Pydantic ResearchState
│   ├── prompts.py    # All LLM prompts in one place
│   └── concepts.py   # 252 curated DS/ML concepts for concept_of_the_day selection
├── rag/
│   ├── embeddings.py # HuggingFaceEmbeddings (local, free)
│   ├── store.py      # ChromaDB read/write + deduplication
│   └── retriever.py  # Semantic retrieval (similarity search)
├── mcp_server/
│   └── server.py     # MCP server exposing 4 tools
├── schemas/
│   ├── paper.py      # PaperSummary Pydantic model
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

**Groq** (required)
- Go to https://console.groq.com
- Create a free account and generate an API key
- Free tier: 1,000 requests/day on llama-3.3-70b, no credit card needed

**Tavily** (strongly recommended — powers paper ranking + news)
- Go to https://tavily.com
- Sign up for free tier (1,000 searches/month)

HuggingFace Papers API requires no key.

### 2. Local Setup

```bash
git clone <your-repo-url>
cd banyan

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — add GROQ_API_KEY and TAVILY_API_KEY

python main.py
```

### 3. GitHub Actions Deployment

1. Push to GitHub (public repo = free Actions minutes)
2. Go to **Settings → Secrets and variables → Actions**
3. Add secrets:

| Secret | Where to get it |
|--------|----------------|
| `GROQ_API_KEY` | https://console.groq.com |
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
        "GROQ_API_KEY": "gsk_your_key",
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
| `GROQ_API_KEY` | Groq API key **(required)** — free at console.groq.com | — |
| `TAVILY_API_KEY` | Tavily search API key **(strongly recommended)** | — |
| `PRIMARY_MODEL` | Groq model for research + synthesis | `llama-3.3-70b-versatile` |
| `FALLBACK_MODEL` | Groq fallback model | `llama-3.1-8b-instant` |
| `EMBEDDING_MODEL` | Local sentence-transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./chroma_db` |

## Error Resilience

The agent degrades gracefully at every stage:
- If **Tavily** key is missing → skips news fetch, paper enrichment, and concept links (all non-fatal)
- If **HuggingFace Papers API** is unreachable → falls back to arXiv search
- If **arXiv** is unreachable → continues with web search only
- If **primary LLM** fails → automatically retries with fallback model
- If **synthesis JSON** is malformed → falls back to a minimal briefing with raw data
- All non-fatal errors are appended to `state.errors` and shown in the briefing footer

## Disclaimer

This agent fetches public data from HuggingFace Papers, arXiv, and the web. Respect rate limits and API terms of service. Briefings are AI-generated summaries, not expert analysis.
