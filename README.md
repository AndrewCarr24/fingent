# FinGent

A LangGraph QA agent for knowledge bases. You give it a folder of markdown
documents; it builds a hybrid (BM25 + semantic) knowledge base; then you
ask it questions from the CLI or a Chainlit UI.

The differentiated piece is the retrieval tool — built on top of
[dsRAG](https://github.com/D-Star-AI/dsRAG) — which adds:

- **Hybrid retrieval**: BM25 (lexical) + Bedrock-Titan embeddings (semantic),
  fused via Reciprocal Rank Fusion. Fixes the silent `metadata_filter` drop
  in dsRAG's `BasicVectorDB.search`.
- **Smart α**: an LLM picks the BM25 / semantic weighting per-question, so
  the retriever adapts to question shape (specific dollar figures and named
  line items push toward BM25; conceptual / abstract questions push toward
  semantic).
- **Auto-query expansion**: dsRAG's auto-query helper, but routed through
  DeepSeek instead of the legacy Claude Sonnet 3.5 wiring.
- **Cross-call chunk dedup** within a conversation, optional.

The agent itself is a small LangGraph: a router classifies intent (RAG-query
vs simple greeting), a ReAct agent uses the retrieval tool to answer, and a
finalize node forces a text answer if the tool budget runs out. Bedrock
prefix caching is wired in by default to keep cost down.

---

## Quick start

```bash
git clone <your-fork>
cd fingent
uv sync                      # creates .venv with all deps
cp .env.example .env         # then fill in DEEPSEEK_API_KEY + AWS creds

# 1. Build the KB from a folder of markdown
fingent-build-kb ./my_docs --kb-id my_kb --store-dir ./fingent_store

# 2. Ask questions
fingent-chat --kb-id my_kb --store-dir ./fingent_store \
  "What did the company say about Q3 revenue?"
```

## What you need

- **Python ≥ 3.11** (matches the dsRAG / LangChain stack).
- **DeepSeek API key** (for the orchestrator, auto-query, and smart-α).
- **AWS credentials with Bedrock access** (for Titan v2 embeddings used at
  build and retrieval time). If you set `ORCHESTRATOR_PROVIDER=bedrock`,
  also for the orchestrator model.

## Workflow

1. **Markdown → KB.** `fingent-build-kb <markdown-dir>` ingests every `*.md`
   under the given directory as one dsRAG document. The filename stem
   becomes the `doc_id`, which the agent uses as a metadata filter to
   scope retrieval to a single source. Re-runs are idempotent — already-
   indexed docs are skipped.

2. **Run the agent.** Either:
   - **CLI**: `fingent-chat --kb-id my_kb "<question>"` for one-shot
     questions, or `fingent-chat --kb-id my_kb` (no question) for an
     interactive REPL.
   - **Chainlit UI**: `chainlit run apps/chainlit_app.py` after setting
     `FINGENT_KB_ID` and `FINGENT_STORE_DIR` env vars.
   - **Library**: `from fingent import Agent` (see below).

## Library usage

```python
from fingent import Agent

agent = Agent(kb_id="my_kb", store_dir="./fingent_store")
async for chunk in agent.ask_stream("What did the company say about Q3 revenue?"):
    print(chunk, end="", flush=True)
```

You can also pass extra tools to the agent:

```python
from fingent import Agent
from fingent.retrieval.tool import search_kb

agent = Agent(
    kb_id="my_kb",
    store_dir="./fingent_store",
    extra_tools=[my_custom_tool],   # search_kb is added by default
)
```

## Caching

Bedrock prefix caching is on by default when `ORCHESTRATOR_PROVIDER=bedrock`:
the agent emits `cachePoint` content blocks on the system message and the
last message of each ReAct turn, so Bedrock caches the prefix at ~10% of
input-token price. DeepSeek applies prefix caching server-side
automatically — no client-side markers needed.

The graph also includes a `cache_check_node` placeholder. It's a no-op in
v0.1; the hook is reserved for a future answer-cache implementation.

## Tunable retrieval

Set these env vars to override retrieval defaults (see `.env.example`):

| Var | Default | Effect |
|---|---|---|
| `HYBRID_BM25` | `true` | BM25 + semantic vs semantic-only |
| `RRF_ALPHA` | `smart` | Fixed α (0..1) or `smart` (per-question) |
| `RETRIEVAL_TOP_K` | `200` | Candidates per retriever before fusion |

## Tests

```bash
uv run pytest                # unit tests: imports, graph compile, pricing math
uv run pytest -m e2e         # end-to-end against a real KB (opt-in)
```

The e2e test in `tests/test_smoke.py` is skipped unless the following are set:

```bash
export DEEPSEEK_API_KEY=...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
export FINGENT_TEST_KB_ID=my_kb
export FINGENT_TEST_STORE_DIR=/path/to/fingent_store
# Optional — defaults target a known fact in MGIC's 2025 10-K:
export FINGENT_TEST_QUESTION='What was MGIC primary IIF at year-end 2025?'
export FINGENT_TEST_EXPECT='303.1'
```

## Layout

```
src/fingent/
├── agent.py              # Agent class — public API
├── config.py             # pydantic Settings
├── orchestrator/         # LangGraph: router → agent → tool → finalize
├── retrieval/            # dsRAG-based KB tool with smart-α + hybrid + auto-query
├── models/               # Bedrock + DeepSeek model abstraction; usage tracking
└── build_kb/             # markdown → KB pipeline
apps/
├── cli.py                # interactive REPL / one-shot
└── chainlit_app.py       # Chainlit UI
```

## Acknowledgements

The retrieval layer is built on [dsRAG](https://github.com/D-Star-AI/dsRAG)
by D-Star AI. FinGent contributes the hybrid + smart-α + auto-query layer
on top, but the core chunking, AutoContext, and storage are dsRAG's.
