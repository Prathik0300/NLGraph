```
███╗   ██╗██╗      ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
████╗  ██║██║     ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
██╔██╗ ██║██║     ██║  ███╗██████╔╝███████║██████╔╝███████║
██║╚██╗██║██║     ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
```

<div align="center">

**v0.1.0** &nbsp;·&nbsp; Query Decomposition · Multi-Agent Orchestration

*Vector-native &nbsp;·&nbsp; Framework-agnostic &nbsp;·&nbsp; Self-improving*

🔍 Gather &nbsp;&nbsp; 🧠 Analyze &nbsp;&nbsp; 📐 Plan &nbsp;&nbsp; ⚡ Execute &nbsp;&nbsp; ✅ Verify &nbsp;&nbsp; 🔧 Refine &nbsp;&nbsp; 📝 Communicate

</div>

---

NLGraph takes a single complex natural language query, breaks it into semantically coherent phrases, identifies the cognitive work each phrase requires, detects dependencies between those tasks, and coordinates a dynamic team of specialised AI agents to execute the full plan — in parallel where possible, sequentially where required.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Capabilities and Dependency Types](#capabilities-and-dependency-types)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [One-Time Setup](#one-time-setup)
- [Running NLGraph](#running-nlgraph)
- [CLI Commands](#cli-commands)
- [Interactive Plan Editing](#interactive-plan-editing)
- [Configuration](#configuration)
- [Capability Model Routing](#capability-model-routing)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## How It Works

```
Your query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. EMBED  ──  bge-m3 hybrid vector (dense 1024-dim + sparse)   │
│  2. CACHE CHECK  ──  Qdrant cosine similarity ≥ 0.97            │
│                        ↓ miss                                    │
│  3. PHRASE SPLIT  ──  spaCy dep-parse + Benepar + PDTB lexicon  │
│  4. PROJECT  ──  7 capability centroids × K-means prototypes     │
│  5. DETECT DEPS  ──  PDTB lexicon + parse + cross-encoder        │
│  6. IMPLICIT INJECT  ──  implication rules (e.g. verify→execute) │
│  7. BUILD AGENTS  ──  one Agent per capability, domain-labelled  │
│  8. VALIDATE DAG  ──  cycle check, orphan repair, canon fallback │
│                        ↓                                         │
│  9. PREVIEW  ──  show plan, await user confirm / edit            │
│ 10. EXECUTE  ──  topological generations, asyncio.gather()       │
│ 11. AGGREGATE  ──  merge outputs, compute coherence score        │
│ 12. MEMORY  ──  store in Qdrant + SQLite, collect feedback       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Final answer from coordinated agent team
```

---

## Architecture

```
nlgraph/
├── cli/
│   ├── __main__.py          Entry point (python -m cli)
│   ├── interactive.py       Main REPL loop, plan confirm/edit, feedback
│   ├── display.py           Rich-based rendering (tables, panels, spinners)
│   ├── banner.py            Startup banner and system status panel
│   └── theme.py             Colour theme constants
│
├── core/
│   ├── models.py            All Pydantic data models
│   │                          Capability, DependencyType, Agent, IntentMap,
│   │                          ExecutionPreview, ExecutionResult, HybridVector …
│   ├── exceptions.py        Custom exception hierarchy (NLGraphError base)
│   │
│   ├── intent/
│   │   ├── embedding_engine.py    bge-m3 embed + LRU cache (50 k entries)
│   │   ├── phrase_splitter.py     3-layer split: dep-parse / Benepar / PDTB
│   │   ├── capability_projector.py  Multi-prototype cosine + SetFit + cross-enc
│   │   ├── dependency_detector.py   3-layer dep signal cascade
│   │   ├── implicit_detector.py     Implication rule engine
│   │   ├── agent_constructor.py     Build Agent objects with domain-aware labels
│   │   └── dag_validator.py         Cycle detection + topological repair
│   │
│   ├── execution/
│   │   ├── orchestrator.py          End-to-end pipeline coordinator
│   │   ├── agent_executor.py        Single-agent LLM call + blackboard write
│   │   ├── result_aggregator.py     Merge outputs into ExecutionResult
│   │   └── llm_providers/
│   │       └── ollama.py            Ollama HTTP provider + health check
│   │
│   ├── blackboard/
│   │   ├── blackboard.py            In-session FAISS fact store + artifact log
│   │   └── context_builder.py       Assemble per-agent prompt context
│   │
│   └── memory/
│       ├── global_store.py          Qdrant persistent vector memory
│       ├── performance_tracker.py   SQLite execution + quality metrics
│       └── feedback_collector.py    User score → DB + Qdrant
│
├── setup/
│   ├── capability_seeds.py     Positive + contrastive seed phrases (7 caps)
│   ├── dependency_seeds.py     Positive + contrastive seed phrases (4 dep types)
│   └── centroid_builder.py     Embed seeds → K-means prototypes → centroids.npz
│
├── config.py                   Single-source settings (pydantic-settings)
└── pyproject.toml              Project metadata and dependencies
```

---

## Capabilities and Dependency Types

### 7 Capability Primitives

| Capability | Emoji | Meaning | Example phrases |
|---|---|---|---|
| **GATHER** | 🔍 | Retrieve, search, collect, fetch information | "find papers on…", "query the database", "collect the dataset" |
| **ANALYZE** | 🧠 | Understand deeply, synthesize, find patterns | "analyze the results", "identify the root cause", "compare approaches" |
| **PLAN** | 📐 | Design, architect, create a roadmap | "design the system", "outline the steps", "define the strategy" |
| **EXECUTE** | ⚡ | Build, implement, produce, generate output | "implement the algorithm", "write the code", "build the model" |
| **VERIFY** | ✅ | Test, validate, check, ensure correctness | "run the tests", "validate the output", "fact-check" |
| **REFINE** | 🔧 | Improve, polish, optimize, iterate | "refactor", "optimize performance", "edit the draft" |
| **COMMUNICATE** | 📝 | Write up, present, document, explain | "write the report", "explain to stakeholders", "document the API" |

Agents are also domain-labelled. The same capability gets a contextually appropriate role title:

| Capability | software | research | business | data | general |
|---|---|---|---|---|---|
| GATHER | Researcher | Literature Reviewer | Market Analyst | Data Collector | Information Gatherer |
| ANALYZE | Synthesizer | Concept Analyst | Business Analyst | Data Analyst | Analyst |
| PLAN | Architect | Research Planner | Strategist | — | Planner |
| EXECUTE | Coder | Author | Producer | Pipeline Engineer | Builder |
| VERIFY | Tester | Fact Checker | QA Reviewer | — | Verifier |
| REFINE | Refactorer | Editor | Optimizer | — | Refiner |
| COMMUNICATE | Documenter | Writer | Presenter | — | Communicator |

### 4 Dependency Types

| Type | Symbol | Meaning | Example signal |
|---|---|---|---|
| **SEQUENTIAL** | `→` | Strict time ordering; B starts after A finishes | "then", "next", "after that", "subsequently" |
| **PREREQUISITE** | `⟹` | Knowledge dependency; B needs A's understanding to proceed | "based on that", "informed by", "building on", "given what we know" |
| **PARALLEL** | `∥` | Independent; A and B can run concurrently | "at the same time", "simultaneously", "in parallel", "alongside" |
| **FEEDS\_INTO** | `↣` | Data pipeline; A's output is B's direct input | "which feeds into", "as input to", "providing data for", "whose output goes to" |

**Learned Bayesian priors** (used when signals are weak):

| Pair | Default | Confidence |
|---|---|---|
| gather → analyze | feeds\_into | 0.80 |
| analyze → plan | feeds\_into | 0.75 |
| plan → execute | prerequisite | 0.85 |
| execute → verify | sequential | 0.90 |
| verify → refine | feeds\_into | 0.80 |
| refine → communicate | feeds\_into | 0.70 |

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd NLGraph
python3.11 -m venv venv && source venv/bin/activate
pip install -e .

# 2. Download NLP models
python -m spacy download en_core_web_trf
python -c "import benepar; benepar.download('benepar_en3')"

# 3. Start Ollama and pull at least the fallback model
ollama serve &
ollama pull mistral:7b

# 4. Build capability centroids (run once, ~5 min)
python -m setup.centroid_builder

# 5. Launch
python -m cli
```

---

## Installation

**Requirements:** Python 3.11+, [Ollama](https://ollama.ai)

```bash
# Development install (editable)
pip install -e .

# With test/dev dependencies
pip install -e ".[dev]"
```

**Key dependencies installed automatically:**

| Stack | Packages |
|---|---|
| Phrase splitting | `spacy`, `benepar` |
| Embeddings | `FlagEmbedding` (bge-m3), `sentence-transformers`, `torch`, `transformers` |
| Vector DB | `qdrant-client` (embedded, no server needed) |
| In-session search | `faiss-cpu` |
| Classification learning | `setfit` |
| Data models | `pydantic`, `pydantic-settings` |
| Persistence | `sqlmodel` |
| Graph / DAG | `networkx` |
| LLM calls | `httpx` (Ollama HTTP) |
| CLI | `rich`, `prompt_toolkit` |

---

## One-Time Setup

These steps must be completed before the first query.

### 1. spaCy transformer model

```bash
python -m spacy download en_core_web_trf
```

Used for dependency-parse-based phrase splitting. The transformer variant (`en_core_web_trf`) gives the best boundary quality.

### 2. Benepar constituency parser (optional but recommended)

```bash
python -c "import benepar; benepar.download('benepar_en3')"
```

Adds a second phrase-splitting layer via constituency parse (VP/SBAR boundaries). NLGraph degrades gracefully if Benepar is unavailable — spaCy dependency parse is used alone.

> **Note:** Benepar may conflict with `transformers >= 4.36`. If it fails to load, phrase splitting still works via spaCy.

### 3. Ollama and LLM models

Ollama must be running before NLGraph starts:

```bash
ollama serve
```

Pull models for each capability (or at minimum the fallback):

```bash
# Required — fallback for any missing model
ollama pull mistral:7b

# Recommended — used for GATHER / EXECUTE / VERIFY / REFINE
ollama pull qwen2.5:14b

# Recommended — used for ANALYZE / PLAN (reasoning-heavy)
ollama pull deepseek-r1:14b
```

If a capability-specific model is not pulled, NLGraph **automatically falls back to `mistral:7b`** without crashing.

### 4. Build capability centroids

```bash
python -m setup.centroid_builder
```

This is the most important setup step. It:

1. Loads bge-m3 and embeds all seed phrases (~370 seeds across 7 capabilities + 4 dependency types)
2. Runs K-means per capability (auto-selects best K by silhouette score, up to K=5)
3. Applies contrastive repulsion (α = 0.70) to push capability centroids apart
4. Validates pairwise centroid separation — warns if any pair similarity > 0.75
5. Saves `data/centroids.npz` with model metadata (staleness detection on load)

Expected output (all pairs should show **well separated**):

```
GATHER ↔ VERIFY         0.577   well separated
GATHER ↔ ANALYZE        0.550   well separated
PLAN   ↔ COMMUNICATE    0.545   well separated
...
EXECUTE ↔ REFINE        0.421   well separated
✓ All capability centroids are well separated.
```

> **Rebuild when:** you modify `capability_seeds.py`, change the embedding model, or see stale centroid warnings at startup.

---

## Running NLGraph

```bash
# Standard launch
python -m cli

# Or if installed as package
nlgraph
```

On startup, NLGraph checks all subsystems and displays a status panel:

```
╭──────────────────────── System Status ─────────────────────────╮
│  ✓ Ollama (LLM)      ready  [mistral:7b]                       │
│  ● bge-m3 (Embed)    ready  [BAAI/bge-m3]                      │
│  ● Qdrant (Memory)   ready  [embedded]                         │
│  ● Centroids         ready  [capability vectors]               │
│  ● spaCy (NLP)       ready  [phrase splitting]                 │
╰────────────────────────────────────────────────────────────────╯
```

All five must be ready before queries are accepted.

---

## CLI Commands

### Slash commands (type at the `NLGraph ›` prompt)

| Command | Description |
|---|---|
| `/help` | Show all available commands |
| `/quit` · `/exit` · `/q` | Exit NLGraph |
| `/clear` | Clear the screen and re-render the banner |
| `/history` | Show your last 10 queries (most recent first) |
| `/stats` | Display system metrics: total queries, average quality score, cache hit rate, intent library size, average latency, feedback count |
| `/internals on` | Show internal details — capability projection scores, dependency signals, phrase assignments |
| `/internals off` | Hide internals (clean output mode) |
| `/setup` | Remind you how to rebuild the capability centroids |

### Query flow

After typing a query and pressing Enter, NLGraph runs the full decomposition pipeline and presents an execution plan:

```
NLGraph › research security concepts and explain them clearly

  [1] Decomposing query…
      ✓ Decomposed into 3 agents   2340ms · vector

  ┏━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
  ┃ # ┃ Role               ┃ Capability     ┃ Confidence ┃ Depends on┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
  │ 1 │ Literature Reviewer│ 🔍 GATHER      │ 0.78       │ —         │
  │ 2 │ Concept Analyst    │ 🧠 ANALYZE     │ 0.64       │ 1         │
  │ 3 │ Writer             │ 📝 COMMUNICATE │ 0.71       │ 2         │
  └───┴────────────────────┴────────────────┴────────────┴───────────┘

  Confirm plan?
    y / Enter  — run it
    e          — edit (remove/add agents)
    n          — cancel
```

---

## Interactive Plan Editing

Type `e` at the confirmation prompt to modify the plan before execution.

### Remove agents

Enter the agent numbers to remove, comma-separated:

```
  remove › 2, 3
  − Removed: Concept Analyst (analyze)
  − Removed: Writer (communicate)
```

Press Enter without typing anything to keep all existing agents.

### Add capabilities

Enter capability names to add, comma-separated or space-separated (case-insensitive):

```
  Available: analyze  communicate  execute  gather  plan  refine  verify

  add    › gather, analyze, verify, communicate
  + Added: 🔍 GATHER agent
  + Added: 🧠 ANALYZE agent
  + Added: ✅ VERIFY agent
  + Added: 📝 COMMUNICATE agent
```

Press Enter without typing anything to skip adding agents.

Added agents are automatically chained sequentially (each depends on the previous one). The execution plan is rebuilt before running.

### Feedback

After execution, NLGraph asks for a quality score:

```
  How did this go? [1–5]  or press Enter to skip:
```

Scores below 3 prompt an optional correction description. Feedback is stored in SQLite and used to recalibrate per-capability activation thresholds over time.

---

## Configuration

All settings live in `config.py` and can be overridden via environment variables or a `.env` file.

### Core thresholds

| Setting | Default | Description |
|---|---|---|
| `ACTIVATION_THRESHOLD` | `0.55` | Minimum cosine similarity for a capability to activate |
| `AMBIGUITY_THRESHOLD` | `0.08` | Score gap below which cross-encoder is called |
| `CACHE_HIT_THRESHOLD` | `0.97` | Dense cosine similarity to reuse a cached plan |
| `DEP_MIN_CONFIDENCE` | `0.55` | Below this, dependency defaults to SEQUENTIAL |
| `CONTRASTIVE_ALPHA` | `0.70` | Centroid repulsion strength during build |
| `PROTOTYPE_K` | `3` | K-means clusters per capability |

### Dependency signal weights

| Setting | Default | Layer |
|---|---|---|
| `DEP_LEXICAL_WEIGHT` | `0.45` | PDTB connective lexicon |
| `DEP_PARSE_WEIGHT` | `0.30` | spaCy dependency grammar |
| `DEP_CROSSENC_WEIGHT` | `0.25` | Cross-encoder classifier |

### Memory and caching

| Setting | Default | Description |
|---|---|---|
| `EMBED_CACHE_MAX_SIZE` | `50000` | Max embedding cache entries (LRU) |
| `EMBED_CACHE_TTL_SECONDS` | `86400` | Cache TTL (24 hours) |
| `AGENT_OUTPUT_MIN_QUALITY` | `4.0` | Min feedback score to store in Qdrant |
| `FAISS_MAX_FACTS` | `10000` | Max in-session blackboard facts |
| `CONTEXT_TOP_K_FACTS` | `8` | Facts retrieved per agent context build |
| `CONTEXT_TOKEN_BUDGET` | `3000` | Max tokens in agent prompt context |

### Execution

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_LLM_MODEL` | `mistral:7b` | Default LLM |
| `OLLAMA_FALLBACK_MODEL` | `mistral:7b` | Used when capability model is not pulled |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Per-agent generation timeout |
| `OLLAMA_AUTO_PULL` | `True` | Auto-pull missing models |
| `MAX_DAG_REPAIR_ATTEMPTS` | `3` | Attempts before canonical fallback |

---

## Capability Model Routing

Each capability is routed to a specialised model by default. If a model is not available, the executor falls back to `OLLAMA_FALLBACK_MODEL` automatically.

| Capability | Default model | Rationale |
|---|---|---|
| GATHER | `qwen2.5:14b` | Strong factual retrieval and summarisation |
| ANALYZE | `deepseek-r1:14b` | Chain-of-thought reasoning |
| PLAN | `deepseek-r1:14b` | Multi-step structured planning |
| EXECUTE | `qwen2.5:14b` | Code generation and task execution |
| VERIFY | `qwen2.5:14b` | Careful checking and validation |
| REFINE | `qwen2.5:14b` | Editing and quality improvement |
| COMMUNICATE | `mistral:7b` | Clear, fluent natural language output |

Override any mapping in `config.py`:

```python
CAPABILITY_MODEL_MAP = {
    "gather":      "llama3.2:latest",
    "analyze":     "deepseek-r1:14b",
    "plan":        "deepseek-r1:14b",
    "execute":     "qwen2.5:14b",
    "verify":      "qwen2.5:14b",
    "refine":      "qwen2.5:14b",
    "communicate": "mistral:7b",
}
```

---

## Project Structure

```
NLGraph/
├── config.py                  All tunable constants (pydantic-settings singleton)
├── pyproject.toml             Package metadata, dependencies, entry points
│
├── cli/
│   ├── __main__.py            python -m cli entry point
│   ├── interactive.py         REPL: query loop, plan editing, feedback
│   ├── display.py             Rich rendering helpers
│   ├── banner.py              Startup banner, system status panel
│   └── theme.py               Colour definitions
│
├── core/
│   ├── models.py              Pydantic models: Agent, IntentMap, etc.
│   ├── exceptions.py          Exception hierarchy
│   ├── intent/
│   │   ├── embedding_engine.py     bge-m3 + LRU embed cache
│   │   ├── phrase_splitter.py      3-layer query decomposition
│   │   ├── capability_projector.py Multi-prototype + SetFit + cross-encoder
│   │   ├── dependency_detector.py  PDTB + parse + cross-encoder cascade
│   │   ├── implicit_detector.py    Implication rule engine
│   │   ├── agent_constructor.py    Agent assembly, domain labelling
│   │   └── dag_validator.py        Cycle detection + repair
│   ├── execution/
│   │   ├── orchestrator.py         Full pipeline coordinator
│   │   ├── agent_executor.py       LLM call, blackboard write, fallback
│   │   ├── result_aggregator.py    Merge agent outputs
│   │   └── llm_providers/
│   │       └── ollama.py           Ollama HTTP provider
│   ├── blackboard/
│   │   ├── blackboard.py           FAISS fact store, artifact log
│   │   └── context_builder.py      Per-agent prompt construction
│   └── memory/
│       ├── global_store.py         Qdrant: intent + output cache
│       ├── performance_tracker.py  SQLite: metrics, thresholds
│       └── feedback_collector.py   Score storage + correction logging
│
├── setup/
│   ├── capability_seeds.py     Seed phrases for capability centroids
│   ├── dependency_seeds.py     Seed phrases for dependency centroids
│   └── centroid_builder.py     Build + validate + save centroids.npz
│
└── data/                       Runtime artefacts (git-ignored)
    ├── centroids.npz           Capability + dependency prototype vectors
    ├── nlgraph.db              SQLite: execution log, feedback, thresholds
    ├── qdrant/                 Qdrant embedded vector store
    └── models/
        └── setfit/             SetFit classifier checkpoint (after training)
```

---

## Future Work

### Short-term improvements

**Smarter phrase splitting**
The current 3-layer cascade (spaCy dep-parse → Benepar → PDTB) works well for compound queries with explicit connectives but struggles with short, ambiguous phrases. Integrating a sentence-boundary model trained on instruction-style text (e.g. NeuralEDUSeg or a fine-tuned discourse segmenter) would improve splitting quality on terse queries.

**Cross-encoder result caching**
The cross-encoder (`ms-marco-MiniLM-L-6-v2`) is invoked on every ambiguous phrase–capability pair at query time. A small in-memory cache keyed on `(text, capability)` would eliminate redundant inference on repeated phrases across sessions.

**Streaming agent output**
Agent responses are currently collected in full before display. Streaming each agent's tokens to the terminal as they arrive would give much faster time-to-first-token and allow the user to interrupt long-running agents mid-generation.

**Per-capability threshold auto-calibration**
Feedback scores are stored in SQLite, and the recalibration trigger (`RECALIB_THRESHOLD = 50`) is wired but the actual threshold-update logic is minimal. A proper Bayesian update — lowering the activation threshold for consistently high-scoring capabilities and raising it for noisy ones — would improve projection accuracy over time without manual tuning.

---

### Medium-term features

**SetFit online learning**
A SetFit classifier checkpoint slot exists at `data/models/setfit/`. The retraining trigger (`SETFIT_RETRAIN_THRESHOLD = 20` new labelled examples) fires but the training loop itself is incomplete. Closing this loop would allow the capability projector to adapt to domain-specific language that the centroid seeds don't cover.

**Parallel phrase embedding**
`embed_batch` is already vectorised for a list of phrases, but the phrase splitter processes the query synchronously before the embedding step. Pipelining the two — start embedding phrase 1 while splitting phrase 2 — would reduce decomposition latency on long queries.

**Dependency graph visualisation**
The DAG structure is computed but only displayed as a flat table. A terminal-rendered dependency graph (using Rich's `Tree` or a compact ASCII box layout) would let users understand execution ordering at a glance before confirming.

**Multi-turn conversation context**
Each query is currently treated independently. Maintaining a rolling conversation context across turns — so that "now explain it to beginners" after a technical analysis correctly refers to the previous answer — would make NLGraph far more useful in iterative research sessions.

**Agent retry and partial re-execution**
When an agent fails, the entire plan stops. A retry-with-backoff policy plus the ability to re-run only the failed subtree (while keeping outputs from successful agents on the blackboard) would make execution more resilient.

---

### Long-term directions

**Learned dependency priors**
The Bayesian prior table in `dependency_detector.py` is hand-crafted. Training a small neural classifier on pairs of `(capability_A, capability_B, query_context) → dependency_type` from accumulated execution logs would replace the static priors with data-driven ones.

**Tool use and external agents**
Today every agent calls a local LLM. Extending `AgentExecutor` with a tool-use protocol — so a GATHER agent can call a web search API, a VERIFY agent can run a Python sandbox, or an EXECUTE agent can push code to a repository — would move NLGraph from a text-only system to a full agentic workflow engine.

**Plan optimisation**
The current dependency detection produces a valid execution order but does not optimise for latency. A planner that identifies which agents can overlap (e.g. start COMMUNICATE drafting while VERIFY is still running) and schedules them accordingly could significantly reduce wall-clock time on large plans.

**Distributed execution**
With many agents and long-running LLM calls, a single asyncio event loop on one machine becomes a bottleneck. A worker-queue backend (Celery, Ray, or a simple task queue over Redis) would allow agents to run on separate processes or machines while the orchestrator coordinates through shared Qdrant + SQLite state.

**Evaluation harness**
There is no automated benchmark for decomposition quality. A test suite that feeds canonical queries through the pipeline and asserts expected capabilities, dependency types, and agent counts would enable regression testing when seeds, thresholds, or models change.

**Graphical interface**
The Rich-based terminal UI is functional but limited. A lightweight web frontend (FastAPI backend + React or Svelte) exposing the same orchestrator API would make NLGraph accessible to non-terminal users and enable richer plan visualisations (interactive DAG editor, real-time streaming output per agent).
