# GraphRAG

A Retrieval-Augmented Generation (RAG) pipeline for plant biology research, built on **Neo4j** (graph + vector + full-text store), **Google Gemini** (LLM + embeddings), and **LangChain**. It combines three knowledge sources:

- **Literature graph** — chunks and extracted entity/relationship triples from scientific papers (Markdown/PDF-derived `.md` files).
- **Metadata graph** — structured dataset/project/accession metadata loaded from CSV exports.
- **Pretzel graph** — documentation for the Pretzel genomics visualisation tool.

## Prerequisites

- Python `3.14` (see `.python-version`) — managed via [`uv`](https://docs.astral.sh/uv/).
- A running **Neo4j** instance (with vector index support, e.g. Neo4j 5.x+).
- A **Google API key** with access to Gemini models (chat + embeddings).

## Setup

1. Install dependencies (creates/updates `.venv` in this folder):

   ```bash
   cd GraphRAG
   uv sync
   ```

2. Create a `.env` file at the **repository root** (`bakery-research/.env`) with:

   ```bash
   export GOOGLE_API_KEY=your-gemini-api-key
   export NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   export NEO4J_USERNAME=neo4j
   export NEO4J_PASSWORD=your-password
   ```

   `BuildLiteraturePretzelDocuementationGraph.py` and `main.py` (via `Query.py`) read these at import time. `.env` is git-ignored — never commit it.

3. Run all commands below with `uv run` (or activate `GraphRAG/.venv` first) so the correct Python/deps are used.

## Components & how to run them

### Build the literature graph — `BuildLiteraturePretzelDocuementationGraph.py`

Loads Markdown files, splits them into chunks, embeds + upserts them into Neo4j, then uses an LLM to extract entities/relationships according to a schema.

```bash
uv run python BuildLiteraturePretzelDocuementationGraph.py
```

Before running, check/edit the `mode` variable and paths inside `main()` at the bottom of the file:

- `mode = "build"` — full build from `md_dir/` (initial load).
- `mode = "add"` — incrementally add new documents from `add_dir/`.
- `mode = "add_pretzel_functions"` — load Pretzel documentation chunks (set `pretzel_functions_dir` first).

This script also requires a schema file (`schema.yaml` by default, referenced in `main()`) describing allowed node types and relationships as YAML, e.g.:

```yaml
entities:
  Gene: {}
  Trait: {}
relationships:
  ASSOCIATED_WITH:
    from: Gene
    to: Trait
```

This file does not currently exist in the repo — create it before running, or point `schema` in `main()` at your own schema file.

Progress is appended to `build.log` / `add.log` / `addpretzelfunctions.log`, and any chunks that fail extraction after all retries are appended to `failed_docs.jsonl`.

### Ask a question from the command line — `Query.py`

Runs the full retrieval + generation pipeline once and prints the streamed answer.

```bash
uv run python Query.py "Is the wheat variety Wyalkatchem available in the Australian Grains Genebank?"
```

### Run the API server — `main.py`

Exposes the RAG pipeline as a streaming [AG-UI](https://github.com/ag-ui-protocol/ag-ui) endpoint for a front-end (e.g. a chat UI) to consume.

```bash
uv run uvicorn main:app --reload --port 8000
```

This starts `POST /agent`, which accepts an AG-UI `RunAgentInput` payload and streams back Server-Sent Events (thinking, text, state snapshots, errors).

## Data directories

- `md_dir/` — source Markdown documents used for the initial literature graph `build`.
- `add_dir/` — new documents to incrementally `add` to the literature graph.

## Logs

Each pipeline stage writes its own append-only log file in this folder: `build.log`, `add.log`, `addpretzelfunctions.log`. These (and `failed_docs.jsonl`) are useful for resuming/debugging long-running ingestion jobs that retry with backoff on transient API failures.
