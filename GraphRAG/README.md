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

Loads Markdown files (or a single Markdown file), splits them into chunks, embeds + upserts them into Neo4j, then uses an LLM to extract entities/relationships according to a schema.

The script is driven by a CLI (`argparse`) — no need to edit the file to change modes/paths:

```bash
uv run python BuildLiteraturePretzelDocuementationGraph.py --mode build
uv run python BuildLiteraturePretzelDocuementationGraph.py --mode add
uv run python BuildLiteraturePretzelDocuementationGraph.py --mode add_sup
uv run python BuildLiteraturePretzelDocuementationGraph.py --mode add_pretzel
```

Modes (`--mode`, required):

- `build` — full build from `md_dir/` (initial load): chunk + embed + upsert, create vector/full-text indexes, then extract entities/relationships.
- `add` — incrementally add new documents from `add_dir/`, embedding **and** extracting nodes/relationships.
- `add_sup` — incrementally add new documents from `add_dir/` **without** node/relationship extraction (chunks + embeddings only).
- `add_pretzel` — load Pretzel documentation chunks from `add_pretzel_functions/` into the separate `PretzelFunction` vector/full-text indexes.

Optional flags:

- `--dir PATH` — override the input directory for the selected mode (defaults: `md_dir/` for `build`, `add_dir/` for `add`/`add_sup`, `add_pretzel_functions/` for `add_pretzel`).
- `--file PATH` — process a single Markdown file instead of a directory (mutually exclusive with `--dir`).
- `--schema PATH` — path to the schema YAML (defaults to `schema.yaml` in this folder).

This script requires a schema file (`schema.yaml` by default) describing allowed node types and relationships as YAML, e.g.:

```yaml
entities:
  Gene: {}
  Trait: {}
relationships:
  ASSOCIATED_WITH:
    from: Gene
    to: Trait
```

`schema.yaml` is git-ignored (see `.gitignore`) and not included in the repo — create your own before running, or point `--schema` at a different file.

Logging (`setup_logging`) writes to both the console and the mode's log file (`build.log` / `add.log` / `addpretzelfunctions.log`) simultaneously. Any chunks that fail extraction after all retries are appended to `failed_docs.jsonl`.

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

This starts `POST /agent`, which accepts an AG-UI `RunAgentInput` payload and streams back Server-Sent Events (thinking, text, state snapshots, errors). Point the Next.js UI in `frontend/` at this URL (`NEXT_PUBLIC_AGUI_AGENT_URL=http://localhost:8000/agent`; see `frontend/README.md`).

## Data directories

- `md_dir/` — source Markdown documents used for the initial literature graph `build` (e.g. `Abbott 1991.pdf.md`, `Coulter 2018.pdf.md`, `Wang 2020.pdf.md`).
- `add_dir/` — new documents to incrementally `add`/`add_sup` to the literature graph.
- `add_pretzel_functions/` — Pretzel documentation chunks used by `add_pretzel`.

Any of these can be overridden per-run with `--dir`/`--file` (see above).

## Logs

Each pipeline mode writes its own append-only log file in this folder — `build.log`, `add.log`, `addpretzelfunctions.log` — and mirrors the same output to the console. These (and `failed_docs.jsonl`) are useful for resuming/debugging long-running ingestion jobs that retry with backoff on transient API failures.
