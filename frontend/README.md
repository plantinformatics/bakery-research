# GraphRAG chat UI

Next.js chat frontend for the plant biology GraphRAG pipeline. It speaks the [AG-UI](https://docs.ag-ui.com) protocol to the FastAPI server in `GraphRAG/main.py`.

## Quick start

### 1. Start the FastAPI agent

From `GraphRAG/` (see that folder's README for `.env` / Neo4j / Gemini setup):

```bash
cd GraphRAG
uv run uvicorn main:app --reload --port 8000
```

This serves `POST /agent` at `http://localhost:8000/agent`.

### 2. Configure the frontend

`frontend/.env.example` is the template. For local dev, `frontend/.env.local` should contain:

```
NEXT_PUBLIC_AGUI_AGENT_URL=http://localhost:8000/agent
```

The browser posts AG-UI `RunAgentInput` directly to that URL (CORS is enabled on the FastAPI app).

### 3. Run the UI

```bash
cd frontend
bun install
bun dev
```

Open `http://localhost:3000`. The chat streams thinking traces, the answer, and pipeline stage snapshots (expanding question → retrieval → answer → accession lookup).

## What is wired

- `@assistant-ui/react-ag-ui` + `HttpAgent` → FastAPI `POST /agent`
- `STATE_SNAPSHOT` events (`RunState` from `Query.py`) → the pipeline status bar
- `THINKING_*` / `TEXT_MESSAGE_*` events → reasoning + answer in the thread
- Suggested prompts are GraphRAG example questions, not the stock assistant-ui demo tools

The pipeline is single-turn: each send uses the latest user message. Threads are kept in the browser only.

## Related

- [assistant-ui AG-UI runtime](https://www.assistant-ui.com/docs/runtimes/ag-ui)
- [AG-UI protocol](https://docs.ag-ui.com)
- `GraphRAG/README.md` — API server and graph build
