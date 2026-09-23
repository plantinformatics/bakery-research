/** FastAPI AG-UI endpoint (`GraphRAG/main.py` → `POST /agent`). */
export const AGUI_AGENT_URL =
  process.env.NEXT_PUBLIC_AGUI_AGENT_URL ?? "http://localhost:8000/agent";

/**
 * Model/reasoning-level selector options (`GraphRAG/main.py` →
 * `GET /options`), kept in sync with `GraphRAG/Query.py`'s
 * `AVAILABLE_MODELS`/`AVAILABLE_REASONING_LEVELS`. Defaults to the `/agent`
 * origin with `/options` swapped in, so only one env var is usually needed.
 */
export const AGUI_OPTIONS_URL =
  process.env.NEXT_PUBLIC_AGUI_OPTIONS_URL ??
  AGUI_AGENT_URL.replace(/\/agent\/?$/, "/options");
