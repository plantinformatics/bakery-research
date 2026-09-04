/** FastAPI AG-UI endpoint (`GraphRAG/main.py` → `POST /agent`). */
export const AGUI_AGENT_URL =
  process.env.NEXT_PUBLIC_AGUI_AGENT_URL ?? "http://localhost:8000/agent";
