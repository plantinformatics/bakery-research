"use client";

import { useMemo, useRef, useState, useEffect, type ReactNode } from "react";
import {
  AssistantRuntimeProvider,
  type ThreadMessage,
} from "@assistant-ui/react";
import { HttpAgent } from "@ag-ui/client";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { AGUI_AGENT_URL } from "@/lib/agent";

type StoredThread = {
  id: string;
  messages: readonly ThreadMessage[];
};

/**
 * AG-UI runtime pointed at the GraphRAG FastAPI `POST /agent` endpoint.
 */
export function MyRuntimeProvider({
  children,
}: Readonly<{ children: ReactNode }>) {
  const threadsRef = useRef<Map<string, StoredThread>>(new Map());
  const [currentThreadId, setCurrentThreadId] = useState<string>(() => {
    const id = crypto.randomUUID();
    threadsRef.current.set(id, { id, messages: [] });
    return id;
  });

  const agent = useMemo(() => {
    return new HttpAgent({
      url: AGUI_AGENT_URL,
      threadId: currentThreadId,
      headers: {
        Accept: "text/event-stream",
      },
    });
  }, [currentThreadId]);

  const threadListAdapter = useMemo(
    () => ({
      threadId: currentThreadId,
      onSwitchToNewThread: async () => {
        const newId = crypto.randomUUID();
        threadsRef.current.set(newId, { id: newId, messages: [] });
        setCurrentThreadId(newId);
      },
      onSwitchToThread: async (threadId: string) => {
        const thread = threadsRef.current.get(threadId);
        if (!thread) {
          throw new Error(`Thread ${threadId} not found`);
        }
        setCurrentThreadId(threadId);
        return { messages: thread.messages };
      },
    }),
    [currentThreadId],
  );

  const runtime = useAgUiRuntime({
    agent,
    showThinking: true,
    logger: {
      debug: (...a: unknown[]) => console.debug("[agui]", ...a),
      error: (...a: unknown[]) => console.error("[agui]", ...a),
    },
    adapters: {
      threadList: threadListAdapter,
    },
  });

  useEffect(() => {
    return runtime.thread.subscribe(() => {
      threadsRef.current.set(currentThreadId, {
        id: currentThreadId,
        messages: runtime.thread.getState().messages,
      });
    });
  }, [runtime, currentThreadId]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
}
