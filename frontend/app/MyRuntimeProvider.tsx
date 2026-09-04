"use client";

import { useMemo, useRef, useState, useEffect, type ReactNode } from "react";
import {
  AssistantRuntimeProvider,
  type ThreadMessage,
} from "@assistant-ui/react";
import {
  BackwardCompatibility_0_0_45,
  HttpAgent,
} from "@ag-ui/client";
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
    // GraphRAG emits legacy THINKING_* events. @ag-ui/client 0.0.59 only
    // auto-maps those to REASONING_* when maxVersion <= 0.0.45, and otherwise
    // drops them — so without this shim the reasoning panel never appears.
    return new HttpAgent({
      url: AGUI_AGENT_URL,
      threadId: currentThreadId,
      headers: {
        Accept: "text/event-stream",
      },
    }).use(new BackwardCompatibility_0_0_45());
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
