"use client";

import {
  useAui,
  AuiProvider,
  AuiConfig,
  Suggestions,
} from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { PipelineStatus } from "@/components/pipeline-status";
import { PlusIcon } from "lucide-react";
import type { FC } from "react";

function NewThreadButton() {
  const aui = useAui();

  return (
    <button
      type="button"
      onClick={() => aui.threads.switchToNewThread()}
      className="bg-background hover:bg-accent absolute top-4 right-4 z-10 flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium shadow-sm transition-colors"
    >
      <PlusIcon className="size-4" />
      New Thread
    </button>
  );
}

const ThreadWelcome: FC = () => {
  return (
    <div className="aui-thread-welcome-root mb-6 flex flex-col items-center px-4 text-center">
      <h1 className="aui-thread-welcome-message-inner fade-in slide-in-from-bottom-1 animate-in fill-mode-both text-2xl font-medium tracking-tight duration-200">
        Plant biology GraphRAG
      </h1>
      <p className="text-muted-foreground fade-in slide-in-from-bottom-1 animate-in mt-2 max-w-md text-sm [animation-delay:80ms]">
        Ask about literature, Pretzel, or Australian Grains Genebank accessions.
      </p>
    </div>
  );
};

function ThreadWithSuggestions() {
  const aui = useAui();
  const config = AuiConfig({
    suggestions: Suggestions([
      {
        title: "Look up an accession",
        label: "in the Australian Grains Genebank",
        prompt:
          "Is the wheat variety Wyalkatchem available in the Australian Grains Genebank?",
      },
      {
        title: "Ask about a trait",
        label: "2NS introgression in wheat",
        prompt:
          "What evidence is there to suggest the wheat variety Wyalkatchem carries the 2NS introgression?",
      },
      {
        title: "Ask about Pretzel",
        label: "how alignments work",
        prompt: "How do I align two genome assemblies in Pretzel?",
      },
    ]),
  });
  return (
    <AuiProvider extends={aui} config={config}>
      <Thread components={{ Welcome: ThreadWelcome }} />
    </AuiProvider>
  );
}

export default function Home() {
  return (
    <main className="relative flex h-dvh flex-col">
      <PipelineStatus />
      <div className="relative min-h-0 flex-1">
        <NewThreadButton />
        <ThreadWithSuggestions />
      </div>
    </main>
  );
}
