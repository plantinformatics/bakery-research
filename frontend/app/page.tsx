"use client";

import {
  useAui,
  AuiProvider,
  AuiConfig,
  Suggestions,
} from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { PipelineStatus } from "@/components/pipeline-status";
import { MessageDataBoxes } from "@/components/message-data-boxes";
import type { FC } from "react";

const ThreadWelcome: FC = () => {
  return (
    <div className="aui-thread-welcome-root mb-6 flex flex-col items-center px-4 text-center">
      <h1 className="aui-thread-welcome-message-inner fade-in slide-in-from-bottom-1 animate-in fill-mode-both text-2xl font-medium tracking-tight duration-200">
        Bakery Research
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
        title: "Do any of the 10 wheat genomes",
        label: "carry Lr46?",
        prompt: "do any of the 10 wheat genomes carry Lr46?",
      },
      {
        title: "Do any of the 10 wheat genomes",
        label: "carry Yr29?",
        prompt: "do any of the 10 wheat genomes carry Yr29?",
      },
      {
        title: "Is there a relationship",
        label: "between Lr46 and Yr29?",
        prompt: "is there a relationship between Lr46 and Yr29?",
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
      <MessageDataBoxes />
      <PipelineStatus />
      <div className="relative min-h-0 flex-1">
        <ThreadWithSuggestions />
      </div>
    </main>
  );
}
