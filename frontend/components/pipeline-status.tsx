"use client";

import { useState } from "react";
import { CheckIcon, CopyIcon } from "lucide-react";
import { useAuiState } from "@assistant-ui/react";
import { useAgUiState } from "@assistant-ui/react-ag-ui";
import { cn } from "@/lib/utils";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import { TooltipIconButton } from "@/components/assistant-ui/elements/tooltip-icon-button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  PIPELINE_STAGES,
  stageLabel,
  type PipelineStage,
  type PlantBioRunState,
} from "@/lib/run-state";

const RETRIEVED_CONTEXT_SECTIONS: Array<{
  key: "literature_context" | "metadata_context" | "pretzel_context";
  label: string;
}> = [
  { key: "literature_context", label: "Literature" },
  { key: "metadata_context", label: "Metadata graph" },
  { key: "pretzel_context", label: "Pretzel documentation" },
];

function RetrievedContext({ state }: { state: PlantBioRunState }) {
  const [open, setOpen] = useState(false);
  const { isCopied, copyToClipboard } = useCopyToClipboard();
  const sections = RETRIEVED_CONTEXT_SECTIONS.flatMap(({ key, label }) => {
    const text = state[key];
    return text ? [{ key, label, text }] : [];
  });

  if (sections.length === 0) return null;

  const totalChars = sections.reduce((sum, { text }) => sum + text.length, 0);
  const combinedText = sections
    .map(({ label, text }) => `### ${label}\n\n${text}`)
    .join("\n\n---\n\n");

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="flex items-center gap-1">
        <CollapsibleTrigger className="text-muted-foreground hover:text-foreground text-xs underline decoration-dotted underline-offset-2">
          {open ? "Hide" : "Show"} retrieved context ({sections.length} source
          {sections.length === 1 ? "" : "s"}, {totalChars.toLocaleString()}{" "}
          chars)
        </CollapsibleTrigger>
        <TooltipIconButton
          tooltip="Copy all retrieved context"
          onClick={() => copyToClipboard(combinedText)}
          className="size-5"
        >
          {isCopied ? (
            <CheckIcon className="animate-in zoom-in-50 fade-in duration-200 ease-out" />
          ) : (
            <CopyIcon className="animate-in zoom-in-75 fade-in duration-150" />
          )}
        </TooltipIconButton>
      </div>
      <CollapsibleContent className="mt-2 flex flex-col gap-2">
        {sections.map(({ key, label, text }) => (
          <div key={key} className="flex flex-col gap-1">
            <p className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
              {label} ({text.length.toLocaleString()} chars)
            </p>
            <pre className="bg-muted max-h-64 overflow-auto rounded-md p-2 text-[11px] whitespace-pre-wrap">
              {text}
            </pre>
          </div>
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}

function stageIndex(stage: string | undefined): number {
  if (!stage) return -1;
  return PIPELINE_STAGES.indexOf(stage as PipelineStage);
}

export function PipelineStatus() {
  const state = useAgUiState<PlantBioRunState>();
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const current = stageIndex(state?.stage);
  const accessions = state?.accessions ?? [];

  if (!isRunning && !state?.stage) {
    return null;
  }

  return (
    <div
      data-slot="pipeline-status"
      className="border-border/60 bg-background/95 shrink-0 border-b px-4 py-3 backdrop-blur-sm"
    >
      <div className="mx-auto flex w-full max-w-(--thread-max-width,44rem) flex-col gap-2">
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm font-medium">
            {isRunning ? stageLabel(state?.stage) : "Last run"}
            {state?.species ? (
              <span className="text-muted-foreground font-normal">
                {" "}
                · {state.species}
              </span>
            ) : null}
          </p>
          {isRunning ? (
            <span className="text-muted-foreground text-xs tracking-wide uppercase">
              Running
            </span>
          ) : null}
        </div>

        <ol className="flex flex-wrap items-center gap-1.5">
          {PIPELINE_STAGES.map((stage, index) => {
            const reached = current >= index;
            const active = current === index;
            return (
              <li key={stage} className="flex items-center gap-1.5">
                {index > 0 ? (
                  <span
                    aria-hidden
                    className={cn(
                      "bg-border h-px w-4",
                      reached && "bg-foreground/40",
                    )}
                  />
                ) : null}
                <span
                  className={cn(
                    "rounded-full px-2 py-0.5 text-[11px] whitespace-nowrap",
                    active &&
                      "bg-foreground text-background font-medium",
                    reached &&
                      !active &&
                      "bg-muted text-foreground",
                    !reached && "text-muted-foreground",
                  )}
                >
                  {stageLabel(stage)}
                </span>
              </li>
            );
          })}
        </ol>

        {state?.expanded_question ? (
          <p className="text-muted-foreground text-xs">
            Expanded: {state.expanded_question}
          </p>
        ) : null}

        {accessions.length > 0 ? (
          <p className="text-muted-foreground line-clamp-2 text-xs">
            Accessions: {accessions.join(", ")}
          </p>
        ) : null}

        {state?.needs_clarification ? (
          <p className="text-xs font-medium">
            Species needed to look up accessions in AGG.
          </p>
        ) : null}

        {state ? <RetrievedContext state={state} /> : null}

        {state?.error ? (
          <p className="text-destructive text-xs">{state.error}</p>
        ) : null}
      </div>
    </div>
  );
}
