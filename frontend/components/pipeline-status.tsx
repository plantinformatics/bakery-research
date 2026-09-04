"use client";

import { useAuiState } from "@assistant-ui/react";
import { useAgUiState } from "@assistant-ui/react-ag-ui";
import { cn } from "@/lib/utils";
import {
  PIPELINE_STAGES,
  stageLabel,
  type PipelineStage,
  type PlantBioRunState,
} from "@/lib/run-state";

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
          <p className="text-muted-foreground line-clamp-2 text-xs">
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

        {state?.error ? (
          <p className="text-destructive text-xs">{state.error}</p>
        ) : null}
      </div>
    </div>
  );
}
