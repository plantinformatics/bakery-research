"use client";

import { useAgUiState } from "@assistant-ui/react-ag-ui";
import { ThinkingOrb, type OrbState } from "thinking-orbs";
import {
  stageLabel,
  type PipelineStage,
  type PlantBioRunState,
} from "@/lib/run-state";

/** Orb animation per pipeline stage. `null` hides the orb: while the answer
 * is generating, the streamed reasoning already shows progress. */
const STAGE_ORB_STATES: Record<PipelineStage, OrbState | null> = {
  expanding_question: "shaping",
  retrieving_context: "searching",
  judging_relevance: "weaving",
  generating_answer: null,
  checking_agg_accessions: "connecting",
  presenting_accessions: "composing",
};

/** Replaces assistant-ui's "thinking" dot with an animated orb that follows
 * the current `PlantBioRAG.query()` stage. */
export function StageOrb() {
  const state = useAgUiState<PlantBioRunState>();
  const stage = state?.stage;
  const orbState =
    stage && stage in STAGE_ORB_STATES
      ? STAGE_ORB_STATES[stage as PipelineStage]
      : "working";

  if (!orbState) return null;

  const label = stage ? stageLabel(stage) : "Working";

  return (
    <span
      data-slot="aui_assistant-message-indicator"
      role="status"
      className="text-muted-foreground inline-flex items-center gap-2 text-sm"
    >
      <ThinkingOrb state={orbState} size={32} aria-hidden />
      <span>{label}…</span>
    </span>
  );
}
