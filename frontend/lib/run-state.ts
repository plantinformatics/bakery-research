/** Mirrors `RunState` from `GraphRAG/Query.py`, as sent in AG-UI STATE_SNAPSHOT events. */

export const PIPELINE_STAGES = [
  "expanding_question",
  "retrieving_context",
  "generating_answer",
  "checking_agg_accessions",
  "presenting_accessions",
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];

export type PlantBioRunState = {
  stage?: PipelineStage | string;
  expanded_question?: string | null;
  /** Answer-generation model/reasoning level actually used for this run
   * (after `Query.py`'s fallback resolution), echoed back so the frontend
   * can confirm the `ModelSelector` choice took effect. */
  model_name?: string;
  reasoning_level?: string;
  species?: string;
  is_agg_accession_query?: boolean;
  needs_clarification?: boolean;
  accessions?: string[];
  usage_metadata?: Record<string, unknown>;
  /** Raw context retrieved from Neo4j and injected into the answer prompt. */
  literature_context?: string | null;
  metadata_context?: string | null;
  pretzel_context?: string | null;
  error?: string | null;
};

export const STAGE_LABELS: Record<PipelineStage, string> = {
  expanding_question: "Expanding question",
  retrieving_context: "Retrieving context",
  generating_answer: "Generating answer",
  checking_agg_accessions: "Checking accessions",
  presenting_accessions: "Presenting accessions",
};

export function stageLabel(stage: string | undefined): string {
  if (!stage) return "Idle";
  if (stage in STAGE_LABELS) {
    return STAGE_LABELS[stage as PipelineStage];
  }
  return stage.replaceAll("_", " ");
}
