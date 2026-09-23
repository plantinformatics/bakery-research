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

function tokenCount(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Compact "1,234 in · 567 out · 1,801 total" label for `usage_metadata`
 * from the answer-generation LLM. Returns null when the model reported
 * no counts (direct AGG lookups, or a run that failed before generation). */
export function formatTokenUsage(
  usage: Record<string, unknown> | undefined,
): string | null {
  if (!usage) return null;
  const input = tokenCount(usage.input_tokens);
  const output = tokenCount(usage.output_tokens);
  const total = tokenCount(usage.total_tokens);
  if (input === null && output === null && total === null) return null;

  const outputDetails = usage.output_token_details;
  const reasoning =
    outputDetails && typeof outputDetails === "object"
      ? tokenCount(
          (outputDetails as Record<string, unknown>).reasoning,
        )
      : null;
  const inputDetails = usage.input_token_details;
  const cached =
    inputDetails && typeof inputDetails === "object"
      ? tokenCount((inputDetails as Record<string, unknown>).cache_read)
      : null;

  const parts: string[] = [];
  if (input !== null) {
    const cachedLabel =
      cached !== null && cached > 0
        ? ` (${cached.toLocaleString()} cached)`
        : "";
    parts.push(`${input.toLocaleString()} in${cachedLabel}`);
  }
  if (output !== null) {
    const reasoningLabel =
      reasoning !== null && reasoning > 0
        ? ` (${reasoning.toLocaleString()} reasoning)`
        : "";
    parts.push(`${output.toLocaleString()} out${reasoningLabel}`);
  }
  if (total !== null) parts.push(`${total.toLocaleString()} total`);
  return parts.join(" · ");
}
