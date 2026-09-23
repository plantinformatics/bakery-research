"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ModelSelector as AuiModelSelector,
  type ModelOption,
} from "@/components/assistant-ui/elements/model-selector.aui";
import { useModelOptions } from "@/hooks/use-model-options";

const REASONING_LABELS: Record<string, string> = {
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
};

/**
 * Model + reasoning-level selector for the answer-generation LLM
 * (`PlantBioRAG._get_answer_llm` in `GraphRAG/Query.py`). Options come from
 * the backend (`useModelOptions`); wraps assistant-ui's `ModelSelector`
 * element (https://www.assistant-ui.com/elements/model-selector), which
 * registers the selection with `ModelContext` itself - no bespoke
 * publishing hook needed. `AgUiThreadRuntimeCore.buildRunInput` forwards
 * that context, which `GraphRAG/main.py`'s `/agent` endpoint reads via
 * `_forwarded_model_selection`.
 */
export function ModelSelector() {
  const { models, defaultModel, reasoningLevels, defaultReasoningLevel } =
    useModelOptions();
  const [model, setModel] = useState<string>(defaultModel);
  const [reasoning, setReasoning] = useState<string>(defaultReasoningLevel);

  // Adopt the fetched defaults/options once they arrive, without
  // clobbering a choice the user already made while `/options` was
  // still in flight.
  useEffect(() => {
    setModel((current) => (models.includes(current) ? current : defaultModel));
  }, [models, defaultModel]);
  useEffect(() => {
    setReasoning((current) =>
      reasoningLevels.includes(current) ? current : defaultReasoningLevel,
    );
  }, [reasoningLevels, defaultReasoningLevel]);

  // Every model currently accepts the same reasoning vocabulary
  // (`AVAILABLE_REASONING_LEVELS` in `Query.py`), so build one custom
  // effort list and share it across all model options.
  const efforts = useMemo(
    () =>
      reasoningLevels.map((level) => ({
        id: level,
        name: REASONING_LABELS[level] ?? level,
      })),
    [reasoningLevels],
  );

  const modelOptions: ModelOption[] = useMemo(
    () => models.map((id) => ({ id, name: id, efforts })),
    [models, efforts],
  );

  return (
    <AuiModelSelector
      models={modelOptions}
      value={model}
      onValueChange={setModel}
      effort={reasoning}
      onEffortChange={setReasoning}
      variant="ghost"
      size="sm"
      side="top"
      align="start"
      className="text-muted-foreground hover:text-foreground h-7 max-w-full rounded-full px-2 font-normal"
    />
  );
}
