"use client";

import { useEffect, useState } from "react";
import { AGUI_OPTIONS_URL } from "@/lib/agent";

export type ModelOptions = {
  models: string[];
  defaultModel: string;
  reasoningLevels: string[];
  defaultReasoningLevel: string;
};

/**
 * Static fallback used until `GET /options` (`GraphRAG/main.py`) resolves,
 * or if the request fails (e.g. backend not running yet). Kept in sync with
 * `Query.py`'s `AVAILABLE_MODELS`/`AVAILABLE_REASONING_LEVELS` defaults.
 */
const FALLBACK_OPTIONS: ModelOptions = {
  models: ["gemini-3.8-flash", "gemini-2.5-flash"],
  defaultModel: "gemini-3.8-flash",
  reasoningLevels: ["minimal", "low", "medium", "high"],
  defaultReasoningLevel: "medium",
};

/** Fetches the selectable models/reasoning levels for `ModelSelector` from
 * the GraphRAG backend, so the frontend never has its own hardcoded list
 * that can drift from what `PlantBioRAG.query()` actually accepts. */
export function useModelOptions(): ModelOptions {
  const [options, setOptions] = useState<ModelOptions>(FALLBACK_OPTIONS);

  useEffect(() => {
    const controller = new AbortController();
    fetch(AGUI_OPTIONS_URL, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`GET /options failed: ${res.status}`);
        return res.json();
      })
      .then((data: Partial<ModelOptions>) => {
        setOptions((prev) => ({
          models: data.models?.length ? data.models : prev.models,
          defaultModel: data.defaultModel ?? prev.defaultModel,
          reasoningLevels: data.reasoningLevels?.length
            ? data.reasoningLevels
            : prev.reasoningLevels,
          defaultReasoningLevel:
            data.defaultReasoningLevel ?? prev.defaultReasoningLevel,
        }));
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        console.error("[use-model-options] falling back to defaults:", err);
      });
    return () => controller.abort();
  }, []);

  return options;
}
