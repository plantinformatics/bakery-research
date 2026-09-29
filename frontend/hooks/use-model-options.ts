"use client";

import { useEffect, useState } from "react";
import modelConfig from "@/config/models.json";
import { AGUI_OPTIONS_URL } from "@/lib/agent";

export type ModelOptions = {
  models: string[];
  defaultModel: string;
  reasoningLevels: string[];
  defaultReasoningLevel: string;
  /** Literature context budget slider bounds (`Query.py`'s
   * `MIN_/MAX_LITERATURE_CONTEXT_CHARS`, `LITERATURE_CONTEXT_CHARS_STEP`,
   * `MAX_CHARACTERS`). */
  minLiteratureContextChars: number;
  maxLiteratureContextChars: number;
  literatureContextCharsStep: number;
  defaultLiteratureContextChars: number;
};

/**
 * Static fallback used until `GET /options` (`GraphRAG/main.py`) resolves,
 * or if the request fails (e.g. backend not running yet). Read from the
 * same `frontend/config/models.json` the backend loads.
 */
const FALLBACK_OPTIONS: ModelOptions = {
  models: modelConfig.models,
  defaultModel: modelConfig.defaultModel,
  reasoningLevels: modelConfig.reasoningLevels,
  defaultReasoningLevel: modelConfig.defaultReasoningLevel,
  // Keep in sync with the literature context constants in `GraphRAG/Query.py`.
  minLiteratureContextChars: 10000,
  maxLiteratureContextChars: 100000,
  literatureContextCharsStep: 10000,
  defaultLiteratureContextChars: 50000,
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
          minLiteratureContextChars:
            data.minLiteratureContextChars ?? prev.minLiteratureContextChars,
          maxLiteratureContextChars:
            data.maxLiteratureContextChars ?? prev.maxLiteratureContextChars,
          literatureContextCharsStep:
            data.literatureContextCharsStep ?? prev.literatureContextCharsStep,
          defaultLiteratureContextChars:
            data.defaultLiteratureContextChars ??
            prev.defaultLiteratureContextChars,
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
