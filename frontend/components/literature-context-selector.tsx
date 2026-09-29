"use client";

import { useEffect, useState } from "react";
import { useAui, type ModelContext } from "@assistant-ui/react";
import { BookOpenIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { useModelOptions } from "@/hooks/use-model-options";

function formatChars(chars: number): string {
  if (chars >= 1_000_000) return `${+(chars / 1_000_000).toFixed(2)}M`;
  return `${Math.round(chars / 1000)}k`;
}

/**
 * Composer button that opens a slider for the literature context character
 * budget (`max_context_chars` in `PlantBioRAG.query()`, `GraphRAG/Query.py`).
 * Bounds/default come from the backend (`useModelOptions`). The value is
 * registered with assistant-ui's `ModelContext` as
 * `config.maxLiteratureContextChars`, which `AgUiThreadRuntimeCore.
 * buildRunInput` spreads into `forwardedProps`, read by `GraphRAG/main.py`'s
 * `_forwarded_max_context_chars`.
 */
export function LiteratureContextSelector() {
  const {
    minLiteratureContextChars: min,
    maxLiteratureContextChars: max,
    literatureContextCharsStep: step,
    defaultLiteratureContextChars: defaultChars,
  } = useModelOptions();
  const [chars, setChars] = useState<number>(defaultChars);
  const [touched, setTouched] = useState(false);
  const api = useAui();

  // Adopt the fetched default once `/options` resolves, unless the user
  // already moved the slider; always keep the value inside the bounds.
  useEffect(() => {
    setChars((current) => {
      const next = touched ? current : defaultChars;
      return Math.min(max, Math.max(min, next));
    });
  }, [defaultChars, min, max, touched]);

  useEffect(() => {
    // `LanguageModelConfig` is a closed type, but the AG-UI runtime spreads
    // every `config` key into `forwardedProps`, so extend it with our field.
    const config = {
      config: { maxLiteratureContextChars: chars } as unknown as NonNullable<
        ModelContext["config"]
      >,
    };
    return api.modelContext.register({
      getModelContext: () => config,
    });
  }, [api, chars]);

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          aria-label="Max literature context"
          className="text-muted-foreground hover:text-foreground h-7 gap-1 rounded-full px-2 font-normal"
        >
          <BookOpenIcon className="size-3.5" />
          <span>{formatChars(chars)}</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent side="top" align="start" className="w-72">
        <div className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between">
            <span className="text-sm font-medium">Max literature context</span>
            <span className="text-muted-foreground text-sm tabular-nums">
              {chars.toLocaleString()} chars
            </span>
          </div>
          <Slider
            value={[chars]}
            min={min}
            max={max}
            step={step}
            onValueChange={([value]) => {
              setTouched(true);
              setChars(value);
            }}
            aria-label="Max literature context characters"
          />
          <div className="text-muted-foreground flex justify-between text-xs">
            <span>{formatChars(min)}</span>
            <span>≈ {formatChars(Math.round(chars / 4))} tokens</span>
            <span>{formatChars(max)}</span>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="self-end"
            disabled={chars === defaultChars}
            onClick={() => {
              setTouched(false);
              setChars(defaultChars);
            }}
          >
            Reset to default ({formatChars(defaultChars)})
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
