"use client";

import { useAgUiState } from "@assistant-ui/react-ag-ui";
import { MapIcon } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useModelOptions } from "@/hooks/use-model-options";
import type { PlantBioRunState } from "@/lib/run-state";

/**
 * Small composer box, next to the literature context slider, shown while
 * the current (or most recent) run is on the Pretzel pathway: Jev judged the
 * question to be about using Pretzel (`is_pretzel_question` in `Query.py`),
 * so the Pretzel documentation is searched. Links to the Pretzel docs.
 */
export function PretzelPathwayIndicator() {
  const state = useAgUiState<PlantBioRunState>();
  const { pretzelDocsUrl } = useModelOptions();

  if (!state?.is_pretzel_question) return null;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <a
            data-slot="pretzel-pathway-indicator"
            href={pretzelDocsUrl}
            target="_blank"
            rel="noreferrer"
            className="fade-in animate-in inline-flex h-7 items-center gap-1 rounded-full bg-sky-100 px-2 text-sm text-sky-900 transition-colors hover:bg-sky-200 dark:bg-sky-950 dark:text-sky-100 dark:hover:bg-sky-900"
          >
            <MapIcon className="size-3.5" />
            <span>Pretzel</span>
          </a>
        </TooltipTrigger>
        <TooltipContent side="top">
          Pretzel how-to question: Pretzel documentation searched. Click to open
          the docs.
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
