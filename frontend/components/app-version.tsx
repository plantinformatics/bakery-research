"use client";

import { useModelOptions } from "@/hooks/use-model-options";
import { commitUrl, formatVersion, versionSummary } from "@/lib/version";

/** Small muted "Version <commit>" line under the composer, linking to the
 * commit on GitHub; hover for the branch. "unknown" (no link) until
 * `/options` loads, or if the backend couldn't read git (see `_git_version`
 * in `GraphRAG/main.py`). */
export function AppVersionLabel() {
  const { backendVersion } = useModelOptions();
  const url = commitUrl(backendVersion);
  const label = formatVersion(backendVersion);

  return (
    <p
      data-slot="app-version"
      title={versionSummary(backendVersion)}
      className="text-muted-foreground/70 text-center font-mono text-[11px]"
    >
      {url ? (
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="hover:text-foreground underline-offset-2 hover:underline"
        >
          {label}
        </a>
      ) : (
        label
      )}
    </p>
  );
}
