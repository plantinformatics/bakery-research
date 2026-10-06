/** Git version of the repo the app runs from, reported by the backend
 * (`GET /options` → `backendVersion`, read in `GraphRAG/main.py`). The
 * frontend lives in the same repo, so one version covers both. */
export type AppVersion = {
  commit: string | null;
  branch: string | null;
};

const REPO_URL = "https://github.com/plantinformatics/bakery-research";

/** "Version 1ee0f3d" (no branch) for the label under the composer. */
export function formatVersion(version: AppVersion | null | undefined): string {
  return `Version ${version?.commit ?? "unknown"}`;
}

/** GitHub page for the running commit, or null when it isn't known. Commits
 * that only exist locally (not pushed yet) will 404 on GitHub. */
export function commitUrl(version: AppVersion | null | undefined): string | null {
  return version?.commit ? `${REPO_URL}/commit/${version.commit}` : null;
}

/** "Version 1ee0f3d (main) - <commit url>" for chat exports and bug-report
 * emails. */
export function versionSummary(version: AppVersion | null | undefined): string {
  const branch = version?.branch ? ` (${version.branch})` : "";
  const url = commitUrl(version);
  return `${formatVersion(version)}${branch}${url ? ` - ${url}` : ""}`;
}
