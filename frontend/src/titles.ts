/**
 * The title the user typed at upload, remembered for this tab only.
 *
 * `JobOut` exposes neither the title nor the original filename -- both live on
 * `JobSpec`, server-side. Before the router that did not show, because the
 * title was passed down from the screen that collected it. A deep link has no
 * such chain.
 *
 * sessionStorage is the honest middle: the title survives a refresh in the tab
 * that submitted the job, and a link opened anywhere else falls back to the
 * detected key, exactly as Phase 6 did. It is NOT a cache pretending to be
 * data -- nothing reads it as authoritative, and the durable fix is putting the
 * title on `JobOut` in `api/models.py`.
 */

const KEY = "ptify.titles";

type TitleMap = Record<string, string>;

function read(): TitleMap {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as TitleMap) : {};
  } catch {
    // Private-mode Safari throws on sessionStorage; a missing title is a
    // cosmetic fallback, never a failure worth surfacing.
    return {};
  }
}

export function rememberTitle(jobId: string, title: string): void {
  const trimmed = title.trim();
  if (!trimmed) return;
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ ...read(), [jobId]: trimmed }));
  } catch {
    /* ignore */
  }
}

export function recalledTitle(jobId: string): string | null {
  return read()[jobId] ?? null;
}
