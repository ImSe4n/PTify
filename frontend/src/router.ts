/**
 * The router.
 *
 * WHY THIS IS HAND-WRITTEN
 *
 * Six screens and one parameter. react-router is ~20kB gzipped and brings a
 * Routes/Outlet/loader model for a flow that is genuinely linear. The house
 * precedent is to hand-roll the small thing and say why: `api/sse.ts` is a
 * fetch-based EventSource for the same reason, and `api/tokens.py` is a
 * hand-written JWT.
 *
 * WHY HASH AND NOT THE HISTORY API
 *
 * Nothing serves index.html on a 404 yet -- deploy is Phase 10. A hash route
 * works identically under `vite dev`, `vite preview` and any static host, with
 * no server cooperation at all. The cost is one `#` in the URL.
 *
 * Every screen navigates through `navigate()`, so switching to `pushState`
 * later is a change to this file alone.
 *
 * WHY THERE IS NO `/j/{id}/waiting`
 *
 * Whether a job shows Waiting or Result is a fact about the job's STATE, which
 * changes while the URL does not. So `/j/{id}` means "show me this job" and the
 * screen decides. Bookmark a running job and you get Waiting, then Result, with
 * the URL never changing -- which is the correct behaviour and falls out of the
 * design rather than being special-cased.
 */

import { useMemo, useSyncExternalStore } from "react";

export type Route =
  | { screen: "upload" }
  | { screen: "auth" }
  | { screen: "history" }
  | { screen: "job"; jobId: string }
  | { screen: "sheet"; jobId: string; page: number };

/** The screens the header's nav can highlight. */
export type NavScreen = Route["screen"];

/**
 * Parse a `location.hash` into a Route. Total: anything unrecognised is the
 * upload screen, so a mangled link lands somewhere useful rather than blank.
 */
export function parseHash(hash: string): Route {
  // "#/j/abc?p=3" -> "/j/abc?p=3". A bare "" or "#" is the root.
  const raw = hash.replace(/^#/, "");
  const [path, query] = raw.split("?");
  const parts = path.split("/").filter(Boolean);

  if (parts.length === 0) return { screen: "upload" };

  if (parts[0] === "sign-in") return { screen: "auth" };
  if (parts[0] === "history") return { screen: "history" };

  if (parts[0] === "j" && parts[1]) {
    const jobId = decodeURIComponent(parts[1]);
    if (parts[2] === "sheet") {
      return { screen: "sheet", jobId, page: parsePage(query) };
    }
    return { screen: "job", jobId };
  }

  return { screen: "upload" };
}

/** `?p=3` -> 3. Floors, and clamps to at least 1 -- a deep link is untrusted. */
function parsePage(query: string | undefined): number {
  if (!query) return 1;
  const value = new URLSearchParams(query).get("p");
  const n = Math.floor(Number(value));
  return Number.isFinite(n) && n >= 1 ? n : 1;
}

/** The inverse of parseHash. */
export function formatRoute(route: Route): string {
  switch (route.screen) {
    case "upload":
      return "#/";
    case "auth":
      return "#/sign-in";
    case "history":
      return "#/history";
    case "job":
      return `#/j/${encodeURIComponent(route.jobId)}`;
    case "sheet": {
      const id = encodeURIComponent(route.jobId);
      return route.page > 1 ? `#/j/${id}/sheet?p=${route.page}` : `#/j/${id}/sheet`;
    }
  }
}

/**
 * Go to a route.
 *
 * `replace` swaps the current entry instead of pushing one. Use it for
 * same-screen state (the sheet pager) so Back leaves the screen rather than
 * walking backwards through twelve pages.
 */
export function navigate(route: Route, { replace = false } = {}): void {
  const href = formatRoute(route);
  if (href === window.location.hash) return;

  if (replace) {
    // replaceState does not fire hashchange, so tell our own subscribers.
    window.history.replaceState(null, "", href);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else {
    window.location.hash = href;
  }
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener("hashchange", onChange);
  window.addEventListener("popstate", onChange);
  return () => {
    window.removeEventListener("hashchange", onChange);
    window.removeEventListener("popstate", onChange);
  };
}

const getSnapshot = () => window.location.hash;

/**
 * The current route.
 *
 * useSyncExternalStore rather than useState+useEffect: it is tear-free under
 * concurrent rendering and needs no mount effect to catch the initial value.
 *
 * The parse is memoised on the raw hash STRING. Without that, `parseHash`
 * returns a fresh object every render and every `useEffect([route])` downstream
 * re-fires forever.
 */
export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return useMemo(() => parseHash(hash), [hash]);
}
