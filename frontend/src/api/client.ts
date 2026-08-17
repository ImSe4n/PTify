/**
 * The HTTP client.
 *
 * THREE ERROR ENVELOPES SHIP, AND THIS IS THE ONLY PLACE THAT KNOWS IT.
 *
 *   1. Routes raise HTTPException(detail=ErrorOut(...)) -> {"detail": {"code", "message"}}
 *   2. The PipelineError handler returns ErrorOut UNWRAPPED -> {"code", "message"}
 *      (api/app.py:185-194)
 *   3. Pydantic validation -> {"detail": [{"loc", "msg", "type"}, ...]}
 *
 * Every caller goes through `parseApiError`, so no screen has to guess which
 * one it got. Verified against a live server in Phase 5.5: a bad token gives
 * shape 1, a short password gives shape 3.
 */

import type {
  EngineOut,
  HealthOut,
  JobAccepted,
  JobOut,
  MeOut,
  OutputFormat,
  SubmitOptions,
  Summary,
  TokenOut,
} from "./types";

/** Same-origin by default: the Vite dev server proxies /v1 to the backend. */
const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  readonly status: number;
  /** Stable machine-readable code. Branch on THIS, never on the message. */
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }

  /**
   * Both limits are 429 and they mean different things, so status alone is not
   * enough to decide what to tell the user (api/security.py:200-233).
   * `rate_limited` clears on its own; `too_many_jobs` needs the user to act.
   */
  get isRateLimit(): boolean {
    return this.code === "rate_limited";
  }

  get isTooManyJobs(): boolean {
    return this.code === "too_many_jobs";
  }

  /** No refresh endpoint exists -- a 401 means "log in again". */
  get isAuthFailure(): boolean {
    return this.status === 401;
  }
}

/** Normalises all three envelopes into one ApiError. */
export async function parseApiError(res: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return new ApiError(res.status, "http_error", res.statusText || "request failed");
  }

  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;

    // Shape 2: unwrapped {code, message}.
    if (typeof b.code === "string" && typeof b.message === "string") {
      return new ApiError(res.status, b.code, b.message);
    }

    const detail = b.detail;

    // Shape 1: {detail: {code, message}}.
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const d = detail as Record<string, unknown>;
      if (typeof d.code === "string" && typeof d.message === "string") {
        return new ApiError(res.status, d.code, d.message);
      }
    }

    // Shape 3: Pydantic's list of validation failures. Join them into one
    // readable sentence rather than showing the caller a loc/msg/type tree.
    if (Array.isArray(detail)) {
      const msg = detail
        .map((item) => {
          if (item && typeof item === "object") {
            const it = item as Record<string, unknown>;
            const loc = Array.isArray(it.loc) ? it.loc.filter((p) => p !== "body").join(".") : "";
            const m = typeof it.msg === "string" ? it.msg : "invalid value";
            return loc ? `${loc}: ${m}` : m;
          }
          return "invalid value";
        })
        .join("; ");
      return new ApiError(res.status, "validation_error", msg || "invalid request");
    }

    if (typeof detail === "string") {
      return new ApiError(res.status, "http_error", detail);
    }
  }

  return new ApiError(res.status, "http_error", res.statusText || "request failed");
}

export type TokenGetter = () => string | null;

let getToken: TokenGetter = () => null;

/** The auth layer registers its accessor here so every request picks it up. */
export function setTokenGetter(fn: TokenGetter): void {
  getToken = fn;
}

export function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { ...authHeaders(), ...(init?.headers ?? {}) },
  });
  if (!res.ok) throw await parseApiError(res);
  return (await res.json()) as T;
}

// --- health and capabilities ------------------------------------------------

export const getHealth = () => request<HealthOut>("/healthz");
export const getEngines = () => request<EngineOut[]>("/v1/engines");

// --- auth -------------------------------------------------------------------

/**
 * `/v1/auth/*` is registered ONLY when the server has both PTIFY_JWT_SECRET and
 * PTIFY_DB_PATH (api/app.py:74-77). Without them every auth path is a plain
 * 404, which is an honest "this server does not do accounts" -- so a 404 here
 * must degrade to anonymous rather than being shown as an error.
 */
export async function accountsAvailable(): Promise<boolean> {
  const res = await fetch(`${BASE}/v1/auth/me`, { headers: authHeaders() });
  return res.status !== 404;
}

const credentials = (email: string, password: string): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ email, password }),
});

export const signup = (email: string, password: string) =>
  request<TokenOut>("/v1/auth/signup", credentials(email, password));

export const login = (email: string, password: string) =>
  request<TokenOut>("/v1/auth/login", credentials(email, password));

export const getMe = () => request<MeOut>("/v1/auth/me");

// --- jobs -------------------------------------------------------------------

export async function submitJob(opts: SubmitOptions): Promise<JobAccepted> {
  const form = new FormData();
  form.append("file", opts.file);
  form.append("formats", opts.formats.join(","));
  if (opts.engine) form.append("engine", opts.engine);
  if (opts.tempo != null) form.append("tempo", String(opts.tempo));
  if (opts.beatsPerBar != null) form.append("beats_per_bar", String(opts.beatsPerBar));
  if (opts.title) form.append("title", opts.title);
  if (opts.composer) form.append("composer", opts.composer);

  // No Content-Type header: the browser must set the multipart boundary.
  const res = await fetch(`${BASE}/v1/jobs`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) throw await parseApiError(res);
  return (await res.json()) as JobAccepted;
}

export const getJob = (id: string) => request<JobOut>(`/v1/jobs/${id}`);
export const listJobs = () => request<JobOut[]>("/v1/jobs");
export const getResultJson = (id: string) => request<Summary>(`/v1/jobs/${id}/result/json`);

/** Cancel. Returns the job: a queued one is already `cancelled`, a running one
 *  stays `running` until the next stage boundary (verified in Phase 5.5). */
export const cancelJob = (id: string) =>
  request<JobOut>(`/v1/jobs/${id}`, { method: "DELETE" });

export function artifactUrl(id: string, fmt: OutputFormat, page?: number): string {
  const q = page != null ? `?page=${page}` : "";
  return `${BASE}/v1/jobs/${id}/result/${fmt}${q}`;
}

/**
 * Fetches an artifact.
 *
 * EVERY artifact goes through here, because the URL needs an Authorization
 * header -- a plain <a href>, an <img src> or an <audio src> cannot carry one,
 * and the API accepts no token in the query string (deliberately: it would put
 * a credential into server logs). So an artifact can only ever be reached by
 * fetch, and playback has to load the MIDI into memory rather than pointing an
 * element at the URL.
 *
 * Errors go through parseApiError like every other call. Before Phase 7 the
 * sheet viewer hand-rolled this fetch and threw a bare Error, which was the one
 * place in the app where an API error lost its code and message.
 */
async function fetchArtifact(id: string, fmt: OutputFormat, page?: number): Promise<Response> {
  const res = await fetch(artifactUrl(id, fmt, page), { headers: authHeaders() });
  if (!res.ok) throw await parseApiError(res);
  return res;
}

export const fetchArtifactBlob = (id: string, fmt: OutputFormat, page?: number) =>
  fetchArtifact(id, fmt, page).then((r) => r.blob());

export const fetchArtifactText = (id: string, fmt: OutputFormat, page?: number) =>
  fetchArtifact(id, fmt, page).then((r) => r.text());

export const fetchArtifactBytes = (id: string, fmt: OutputFormat, page?: number) =>
  fetchArtifact(id, fmt, page).then((r) => r.arrayBuffer());

/** Downloads an artifact through the browser's save flow. */
export async function downloadArtifact(
  id: string,
  fmt: OutputFormat,
  filename: string,
  page?: number,
): Promise<void> {
  const blob = await fetchArtifactBlob(id, fmt, page);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
