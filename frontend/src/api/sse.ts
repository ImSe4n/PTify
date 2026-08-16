/**
 * A fetch-based Server-Sent Events reader.
 *
 * WHY NOT `new EventSource(...)`
 *
 * `GET /v1/jobs/{id}/events` sits behind `get_principal`, which reads headers
 * only (api/routes/jobs.py:235). The native EventSource API cannot set an
 * Authorization header, and the API deliberately accepts no token in the query
 * string -- a credential in a URL reaches server logs and browser history, and
 * the codebase's own rule is that a principal id never carries the credential.
 *
 * So the stream is read over fetch(), which can carry the header. This is ~40
 * lines and needs no dependency.
 *
 * WHAT THE STREAM SENDS (api/events.py:68-72)
 *
 *   state      full JobOut, on connect and whenever (state, progress, stage) changes
 *   heartbeat  {job_id, state, progress, stage, elapsed} during a silent span
 *   end        full JobOut, once, then the stream closes
 *   error      {code, message} -- only if the job disappears mid-stream
 *
 * THE HEARTBEAT IS THE WHOLE POINT, AND PHASE 5.5 MEASURED IT.
 *
 * On a real 67-second Scarlatti recording the stream showed `progress` frozen
 * at 0.09 for ~160 seconds while heartbeats arrived every ~10s with a climbing
 * `elapsed`, then it jumped straight to 0.92. That is not a stall: ByteDance
 * reports nothing at all during inference, and the API refuses to invent a
 * percentage it did not measure. The client must not invent one either -- show
 * an indeterminate indicator and the honest elapsed time.
 */

import { authHeaders, parseApiError } from "./client";
import type { JobOut, JobState } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export interface Heartbeat {
  job_id: string;
  state: JobState;
  progress: number;
  stage: string;
  elapsed: number;
}

export interface JobStreamHandlers {
  onState?: (job: JobOut) => void;
  onHeartbeat?: (beat: Heartbeat) => void;
  onEnd?: (job: JobOut) => void;
  /** Stream-level failure. The caller should fall back to polling. */
  onError?: (err: Error) => void;
}

/**
 * Streams one job's progress until it reaches a terminal state.
 * Returns a function that aborts the stream.
 */
export function streamJob(
  jobId: string,
  handlers: JobStreamHandlers,
  signal?: AbortSignal,
): () => void {
  const controller = new AbortController();
  if (signal) signal.addEventListener("abort", () => controller.abort());

  void (async () => {
    try {
      const res = await fetch(`${BASE}/v1/jobs/${jobId}/events`, {
        headers: { ...authHeaders(), Accept: "text/event-stream" },
        signal: controller.signal,
      });

      // A bad id or another principal's job is a 404 HTTP response here, not an
      // `error` event -- the route authorises before the stream opens.
      if (!res.ok) throw await parseApiError(res);
      if (!res.body) throw new Error("this browser cannot read a streaming response");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line. Anything after the last
        // separator is a partial frame and stays buffered.
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const parsed = parseFrame(frame);
          if (!parsed) continue;
          const { event, data } = parsed;

          if (event === "heartbeat") {
            handlers.onHeartbeat?.(data as Heartbeat);
          } else if (event === "end") {
            handlers.onEnd?.(data as JobOut);
            return;
          } else if (event === "error") {
            const e = data as { code?: string; message?: string };
            throw new Error(e.message ?? "the job stream failed");
          } else {
            // Default event name is "message"; the API always names it "state".
            handlers.onState?.(data as JobOut);
          }
        }
      }
    } catch (err) {
      if ((err as Error)?.name === "AbortError") return; // deliberate teardown
      handlers.onError?.(err as Error);
    }
  })();

  return () => controller.abort();
}

/** Parses one SSE frame into its event name and JSON payload. */
function parseFrame(frame: string): { event: string; data: unknown } | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of frame.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    // One optional leading space after the colon is part of the framing.
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }

  if (dataLines.length === 0) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}
