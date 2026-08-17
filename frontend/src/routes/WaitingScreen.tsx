/**
 * Job progress.
 *
 * THE WHOLE DESIGN OF THIS SCREEN COMES FROM ONE MEASURED FACT.
 *
 * Phase 5.5 watched a real 67-second recording through this endpoint. `progress`
 * sat at exactly 0.09 for ~160 seconds while heartbeats arrived every ~10s with
 * a climbing `elapsed`, then jumped straight to 0.92. ByteDance reports nothing
 * during inference, and the API refuses to interpolate a percentage it did not
 * measure (api/events.py:25-33).
 *
 * So: when the stage is one the engine does not report through, this shows an
 * INDETERMINATE sweep and the honest elapsed clock -- never a progress bar
 * frozen at 9%, which reads as a hang, and never a fake percentage.
 */

import { useEffect, useRef, useState } from "react";

import { ApiError, cancelJob, getJob } from "../api/client";
import { streamJob } from "../api/sse";
import type { JobOut } from "../api/types";
import { isTerminal } from "../api/types";
import { fmtClock } from "../ui/format";

/** Stages where the engine genuinely reports no sub-progress. */
const SILENT_STAGES = ["transcribing", "loading model"];

function isSilent(stage: string): boolean {
  return SILENT_STAGES.some((s) => stage.startsWith(s));
}

interface Props {
  jobId: string;
  onDone: () => void;
  onLeave: () => void;
}

export function WaitingScreen({ jobId, onDone, onLeave }: Props) {
  const [job, setJob] = useState<JobOut | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const doneRef = useRef(false);

  useEffect(() => {
    doneRef.current = false;

    const settle = (j: JobOut) => {
      setJob(j);
      setElapsed(j.elapsed);
      if (isTerminal(j.state) && !doneRef.current) {
        doneRef.current = true;
        if (j.state === "succeeded") setTimeout(onDone, 450);
      }
    };

    const stop = streamJob(
      jobId,
      {
        onState: settle,
        onHeartbeat: (b) => {
          setElapsed(b.elapsed);
          setJob((prev) =>
            prev ? { ...prev, progress: b.progress, stage: b.stage, state: b.state } : prev,
          );
        },
        onEnd: settle,
        onError: (err) => {
          // The stream has no reconnect/Last-Event-ID support, so fall back to
          // polling rather than leaving the screen dead.
          setError(err.message);
          const poll = setInterval(async () => {
            try {
              const j = await getJob(jobId);
              settle(j);
              if (isTerminal(j.state)) clearInterval(poll);
            } catch {
              clearInterval(poll);
            }
          }, 2000);
        },
      },
    );

    return stop;
  }, [jobId, onDone]);

  // A local clock so elapsed time moves between heartbeats. It is reconciled to
  // the server's value on every event, so it can drift by at most one interval.
  useEffect(() => {
    if (job && isTerminal(job.state)) return;
    const t = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(t);
  }, [job]);

  async function onCancel() {
    if (cancelling) return;
    setCancelling(true);
    try {
      const j = await cancelJob(jobId);
      setJob(j);
      if (isTerminal(j.state)) setTimeout(onLeave, 900);
    } catch (err) {
      setError((err as ApiError).message);
      setCancelling(false);
    }
  }

  const stage = job?.stage ?? "queued";
  const state = job?.state ?? "queued";
  const indeterminate = !isTerminal(state) && isSilent(stage);
  const pct = Math.round((job?.progress ?? 0) * 100);
  const failed = state === "failed";
  const cancelled = state === "cancelled";

  return (
    <div className="page waiting fade-in">
      <div className="waiting-inner">
        <p className="eyebrow">{cancelled ? "cancelled" : failed ? "failed" : state}</p>

        <h1 className="h1 waiting-stage">
          {cancelled
            ? "Cancelled"
            : failed
              ? "That did not work"
              : state === "succeeded"
                ? "Done"
                : stage}
        </h1>

        <p className="lede waiting-note">
          {failed
            ? (job?.error_message ?? "The job failed.")
            : cancelled
              ? "Stopped at a stage boundary."
              : indeterminate
                ? "The model reports no progress while it runs. This is expected; the elapsed clock is the real signal."
                : "Working through the pipeline."}
        </p>

        <div
          className="progress"
          role="progressbar"
          aria-valuetext={indeterminate ? "working" : `${pct}%`}
          {...(indeterminate ? {} : { "aria-valuenow": pct, "aria-valuemin": 0, "aria-valuemax": 100 })}
        >
          {indeterminate ? (
            <span className="progress-sweep sweep" />
          ) : (
            <span className="progress-fill" style={{ width: `${pct}%` }} />
          )}
        </div>

        <dl className="waiting-stats">
          <div>
            <dt>Elapsed</dt>
            <dd className="numeral waiting-clock">{fmtClock(elapsed)}</dd>
          </div>
          <div>
            <dt>Engine</dt>
            <dd className="mono">{job?.engine ?? "—"}</dd>
          </div>
          <div>
            <dt>Progress</dt>
            <dd className="mono">{indeterminate ? "—" : `${pct}%`}</dd>
          </div>
        </dl>

        {error && (
          <p className="form-error" role="alert">
            {error}. Falling back to polling.
          </p>
        )}

        <div className="waiting-actions">
          {isTerminal(state) ? (
            <button className="btn-ghost btn" onClick={onLeave}>
              Back to transcriptions
            </button>
          ) : (
            <button className="btn-link" onClick={onCancel} disabled={cancelling}>
              {cancelling ? "Cancelling…" : "Cancel job"}
            </button>
          )}
        </div>

        {!isTerminal(state) && (
          <p className="waiting-fineprint">
            Cancelling takes effect at the next stage boundary. The model cannot
            be interrupted mid-inference, so a running stage finishes first.
          </p>
        )}
      </div>
    </div>
  );
}
