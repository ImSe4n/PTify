/**
 * Past transcriptions, from `GET /v1/jobs` (scoped to the caller, newest first).
 *
 * Two API facts shape this screen:
 *  - Finished jobs and their artifacts expire after PTIFY_JOB_TTL_SECONDS
 *    (1 hour by default), so a row that worked can later 404. The header says
 *    so rather than letting it look like data loss.
 *  - `JobOut` carries no filename, so a row is identified by what the API does
 *    return: engine, state, note count, and when it was created.
 */

import { useEffect, useState } from "react";

import { ApiError, listJobs } from "../api/client";
import type { JobOut, Summary } from "../api/types";
import { Reveal } from "../ui/Reveal";
import { navigate } from "../router";

function relative(epochSeconds: number): string {
  const diff = Date.now() / 1000 - epochSeconds;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  return new Date(epochSeconds * 1000).toLocaleDateString();
}

function describe(job: JobOut): string {
  const r = job.result as Partial<Summary>;
  if (job.state === "failed") return job.error_code ?? "failed";
  if (job.state === "cancelled") return "cancelled at a stage boundary";
  if (job.state !== "succeeded") return job.stage;

  const bits: string[] = [];
  if (r.note_count != null) bits.push(`${r.note_count.toLocaleString()} notes`);
  if (r.key?.name) bits.push(r.key.name);
  else if (r.key === null) bits.push("no key signature");
  if (r.pedalled_fraction != null) {
    bits.push(`${Math.round(r.pedalled_fraction * 100)}% estimated`);
  }
  return bits.join(" · ") || "done";
}

export function HistoryScreen() {
  const [jobs, setJobs] = useState<JobOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listJobs()
      .then(setJobs)
      .catch((e: ApiError) => setError(e.message));
  }, []);

  return (
    <div className="page history enter-stagger">
      <header className="screen-head history-head" style={{ "--i": 0 } as React.CSSProperties}>
        <div>
          <p className="eyebrow">your work</p>
          <Reveal as="h1" className="h1">Transcriptions.</Reveal>
          <p className="lede">
            Finished jobs and their files expire an hour after they complete.
          </p>
        </div>
        <button className="btn" onClick={() => navigate({ screen: "upload", step: "file" })}>
          New transcription
        </button>
      </header>

      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}

      {jobs && jobs.length === 0 && (
        <div className="empty" style={{ "--i": 1 } as React.CSSProperties}>
          <p className="h2 serif">Nothing here yet.</p>
          <p className="prose">
            Upload a piano recording and it will show up here — with the notes,
            the score, and an honest account of which rhythms were measured.
          </p>
        </div>
      )}

      {jobs && jobs.length > 0 && (
        <ul className="job-list" style={{ "--i": 1 } as React.CSSProperties}>
          {jobs.map((job) => {
            const openable = job.state === "succeeded" || job.state === "running";
            return (
              <li key={job.job_id}>
                <button
                  className={`job-row${openable ? "" : " is-inert"}`}
                  // One route for both states: JobScreen reads the job and
                  // picks Waiting or Result. The URL says which job, not which
                  // screen.
                  onClick={() => openable && navigate({ screen: "job", jobId: job.job_id })}
                  disabled={!openable}
                >
                  <span className="job-when">
                    <span className="mono job-id">{job.job_id.slice(0, 8)}</span>
                    <span className="job-time">{relative(job.created_at)}</span>
                  </span>
                  <span className="mono job-engine">{job.engine}</span>
                  <span className={`job-state is-${job.state}`}>{job.state}</span>
                  <span className="job-desc">{describe(job)}</span>
                  <span className="job-arrow" aria-hidden="true">
                    {openable ? "→" : ""}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
