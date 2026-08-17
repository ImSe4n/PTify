/**
 * One job, at whichever stage it is in.
 *
 * The URL `#/j/{id}` says WHICH job, never which screen -- because that is a
 * fact about the job's state, and the state changes while the URL does not.
 * So this fetches the job once and picks: running -> Waiting, done -> Result.
 * Deep-link a running job and you get Waiting, then Result, with the URL never
 * changing.
 *
 * Phase 6 made this choice in two places (App's history callback and Waiting's
 * onDone). Here it is made once.
 */

import { useEffect, useState } from "react";

import { ApiError, getJob } from "../api/client";
import type { JobState } from "../api/types";
import { navigate } from "../router";
import { recalledTitle } from "../titles";
import { ResultScreen } from "./ResultScreen";
import { WaitingScreen } from "./WaitingScreen";

export function JobScreen({ jobId }: { jobId: string }) {
  const [state, setState] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Only the FIRST state matters here. Once Waiting is mounted it owns the SSE
  // stream and tells us when it finishes -- re-fetching would race it.
  useEffect(() => {
    let cancelled = false;
    setState(null);
    setError(null);

    getJob(jobId)
      .then((job) => {
        if (!cancelled) setState(job.state);
      })
      .catch((e: ApiError) => {
        if (!cancelled) setError(e.message);
      });

    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (error) {
    return (
      <div className="page fade-in">
        <header className="screen-head">
          <p className="eyebrow">not found</p>
          <h1 className="h1">That transcription isn’t available.</h1>
          <p className="lede">{error}</p>
        </header>
        <button className="btn btn-ghost" onClick={() => navigate({ screen: "history" })}>
          Your transcriptions
        </button>
      </div>
    );
  }

  if (state === null) {
    return (
      <div className="page">
        <div className="boot-inline" role="status">
          <span className="sr-only">Loading transcription</span>
        </div>
      </div>
    );
  }

  if (state === "succeeded") {
    return (
      <ResultScreen
        jobId={jobId}
        title={recalledTitle(jobId)}
        onOpenSheet={() => navigate({ screen: "sheet", jobId, page: 1 })}
      />
    );
  }

  return (
    <WaitingScreen
      jobId={jobId}
      onDone={() => setState("succeeded")}
      onLeave={() => navigate({ screen: "history" })}
    />
  );
}
