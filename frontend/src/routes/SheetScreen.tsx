/**
 * The engraved score.
 *
 * A thin viewer by design -- Phase 6 shows what the backend already renders and
 * does not re-engrave anything in the browser. Verovio runs server-side (and is
 * not thread-safe, which is why it is funnelled onto one thread there).
 *
 * SVG pages are fetched rather than linked because the URL needs an
 * Authorization header. `job.artifacts.svg` is a list of filenames, one per
 * page, in page order -- the page count comes from its length, and `?page=` is
 * 1-indexed (verified in Phase 5.5: page 9 of a 5-page score is a 404).
 */

import { useEffect, useState } from "react";

import { ApiError, artifactUrl, authHeaders, downloadArtifact, getJob } from "../api/client";
import type { JobOut } from "../api/types";

export function SheetScreen({ jobId, onBack }: { jobId: string; onBack: () => void }) {
  const [job, setJob] = useState<JobOut | null>(null);
  const [page, setPage] = useState(1);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getJob(jobId)
      .then(setJob)
      .catch((e: ApiError) => setError(e.message));
  }, [jobId]);

  const pageCount = job?.artifacts.svg?.length ?? 0;

  useEffect(() => {
    if (!job || pageCount === 0) return;
    let cancelled = false;
    setSvg(null);

    (async () => {
      try {
        const res = await fetch(artifactUrl(jobId, "svg", page), { headers: authHeaders() });
        if (!res.ok) throw new Error(`could not load page ${page}`);
        const text = await res.text();
        if (!cancelled) setSvg(text);
      } catch (err) {
        if (!cancelled) setError((err as Error).message);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [jobId, job, page, pageCount]);

  if (error) {
    return (
      <div className="page">
        <p className="form-error" role="alert">
          {error}
        </p>
        <button className="btn-ghost btn" onClick={onBack}>
          Back
        </button>
      </div>
    );
  }

  if (job && pageCount === 0) {
    return (
      <div className="page fade-in">
        <header className="screen-head">
          <p className="eyebrow">no score</p>
          <h1 className="h1">This job produced no engraved pages.</h1>
          <p className="lede">
            Request the SVG or PDF format when submitting to get a readable score.
          </p>
        </header>
        <button className="btn-ghost btn" onClick={onBack}>
          Back to the roll
        </button>
      </div>
    );
  }

  return (
    <div className="sheet fade-in">
      <header className="sheet-bar">
        <button className="btn-link" onClick={onBack}>
          ← Roll
        </button>
        <span className="sheet-bar-spacer" />
        <div className="sheet-pager">
          <button
            className="icon-btn"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            aria-label="Previous page"
          >
            ‹
          </button>
          <span className="mono sheet-page-label">
            {page} / {pageCount || "—"}
          </span>
          <button
            className="icon-btn"
            onClick={() => setPage((p) => Math.min(pageCount, p + 1))}
            disabled={page >= pageCount}
            aria-label="Next page"
          >
            ›
          </button>
        </div>
        {(job?.artifacts.pdf?.length ?? 0) > 0 && (
          <button
            className="btn btn-sm"
            onClick={() =>
              downloadArtifact(jobId, "pdf", "score.pdf").catch((e: ApiError) =>
                setError(e.message),
              )
            }
          >
            Download PDF
          </button>
        )}
      </header>

      <div className="sheet-desk">
        <div className="sheet-page">
          {svg ? (
            // The SVG comes from our own Verovio render, not from user input.
            <div className="sheet-svg" dangerouslySetInnerHTML={{ __html: svg }} />
          ) : (
            <div className="sheet-loading">
              <span className="sr-only">Loading page {page}</span>
            </div>
          )}
        </div>
        <p className="sheet-caption">
          Engraved from the transcription. Multi-page scores are normal — SVG
          exports one file per page.
        </p>
      </div>
    </div>
  );
}
