/**
 * The result: piano roll, what was detected, and how much to trust it.
 *
 * This is the dense TOOL screen, so it does not take the display-scale type the
 * marketing-adjacent screens use. Character comes from the typography and from
 * the trust panel, while the layout stays a working instrument.
 *
 * The trust panel is the product. `pedalled_fraction` is the share of notes
 * whose printed LENGTH is interpolation rather than measurement, and a heavily
 * pedalled piece can be 91%. Saying so plainly is the thing that distinguishes
 * this from a transcriber that prints confident nonsense.
 */

import { useEffect, useMemo, useState } from "react";

import { ApiError, downloadArtifact, getJob, getResultJson } from "../api/client";
import type { JobOut, OutputFormat, Summary } from "../api/types";
import { PianoRoll, noteName } from "../roll/PianoRoll";

const DOWNLOADS: { fmt: OutputFormat; label: string; desc: string; file: string }[] = [
  { fmt: "midi", label: "MIDI", desc: "notes + pedal (CC64)", file: "transcription.mid" },
  { fmt: "musicxml", label: "MusicXML", desc: "opens in notation apps", file: "score.musicxml" },
  { fmt: "pdf", label: "PDF", desc: "engraved score", file: "score.pdf" },
  { fmt: "json", label: "JSON", desc: "the piano-roll payload", file: "transcription.json" },
];

function fmtClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function ResultScreen({
  jobId,
  title,
  onOpenSheet,
}: {
  jobId: string;
  /** What the user typed on upload, if this session submitted the job. */
  title?: string | null;
  onOpenSheet: () => void;
}) {
  const [job, setJob] = useState<JobOut | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [explain, setExplain] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [j, s] = await Promise.all([getJob(jobId), getResultJson(jobId)]);
        if (cancelled) return;
        setJob(j);
        setSummary(s);
      } catch (err) {
        if (!cancelled) setError((err as ApiError).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  const facts = useMemo(() => {
    if (!summary) return [];
    const rows: [string, string][] = [
      ["Notes", String(summary.note_count)],
      ["Duration", fmtClock(summary.duration)],
      ["Range", `${noteName(summary.pitch_range[0])}–${noteName(summary.pitch_range[1])}`],
    ];
    if (summary.bpm) rows.push(["Tempo", `${Math.round(summary.bpm)} bpm`]);
    if (summary.time_signature) rows.push(["Metre", summary.time_signature]);
    if (summary.measures) rows.push(["Measures", String(summary.measures)]);
    rows.push(["Pedal spans", String(summary.pedal_count)]);
    return rows;
  }, [summary]);

  if (error) {
    return (
      <div className="page">
        <p className="form-error" role="alert">
          {error}
        </p>
      </div>
    );
  }

  if (!summary || !job) {
    return (
      <div className="page">
        <p className="lede">Loading transcription…</p>
      </div>
    );
  }

  // Zero notes is a SUCCEEDED job with a warning, not a failure.
  if (summary.note_count === 0) {
    return (
      <div className="page fade-in">
        <header className="screen-head">
          <p className="eyebrow">no notes detected</p>
          <h1 className="h1">Nothing came back from this recording.</h1>
          <p className="lede">
            {job.warnings[0] ??
              "The transcription succeeded but found no notes. Is the recording silent or very quiet?"}
          </p>
        </header>
      </div>
    );
  }

  const pedalled = summary.pedalled_fraction;
  const heavy = pedalled != null && pedalled >= 0.5;
  const availableSvg = job.artifacts.svg?.length ?? 0;

  // `JobOut` carries neither the title the user typed nor the original
  // filename -- both live on JobSpec, which is server-side only. Rather than
  // showing a job-id fragment, describe the piece from what the API does
  // return. `title` is passed through from the upload screen when we have it.
  const displayTitle =
    title ??
    (summary.key
      ? `${summary.key.name}${summary.time_signature ? ` · ${summary.time_signature}` : ""}`
      : "Transcription");

  return (
    <div className="result fade-in">
      <header className="result-head">
        <div className="result-title">
          <h1 className="h2">{displayTitle}</h1>
          <span className="mono result-engine">{summary.engine}</span>
        </div>
        <div className="result-head-end">
          {availableSvg > 0 && (
            <button className="btn-ghost btn btn-sm" onClick={onOpenSheet}>
              <span className="serif" aria-hidden="true">
                𝄞
              </span>
              Sheet music
            </button>
          )}
        </div>
      </header>

      <div className="result-body">
        <section className="result-roll">
          <div className="roll-toolbar">
            <span className="mono roll-duration">{fmtClock(summary.duration)}</span>
            <span className="roll-toolbar-spacer" />
            <span className="mono roll-zoom-label">zoom</span>
            <div className="roll-zoom">
              <button
                className="icon-btn"
                onClick={() => setZoom((z) => Math.max(0.35, +(z / 1.4).toFixed(3)))}
                aria-label="Zoom out"
              >
                −
              </button>
              <span className="mono roll-zoom-value">{Math.round(zoom * 100)}%</span>
              <button
                className="icon-btn"
                onClick={() => setZoom((z) => Math.min(4, +(z * 1.4).toFixed(3)))}
                aria-label="Zoom in"
              >
                +
              </button>
            </div>
          </div>

          <PianoRoll summary={summary} zoom={zoom} />

          <div className="roll-legend">
            <span className="legend-item">
              <span className="swatch swatch-measured" aria-hidden="true" />
              measured
            </span>
            <span className="legend-item">
              <span className="swatch swatch-estimated" aria-hidden="true" />
              length estimated under pedal
            </span>
            <span className="legend-item">
              <span className="swatch swatch-pedal" aria-hidden="true" />
              sustain pedal
            </span>
            <span className="roll-toolbar-spacer" />
            <span className="mono legend-note">darker = louder</span>
          </div>
        </section>

        <aside className="result-aside">
          <section className="aside-block">
            <h2 className="section-title serif">Detected</h2>
            <dl className="fact-list">
              {facts.map(([label, value]) => (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd className="mono">{value}</dd>
                </div>
              ))}
            </dl>
          </section>

          {/* The trust block is deliberately heavier than the rest: a bordered,
              filled panel rather than another flat list. Importance shows. */}
          <section className="aside-block trust">
            <div className="trust-head">
              <h2 className="section-title serif">Confidence</h2>
              <button className="btn-link" onClick={() => setExplain((v) => !v)}>
                {explain ? "Hide" : "What do these mean?"}
              </button>
            </div>

            <div className="trust-item">
              <div className="trust-row">
                <span className="trust-label">Key signature</span>
                <span className={`mono trust-pill ${summary.key ? "is-good" : "is-warn"}`}>
                  {summary.key ? `${Math.round(summary.key.confidence * 100)}% conf.` : "unclear"}
                </span>
              </div>
              <p className="trust-value serif">{summary.key?.name ?? "No key signature"}</p>
              <p className="trust-caption serif">
                {summary.key
                  ? "Printed on the score."
                  : "Too chromatic to call — printing no signature is the honest answer."}
              </p>
              {explain && (
                <p className="trust-detail">
                  Krumhansl-Schmuckler over the note content. A wrong key misspells
                  every accidental, so only confident readings are printed.
                </p>
              )}
            </div>

            {pedalled != null && (
              <div className="trust-item">
                <div className="trust-row">
                  <span className="trust-label">Rhythm reliability</span>
                  <span className={`mono trust-pill ${heavy ? "is-warn" : "is-good"}`}>
                    {Math.round(pedalled * 100)}% estimated
                  </span>
                </div>
                <div className="trust-bar">
                  <span
                    className={`trust-bar-fill${heavy ? " is-warn" : ""}`}
                    style={{ width: `${Math.round(pedalled * 100)}%` }}
                  />
                </div>
                <p className="trust-caption serif">
                  {heavy
                    ? "Most note lengths are estimated under pedal. Trust the onsets, not the durations."
                    : "Most note lengths were measured directly. These rhythms are reliable."}
                </p>
                {explain && (
                  <p className="trust-detail">
                    Share of notes whose release fell under sustain, where the
                    printed length is interpolation rather than measurement.
                    Onsets are reliable regardless.
                  </p>
                )}
              </div>
            )}

            {(summary.trills != null || summary.staccato != null) && (
              <div className="trust-item">
                <div className="trust-row">
                  <span className="trust-label">Notation markings</span>
                  <span className="mono trust-pill">
                    {summary.trills ?? 0} trills · {summary.staccato ?? 0} staccato
                  </span>
                </div>
                <p className="trust-caption serif">Detected conservatively.</p>
                {explain && (
                  <p className="trust-detail">
                    The system under-reports rather than inventing symbols nobody
                    played — a missing mark still leaves the notes readable.
                  </p>
                )}
              </div>
            )}
          </section>

          <section className="aside-block">
            <h2 className="section-title serif">Download</h2>
            <ul className="download-list">
              {DOWNLOADS.filter(
                (d) => d.fmt === "json" || (job.artifacts[d.fmt]?.length ?? 0) > 0,
              ).map((d) => (
                <li key={d.fmt}>
                  <button
                    className="download"
                    onClick={() =>
                      downloadArtifact(jobId, d.fmt, d.file).catch((e: ApiError) =>
                        setError(e.message),
                      )
                    }
                  >
                    <span className="mono download-fmt">{d.label}</span>
                    <span className="download-desc">{d.desc}</span>
                  </button>
                </li>
              ))}
            </ul>
            {job.warnings.length > 0 && (
              <p className="aside-note">{job.warnings.join(" ")}</p>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}

