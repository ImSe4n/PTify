/**
 * Submit a recording.
 *
 * The engine list comes from `GET /v1/engines`, never from a hardcoded array:
 * `available` is a per-deployment fact (ptify needs a 172MB checkpoint that is
 * not bundled), and `notes` is free text because the project documents a single
 * accuracy number as a comparison that would be meaningless.
 */

import { useEffect, useMemo, useState } from "react";

import { ApiError, getEngines, submitJob } from "../api/client";
import type { EngineOut, OutputFormat } from "../api/types";

const ACCEPT = ".mp3,.wav,.m4a,.flac,.ogg,.aiff,.aif";

const FORMATS: { id: OutputFormat; label: string; hint: string }[] = [
  { id: "midi", label: "MIDI", hint: "notes + pedal as CC64" },
  { id: "musicxml", label: "MusicXML", hint: "opens in notation apps" },
  { id: "pdf", label: "PDF", hint: "engraved score" },
  { id: "svg", label: "SVG", hint: "one file per page" },
  { id: "json", label: "JSON", hint: "the piano-roll payload" },
];

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function UploadScreen({ onSubmitted }: { onSubmitted: (jobId: string) => void }) {
  const [engines, setEngines] = useState<EngineOut[]>([]);
  const [engine, setEngine] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [formats, setFormats] = useState<Set<OutputFormat>>(
    () => new Set<OutputFormat>(["midi", "musicxml", "pdf"]),
  );
  const [title, setTitle] = useState("");
  const [composer, setComposer] = useState("");
  const [tempo, setTempo] = useState("");
  const [beatsPerBar, setBeatsPerBar] = useState("4");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEngines()
      .then((list) => {
        setEngines(list);
        const preferred = list.find((e) => e.default && e.available) ?? list.find((e) => e.available);
        if (preferred) setEngine(preferred.name);
      })
      .catch((e: ApiError) => setError(e.message));
  }, []);

  const canStart = useMemo(
    () => !!file && !!engine && formats.size > 0 && !pending,
    [file, engine, formats, pending],
  );

  function toggleFormat(id: OutputFormat) {
    setFormats((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function start() {
    if (!file || !engine || !canStart) return;
    setPending(true);
    setError(null);
    try {
      const res = await submitJob({
        file,
        engine,
        formats: [...formats],
        tempo: tempo.trim() ? Number(tempo) : null,
        beatsPerBar: Number(beatsPerBar) || 4,
        title: title.trim(),
        composer: composer.trim(),
      });
      onSubmitted(res.job_id);
    } catch (err) {
      const e = err as ApiError;
      // Both limits are 429 and mean different things, so branch on the code.
      setError(
        e.isTooManyJobs
          ? `${e.message} (this is a per-account limit, not a failure)`
          : e.message,
      );
      setPending(false);
    }
  }

  return (
    <div className="page upload fade-in">
      <header className="screen-head">
        <p className="eyebrow">new transcription</p>
        <h1 className="h1">Upload a recording.</h1>
        <p className="lede">
          A single piano performance works best. Transcription runs on a worker
          and takes a few minutes — roughly twice the length of the recording.
        </p>
      </header>

      <div className="upload-grid">
        <div className="upload-main">
          <label
            className={`drop${dragging ? " is-dragging" : ""}${file ? " has-file" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const f = e.dataTransfer.files?.[0];
              if (f) setFile(f);
            }}
          >
            <input
              type="file"
              accept={ACCEPT}
              className="sr-only"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <span className="drop-mark serif" aria-hidden="true">
              ♪
            </span>
            {file ? (
              <>
                <span className="drop-title serif">{file.name}</span>
                <span className="drop-sub mono">{humanSize(file.size)} · ready</span>
              </>
            ) : (
              <>
                <span className="drop-title serif">Drop a recording</span>
                <span className="drop-sub">or click to browse</span>
              </>
            )}
            <span className="drop-formats mono">
              mp3 · wav · m4a · flac · ogg · aiff — up to 15 minutes
            </span>
          </label>

          <section className="upload-section">
            <h2 className="section-title serif">Engine</h2>
            <p className="section-hint">They trade off. Pick one.</p>

            <div className="engine-list">
              {engines.map((e) => {
                const selected = engine === e.name;
                return (
                  <button
                    key={e.name}
                    type="button"
                    disabled={!e.available}
                    onClick={() => e.available && setEngine(e.name)}
                    className={`engine${selected ? " is-selected" : ""}${
                      e.available ? "" : " is-unavailable"
                    }`}
                  >
                    <span className="engine-head">
                      <span className="engine-radio" aria-hidden="true" />
                      <span className="engine-name serif">{e.name}</span>
                      {e.default && <span className="engine-badge mono">default</span>}
                      {!e.available && (
                        <span className="engine-badge mono is-warn">checkpoint missing</span>
                      )}
                      {e.supports_pedal && (
                        <span className="engine-tag mono">pedal + velocity</span>
                      )}
                    </span>
                    <span className="engine-notes">{e.notes}</span>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="upload-section">
            <h2 className="section-title serif">Output formats</h2>
            <div className="format-row">
              {FORMATS.map((f) => {
                const on = formats.has(f.id);
                return (
                  <button
                    key={f.id}
                    type="button"
                    className={`chip${on ? " is-on" : ""}`}
                    onClick={() => toggleFormat(f.id)}
                    aria-pressed={on}
                  >
                    <span className="chip-box" aria-hidden="true">
                      {on ? "✓" : ""}
                    </span>
                    <span className="chip-label">{f.label}</span>
                    <span className="chip-hint">{f.hint}</span>
                  </button>
                );
              })}
            </div>
          </section>
        </div>

        <aside className="upload-aside">
          <h2 className="section-title serif">
            Score details <span className="section-optional">optional</span>
          </h2>

          <label className="field">
            <span className="field-label">Title</span>
            <input
              className="field-input serif"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Sonata in D"
            />
          </label>

          <label className="field">
            <span className="field-label">Composer</span>
            <input
              className="field-input serif"
              value={composer}
              onChange={(e) => setComposer(e.target.value)}
              placeholder="D. Scarlatti"
            />
          </label>

          <div className="field-pair">
            <label className="field">
              <span className="field-label">Tempo</span>
              <input
                className="field-input mono"
                value={tempo}
                onChange={(e) => setTempo(e.target.value)}
                placeholder="auto"
                inputMode="decimal"
              />
            </label>
            <label className="field">
              <span className="field-label">Beats / bar</span>
              <input
                className="field-input mono"
                value={beatsPerBar}
                onChange={(e) => setBeatsPerBar(e.target.value)}
                inputMode="numeric"
              />
            </label>
          </div>

          <p className="aside-note">
            Leave tempo blank to beat-track the audio. A fixed tempo skips
            detection entirely.
          </p>

          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}

          <button className="btn upload-start" onClick={start} disabled={!canStart}>
            {pending ? "Uploading…" : "Start transcription"}
          </button>
          <p className="aside-note">
            {!file
              ? "Choose a recording to continue."
              : formats.size === 0
                ? "Pick at least one output format."
                : "Runs on a worker — you can leave this page."}
          </p>
        </aside>
      </div>
    </div>
  );
}
