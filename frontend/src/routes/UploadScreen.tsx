/**
 * Submit a recording, as a sequence rather than a wall.
 *
 * WHY THIS IS STEPPED
 *
 * Every control used to be on one screen at once: a dropzone, three engine
 * cards, five format chips, four metadata fields and a start button. That is
 * six decisions presented simultaneously, five of which have a sensible default
 * and only one of which is actually required -- so the page asked the user to
 * read all of it to discover that most of it was optional.
 *
 * The flow now asks one thing at a time:
 *
 *   1. RECORDING  the only required input. Advances by itself once a file is
 *                 chosen, because there is nothing else to decide here.
 *   2. OUTPUT     engine and formats. Both are pre-filled, so this step can be
 *                 passed through without touching anything.
 *   3. DETAILS    title, composer, tempo -- entirely optional, and last, which
 *                 is where optional things belong.
 *
 * Each step is a real URL (`#/new/output`), so Back works, a refresh keeps its
 * place, and the browser's own history is the undo. That falls out of the
 * router from 7a rather than needing wizard state.
 *
 * The engine list still comes from `GET /v1/engines`, never from a hardcoded
 * array: `available` is a per-deployment fact (ptify needs a 172MB checkpoint
 * that is not bundled), and `notes` is free text because the project documents
 * a single accuracy number as a comparison that would be meaningless.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { ApiError, getEngines, submitJob } from "../api/client";
import type { EngineOut, OutputFormat } from "../api/types";
import { Reveal } from "../ui/Reveal";
import { navigate, type UploadStep } from "../router";
import { rememberTitle } from "../titles";

const ACCEPT = ".mp3,.wav,.m4a,.flac,.ogg,.aiff,.aif";

const FORMATS: { id: OutputFormat; label: string; hint: string }[] = [
  { id: "midi", label: "MIDI", hint: "notes + pedal as CC64" },
  { id: "musicxml", label: "MusicXML", hint: "opens in notation apps" },
  { id: "pdf", label: "PDF", hint: "engraved score" },
  { id: "svg", label: "SVG", hint: "one file per page" },
  { id: "json", label: "JSON", hint: "the piano-roll payload" },
];

const STEPS: { id: UploadStep; label: string }[] = [
  { id: "file", label: "Recording" },
  { id: "output", label: "Output" },
  { id: "details", label: "Details" },
];

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function UploadScreen({ step }: { step: UploadStep }) {
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

  // A File cannot survive a reload, so a deep link to a later step with nothing
  // chosen has to fall back rather than show a flow with no subject.
  useEffect(() => {
    if (step !== "file" && !file) navigate({ screen: "upload", step: "file" }, { replace: true });
  }, [step, file]);

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

  // Choosing a file IS the decision this step exists for, so it advances on its
  // own rather than making the user confirm what they just did.
  const advanceTimer = useRef<number | null>(null);
  function chooseFile(f: File | null) {
    setFile(f);
    if (!f) return;
    if (advanceTimer.current) clearTimeout(advanceTimer.current);
    // Long enough to see the filename land, short enough not to feel held up.
    advanceTimer.current = window.setTimeout(
      () => navigate({ screen: "upload", step: "output" }),
      620,
    );
  }
  useEffect(() => () => {
    if (advanceTimer.current) clearTimeout(advanceTimer.current);
  }, []);

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
      // The server does not echo the title back on JobOut, so keep it for this
      // tab. See titles.ts -- the durable fix is an api/models.py change.
      rememberTitle(res.job_id, title);
      navigate({ screen: "job", jobId: res.job_id });
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

  const index = STEPS.findIndex((s) => s.id === step);
  const go = (to: UploadStep) => navigate({ screen: "upload", step: to });

  return (
    <div className="page upload">
      <header className="screen-head upload-head">
        <p className="eyebrow">new transcription</p>
        <Reveal as="h1" className="h1" key={step}>
          {step === "file"
            ? "Choose a recording."
            : step === "output"
              ? "What should come back?"
              : "Anything to name it?"}
        </Reveal>
        <p className="lede">
          {step === "file"
            ? "A single piano performance works best. Transcription runs on a worker and takes roughly twice the length of the recording."
            : step === "output"
              ? "Both of these are already set. Change them only if you want something specific."
              : "All optional. They are printed on the score, and skipping them costs nothing."}
        </p>
      </header>

      <StepBar steps={STEPS} index={index} file={file} onGo={go} />

      {/* Keyed on the step so each panel mounts fresh and animates in. */}
      <div className="step-panel" key={step}>
        {step === "file" && (
          <section className="step-file">
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
                if (f) chooseFile(f);
              }}
            >
              <input
                type="file"
                accept={ACCEPT}
                className="sr-only"
                onChange={(e) => chooseFile(e.target.files?.[0] ?? null)}
              />
              <span className="drop-mark" aria-hidden="true">
                ♪
              </span>
              {file ? (
                <>
                  <span className="drop-title">{file.name}</span>
                  <span className="drop-sub mono">{humanSize(file.size)} · ready</span>
                </>
              ) : (
                <>
                  <span className="drop-title">Drop a recording</span>
                  <span className="drop-sub">or click to browse</span>
                </>
              )}
              <span className="drop-formats mono">
                mp3 · wav · m4a · flac · ogg · aiff · up to 15 minutes
              </span>
            </label>

            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
          </section>
        )}

        {step === "output" && (
          <section className="step-output">
            <div className="step-block">
              <h2 className="section-title">Engine</h2>
              <p className="section-hint">They trade off. Pick one.</p>

              <div className="engine-list">
                {engines.map((e, i) => {
                  const selected = engine === e.name;
                  return (
                    <button
                      key={e.name}
                      type="button"
                      disabled={!e.available}
                      style={{ "--i": i } as React.CSSProperties}
                      onClick={() => e.available && setEngine(e.name)}
                      className={`engine${selected ? " is-selected" : ""}${
                        e.available ? "" : " is-unavailable"
                      }`}
                    >
                      <span className="engine-head">
                        <span className="engine-radio" aria-hidden="true" />
                        <span className="engine-name">{e.name}</span>
                        {e.default && <span className="engine-badge mono">default</span>}
                        {!e.available && (
                          /* WHY TWO MESSAGES. `available` is false for two
                             unrelated reasons and `requires_weights` is the
                             field that tells them apart: ptify needs a 172MB
                             checkpoint that is not bundled, while remote needs
                             PTIFY_REMOTE_URL set. Printing "checkpoint
                             missing" for both sent a real debugging session
                             looking at the GPU host and its checkpoint, when
                             the actual cause was an unset env var. */
                          <span className="engine-badge mono is-warn">
                            {e.requires_weights ? "checkpoint missing" : "not configured"}
                          </span>
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
            </div>

            <div className="step-block">
              <h2 className="section-title">Output formats</h2>
              <p className="section-hint">
                MIDI, MusicXML and an engraved PDF are on by default.
              </p>
              <div className="format-row">
                {FORMATS.map((f, i) => {
                  const on = formats.has(f.id);
                  return (
                    <button
                      key={f.id}
                      type="button"
                      style={{ "--i": i } as React.CSSProperties}
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
            </div>
          </section>
        )}

        {step === "details" && (
          <section className="step-details">
            <div className="details-card">
              <label className="field">
                <span className="field-label">Title</span>
                <input
                  className="field-input"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Sonata in D"
                />
              </label>

              <label className="field">
                <span className="field-label">Composer</span>
                <input
                  className="field-input"
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
            </div>

            <Summary
              file={file}
              engine={engine}
              formats={formats}
              error={error}
              pending={pending}
              canStart={canStart}
              onStart={start}
            />
          </section>
        )}
      </div>

      {step !== "details" && (
        <footer className="step-nav">
          {index > 0 ? (
            <button className="btn btn-ghost" onClick={() => go(STEPS[index - 1].id)}>
              Back
            </button>
          ) : (
            <span />
          )}
          <button
            className="btn"
            onClick={() => go(STEPS[index + 1].id)}
            disabled={!file || (step === "output" && formats.size === 0)}
          >
            {step === "file" ? "Continue" : "Continue"}
          </button>
        </footer>
      )}
    </div>
  );
}

/** The progress rail. Steps already passed are clickable; later ones are not. */
function StepBar({
  steps,
  index,
  file,
  onGo,
}: {
  steps: { id: UploadStep; label: string }[];
  index: number;
  file: File | null;
  onGo: (s: UploadStep) => void;
}) {
  return (
    <ol className="step-bar" aria-label="Progress">
      {steps.map((s, i) => {
        const state = i < index ? "is-done" : i === index ? "is-current" : "is-todo";
        const reachable = i <= index || !!file;
        return (
          <li key={s.id} className={`step-item ${state}`}>
            <button
              className="step-dot"
              onClick={() => reachable && onGo(s.id)}
              disabled={!reachable}
              aria-current={i === index ? "step" : undefined}
            >
              <span className="step-num mono">{i < index ? "✓" : i + 1}</span>
              <span className="step-label">{s.label}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

/** What is about to be sent, restated before the one irreversible click. */
function Summary({
  file,
  engine,
  formats,
  error,
  pending,
  canStart,
  onStart,
}: {
  file: File | null;
  engine: string | null;
  formats: Set<OutputFormat>;
  error: string | null;
  pending: boolean;
  canStart: boolean;
  onStart: () => void;
}) {
  return (
    <aside className="summary-card">
      <h2 className="section-title">Ready to transcribe</h2>

      <dl className="summary-list">
        <div>
          <dt>Recording</dt>
          <dd className="summary-file">{file?.name ?? "—"}</dd>
        </div>
        <div>
          <dt>Engine</dt>
          <dd className="mono">{engine ?? "—"}</dd>
        </div>
        <div>
          <dt>Formats</dt>
          <dd>
            <span className="summary-chips">
              {[...formats].map((f) => (
                <span className="pill" key={f}>
                  {f}
                </span>
              ))}
            </span>
          </dd>
        </div>
      </dl>

      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}

      <button className="btn btn-lg upload-start" onClick={onStart} disabled={!canStart}>
        {pending ? "Uploading…" : "Start transcription"}
      </button>
      <p className="aside-note">
        {!file
          ? "Choose a recording to continue."
          : formats.size === 0
            ? "Pick at least one output format."
            : "Runs on a worker, so you can leave this page."}
      </p>
    </aside>
  );
}
