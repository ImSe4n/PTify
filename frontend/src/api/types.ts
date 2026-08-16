/**
 * Types for the PTify HTTP API.
 *
 * Hand-written from `api/models.py` rather than generated, and CHECKED against
 * a live server during Phase 5.5 -- the shapes below are what the wire actually
 * carried, not what a schema promised.
 *
 * The one that matters: `api/models.py: TranscriptionOut` documents the result
 * payload but is never wired as a `response_model`, so the real shape is what
 * `pipeline._summarise` emits (api/pipeline.py:306-328). `Summary` below is
 * that dict, verified on a real Scarlatti transcription.
 */

export type JobState =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export const TERMINAL_STATES: readonly JobState[] = [
  "succeeded",
  "failed",
  "cancelled",
];

export function isTerminal(state: JobState): boolean {
  return TERMINAL_STATES.includes(state);
}

export type OutputFormat = "midi" | "json" | "musicxml" | "pdf" | "svg";

export interface Note {
  pitch: number; // MIDI note number, 21-108
  onset: number; // seconds
  offset: number; // seconds
  /**
   * RAW MIDI velocity, 0-127 -- deliberately NOT normalised. `api/models.py`
   * keeps one convention end to end because normalising velocities is a
   * documented trap upstream (mir_eval returns 1.0 for everything).
   */
  velocity: number;
}

export interface Pedal {
  onset: number;
  offset: number;
}

export interface DetectedKey {
  name: string; // e.g. "C major"
  confidence: number; // Krumhansl-Schmuckler correlation
  margin: number; // gap to the runner-up
}

/**
 * `GET /v1/jobs/{id}/result/json`, and the same object as `JobOut.result`.
 *
 * The notation block is ABSENT (not null) unless a notation format was
 * requested -- see `wants_notation` in api/pipeline.py:122-142. That is why
 * those fields are optional rather than nullable.
 */
export interface Summary {
  engine: string;
  duration: number;
  note_count: number;
  pedal_count: number;
  pitch_range: [number, number];
  notes: Note[];
  pedals: Pedal[];

  /**
   * Share of notes whose release fell under sustain, 0.0-1.0 -- the score's
   * health metric. Measured 0.09 on Scarlatti and up to 0.91 on a Schubert
   * impromptu, where 91% of printed durations are interpolation rather than
   * measurement. The UI must say so; this is the product's honesty claim.
   */
  pedalled_fraction?: number;
  bpm?: number;
  measures?: number;
  time_signature?: string;
  trills?: number;
  staccato?: number;

  /**
   * `null` means "print NO key signature" -- an honest refusal, not "unknown".
   * A wrong key misspells every accidental in the piece, so a weak reading is
   * deliberately not printed (api/models.py:64-69).
   */
  key?: DetectedKey | null;
}

export interface JobOut {
  job_id: string;
  state: JobState;
  /** 0.0-1.0, and COARSE. See the note on indeterminate rendering in sse.ts. */
  progress: number;
  stage: string;
  /** Seconds. The only signal that moves during ByteDance's silent span. */
  elapsed: number;
  engine: string;
  formats: string[];
  created_at: number; // unix epoch seconds
  started_at: number | null;
  finished_at: number | null;
  error_code: string | null;
  error_message: string | null;
  /** format -> filenames. `json` is deliberately an empty list. */
  artifacts: Partial<Record<OutputFormat, string[]>>;
  result: Summary | Record<string, never>;
  warnings: string[];
}

export interface JobAccepted {
  job_id: string;
  state: JobState;
}

export interface EngineOut {
  name: string;
  supports_pedal: boolean;
  native_sample_rate: number;
  default: boolean;
  /** False when this deployment cannot serve it -- grey the option out. */
  available: boolean;
  requires_weights: boolean;
  /** Free text. There is deliberately no single accuracy number. */
  notes: string;
}

export interface TokenOut {
  access_token: string;
  token_type: string;
  expires_in: number;
  user_id: string;
  email: string;
}

export interface MeOut {
  id: string; // "user:<uuid>" | "key:<digest>" | "anonymous"
  kind: "anonymous" | "api_key" | "user";
  email: string | null;
}

export interface HealthOut {
  status: string;
  queue: string;
  workers: number;
  /** Only reflects the SHARED-KEY path, not accounts. Verified in 5.5. */
  auth_enabled: boolean;
  jobs_tracked: number;
}

/** Job submission options. Sent as multipart/form-data, never JSON. */
export interface SubmitOptions {
  file: File;
  engine?: string;
  formats: OutputFormat[];
  tempo?: number | null;
  beatsPerBar?: number;
  title?: string;
  composer?: string;
}
