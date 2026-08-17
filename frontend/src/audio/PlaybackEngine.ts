/**
 * Playing a transcription back.
 *
 * WHAT PLAYS, AND WHY IT IS NOT THE MIDI ARTIFACT
 *
 * Playback was built to fetch `/result/midi` -- it is the file the user
 * downloads, so playing it seemed like the honest preview. Driving it in a
 * browser proved otherwise, and the dev drift guard below is what caught it:
 * onsets differed from the roll by up to 0.908s on a 25-second clip.
 *
 * The cause is deliberate backend behaviour, not a bug. When a job requests any
 * notation format, `api/pipeline.py:264-273` exports the QUANTISED notes so the
 * MIDI and the engraved page agree. `Summary.notes` is the raw measurement. So
 * for a job with a score the two artifacts are legitimately in different time
 * bases -- and the roll draws the raw one.
 *
 * Playing the MIDI would therefore put the playhead visibly out of step with
 * the sound on exactly those jobs, and only on those jobs, which is the kind of
 * defect that looks like a mysterious "sometimes it feels laggy". Playing
 * `summary.notes` makes the disagreement structurally impossible: the ear and
 * the eye read one array.
 *
 * `checkDrift` is kept, pointed the other way -- it now watches for the roll
 * and the scheduler falling out of step, which after this change should be
 * exactly never.
 *
 * THE CLOCK IS THE AUDIO CLOCK
 *
 * `position` is derived from `ctx.currentTime`, never from Date.now() (which
 * drifts against the audio hardware over a long piece) and never from a frame
 * counter (which stops in a background tab while the audio keeps going).
 *
 * SCHEDULING IS A LOOKAHEAD LOOP, NOT A TIMER PER NOTE
 *
 * A few thousand setTimeouts is not a schedule. A 50ms interval walks a cursor
 * through the sorted notes and hands the next 1.5 seconds to WebAudio, which
 * has a sample-accurate clock of its own. The window is 1.5s because background
 * tabs clamp setInterval to ~1000ms -- anything shorter drops notes the moment
 * the tab loses focus. requestAnimationFrame cannot be used here at all: it is
 * throttled to zero in a background tab and would starve the scheduler.
 */

import { SplendidGrandPiano } from "smplr";

import type { Summary } from "../api/types";

/** How far ahead of the clock notes are handed to WebAudio. */
const LOOKAHEAD_SEC = 1.5;
/** How often the scheduler wakes. */
const TICK_MS = 50;
/** Hard ceiling on a single note, so a long pedal span cannot become a drone. */
const MAX_NOTE_SEC = 8;
/** Sustain spans shorter than this are pedal noise, not a pedalling gesture. */
const MIN_PEDAL_SEC = 0.05;
/** How long to wait for the sample CDN before falling back to the synth. */
const SAMPLE_LOAD_TIMEOUT_MS = 12000;

/** Reject if `promise` has not settled in time. A hung fetch is not an error. */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error("sample load timed out")), ms),
    ),
  ]);
}

export interface PlayableNote {
  /** MIDI pitch. */
  midi: number;
  /** Seconds from the start of the piece. */
  time: number;
  /** Seconds. Already extended through any sustain span that covers it. */
  duration: number;
  /** 0-127. See the unit note in `notesFromSummary`. */
  velocity: number;
}

export type EngineStatus = "idle" | "loading" | "ready" | "fallback" | "failed";

/**
 * What actually makes the sound.
 *
 * Two implementations: the sampled piano, and a synthesised voice for when its
 * samples cannot be reached.
 */
interface Voice {
  note(midi: number, velocity: number, at: number, duration: number): void;
  sustain(down: boolean): void;
  allOff(): void;
  dispose(): void;
}

const A4 = 440;
const hz = (midi: number) => A4 * 2 ** ((midi - 69) / 12);

/**
 * The offline voice.
 *
 * The sampled piano loads from a third-party CDN, which is the one part of
 * playback this app does not control -- a blocked host, a corporate proxy or an
 * offline laptop would otherwise leave a dead play button. This is deliberately
 * modest: a triangle with two partials and a struck-string decay. It does not
 * pretend to be a Steinway, it just means the transport always works.
 */
class SynthVoice implements Voice {
  private readonly master: GainNode;
  private readonly live = new Set<{ osc: OscillatorNode[]; gain: GainNode }>();

  constructor(private readonly ctx: AudioContext) {
    // A compressor keeps a pedalled twelve-note chord from clipping into fuzz.
    const comp = ctx.createDynamicsCompressor();
    this.master = ctx.createGain();
    this.master.gain.value = 0.5;
    this.master.connect(comp);
    comp.connect(ctx.destination);
  }

  note(midi: number, velocity: number, at: number, duration: number): void {
    // A runaway cursor must never be able to hang the tab.
    if (this.live.size > 64) return;

    const f = hz(midi);
    const gain = this.ctx.createGain();
    gain.connect(this.master);

    // Velocity is roughly perceptual, so a linear map makes everything sound
    // uniformly mezzo-forte.
    const peak = 0.16 * (velocity / 127) ** 1.6;
    const end = at + duration;

    gain.gain.setValueAtTime(0, at);
    // 6ms attack: shorter clicks, longer loses the percussive onset.
    gain.gain.linearRampToValueAtTime(peak, at + 0.006);
    // NEVER ramp exponentially to 0 -- WebAudio throws. Approach it instead.
    gain.gain.exponentialRampToValueAtTime(Math.max(peak * 0.28, 0.0001), at + 0.35);
    gain.gain.exponentialRampToValueAtTime(0.0001, Math.max(end, at + 0.4));
    gain.gain.linearRampToValueAtTime(0, Math.max(end, at + 0.4) + 0.06);

    const specs: [OscillatorType, number, number][] = [
      ["triangle", 1, 1],
      ["sine", 2, 0.32],
      ["sine", 3, 0.1],
    ];
    const oscs: OscillatorNode[] = [];
    for (const [type, mult, level] of specs) {
      const osc = this.ctx.createOscillator();
      osc.type = type;
      osc.frequency.value = f * mult;
      const partial = this.ctx.createGain();
      partial.gain.value = level;
      osc.connect(partial);
      partial.connect(gain);
      osc.start(at);
      osc.stop(Math.max(end, at + 0.4) + 0.1);
      oscs.push(osc);
    }

    const entry = { osc: oscs, gain };
    this.live.add(entry);
    oscs[0].onended = () => {
      this.live.delete(entry);
      try {
        gain.disconnect();
      } catch {
        /* already gone */
      }
    };
  }

  // Sustain needs no action here: extendUnderPedal has already baked each
  // pedalled note's held length into its scheduled duration.
  sustain(): void {}

  allOff(): void {
    const now = this.ctx.currentTime;
    for (const { osc, gain } of this.live) {
      try {
        gain.gain.cancelScheduledValues(now);
        gain.gain.setValueAtTime(gain.gain.value, now);
        // A 20ms fade, because stopping a ringing oscillator outright clicks.
        gain.gain.linearRampToValueAtTime(0, now + 0.02);
        for (const o of osc) o.stop(now + 0.03);
      } catch {
        /* already stopped */
      }
    }
    this.live.clear();
  }

  dispose(): void {
    this.allOff();
    try {
      this.master.disconnect();
    } catch {
      /* already gone */
    }
  }
}

/** The sampled piano. */
class SamplerVoice implements Voice {
  constructor(private readonly piano: SplendidGrandPiano) {}

  note(midi: number, velocity: number, at: number, duration: number): void {
    this.piano.start({ note: midi, velocity, time: at, duration });
  }
  sustain(down: boolean): void {
    this.piano.setCC(64, down ? 127 : 0);
  }
  allOff(): void {
    this.piano.stop();
    this.piano.setCC(64, 0);
  }
  dispose(): void {
    this.piano.dispose();
  }
}

/**
 * The roll payload, ready to schedule.
 *
 * `Summary.notes` carries RAW MIDI velocity 0-127 (deliberately -- see
 * api/types.ts) and smplr wants the same scale, so no conversion happens here.
 * That is worth stating because it is the trap the other way round: @tonejs/midi
 * normalises velocity to 0-1, and feeding those numbers to smplr would produce
 * perfectly plausible playback in which every note is pianissimo.
 *
 * The array is sorted defensively -- the API does not promise an order, and the
 * scheduler's cursor is meaningless without one.
 */
export function notesFromSummary(summary: Summary): PlayableNote[] {
  const notes: PlayableNote[] = summary.notes.map((n) => ({
    midi: n.pitch,
    time: n.onset,
    duration: Math.max(0.02, n.offset - n.onset),
    velocity: Math.max(1, Math.min(127, Math.round(n.velocity))),
  }));

  notes.sort((a, b) => a.time - b.time);
  return extendUnderPedal(notes, summary.pedals);
}

/**
 * Hold notes whose release falls under sustain.
 *
 * This is not an embellishment. The Result screen's whole claim is that under
 * pedal a note's printed LENGTH is interpolation rather than measurement --
 * that is what `pedalled_fraction` counts and why the roll draws those notes
 * with a translucent tail. Releasing exactly at `offset` on a piece that is 91%
 * pedalled would sound staccato and contradict the screen saying so.
 *
 * The same binary search the roll uses lives in roll/pedal.ts, so the ear and
 * the eye cannot disagree about which notes are affected.
 */
function extendUnderPedal(notes: PlayableNote[], pedals: Summary["pedals"]): PlayableNote[] {
  const spans = pedals.filter((p) => p.offset - p.onset >= MIN_PEDAL_SEC);
  if (spans.length === 0) {
    return notes.map((n) => ({ ...n, duration: Math.min(n.duration, MAX_NOTE_SEC) }));
  }

  const starts = spans.map((p) => p.onset);
  return notes.map((n) => {
    const end = n.time + n.duration;

    // Last span starting at or before this release.
    let lo = 0;
    let hi = starts.length - 1;
    let idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (starts[mid] <= end) {
        idx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }

    const held = idx >= 0 && spans[idx].offset >= end ? spans[idx].offset - n.time : n.duration;
    return { ...n, duration: Math.min(Math.max(held, n.duration), MAX_NOTE_SEC) };
  });
}

/**
 * Does the audio agree with the picture?
 *
 * Dev only, and a warning rather than a throw. This found a real defect on its
 * first run -- see the note at the top of this file -- and it is kept because
 * the failure it detects is invisible both on screen and in a type check: two
 * time bases that look right individually and disagree with each other.
 *
 * It should now be silent, because both sides read `summary.notes`. If it ever
 * speaks again, something has reintroduced a second source of truth.
 */
export function checkDrift(played: PlayableNote[], summary: Summary): void {
  if (!import.meta.env.DEV) return;

  const drawn = summary.notes;
  if (played.length !== drawn.length) {
    console.warn(
      `[ptify] playback/roll drift: the scheduler has ${played.length} notes, ` +
        `the roll draws ${drawn.length}. The playhead may not match the sound.`,
    );
    return;
  }
  if (drawn.length === 0) return;

  const sorted = [...drawn].sort((a, b) => a.onset - b.onset);
  let worst = 0;
  for (let i = 0; i < sorted.length; i++) {
    worst = Math.max(worst, Math.abs(played[i].time - sorted[i].onset));
  }
  if (worst > 0.02) {
    console.warn(
      `[ptify] playback/roll drift: onsets differ by up to ${worst.toFixed(3)}s ` +
        `between the scheduler and the roll payload.`,
    );
  }
}

export interface EngineHandlers {
  onEnd?: () => void;
  onStatus?: (status: EngineStatus, detail?: string) => void;
}

export class PlaybackEngine {
  private ctx: AudioContext | null = null;
  private voice: Voice | null = null;
  private notes: PlayableNote[] = [];
  private pedals: Summary["pedals"] = [];

  /** ctx.currentTime at which the current run started. */
  private startedAt = 0;
  /** Seconds into the piece that the current run started from. */
  private offset = 0;
  private playing = false;
  private cursor = 0;
  private timer: number | null = null;
  private disposed = false;

  constructor(
    private readonly summary: Summary,
    private readonly handlers: EngineHandlers = {},
  ) {}

  get duration(): number {
    return this.summary.duration;
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  get position(): number {
    if (!this.playing || !this.ctx) return this.offset;
    return Math.min(this.ctx.currentTime - this.startedAt + this.offset, this.duration);
  }

  private setStatus(status: EngineStatus, detail?: string): void {
    this.handlers.onStatus?.(status, detail);
  }

  /**
   * Build the AudioContext and load the samples.
   *
   * Called from the play handler, never at mount: a context constructed before
   * any user gesture starts `suspended`, which looks exactly like a play button
   * that does nothing.
   */
  private async prepare(): Promise<void> {
    if (this.voice && this.ctx) return;
    this.setStatus("loading");

    const ctx = new AudioContext();
    this.ctx = ctx;

    this.notes = notesFromSummary(this.summary);
    this.pedals = this.summary.pedals;
    checkDrift(this.notes, this.summary);

    // The samples come from a third-party CDN -- the one part of playback that
    // can fail for reasons nothing here controls. Falling back to the synth
    // keeps the transport working; failing outright would turn an unreachable
    // host into a dead button.
    try {
      const piano = new SplendidGrandPiano(ctx);
      await withTimeout(piano.ready, SAMPLE_LOAD_TIMEOUT_MS);
      if (this.disposed) {
        piano.dispose();
        return;
      }
      this.voice = new SamplerVoice(piano);
      this.setStatus("ready");
    } catch {
      if (this.disposed) return;
      this.voice = new SynthVoice(ctx);
      this.setStatus("fallback");
    }
  }

  async play(): Promise<void> {
    if (this.disposed || this.playing) return;

    try {
      await this.prepare();
    } catch (err) {
      this.setStatus("failed", (err as Error).message);
      return;
    }
    if (this.disposed || !this.ctx || !this.voice) return;

    // Resume on EVERY play, not just the first: the OS, another tab taking
    // audio focus, or an iOS interruption can suspend a running context.
    await this.ctx.resume();
    if (this.disposed) return;

    if (this.offset >= this.duration) this.offset = 0;

    this.startedAt = this.ctx.currentTime;
    this.playing = true;
    this.seekCursor(this.offset);
    this.applyPedalState(this.offset);
    this.pump();
    this.timer = window.setInterval(() => this.pump(), TICK_MS);
  }

  pause(): void {
    if (!this.playing) return;
    this.offset = this.position;
    this.playing = false;
    this.stopTimer();
    this.silence();
  }

  seek(seconds: number): void {
    const target = Math.max(0, Math.min(seconds, this.duration));

    if (!this.playing || !this.ctx) {
      this.offset = target;
      return;
    }

    this.silence();
    this.offset = target;
    this.startedAt = this.ctx.currentTime;
    this.seekCursor(target);
    this.applyPedalState(target);
    this.pump();
  }

  dispose(): void {
    this.disposed = true;
    this.playing = false;
    this.stopTimer();
    try {
      this.voice?.dispose();
    } catch {
      /* already disposed */
    }
    // Leaking contexts hits Chrome's per-page cap after a handful of hot
    // reloads, and then playback fails with an opaque error.
    this.ctx?.close().catch(() => {});
    this.voice = null;
    this.ctx = null;
  }

  private stopTimer(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private silence(): void {
    try {
      this.voice?.allOff();
    } catch {
      /* disposed mid-flight */
    }
  }

  /** Binary-search the note cursor to the first onset at or after `seconds`. */
  private seekCursor(seconds: number): void {
    let lo = 0;
    let hi = this.notes.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.notes[mid].time < seconds) lo = mid + 1;
      else hi = mid;
    }
    this.cursor = lo;
  }

  /** Put the sustain pedal into whatever state it is in at `seconds`. */
  private applyPedalState(seconds: number): void {
    const down = this.pedals.some((p) => p.onset <= seconds && p.offset > seconds);
    this.voice?.sustain(down);
  }

  /** Hand WebAudio everything that starts inside the lookahead window. */
  private pump(): void {
    if (!this.playing || !this.ctx || !this.voice) return;

    const now = this.position;
    if (now >= this.duration) {
      this.finish();
      return;
    }

    const horizon = now + LOOKAHEAD_SEC;
    while (this.cursor < this.notes.length && this.notes[this.cursor].time < horizon) {
      const n = this.notes[this.cursor++];
      // Absolute time on the audio clock. Notes fractionally in the past (the
      // first tick after a seek) are clamped to "now" rather than dropped.
      const when = Math.max(this.ctx.currentTime, this.startedAt + n.time - this.offset);
      try {
        this.voice.note(n.midi, n.velocity, when, n.duration);
      } catch {
        // One unplayable note must never take down the transport.
      }
    }
  }

  private finish(): void {
    this.playing = false;
    this.stopTimer();
    this.silence();
    this.offset = this.duration;
    this.handlers.onEnd?.();
  }
}
