/**
 * The piano roll.
 *
 * Canvas, not DOM: a real transcription is ~850 notes for one minute of music
 * (measured -- the Phase 5.5 Scarlatti returned 854), and a Brahms sonata would
 * be tens of thousands. That is far past what one div per note survives.
 *
 * THE THING THIS DRAWS THAT COMPETITORS DO NOT
 *
 * Under sustain pedal a note's release and its decay are acoustically
 * indistinguishable, so its printed LENGTH is interpolation rather than
 * measurement. `pedalled_fraction` is the share of notes in that state -- 0.09
 * on the Scarlatti measured in Phase 5.5, up to 0.91 on a Schubert impromptu.
 *
 * So notes whose release falls under pedal are drawn differently: a solid
 * onset cap (the part that IS measured) fading into a translucent tail (the
 * part that is inferred). Onsets are reliable regardless, which is exactly what
 * the legend says. Drawing all notes identically would claim a precision the
 * data does not have.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Note, Summary } from "../api/types";

/** Vertical pixels per semitone. */
const ROW_H = 8;
/** Horizontal pixels per second at zoom 1. */
const PPS_BASE = 44;
/** Width of the sticky keyboard gutter. */
const KEY_W = 54;
/** Padding in semitones above and below the piece's range. */
const PITCH_PAD = 3;

const isBlackKey = (midi: number) => [1, 3, 6, 8, 10].includes(((midi % 12) + 12) % 12);

const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
export function noteName(midi: number): string {
  return `${NOTE_NAMES[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

function cssVar(el: HTMLElement, name: string): string {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

/** A note whose release falls inside a sustain span has an estimated length. */
function markEstimated(notes: Note[], pedals: Summary["pedals"]): boolean[] {
  if (pedals.length === 0) return notes.map(() => false);
  const starts = pedals.map((p) => p.onset);
  return notes.map((n) => {
    // Binary search for the last pedal span starting at or before the release.
    let lo = 0;
    let hi = starts.length - 1;
    let idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (starts[mid] <= n.offset) {
        idx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return idx >= 0 && pedals[idx].offset >= n.offset;
  });
}

/**
 * A stream of playhead positions.
 *
 * The playhead is driven by subscription rather than by a prop, because the
 * prop path would mean a React render per frame. See audio/usePlayback.ts.
 */
export interface PositionSource {
  subscribe(cb: (seconds: number) => void): () => void;
  read(): number;
}

export interface PianoRollProps {
  summary: Summary;
  zoom: number;
  /** Current playhead position in seconds. Static; use positionSource to animate. */
  position?: number;
  /** Live positions during playback. Takes precedence over `position`. */
  positionSource?: PositionSource;
  /** Whether to keep the playhead in view. */
  follow?: boolean;
  onSeek?: (seconds: number) => void;
}

/** Scroll when the playhead passes this fraction of the visible width... */
const FOLLOW_TRIGGER = 0.75;
/** ...and put it here afterwards. Not centred: a permanently centred playhead
    means the canvas never stops moving, which is exhausting to read against. */
const FOLLOW_LAND = 0.35;
/** How long to leave the user alone after they scroll by hand. */
const FOLLOW_YIELD_MS = 2500;

export function PianoRoll({
  summary,
  zoom,
  position = 0,
  positionSource,
  follow = false,
  onSeek,
}: PianoRollProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const rollRef = useRef<HTMLCanvasElement>(null);
  const keysRef = useRef<HTMLCanvasElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);

  const [lo, hi] = useMemo(() => {
    const [a, b] = summary.pitch_range;
    return [Math.max(21, a - PITCH_PAD), Math.min(108, b + PITCH_PAD)];
  }, [summary.pitch_range]);

  const estimated = useMemo(
    () => markEstimated(summary.notes, summary.pedals),
    [summary.notes, summary.pedals],
  );

  const pps = PPS_BASE * zoom;
  const rows = hi - lo + 1;
  const height = rows * ROW_H;
  const width = Math.ceil(summary.duration * pps) + 30;

  // The canvas paints with resolved CSS variable VALUES, so unlike real DOM it
  // does not restyle itself when the theme changes -- it would keep the light
  // palette inside dark chrome until something else forced a redraw. Watching
  // the attribute the theme is set on gives the draw effect a dependency.
  const [themeTick, setThemeTick] = useState(0);
  useEffect(() => {
    const target = document.documentElement;
    const observer = new MutationObserver(() => setThemeTick((n) => n + 1));
    observer.observe(target, { attributes: true, attributeFilter: ["data-theme"] });

    // The theme also follows the OS when no explicit choice is stored.
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setThemeTick((n) => n + 1);
    mq.addEventListener("change", onChange);

    return () => {
      observer.disconnect();
      mq.removeEventListener("change", onChange);
    };
  }, []);

  useEffect(() => {
    const wrap = wrapRef.current;
    const roll = rollRef.current;
    const keys = keysRef.current;
    if (!wrap || !roll || !keys) return;

    // Cap DPR at 2: beyond that the memory cost grows quadratically for no
    // visible gain on a chart of flat rectangles.
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const c = {
      rowA: cssVar(wrap, "--roll-row-a"),
      rowB: cssVar(wrap, "--roll-row-b"),
      black: cssVar(wrap, "--roll-black"),
      beat: cssVar(wrap, "--roll-beat"),
      bar: cssVar(wrap, "--roll-bar"),
      note: cssVar(wrap, "--note"),
      est: cssVar(wrap, "--note-est"),
      pedal: cssVar(wrap, "--pedal-band"),
      panel: cssVar(wrap, "--panel"),
      rule: cssVar(wrap, "--rule"),
      muted: cssVar(wrap, "--muted"),
      faint: cssVar(wrap, "--faint"),
    };

    roll.width = width * dpr;
    roll.height = height * dpr;
    roll.style.width = `${width}px`;
    roll.style.height = `${height}px`;

    const g = roll.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Pitch lanes, with black-key rows shaded so the octave is readable.
    for (let m = hi; m >= lo; m--) {
      const y = (hi - m) * ROW_H;
      g.fillStyle = isBlackKey(m) ? c.black : Math.floor(m / 12) % 2 ? c.rowA : c.rowB;
      g.fillRect(0, y, width, ROW_H);
    }

    // Beat and bar lines, when a tempo was detected.
    if (summary.bpm && summary.bpm > 0) {
      const spb = 60 / summary.bpm;
      const beatsPerBar = parseInt(summary.time_signature?.split("/")[0] ?? "4", 10) || 4;
      let i = 0;
      for (let t = 0; t <= summary.duration + 0.01; t += spb) {
        const x = Math.round(t * pps) + 0.5;
        g.strokeStyle = i % beatsPerBar === 0 ? c.bar : c.beat;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(x, 0);
        g.lineTo(x, height);
        g.stroke();
        i++;
      }
    }

    // Octave rules.
    for (let m = hi; m >= lo; m--) {
      if (m % 12 === 0) {
        const y = Math.round((hi - m) * ROW_H) + 0.5;
        g.strokeStyle = c.bar;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(0, y);
        g.lineTo(width, y);
        g.stroke();
      }
    }

    // Sustain spans behind the notes: the reason some lengths are estimates.
    g.fillStyle = c.pedal;
    for (const p of summary.pedals) {
      g.fillRect(p.onset * pps, 0, Math.max(1, (p.offset - p.onset) * pps), height);
    }

    // Notes. Velocity drives alpha, so dynamics are visible as weight.
    summary.notes.forEach((n, i) => {
      const x = n.onset * pps;
      const w = Math.max(2.5, (n.offset - n.onset) * pps);
      const y = (hi - n.pitch) * ROW_H;
      const alpha = 0.45 + 0.5 * (n.velocity / 127);

      if (estimated[i]) {
        // Solid onset cap = measured. Translucent tail = interpolated.
        g.globalAlpha = alpha * 0.45;
        g.fillStyle = c.est;
        g.fillRect(x, y, w, ROW_H - 1);
        g.globalAlpha = alpha;
        g.fillRect(x, y, Math.min(2.5, w), ROW_H - 1);
      } else {
        g.globalAlpha = alpha;
        g.fillStyle = c.note;
        g.fillRect(x, y, w, ROW_H - 1);
      }
    });
    g.globalAlpha = 1;

    // --- the sticky keyboard gutter ---
    keys.width = KEY_W * dpr;
    keys.height = height * dpr;
    keys.style.width = `${KEY_W}px`;
    keys.style.height = `${height}px`;

    const k = keys.getContext("2d");
    if (!k) return;
    k.setTransform(dpr, 0, 0, dpr, 0, 0);
    k.fillStyle = c.panel;
    k.fillRect(0, 0, KEY_W, height);

    for (let m = hi; m >= lo; m--) {
      const y = (hi - m) * ROW_H;
      if (isBlackKey(m)) {
        k.fillStyle = c.black;
        k.fillRect(0, y, KEY_W - 14, ROW_H - 0.5);
      }
      k.strokeStyle = c.rule;
      k.lineWidth = 1;
      k.beginPath();
      k.moveTo(0, Math.round(y) + 0.5);
      k.lineTo(KEY_W, Math.round(y) + 0.5);
      k.stroke();

      if (m % 12 === 0) {
        k.fillStyle = c.muted;
        k.font = '9px "IBM Plex Mono", monospace';
        k.textAlign = "right";
        k.fillText(noteName(m), KEY_W - 5, y + ROW_H - 1.5);
      }
    }
    k.strokeStyle = c.rule;
    k.beginPath();
    k.moveTo(KEY_W - 0.5, 0);
    k.lineTo(KEY_W - 0.5, height);
    k.stroke();
  }, [summary, estimated, lo, hi, pps, width, height, themeTick]);

  // `pps` changes when the user zooms. The subscription below outlives that,
  // so reading it from a ref is what keeps a zoom mid-playback from leaving the
  // playhead behind at the old scale -- the classic stale-closure bug.
  const ppsRef = useRef(pps);
  ppsRef.current = pps;

  // The playhead is a transformed div, never a canvas redraw -- moving it must
  // not cost a repaint of several thousand rectangles.
  const paintPlayhead = useCallback((seconds: number) => {
    const el = playheadRef.current;
    if (el) el.style.transform = `translateX(${seconds * ppsRef.current}px)`;
  }, []);

  // Repaint on a scale change. Without this a zoom while PAUSED leaves the
  // playhead at the old pixel position -- the live path only repaints on the
  // next frame of playback, and while paused there is no next frame.
  useEffect(() => {
    paintPlayhead(positionSource ? positionSource.read() : position);
  }, [position, pps, positionSource, paintPlayhead]);

  // Live positions. This is the 60fps path and it never re-renders React.
  useEffect(() => {
    if (!positionSource) return;
    return positionSource.subscribe(paintPlayhead);
  }, [positionSource, paintPlayhead]);

  // Scroll-follow.
  useEffect(() => {
    if (!positionSource || !follow) return;
    const scroller = scrollRef.current;
    if (!scroller) return;

    let yieldUntil = 0;
    // Listen for INPUT, never for `scroll`: our own scrollLeft writes fire
    // scroll events, so reading those as "the user moved" would suppress
    // following forever after the first automatic scroll.
    const yieldToUser = () => {
      yieldUntil = performance.now() + FOLLOW_YIELD_MS;
    };
    for (const ev of ["wheel", "touchstart", "pointerdown"] as const) {
      scroller.addEventListener(ev, yieldToUser, { passive: true });
    }

    const unsubscribe = positionSource.subscribe((seconds) => {
      if (performance.now() < yieldUntil) return;

      const x = seconds * ppsRef.current;
      const view = scroller.clientWidth;
      const left = scroller.scrollLeft;

      if (x < left || x > left + view * FOLLOW_TRIGGER) {
        // scrollLeft ONLY. `.roll-scroll` scrolls both axes and the vertical
        // position is the user's pitch view -- taking it would be hostile.
        scroller.scrollLeft = Math.max(0, x - view * FOLLOW_LAND);
      }
    });

    return () => {
      unsubscribe();
      for (const ev of ["wheel", "touchstart", "pointerdown"] as const) {
        scroller.removeEventListener(ev, yieldToUser);
      }
    };
  }, [positionSource, follow]);

  const handleClick = (ev: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return;
    const rect = ev.currentTarget.getBoundingClientRect();
    const t = (ev.clientX - rect.left) / pps;
    onSeek(Math.max(0, Math.min(summary.duration, t)));
  };

  return (
    <div ref={wrapRef} className="roll-wrap">
      <div ref={scrollRef} className="roll-scroll">
        <div className="roll-inner">
          <canvas ref={keysRef} className="roll-keys" aria-hidden="true" />
          <div className="roll-canvas-wrap">
            <canvas
              ref={rollRef}
              onClick={handleClick}
              className="roll-canvas"
              role="img"
              aria-label={`Piano roll: ${summary.note_count} notes from ${noteName(
                summary.pitch_range[0],
              )} to ${noteName(summary.pitch_range[1])}`}
            />
            <div ref={playheadRef} className="roll-playhead" style={{ height }} />
          </div>
        </div>
      </div>
    </div>
  );
}
