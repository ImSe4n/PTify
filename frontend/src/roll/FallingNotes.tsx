/**
 * The falling-notes view.
 *
 * The other view (PianoRoll) is a DAW-style editor: time runs left to right and
 * you read the whole piece at once. This one is the performance view -- time
 * falls DOWN onto a keyboard, pitch maps to key position, and a note arrives at
 * the strike line at the instant it sounds. It answers a different question:
 * not "what is in this piece" but "what is being played right now".
 *
 * HOW IT MOVES, AND WHY NOTHING REPAINTS
 *
 * The whole piece is drawn ONCE into a tall canvas, which is then moved by a
 * CSS transform each frame. So playback costs zero repaints of the note field
 * no matter how many notes there are -- a 20,000-note sonata animates exactly
 * as cheaply as a 300-note sonatina. This is the same technique the horizontal
 * view uses for its playhead, and the reason HANDOFF:1296 says not to redraw a
 * canvas to animate one.
 *
 * The keyboard IS repainted per frame, but it is 88 small rectangles rather
 * than several thousand, and it is a separate canvas so the note field is never
 * touched.
 *
 * GEOMETRY. The canvas is translated so that the note at `position` sits on the
 * strike line. A note at time t appears at canvas y = (duration - t) * PPS,
 * i.e. later notes are HIGHER, which is what makes them fall as time advances.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { Summary } from "../api/types";
import type { PositionSource } from "./PianoRoll";
import { assignHands } from "./hands";
import { noteAlpha, noteColour, octaveHues } from "./noteColour";
import { DEFAULT_VIEW, type ViewOptions } from "./viewOptions";

/** Vertical pixels per second at zoom 1. Falling views want more room per
    second than a horizontal roll: notes are read as they arrive, not scanned. */
const PPS_BASE = 190;
/** Height of the keyboard strip along the bottom. Sized to read as a keyboard
    without taking room from the notes -- the notes are the content. */
const KEY_H = 46;
/** Where the strike line sits, as a fraction of the visible height. Not at the
    very bottom: a little space below lets a note be seen finishing. */
const STRIKE_AT = 1;
/** Full 88-key piano. Falling views always show the whole keyboard, because the
    keyboard is the spatial reference -- cropping it moves middle C around. */
const LOW = 21;
const HIGH = 108;

const isBlackKey = (midi: number) => [1, 3, 6, 8, 10].includes(((midi % 12) + 12) % 12);

function cssVar(el: HTMLElement, name: string): string {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

/**
 * Horizontal layout of an 88-key keyboard.
 *
 * White keys tile the width evenly; black keys straddle the seam between their
 * neighbours at 62% width. Computing this once and sharing it between the note
 * field and the keyboard is what keeps a falling note aligned with the key it
 * lands on -- deriving them separately is how they drift apart.
 */
interface KeyGeometry {
  x: (midi: number) => number;
  w: (midi: number) => number;
  whiteCount: number;
}

function keyGeometry(width: number): KeyGeometry {
  const whites: number[] = [];
  for (let m = LOW; m <= HIGH; m++) if (!isBlackKey(m)) whites.push(m);

  const ww = width / whites.length;
  const whiteIndex = new Map<number, number>();
  whites.forEach((m, i) => whiteIndex.set(m, i));

  const bw = ww * 0.62;

  return {
    whiteCount: whites.length,
    w: (midi) => (isBlackKey(midi) ? bw : ww),
    x: (midi) => {
      if (!isBlackKey(midi)) return whiteIndex.get(midi)! * ww;
      // A black key sits on the seam after the white key below it.
      const below = whiteIndex.get(midi - 1);
      if (below != null) return (below + 1) * ww - bw / 2;
      const above = whiteIndex.get(midi + 1);
      return above != null ? above * ww - bw / 2 : 0;
    },
  };
}

export interface FallingNotesProps {
  summary: Summary;
  zoom: number;
  position?: number;
  positionSource?: PositionSource;
  /** Colour scheme and transposition. Presentation only. */
  view?: ViewOptions;
  onSeek?: (seconds: number) => void;
}

export function FallingNotes({
  summary,
  zoom,
  position = 0,
  positionSource,
  view = DEFAULT_VIEW,
  onSeek,
}: FallingNotesProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const fieldRef = useRef<HTMLCanvasElement>(null);
  const keysRef = useRef<HTMLCanvasElement>(null);

  const [size, setSize] = useState({ w: 0, h: 0 });
  const pps = PPS_BASE * zoom;

  // The drawn canvas covers the whole piece plus one screen of lead-in, so the
  // first note falls from off-screen rather than appearing at the strike line.
  const fieldH = Math.ceil(summary.duration * pps) + Math.max(size.h, 1);

  const notes = useMemo(
    () => [...summary.notes].sort((a, b) => a.onset - b.onset),
    [summary.notes],
  );

  // Same assignment as the roll. See hands.ts.
  const hands = useMemo(() => assignHands(notes), [notes]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const r = entry.contentRect;
      setSize({ w: Math.floor(r.width), h: Math.floor(r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // The canvas paints with RESOLVED variable values, so it does not restyle
  // itself on a theme change. Same trap, same fix as PianoRoll.
  const [themeTick, setThemeTick] = useState(0);
  useEffect(() => {
    const observer = new MutationObserver(() => setThemeTick((n) => n + 1));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setThemeTick((n) => n + 1);
    mq.addEventListener("change", onChange);
    return () => {
      observer.disconnect();
      mq.removeEventListener("change", onChange);
    };
  }, []);

  const geo = useMemo(() => keyGeometry(size.w || 1), [size.w]);

  // --- the note field: drawn ONCE per layout change, never during playback ---
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = fieldRef.current;
    if (!wrap || !canvas || size.w === 0) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const c = {
      note: cssVar(wrap, "--note"),
      est: cssVar(wrap, "--note-est"),
      bar: cssVar(wrap, "--roll-bar"),
      beat: cssVar(wrap, "--roll-beat"),
      black: cssVar(wrap, "--roll-black"),
      handLeft: cssVar(wrap, "--hand-left"),
    };
    const palette = {
      note: c.note,
      est: c.est,
      left: c.handLeft,
      right: c.note,
      octave: octaveHues(c.note),
    };

    canvas.width = size.w * dpr;
    canvas.height = fieldH * dpr;
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${fieldH}px`;

    const g = canvas.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size.w, fieldH);

    // Lanes behind the black keys, so the octave is readable while falling.
    for (let m = LOW; m <= HIGH; m++) {
      if (!isBlackKey(m)) continue;
      g.fillStyle = c.black;
      g.globalAlpha = 0.35;
      g.fillRect(geo.x(m), 0, geo.w(m), fieldH);
    }
    g.globalAlpha = 1;

    // Bar lines, so the pulse is visible as the music falls.
    if (summary.bpm && summary.bpm > 0) {
      const spb = 60 / summary.bpm;
      const beatsPerBar = parseInt(summary.time_signature?.split("/")[0] ?? "4", 10) || 4;
      let i = 0;
      for (let t = 0; t <= summary.duration + 0.01; t += spb) {
        const y = Math.round(fieldH - t * pps) + 0.5;
        g.strokeStyle = i % beatsPerBar === 0 ? c.bar : c.beat;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(0, y);
        g.lineTo(size.w, y);
        g.stroke();
        i++;
      }
    }

    // The notes. y is measured from the BOTTOM so later notes sit higher.
    notes.forEach((n, i) => {
      const pitch = Math.max(21, Math.min(108, n.pitch + view.transpose));
      const h = Math.max(3, (n.offset - n.onset) * pps);
      const y = fieldH - n.offset * pps;
      const x = geo.x(pitch);
      const w = geo.w(pitch);
      // Velocity drives alpha here too, so dynamics read as weight.
      g.globalAlpha = noteAlpha(view.scheme, n.velocity);
      g.fillStyle = noteColour(view.scheme, palette, { ...n, pitch }, hands[i]);

      // Fill the key's width bar a hairline, so a falling note visibly lines
      // up with the key it is about to strike.
      const inset = Math.min(1.5, w * 0.12);
      const bw = Math.max(2, w - inset * 2);
      const r = Math.min(3, bw / 2, h / 2);
      g.beginPath();
      g.roundRect(x + inset, y, bw, h, r);
      g.fill();
    });
    g.globalAlpha = 1;
  }, [notes, hands, view, summary, geo, pps, size.w, fieldH, themeTick]);

  // --- the keyboard: repainted per frame, but only 88 small rectangles -------
  const paintKeys = useCallback(
    (seconds: number) => {
      const wrap = wrapRef.current;
      const canvas = keysRef.current;
      if (!wrap || !canvas || size.w === 0) return;

      const g = canvas.getContext("2d");
      if (!g) return;

      const c = {
        panel: cssVar(wrap, "--panel"),
        rule: cssVar(wrap, "--rule"),
        ink: cssVar(wrap, "--ink"),
        accent: cssVar(wrap, "--accent"),
        black: cssVar(wrap, "--roll-black"),
        bg: cssVar(wrap, "--bg"),
      };

      // Which pitches are sounding right now. Linear over the sorted notes is
      // fine at these sizes and avoids an index that could fall out of step.
      const lit = new Set<number>();
      for (const n of notes) {
        if (n.onset > seconds) break;
        if (n.offset > seconds) {
          lit.add(Math.max(21, Math.min(108, n.pitch + view.transpose)));
        }
      }

      g.fillStyle = c.bg;
      g.fillRect(0, 0, size.w, KEY_H);

      // White keys first, then black over them.
      for (let m = LOW; m <= HIGH; m++) {
        if (isBlackKey(m)) continue;
        g.fillStyle = lit.has(m) ? c.accent : c.panel;
        g.fillRect(geo.x(m), 0, geo.w(m), KEY_H);
        g.strokeStyle = c.rule;
        g.lineWidth = 1;
        g.strokeRect(Math.round(geo.x(m)) + 0.5, 0.5, Math.round(geo.w(m)) - 1, KEY_H - 1);
      }
      for (let m = LOW; m <= HIGH; m++) {
        if (!isBlackKey(m)) continue;
        g.fillStyle = lit.has(m) ? c.accent : c.ink;
        g.fillRect(geo.x(m), 0, geo.w(m), KEY_H * 0.62);
      }

      // The strike line: where a falling note becomes a sounding one.
      g.strokeStyle = c.accent;
      g.lineWidth = 2;
      g.beginPath();
      g.moveTo(0, 1);
      g.lineTo(size.w, 1);
      g.stroke();
    },
    [notes, geo, size.w, view.transpose],
  );

  // Sizing a canvas CLEARS it, so the repaint has to happen in the same effect.
  // Leaving it to the render loop below meant a theme toggle while paused wiped
  // the keyboard and nothing drew it back -- there is no next frame when paused.
  useEffect(() => {
    const canvas = keysRef.current;
    if (!canvas || size.w === 0) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = size.w * dpr;
    canvas.height = KEY_H * dpr;
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${KEY_H}px`;
    canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0);
    paintKeys(positionSource ? positionSource.read() : position);
  }, [size.w, themeTick, paintKeys, position, positionSource]);

  // --- motion: one transform per frame, plus the keyboard --------------------
  const render = useCallback(
    (seconds: number) => {
      const canvas = fieldRef.current;
      if (canvas) {
        // Translate so the moment `seconds` sits on the strike line.
        const y = seconds * pps - fieldH + (size.h - KEY_H) * STRIKE_AT;
        canvas.style.transform = `translateY(${y}px)`;
      }
      paintKeys(seconds);
    },
    [pps, fieldH, size.h, paintKeys],
  );

  // Static (paused, seeking, zooming) -- and the initial paint.
  useEffect(() => {
    render(positionSource ? positionSource.read() : position);
  }, [render, position, positionSource]);

  // Live. This is the 60fps path and it never re-renders React.
  useEffect(() => {
    if (!positionSource) return;
    return positionSource.subscribe(render);
  }, [positionSource, render]);

  const handleClick = (ev: React.MouseEvent<HTMLDivElement>) => {
    if (!onSeek) return;
    const rect = ev.currentTarget.getBoundingClientRect();
    // Distance ABOVE the strike line is time still to come.
    const strikeY = rect.top + (size.h - KEY_H) * STRIKE_AT;
    const at = positionSource ? positionSource.read() : position;
    const t = at + (strikeY - ev.clientY) / pps;
    onSeek(Math.max(0, Math.min(summary.duration, t)));
  };

  return (
    <div ref={wrapRef} className="falling-wrap">
      <div
        className="falling-field"
        onClick={handleClick}
        role="img"
        aria-label={`Falling notes: ${summary.note_count} notes`}
      >
        <canvas ref={fieldRef} className="falling-canvas" />
      </div>
      <canvas ref={keysRef} className="falling-keys" aria-hidden="true" />
    </div>
  );
}
