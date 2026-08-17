/**
 * The React seam for playback.
 *
 * THE POSITION NEVER ENTERS REACT STATE.
 *
 * At 60fps a `setPosition` would re-render the whole Result screen -- trust
 * panel, detected facts, download list -- sixty times a second, and twice that
 * under StrictMode. So the playhead is driven the way PianoRoll was already
 * built to be driven: a rAF loop reads the audio clock and writes
 * `style.transform` directly, entirely outside React.
 *
 * What DOES go into state is what a human reads at human speed: whether it is
 * playing, and the clock string -- and the clock only when the rendered text
 * actually changes, which is about once a second rather than sixty times.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { Summary } from "../api/types";
import { fmtClock } from "../ui/format";
import { PlaybackEngine, type EngineStatus } from "./PlaybackEngine";

export interface PositionSource {
  /** Returns an unsubscribe function. */
  subscribe(cb: (seconds: number) => void): () => void;
  /** The position right now, for a subscriber that needs it before the next frame. */
  read(): number;
}

export interface Playback {
  isPlaying: boolean;
  status: EngineStatus;
  error: string | null;
  /** `M:SS`, updated only when the string changes. */
  clock: string;
  positionSource: PositionSource;
  toggle(): void;
  seek(seconds: number): void;
}

export function usePlayback(summary: Summary | null): Playback {
  const engineRef = useRef<PlaybackEngine | null>(null);
  const subscribersRef = useRef(new Set<(t: number) => void>());
  const rafRef = useRef<number | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [status, setStatus] = useState<EngineStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [clock, setClock] = useState("0:00");

  // Build the engine when the transcription arrives. It does NOT touch
  // WebAudio here -- the AudioContext is created inside play(), because one
  // constructed before a user gesture starts suspended.
  useEffect(() => {
    if (!summary) return;

    const engine = new PlaybackEngine(summary, {
      onEnd: () => setIsPlaying(false),
      onStatus: (s, detail) => {
        setStatus(s);
        setError(s === "failed" ? (detail ?? "playback is unavailable") : null);
      },
    });
    engineRef.current = engine;

    return () => {
      engine.dispose();
      engineRef.current = null;
      setIsPlaying(false);
      setStatus("idle");
    };
  }, [summary]);

  // One rAF loop for every subscriber, running only while something is playing.
  const publish = useCallback((seconds: number) => {
    for (const cb of subscribersRef.current) cb(seconds);
    const next = fmtClock(seconds);
    setClock((prev) => (prev === next ? prev : next));
  }, []);

  useEffect(() => {
    if (!isPlaying) return;

    const tick = () => {
      const engine = engineRef.current;
      if (engine) publish(engine.position);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [isPlaying, publish]);

  const positionSource = useMemo<PositionSource>(
    () => ({
      subscribe(cb) {
        subscribersRef.current.add(cb);
        // Paint the current position immediately, so a subscriber mounting
        // while paused does not sit at zero until the next play.
        cb(engineRef.current?.position ?? 0);
        return () => {
          subscribersRef.current.delete(cb);
        };
      },
      read: () => engineRef.current?.position ?? 0,
    }),
    [],
  );

  const toggle = useCallback(() => {
    const engine = engineRef.current;
    if (!engine) return;

    if (engine.isPlaying) {
      engine.pause();
      setIsPlaying(false);
      publish(engine.position);
    } else {
      // play() awaits the sample load; reflect the intent immediately so the
      // button does not look inert for the second that takes.
      setIsPlaying(true);
      engine.play().then(() => {
        if (!engine.isPlaying) setIsPlaying(false);
      });
    }
  }, [publish]);

  const seek = useCallback(
    (seconds: number) => {
      const engine = engineRef.current;
      if (!engine) return;
      engine.seek(seconds);
      publish(engine.position);
    },
    [publish],
  );

  return { isPlaying, status, error, clock, positionSource, toggle, seek };
}
