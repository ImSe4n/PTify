/**
 * The transport: play, scrub, and where you are.
 *
 * The scrub bar is hand-built rather than an <input type="range"> because every
 * other control on this screen is, and restyling a native range across browsers
 * costs more CSS than the twenty lines this takes.
 *
 * Like the playhead, the scrub thumb is moved by subscription rather than by a
 * prop -- it is on the same 60fps path and must not re-render the screen.
 */

import { useCallback, useEffect, useRef } from "react";

import type { PositionSource } from "../audio/usePlayback";
import type { EngineStatus } from "../audio/PlaybackEngine";
import { fmtClock } from "../ui/format";

interface Props {
  duration: number;
  isPlaying: boolean;
  status: EngineStatus;
  error: string | null;
  clock: string;
  positionSource: PositionSource;
  onToggle: () => void;
  onSeek: (seconds: number) => void;
}

export function Transport({
  duration,
  isPlaying,
  status,
  error,
  clock,
  positionSource,
  onToggle,
  onSeek,
}: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const fillRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  useEffect(() => {
    return positionSource.subscribe((seconds) => {
      const pct = duration > 0 ? Math.min(1, seconds / duration) : 0;
      if (fillRef.current) fillRef.current.style.transform = `scaleX(${pct})`;
    });
  }, [positionSource, duration]);

  const seekFromEvent = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el || duration <= 0) return;
      const rect = el.getBoundingClientRect();
      const ratio = (clientX - rect.left) / rect.width;
      onSeek(Math.max(0, Math.min(1, ratio)) * duration);
    },
    [duration, onSeek],
  );

  const onPointerDown = (ev: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = true;
    ev.currentTarget.setPointerCapture(ev.pointerId);
    seekFromEvent(ev.clientX);
  };

  const onPointerMove = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (draggingRef.current) seekFromEvent(ev.clientX);
  };

  const endDrag = (ev: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = false;
    if (ev.currentTarget.hasPointerCapture(ev.pointerId)) {
      ev.currentTarget.releasePointerCapture(ev.pointerId);
    }
  };

  const onKeyDown = (ev: React.KeyboardEvent<HTMLDivElement>) => {
    const step = ev.shiftKey ? 1 : 5;
    const at = positionSource.read();
    if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      onSeek(at - step);
    } else if (ev.key === "ArrowRight") {
      ev.preventDefault();
      onSeek(at + step);
    } else if (ev.key === "Home") {
      ev.preventDefault();
      onSeek(0);
    } else if (ev.key === "End") {
      ev.preventDefault();
      onSeek(duration);
    }
  };

  const failed = status === "failed";

  return (
    <div className="transport">
      <button
        className="transport-play"
        onClick={onToggle}
        disabled={failed}
        aria-label={isPlaying ? "Pause" : "Play"}
        aria-pressed={isPlaying}
      >
        <span aria-hidden="true">{isPlaying ? "❙❙" : "▶"}</span>
      </button>

      <span className="mono transport-clock">
        {clock} <span className="transport-clock-total">/ {fmtClock(duration)}</span>
      </span>

      <div
        ref={trackRef}
        className="transport-track"
        role="slider"
        tabIndex={0}
        aria-label="Playback position"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={undefined}
        aria-valuetext={clock}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onKeyDown={onKeyDown}
      >
        <div ref={fillRef} className="transport-fill" />
      </div>

      <span className="transport-note">
        {failed ? (
          <span className="transport-warn">{error ?? "playback unavailable"}</span>
        ) : status === "loading" ? (
          "loading piano…"
        ) : status === "fallback" ? (
          // Say why it sounds thin, rather than letting it read as bad audio.
          <span className="transport-warn" title="The sampled piano could not be reached.">
            synthesised, samples unavailable
          </span>
        ) : (
          <>
            <span className="transport-hint-keys mono">space</span> play ·{" "}
            <span className="transport-hint-keys mono">← →</span> seek
          </>
        )}
      </span>
    </div>
  );
}
