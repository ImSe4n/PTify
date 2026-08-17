/**
 * Speed, transposition and colour.
 *
 * These are PRACTICE controls, and the copy says so where it matters: they
 * change what you see and hear, never what you download. The transposition note
 * is not politeness -- shipping a shifted MIDI under the same job would make
 * the artifact disagree with the model that produced it, which is the one thing
 * this project does not do.
 */

import {
  SCHEMES,
  SPEEDS,
  TRANSPOSE_MAX,
  clampTranspose,
  formatTranspose,
  type ColourScheme,
  type ViewOptions,
} from "../roll/viewOptions";

interface Props {
  view: ViewOptions;
  speed: number;
  transpose: number;
  onSpeed: (rate: number) => void;
  onTranspose: (semitones: number) => void;
  onScheme: (scheme: ColourScheme) => void;
}

export function ViewControls({
  view,
  speed,
  transpose,
  onSpeed,
  onTranspose,
  onScheme,
}: Props) {
  return (
    <div className="view-controls">
      <div className="vc-group">
        <span className="vc-label">Speed</span>
        <div className="vc-seg" role="group" aria-label="Playback speed">
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={speed === s ? "is-active" : ""}
              onClick={() => onSpeed(s)}
              aria-pressed={speed === s}
            >
              {s === 1 ? "1×" : `${s}×`}
            </button>
          ))}
        </div>
      </div>

      <div className="vc-group">
        <span className="vc-label">Transpose</span>
        <div className="vc-stepper">
          <button
            className="icon-btn"
            onClick={() => onTranspose(clampTranspose(transpose - 1))}
            disabled={transpose <= -TRANSPOSE_MAX}
            aria-label="Transpose down a semitone"
          >
            −
          </button>
          <button
            className="mono vc-value"
            onClick={() => onTranspose(0)}
            disabled={transpose === 0}
            title="Reset to the recorded pitch"
          >
            {formatTranspose(transpose)}
          </button>
          <button
            className="icon-btn"
            onClick={() => onTranspose(clampTranspose(transpose + 1))}
            disabled={transpose >= TRANSPOSE_MAX}
            aria-label="Transpose up a semitone"
          >
            +
          </button>
        </div>
      </div>

      <div className="vc-group">
        <span className="vc-label">Colour</span>
        <div className="vc-seg" role="group" aria-label="Colour scheme">
          {SCHEMES.map((s) => (
            <button
              key={s.id}
              className={view.scheme === s.id ? "is-active" : ""}
              onClick={() => onScheme(s.id)}
              aria-pressed={view.scheme === s.id}
              title={s.hint}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <p className="vc-note">
        {transpose !== 0
          ? "Transposition changes playback and the roll. The MIDI download stays at the recorded pitch."
          : "Practice controls. Downloads are always the original transcription."}
      </p>
    </div>
  );
}
