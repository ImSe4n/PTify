/**
 * How the roll is drawn, and how playback sounds.
 *
 * These are presentation choices, deliberately kept out of `Summary`: nothing
 * here changes the transcription, only how it is shown and heard. Transposing
 * the view does NOT transpose the exported MIDI, and the UI says so -- the
 * download is the measurement, and quietly shipping a shifted file would make
 * the artifact disagree with the model that produced it.
 */

export type ColourScheme = "ink" | "hands" | "velocity" | "register";

export interface ViewOptions {
  /** Playback rate. 1 is the recorded tempo. */
  speed: number;
  /** Semitones added at draw and schedule time. Never written to a file. */
  transpose: number;
  /** How notes are coloured. */
  scheme: ColourScheme;
}

export const DEFAULT_VIEW: ViewOptions = {
  speed: 1,
  transpose: 0,
  scheme: "ink",
};

/**
 * The rates offered.
 *
 * Practice speeds, not a continuous slider: a pianist wants "half speed" and
 * "three-quarter speed", and a slider invites 0.87x, which is not a thing
 * anyone means. Bounded at 0.5 because the sampler's pitch-preserving quality
 * falls apart below it, and at 2 because past that the notes stop being
 * separable by ear.
 */
export const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2] as const;

/** Transposition limit, in semitones. An octave either way. */
export const TRANSPOSE_MAX = 12;

export const SCHEMES: { id: ColourScheme; label: string; hint: string }[] = [
  { id: "ink", label: "Uniform", hint: "one colour; velocity as weight" },
  { id: "hands", label: "Hands", hint: "left and right, split by register" },
  { id: "velocity", label: "Dynamics", hint: "quiet to loud" },
  { id: "register", label: "Octaves", hint: "one hue per octave" },
];

/** Clamp a transposition to the supported range. */
export const clampTranspose = (n: number) =>
  Math.max(-TRANSPOSE_MAX, Math.min(TRANSPOSE_MAX, Math.round(n)));

/** `+3` / `-5` / `0`, for display. */
export const formatTranspose = (n: number) => (n > 0 ? `+${n}` : String(n));
