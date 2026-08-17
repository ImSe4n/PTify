/**
 * What colour is a note?
 *
 * THE ONE RULE THAT SURVIVES EVERY SCHEME
 *
 * A note whose release fell under sustain has an *interpolated* length, and the
 * roll has always drawn that differently -- it is the product's central honesty
 * claim, not decoration. So no scheme is allowed to overwrite it: schemes pick
 * the BASE hue, and the estimated marking is applied on top by the caller as it
 * always was. Colouring by hand and losing "this length is a guess" would trade
 * the thing that distinguishes this tool for a thing every tool has.
 */

import type { Hand } from "./hands";
import type { ColourScheme } from "./viewOptions";

/** Resolved palette entries the schemes draw from. */
export interface Palette {
  note: string;
  est: string;
  left: string;
  right: string;
  octave: string[];
}

/**
 * Per-octave hues for the `register` scheme.
 *
 * Derived from the accent by rotation rather than picked by hand, so the scheme
 * stays inside the product's palette in both themes instead of introducing a
 * second, unrelated colour language.
 */
export function octaveHues(accent: string): string[] {
  // Rotate hue around the accent. Parsed rather than hardcoded so a palette
  // change carries through.
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(accent.trim());
  if (!m) return Array(9).fill(accent);
  const [r, g, b] = [1, 2, 3].map((i) => parseInt(m[i], 16) / 255);

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  const sat = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
  let hue = 0;
  if (d !== 0) {
    if (max === r) hue = ((g - b) / d) % 6;
    else if (max === g) hue = (b - r) / d + 2;
    else hue = (r - g) / d + 4;
    hue *= 60;
    if (hue < 0) hue += 360;
  }

  // Nine octaves across the 88-key range, spread over 200 degrees so the ends
  // stay distinguishable without wrapping back onto the start.
  return Array.from({ length: 9 }, (_, i) => {
    const h = (hue + (i - 4) * 25 + 360) % 360;
    return `hsl(${h.toFixed(0)} ${Math.round(sat * 100)}% ${Math.round(l * 100)}%)`;
  });
}

/** The base colour for a note, before the estimated-length marking. */
export function noteColour(
  scheme: ColourScheme,
  palette: Palette,
  note: { pitch: number; velocity: number },
  hand: Hand,
): string {
  switch (scheme) {
    case "hands":
      return hand === "left" ? palette.left : palette.right;
    case "register":
      return palette.octave[Math.max(0, Math.min(8, Math.floor(note.pitch / 12) - 1))];
    case "velocity":
    case "ink":
    default:
      // Both use the single ink; velocity already drives alpha at the call
      // site, which is what "Dynamics" means here.
      return palette.note;
  }
}

/**
 * Alpha for a note.
 *
 * `velocity` leans on it hard (that is the point of the scheme); the others
 * keep the gentler weighting the roll has always used, so dynamics stay
 * legible without swamping the hue.
 */
export function noteAlpha(scheme: ColourScheme, velocity: number): number {
  const v = velocity / 127;
  return scheme === "velocity" ? 0.25 + 0.75 * v ** 1.4 : 0.45 + 0.5 * v;
}
