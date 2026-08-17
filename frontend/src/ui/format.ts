/**
 * Formatting shared across screens.
 *
 * `fmtClock` existed verbatim in both ResultScreen and WaitingScreen. Phase 7
 * adds a third caller (the transport), which is the point at which a duplicated
 * four-line function stops being cheaper than a module.
 */

/** Seconds -> `M:SS`. Floors, and never renders a negative clock. */
export function fmtClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
