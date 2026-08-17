/**
 * Does this user want motion reduced?
 *
 * styles/tokens.css already collapses CSS animations and transitions under
 * `prefers-reduced-motion`, but that rule cannot reach anything driven from
 * JavaScript -- a requestAnimationFrame loop runs at full motion regardless.
 * Scroll-follow during playback and the roll's draw-in are both JS, so they
 * have to ask.
 *
 * The playhead itself deliberately does NOT consult this. It is functional
 * rather than decorative: it says where in the audio you are, and freezing it
 * would remove information rather than calm the page down.
 */

import { useSyncExternalStore } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

function subscribe(onChange: () => void): () => void {
  const mq = window.matchMedia(QUERY);
  mq.addEventListener("change", onChange);
  return () => mq.removeEventListener("change", onChange);
}

const getSnapshot = () => window.matchMedia(QUERY).matches;

export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}

/** For non-React code (the engine, the draw loop) that cannot use the hook. */
export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia(QUERY).matches;
}
