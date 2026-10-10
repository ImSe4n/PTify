"""Candidate ways to end a note under sustain (Phase 31, step 3).

Under the sustain pedal a key release is silent -- the dampers are up -- so
the model correctly reports the SOUND, and the key release the reference
records is not in the audio. Phase 30 measured the consequence: of the notes
whose onset matched, 20% end in tolerance under sustain against 70% pedal up,
and almost every miss is too LONG. The release has to be inferred from
context. These are the candidates, scored side by side; a human picks one.

WHAT EVERY RULE IS ALLOWED TO DO
--------------------------------
- Only notes the model's OWN pedal says are pedalled are touched. The
  reference CC64 is never read here: a rule scored on ground-truth pedal would
  report a number no user could get.
- Only offsets move, and only EARLIER. Onsets, pitches, velocities and note
  count are copied, so `onset_f1` is unchanged by construction.
- Inputs are never mutated; a new Transcription is returned.

"UNDER SUSTAIN" HERE IS NOT THE DIAGNOSIS'S DEFINITION
------------------------------------------------------
`offset_diagnosis.under_pedal` asks whether the REFERENCE release falls inside
a pedal span. At inference the reference release is the unknown, and the
model's own offset under pedal typically lands AFTER the lift (it hears the
note ring until the dampers fall). So a note is selected here when the
estimated pedal is down at any moment during the estimated note.

THE RULES
---------
baseline              as transcribed
next_onset_same_hand  end at the next attack in the same hand (`notation.hands`)
next_bass_change      end where the left hand's lowest pitch next moves. A
                      note-level stand-in for "next harmony change":
                      `detect_chords` works on quantised bars, which do not
                      exist at this layer.
pedal_change          end at the first pedal lift after the onset
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Callable

from notation.hands import CHORD_WINDOW_SEC, LEFT, assign_hands, group_by_onset
from transcriber.events import MIN_NOTE_SEC, NoteEvent, PedalEvent, Transcription

RULES = ("baseline", "next_onset_same_hand", "next_bass_change", "pedal_change")


def under_estimated_pedal(note: NoteEvent, pedals: list[PedalEvent]) -> bool:
    """Is the estimated pedal down at any moment during the estimated note?"""
    return any(p.onset < note.offset and p.offset > note.onset for p in pedals)


def _next_after(times: list[float], t: float) -> float | None:
    """First time in sorted `times` strictly later than `t`, or None."""
    i = bisect_right(times, t)
    return times[i] if i < len(times) else None


def _next_onset_same_hand(tr: Transcription) -> Callable[[int], float | None]:
    hands = assign_hands(tr.notes)
    onsets = {LEFT: [], "right": []}
    for n, h in sorted(zip(tr.notes, hands), key=lambda x: x[0].onset):
        onsets[h].append(n.onset)

    def end(i: int) -> float | None:
        n = tr.notes[i]
        # Past the chord window, so notes struck together do not end each other.
        return _next_after(onsets[hands[i]], n.onset + CHORD_WINDOW_SEC)
    return end


def _next_bass_change(tr: Transcription) -> Callable[[int], float | None]:
    hands = assign_hands(tr.notes)
    hand_of = {id(n): h for n, h in zip(tr.notes, hands)}
    changes: list[float] = []
    bass = None
    for group in group_by_onset(tr.notes):
        left = [n.pitch for n in group if hand_of[id(n)] == LEFT]
        if not left:
            continue
        low = min(left)
        if bass is not None and low != bass:
            changes.append(group[0].onset)
        bass = low

    def end(i: int) -> float | None:
        return _next_after(changes, tr.notes[i].onset + CHORD_WINDOW_SEC)
    return end


def _pedal_change(tr: Transcription) -> Callable[[int], float | None]:
    lifts = sorted(p.offset for p in tr.pedals)

    def end(i: int) -> float | None:
        return _next_after(lifts, tr.notes[i].onset)
    return end


_BUILDERS = {
    "next_onset_same_hand": _next_onset_same_hand,
    "next_bass_change": _next_bass_change,
    "pedal_change": _pedal_change,
}


def apply_rule(tr: Transcription, rule: str) -> Transcription:
    """A copy of `tr` with pedalled notes ended by `rule`. Offsets only."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")

    end = _BUILDERS[rule](tr) if rule in _BUILDERS else (lambda i: None)
    notes = []
    for i, n in enumerate(tr.notes):
        offset = n.offset
        if under_estimated_pedal(n, tr.pedals):
            candidate = end(i)
            if candidate is not None:
                offset = min(offset, max(candidate, n.onset + MIN_NOTE_SEC))
        notes.append(NoteEvent(n.pitch, n.onset, offset, n.velocity,
                               clamp=False))

    return Transcription(
        notes=notes,
        pedals=[PedalEvent(p.onset, p.offset) for p in tr.pedals],
        duration=tr.duration, engine=tr.engine, source_path=tr.source_path,
    )
