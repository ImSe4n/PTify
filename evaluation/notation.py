"""Notation-level metrics: does the printed page say the right thing?

`metrics.py` scores NOTES -- right pitch, right time, via mir_eval. Nothing in
it can say whether the key signature is correct or whether a trill was printed
where a trill was played, because mir_eval has no concept of a symbol. This
module is the other half: it scores the output of `notation/analysis.py`
against symbolic ground truth.

WHERE THE GROUND TRUTH COMES FROM
---------------------------------
Two sources, for two different reasons.

**Key** is scored against real scores from the music21 core corpus, which ship
with the library -- 3,194 of them, and 200/200 sampled carry an explicit key
signature. Nothing is downloaded.

**Ornaments** cannot be. The same corpus yields 22 trills and 3 mordents across
400 targeted Baroque/Classical scores, which is an anecdote rather than a
benchmark. They are scored instead against SYNTHESISED ground truth:
`music21.expressions.Trill.realize()` expands a notated symbol into the notes a
performer actually plays, so a score→performance pair is exact by construction
and carries no label noise. That is the only honest way to score a detector
whose input is performed notes and whose ground truth is a written symbol.

THE THING THIS MODULE EXISTS TO AVOID
-------------------------------------
A benchmark that flatters itself. Two specific guards, both learned the hard
way elsewhere in this project:

  * A metric that cannot be interpreted serialises as **None, not a number**
    (the `velocity_valid` rule from `metrics.py`). Dynamics on a
    constant-velocity corpus is exactly this case: MAPS gives every note
    velocity 80, so a "dynamics accuracy" computed there would report the
    mapping's opinion of a constant, dressed as a reading.
  * Scores that fail to parse or carry no label are **counted, not dropped**.
    Silent exclusion is how a benchmark reports 0.95 on the eleven files that
    happened to work.
"""

from __future__ import annotations

from dataclasses import dataclass

from transcriber.events import NoteEvent

#: Onset tolerance for matching a detected ornament to a notated one, in
#: seconds. Generous compared to the 50ms of note-level onset scoring, and
#: deliberately so: the question here is "was a trill printed at this
#: ornament", not "was it printed to within a fiftieth of a second". The
#: detected span begins at the first note OF the realised ornament, which is
#: already the notated position, so this only absorbs rounding.
ORNAMENT_ONSET_TOLERANCE = 0.25


@dataclass
class KeyResult:
    """How one key reading compares to the notated signature.

    Signature and tonic are scored SEPARATELY because they disagree, and the
    disagreement is the informative part. Measured on 25 corpus scores:
    signature 0.80, tonic 0.60. The gap is the relative major/minor case --
    D minor and F major share one flat, so a reading can put every accidental
    on the page correctly while naming the wrong tonic.

    For engraving, `signature_match` is the number that matters: it decides
    what prints. `tonic_match` is the stricter musical claim.
    """

    label: str
    truth_sharps: int | None
    est_sharps: int | None
    truth_tonic: str
    est_tonic: str
    correlation: float
    signature_match: bool
    tonic_match: bool
    confident: bool

    def as_row(self) -> dict:
        return {
            "label": self.label,
            "truth_sharps": self.truth_sharps,
            "est_sharps": self.est_sharps,
            "truth_tonic": self.truth_tonic,
            "est_tonic": self.est_tonic,
            "correlation": self.correlation,
            "signature_match": self.signature_match,
            "tonic_match": self.tonic_match,
            "confident": self.confident,
        }


@dataclass
class DetectionResult:
    """Precision/recall/F1 for one symbol type on one piece of material.

    `valid=False` means this material cannot score the detector at all -- not
    that it scored badly. The two must not be averaged together, so the F1
    serialises as None rather than 0.0.
    """

    label: str
    kind: str
    tp: int
    fp: int
    fn: int
    valid: bool = True
    #: Why the result is unscoreable. Empty when it is scoreable.
    invalid_reason: str = ""

    @property
    def n_reference(self) -> int:
        return self.tp + self.fn

    @property
    def n_detected(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> float:
        return self.tp / self.n_detected if self.n_detected else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_reference if self.n_reference else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def as_row(self) -> dict:
        """Flat dict. Every metric is None when the result is unscoreable.

        Not 0.0: a detector that was never given a fair chance and a detector
        that found nothing are different findings, and averaging them together
        understates the first and excuses the second.
        """
        scoreable = self.valid
        return {
            "label": self.label,
            "kind": self.kind,
            "precision": self.precision if scoreable else None,
            "recall": self.recall if scoreable else None,
            "f1": self.f1 if scoreable else None,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_ref": self.n_reference,
            "n_det": self.n_detected,
        }


def _sharps_of(tonic: str, mode: str) -> int | None:
    """Key signature (sharps, negative for flats) for a tonic/mode pair."""
    from music21 import key as m21key

    try:
        return int(m21key.Key(tonic, mode).sharps)
    except Exception:  # noqa: BLE001
        return None


def score_key(
    estimate, truth_sharps: int | None, truth_tonic: str = "", label: str = ""
) -> KeyResult:
    """Score one `KeyEstimate` against a notated key signature.

    `estimate` may be None -- `detect_key` returns None on material too short
    or too chromatic to call, which is a real answer. It is scored as a miss
    on both counts rather than skipped, because declining to print a key
    signature is still a decision the reader lives with.
    """
    if estimate is None:
        return KeyResult(
            label=label,
            truth_sharps=truth_sharps,
            est_sharps=None,
            truth_tonic=truth_tonic,
            est_tonic="",
            correlation=0.0,
            signature_match=False,
            tonic_match=False,
            confident=False,
        )

    est_sharps = _sharps_of(estimate.tonic, estimate.mode)
    return KeyResult(
        label=label,
        truth_sharps=truth_sharps,
        est_sharps=est_sharps,
        truth_tonic=truth_tonic,
        est_tonic=estimate.tonic,
        correlation=float(estimate.correlation),
        signature_match=(est_sharps is not None
                         and truth_sharps is not None
                         and est_sharps == truth_sharps),
        tonic_match=bool(truth_tonic) and estimate.tonic == truth_tonic,
        confident=bool(estimate.confident),
    )


def score_spans(
    detected: list[tuple[float, int]],
    reference: list[tuple[float, int]],
    tolerance: float = ORNAMENT_ONSET_TOLERANCE,
    kind: str = "trill",
    label: str = "",
) -> DetectionResult:
    """Match detected (onset, pitch) pairs against reference ones.

    One-to-one and greedy by proximity: each reference may be claimed by at
    most one detection and vice versa. Matching is by VALUE, never by
    position -- the two lists are independently ordered, and zipping them by
    index would silently compare unrelated events whenever the counts happened
    to agree. (`test_compare_joins_by_key_not_position` in `test_report.py`
    records the same bug found in the baseline differ.)

    Pitch must match exactly. A trill detected a third away from the notated
    one is not a hit, however well the onsets line up.
    """
    unclaimed = list(range(len(reference)))
    tp = 0

    for onset, pitch in sorted(detected):
        best = None
        best_gap = tolerance
        for slot in unclaimed:
            ref_onset, ref_pitch = reference[slot]
            if ref_pitch != pitch:
                continue
            gap = abs(ref_onset - onset)
            if gap <= best_gap:
                best, best_gap = slot, gap
        if best is not None:
            unclaimed.remove(best)
            tp += 1

    return DetectionResult(
        label=label,
        kind=kind,
        tp=tp,
        fp=len(detected) - tp,
        fn=len(unclaimed),
    )


def unscoreable(kind: str, reason: str, label: str = "") -> DetectionResult:
    """A result that must not be averaged in. See `DetectionResult.as_row`."""
    return DetectionResult(label=label, kind=kind, tp=0, fp=0, fn=0,
                           valid=False, invalid_reason=reason)


def notes_from_score(m21_score, bpm: float = 100.0) -> list[NoteEvent]:
    """A music21 score -> the note stream a performer would produce.

    `clamp=False` is load-bearing. `NoteEvent.__post_init__` lengthens any note
    shorter than 20ms when clamping is on, which is right for engine output and
    wrong for ground truth: it silently rewrites the reference before it is
    scored. `read_midi` sets it False for the same reason, and `events.py`
    documents the trap.

    Notes outside the piano range are dropped rather than clamped -- the corpus
    contains vocal and ensemble music, and transposing a cello note into the
    piano's compass would invent evidence.
    """
    from transcriber import config

    spq = 60.0 / bpm if bpm > 0 else 0.6
    out: list[NoteEvent] = []

    for element in m21_score.recurse().notes:
        onset = float(element.offset) * spq
        duration = max(0.05, float(element.quarterLength) * spq)
        pitches = [element.pitch] if element.isNote else list(element.pitches)
        for pitch in pitches:
            midi = int(pitch.midi)
            if config.MIDI_LOWEST <= midi <= config.MIDI_HIGHEST:
                out.append(NoteEvent(pitch=midi, onset=onset,
                                     offset=onset + duration,
                                     velocity=72, clamp=False))

    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


#: music21 expression class name -> the symbol name this project uses.
ORNAMENT_KINDS = {
    "Trill": "trill",
    "Mordent": "mordent",
    "InvertedMordent": "mordent",
    "Turn": "turn",
    "InvertedTurn": "turn",
}


def realise_ornaments(m21_score, bpm: float = 100.0):
    """Expand notated ornaments into the notes they are played as.

    Returns `(notes, reference)` where `notes` is the full performed stream and
    `reference` maps a symbol name to the (onset, pitch) pairs where one is
    notated. This is the score→performance bridge, and it is the only part of
    the benchmark that could quietly measure nothing -- if realisation silently
    failed, every detector would score 0.0 against an empty reference and the
    result would look like a detector problem.

    The pitch recorded for a trill is `min(principal, auxiliary)`, matching
    `detect_trills`, which always reports the LOWER of the pair as the written
    note regardless of which sounded first.
    """
    from music21 import expressions

    from transcriber import config

    spq = 60.0 / bpm if bpm > 0 else 0.6
    notes: list[NoteEvent] = []
    reference: dict[str, list[tuple[float, int]]] = {}

    for element in m21_score.recurse().notes:
        onset = float(element.offset) * spq
        ornament = None
        for expression in getattr(element, "expressions", []):
            kind = ORNAMENT_KINDS.get(type(expression).__name__)
            if kind is not None:
                ornament = (expression, kind)
                break

        if ornament is None or not element.isNote:
            duration = max(0.05, float(element.quarterLength) * spq)
            pitches = [element.pitch] if element.isNote else list(element.pitches)
            for pitch in pitches:
                midi = int(pitch.midi)
                if config.MIDI_LOWEST <= midi <= config.MIDI_HIGHEST:
                    notes.append(NoteEvent(pitch=midi, onset=onset,
                                           offset=onset + duration,
                                           velocity=72, clamp=False))
            continue

        expression, kind = ornament
        try:
            pre, main, post = expression.realize(element)
            realised = list(pre) + ([main] if main is not None else []) + list(post)
        except Exception:  # noqa: BLE001
            # An ornament music21 cannot realise is not ground truth. Emit the
            # plain note and record no reference for it, rather than scoring a
            # detector against a symbol nobody can say the sound of.
            realised = []

        if not realised:
            duration = max(0.05, float(element.quarterLength) * spq)
            midi = int(element.pitch.midi)
            if config.MIDI_LOWEST <= midi <= config.MIDI_HIGHEST:
                notes.append(NoteEvent(pitch=midi, onset=onset,
                                       offset=onset + duration,
                                       velocity=72, clamp=False))
            continue

        cursor = onset
        emitted: list[int] = []
        for piece in realised:
            duration = max(0.02, float(piece.quarterLength) * spq)
            midi = int(piece.pitch.midi)
            if config.MIDI_LOWEST <= midi <= config.MIDI_HIGHEST:
                notes.append(NoteEvent(pitch=midi, onset=cursor,
                                       offset=cursor + duration,
                                       velocity=72, clamp=False))
                emitted.append(midi)
            cursor += duration

        if emitted:
            # The written note, as `detect_trills` reports it: the lower of the
            # alternating pair for a trill, the principal otherwise.
            written = min(emitted) if kind == "trill" else int(element.pitch.midi)
            reference.setdefault(kind, []).append((onset, written))

    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes, reference


def aggregate(results: list[DetectionResult], kind: str = "") -> DetectionResult:
    """Pool results into one, summing counts rather than averaging F1s.

    Averaging per-piece F1 would weight a score with one ornament the same as
    one with fifty. Unscoreable results contribute nothing and, if every result
    is unscoreable, the pooled result is unscoreable too rather than 0.0.
    """
    scoreable = [r for r in results if r.valid]
    if not scoreable:
        reason = results[0].invalid_reason if results else "no material"
        return unscoreable(kind, reason, label="all")

    return DetectionResult(
        label="all",
        kind=kind or scoreable[0].kind,
        tp=sum(r.tp for r in scoreable),
        fp=sum(r.fp for r in scoreable),
        fn=sum(r.fn for r in scoreable),
    )


def key_accuracy(results: list[KeyResult]) -> dict:
    """Signature and tonic accuracy over a list of key readings."""
    if not results:
        return {"n": 0, "signature_accuracy": None, "tonic_accuracy": None,
                "confident_fraction": None}

    n = len(results)
    return {
        "n": n,
        "signature_accuracy": sum(r.signature_match for r in results) / n,
        "tonic_accuracy": sum(r.tonic_match for r in results) / n,
        "confident_fraction": sum(r.confident for r in results) / n,
    }


def format_key_table(results: list[KeyResult]) -> str:
    """Render key readings as a table."""
    if not results:
        return "(no results)"

    width = max([len(r.label) for r in results] + [len("score")])
    lines = [
        f"  {'':<{width}}  {'truth':>6} {'est':>6}  {'sig':>4} {'ton':>4}  {'corr':>5}",
        "  " + "-" * (width + 32),
    ]
    for r in results:
        lines.append(
            f"  {r.label:<{width}}  {str(r.truth_sharps):>6} "
            f"{str(r.est_sharps):>6}  "
            f"{'ok' if r.signature_match else '.':>4} "
            f"{'ok' if r.tonic_match else '.':>4}  {r.correlation:>5.2f}"
        )
    return "\n".join(lines)


def format_detection_table(results: list[DetectionResult]) -> str:
    """Render detection results, with unscoreable rows marked."""
    if not results:
        return "(no results)"

    width = max([len(r.kind) for r in results] + [len("symbol")])
    lines = [
        f"  {'symbol':<{width}}  {'P':>6} {'R':>6} {'F1':>6}  "
        f"{'tp':>5} {'fp':>5} {'fn':>5}",
        "  " + "-" * (width + 40),
    ]
    for r in results:
        if not r.valid:
            lines.append(f"  {r.kind:<{width}}  {'n/a':>6} {'n/a':>6} {'n/a':>6}"
                         f"   ({r.invalid_reason})")
            continue
        lines.append(
            f"  {r.kind:<{width}}  {r.precision:>6.3f} {r.recall:>6.3f} "
            f"{r.f1:>6.3f}  {r.tp:>5} {r.fp:>5} {r.fn:>5}"
        )
    if any(not r.valid for r in results):
        lines.append("")
        lines.append("  n/a: this material cannot score the detector — which is "
                     "not the same")
        lines.append("       as the detector scoring badly, so it is not "
                     "averaged in.")
    return "\n".join(lines)
