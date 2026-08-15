"""Symbolic ground truth for the notation benchmark, from the music21 corpus.

WHY THIS SOURCE
---------------
music21 ships 3,194 scores with the library -- Bach chorales, Palestrina,
Beethoven, Mozart, Haydn. Nothing is downloaded, nothing is licensed
awkwardly, and the labels are the composer's own notation rather than a
machine transcription. Measured over 200 randomly sampled parsed scores:
**200/200 carry an explicit key signature**, which is what makes a key
benchmark possible at all.

WHAT THIS SOURCE CANNOT DO, MEASURED
------------------------------------
Ornaments. A targeted scan of 400 Baroque and Classical scores -- the
repertoire most likely to carry them -- found **22 trills and 3 mordents**.
Staccato appears in 7 of 200 sampled scores, dynamics in 8. Those counts are
anecdotes, not benchmarks, and a precision/recall figure computed over 22
examples would move by 0.05 per example.

So ornaments are NOT scored from this corpus. `evaluation.notation`
synthesises them instead, by realising notated symbols into performed notes.
This module deliberately reports the ornament counts it finds anyway, so the
scarcity stays visible in the artifact rather than being folded away.

WHY PDMX IS NOT USED
--------------------
PDMX (250K MusicXML scores) would supply real notated ornaments in quantity.
It is not fetched here because the binding constraint measured in Phase 21 is
that the detectors do not fire, not that labels are scarce: downloading a
quarter of a million scores to score a detector that returns nothing measures
nothing. Revisit once trills and staccato work.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

#: Same seed convention as `corpus.SELECTION_SEED`, so a selection is
#: reproducible from a clean checkout and a changed sample is a deliberate act.
SELECTION_SEED = 13

DEFAULT_N = 60

#: Parseable by music21 and carrying notation. `.abc` and `.rntxt` are
#: excluded: ABC files in this corpus are folk-tune melodies with no dynamics
#: or articulation, and .rntxt is Roman-numeral analysis, not a score.
SCORE_EXTENSIONS = (".mxl", ".xml", ".musicxml", ".krn")

#: Collections whose notated signature is not a tonal key signature.
#:
#: MEASURED, and the reason this constant exists: Palestrina alone is 1,318 of
#: the 3,194 parseable scores -- 71% of the corpus, and 6 of 8 in the first
#: uniform sample drawn. Krumhansl-Schmuckler is a model of TONAL key, so
#: running it on 16th-century modal polyphony measures the mismatch between
#: two theories of pitch organisation, not the detector. Signature accuracy on
#: a Palestrina-dominated sample read 0.500 against 0.80 on tonal repertoire.
#:
#: These are not excluded from the benchmark. They are reported as a SEPARATE
#: stratum, because "the detector is weak on modal music" is a real finding
#: and deleting the evidence for it would be the flattering choice.
MODAL_COLLECTIONS = ("palestrina", "trecento", "monteverdi", "ciconia",
                     "josquin", "cypriot", "oldEnglish")


def is_modal(path) -> bool:
    """Is this score from a pre-tonal collection? See `MODAL_COLLECTIONS`."""
    text = str(path).lower()
    return any(c.lower() in text for c in MODAL_COLLECTIONS)


@dataclass
class ScoreTruth:
    """The notated facts about one score, and what could not be read.

    `skipped_reason` is empty when the score is usable. A score is never
    silently dropped: an unreadable file is a row in the artifact with a
    reason attached, because a benchmark that quietly excludes what it cannot
    parse reports the accuracy of the subset that happened to work.
    """

    label: str
    path: str
    skipped_reason: str = ""

    #: "tonal" or "modal". Reported separately rather than pooled -- see
    #: MODAL_COLLECTIONS for the measurement that forced the distinction.
    stratum: str = "tonal"

    sharps: int | None = None
    tonic: str = ""
    mode: str = ""
    time_signature: str = ""

    n_notes: int = 0
    n_staccato: int = 0
    n_dynamics: int = 0
    ornaments: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return not self.skipped_reason


def core_paths(extensions: tuple[str, ...] = SCORE_EXTENSIONS) -> list[Path]:
    """Every parseable score bundled with music21, sorted for determinism.

    `corpus.getCorePaths()` returns filesystem order, which differs between
    installs; sorting makes a seeded sample reproducible across machines.
    """
    from music21 import corpus

    return sorted(
        (p for p in corpus.getCorePaths() if p.suffix.lower() in extensions),
        key=str,
    )


def select_scores(
    n: int = DEFAULT_N,
    seed: int = SELECTION_SEED,
    collections: tuple[str, ...] = (),
    paths: list[Path] | None = None,
) -> list[Path]:
    """A reproducible sample of `n` scores.

    `collections` filters by path substring (e.g. `("bach", "beethoven")`).
    Sampling is seeded and applied AFTER filtering, so narrowing the
    collections does not reshuffle what a previous run measured.
    """
    candidates = list(paths) if paths is not None else core_paths()
    if collections:
        wanted = tuple(c.lower() for c in collections)
        candidates = [p for p in candidates
                      if any(c in str(p).lower() for c in wanted)]

    if n >= len(candidates):
        return candidates

    rng = random.Random(seed)

    if collections:
        # An explicit filter is the caller saying what they want measured;
        # stratifying on top of it would silently re-add what they excluded.
        return rng.sample(candidates, n)

    # Stratified: Palestrina alone is 71% of the corpus, so a uniform sample
    # is mostly modal polyphony and the headline number describes repertoire
    # this project does not target. Half and half, so BOTH strata have enough
    # rows to report separately.
    modal = [p for p in candidates if is_modal(p)]
    tonal = [p for p in candidates if not is_modal(p)]

    want_tonal = min(len(tonal), (n + 1) // 2)
    want_modal = min(len(modal), n - want_tonal)
    # Give back any shortfall to the other stratum rather than returning
    # fewer scores than asked for.
    want_tonal = min(len(tonal), n - want_modal)

    picked = rng.sample(tonal, want_tonal) + rng.sample(modal, want_modal)
    picked.sort(key=str)
    return picked


def _first_key(m21_score):
    """The score's opening key signature, as (sharps, tonic, mode).

    Prefers a `Key` (which names a tonic) over a bare `KeySignature` (which
    only counts accidentals). Many corpus scores carry only the latter, so the
    tonic is often unavailable -- reported as "" rather than guessed, since
    inferring it would mean running the very analysis under test against
    itself.
    """
    from music21 import key as m21key

    keys = list(m21_score.recurse().getElementsByClass(m21key.Key))
    if keys:
        first = keys[0]
        return (int(first.sharps), str(first.tonic.name), str(first.mode))

    signatures = list(m21_score.recurse().getElementsByClass(m21key.KeySignature))
    if signatures:
        return (int(signatures[0].sharps), "", "")

    return (None, "", "")


def ground_truth(m21_score, label: str = "", path: str = "") -> ScoreTruth:
    """Extract every notated fact this benchmark can score."""
    from music21 import articulations, dynamics, expressions, meter

    from .notation import ORNAMENT_KINDS

    sharps, tonic, mode = _first_key(m21_score)

    meters = list(m21_score.recurse().getElementsByClass(meter.TimeSignature))
    time_signature = meters[0].ratioString if meters else ""

    n_notes = 0
    n_staccato = 0
    ornaments: dict[str, int] = {}

    for element in m21_score.recurse().notes:
        n_notes += 1
        for articulation in getattr(element, "articulations", []):
            if isinstance(articulation, articulations.Staccato):
                n_staccato += 1
        for expression in getattr(element, "expressions", []):
            kind = ORNAMENT_KINDS.get(type(expression).__name__)
            if kind is not None:
                ornaments[kind] = ornaments.get(kind, 0) + 1

    n_dynamics = len(list(m21_score.recurse().getElementsByClass(dynamics.Dynamic)))

    return ScoreTruth(
        label=label,
        path=path,
        sharps=sharps,
        tonic=tonic,
        mode=mode,
        time_signature=time_signature,
        n_notes=n_notes,
        n_staccato=n_staccato,
        n_dynamics=n_dynamics,
        ornaments=ornaments,
    )


def _default_loader(path: Path):
    from music21 import converter

    return converter.parse(path)


def load_truth(path: Path, loader=None) -> tuple[ScoreTruth, object | None]:
    """Parse one score and extract its ground truth.

    Returns `(truth, parsed_score)`; `parsed_score` is None when the file could
    not be read, and `truth.skipped_reason` says why. Failure is a value here
    rather than an exception: one unparseable file out of sixty must not end a
    benchmark run, but it must not vanish from the count either.

    `loader` is injected the same way `corpus.py` injects `opener`/`downloader`
    -- resolved inside the function, never at import -- so tests can drive the
    failure paths without touching the music21 corpus.
    """
    load = loader or _default_loader
    label = path.stem
    stratum = "modal" if is_modal(path) else "tonal"

    try:
        parsed = load(path)
    except Exception as exc:  # noqa: BLE001
        return (ScoreTruth(label=label, path=str(path), stratum=stratum,
                           skipped_reason=f"parse failed: {type(exc).__name__}"),
                None)

    try:
        truth = ground_truth(parsed, label=label, path=str(path))
        truth.stratum = stratum
    except Exception as exc:  # noqa: BLE001
        return (ScoreTruth(label=label, path=str(path), stratum=stratum,
                           skipped_reason=f"extract failed: {type(exc).__name__}"),
                None)

    if truth.sharps is None:
        truth.skipped_reason = "no key signature"
    elif truth.n_notes < 8:
        # `config.KEY_MIN_NOTES`. Below it `detect_key` returns None by design,
        # so scoring it here would measure the guard, not the detector.
        truth.skipped_reason = "too few notes"

    return truth, parsed


def summarise(truths: list[ScoreTruth]) -> dict:
    """Counts for the artifact, including what was skipped and why."""
    skipped: dict[str, int] = {}
    ornaments: dict[str, int] = {}
    usable = 0
    staccato = 0
    dyn = 0

    for truth in truths:
        if not truth.usable:
            skipped[truth.skipped_reason] = skipped.get(truth.skipped_reason, 0) + 1
            continue
        usable += 1
        staccato += truth.n_staccato
        dyn += truth.n_dynamics
        for kind, count in truth.ornaments.items():
            ornaments[kind] = ornaments.get(kind, 0) + count

    strata: dict[str, int] = {}
    with_ornaments = 0
    for truth in truths:
        if truth.usable:
            strata[truth.stratum] = strata.get(truth.stratum, 0) + 1
            if truth.ornaments:
                with_ornaments += 1

    return {
        "n_selected": len(truths),
        "n_usable": usable,
        "n_skipped": len(truths) - usable,
        "skipped_reasons": skipped,
        "by_stratum": strata,
        "notated_staccato": staccato,
        "notated_dynamics": dyn,
        "notated_ornaments": ornaments,
        #: How many SCORES carry an ornament, not how many ornaments exist.
        #: The distinction decides whether an F1 is possible: 170 ornaments
        #: concentrated in 7 scores (one Beethoven movement carries 67) are a
        #: handful of independent examples, not 170 of them.
        "n_scores_with_ornaments": with_ornaments,
    }
