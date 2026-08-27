"""The notation benchmark: does the scoreboard measure what it claims to?

A benchmark can fail in a way an ordinary module cannot -- by reporting a
number that is not wrong so much as meaningless. Most of these tests defend
against that rather than against a crash:

  * a metric that cannot be interpreted must serialise as None, not 0.0
  * matching must join by value, never by position
  * scores that could not be read must be counted, not dropped
  * the ornament realiser must actually produce notes, since a silent failure
    there would make every detector score 0.0 and look like a detector problem

Everything here is pure -- no audio, no model, no network, no downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.notation import (
    DetectionResult,
    aggregate,
    format_detection_table,
    key_accuracy,
    notes_from_score,
    realise_ornaments,
    score_key,
    score_spans,
    unscoreable,
)
from evaluation import notation_corpus as NC
from notation import analysis


# --- helpers --------------------------------------------------------------

def _ornament_score(factory, quarter_length=2.0, pitch="C5"):
    """A one-note stream carrying a single ornament."""
    from music21 import note, stream

    s = stream.Stream()
    element = note.Note(pitch, quarterLength=quarter_length)
    element.expressions.append(factory())
    s.append(element)
    return s


class _Estimate:
    """Stand-in for `analysis.KeyEstimate`, so scoring is tested alone."""

    def __init__(self, tonic, mode, correlation=0.9, confident=True):
        self.tonic = tonic
        self.mode = mode
        self.correlation = correlation
        self.confident = confident


# --- ornament realisation -------------------------------------------------

@pytest.mark.parametrize("bpm", [60.0, 90.0, 120.0])
def test_a_realised_trill_is_detected_at_every_tempo(bpm):
    """THE bridge the whole ornament benchmark rests on.

    A notated trill is one notehead; the detector consumes performed notes. If
    realisation produced nothing, every ornament would score 0.0 and the
    finding would read as a detector failure rather than a harness failure.
    """
    from music21 import expressions

    notes, reference = realise_ornaments(
        _ornament_score(expressions.Trill), bpm=bpm)

    assert len(notes) > 4, "sanity: the trill realised into a run of notes"
    assert reference["trill"], "sanity: the notated symbol was recorded"
    assert len(analysis.detect_trills(notes)) == 1


@pytest.mark.parametrize("name", ["Mordent", "InvertedMordent", "Turn"])
def test_a_realised_mordent_or_turn_is_not_called_a_trill(name):
    """The conservative bias in analysis.py, measured rather than assumed.

    These realise into 3-4 notes of adjacent pitches -- exactly the shape a
    loose trill detector would claim. Printing a trill where a mordent was
    played rewrites the music.
    """
    from music21 import expressions

    notes, reference = realise_ornaments(
        _ornament_score(getattr(expressions, name)), bpm=90.0)

    assert notes, "sanity: the ornament realised"
    assert "trill" not in reference
    assert analysis.detect_trills(notes) == []


def test_parts_of_a_multi_part_score_do_not_collapse_onto_each_other():
    """REGRESSION: `element.offset` is measured from the element's immediate
    container -- its measure or voice -- so in a multi-part score every part
    restarts at 0 and the parts land on top of each other.

    This made the benchmark measure nothing while reporting a number. A
    Beethoven quartet flattened to notes whose first fourteen all shared onset
    1.3333, so no alternating run could form and trill recall on real scores
    read 0.000 against 122 realisable trills. The synthetic single-voice cases
    were all passing at the time, which is why only real material caught it.
    """
    from music21 import note, stream

    # MEASURES are what make this bite, and a flat Part will not reproduce it:
    # inside a measure `element.offset` is relative to THAT MEASURE, so every
    # note in the piece reports 0.0 and the whole score lands on one instant.
    score = stream.Score()
    for pitch in ("C4", "G4"):
        part = stream.Part()
        for number in range(3):
            measure = stream.Measure(number=number + 1)
            measure.append(note.Note(pitch, quarterLength=4.0))
            part.append(measure)
        # insert at 0, not append: the parts sound SIMULTANEOUSLY.
        score.insert(0, part)

    notes = notes_from_score(score, bpm=60.0)

    assert len(notes) == 6
    # Three distinct onsets, each carrying both parts -- not one pile of six.
    assert len({round(n.onset, 6) for n in notes}) == 3


def test_a_bare_stream_still_uses_its_own_offsets():
    """The hierarchy lookup must fall back cleanly: a flat Stream of notes has
    no containing score, and its local offsets are already absolute."""
    from music21 import note, stream

    s = stream.Stream()
    for _ in range(3):
        s.append(note.Note("C4", quarterLength=1.0))

    notes = notes_from_score(s, bpm=60.0)

    assert [round(n.onset, 3) for n in notes] == [0.0, 1.0, 2.0]


def test_a_trill_on_a_short_note_is_missed_and_the_benchmark_says_so():
    """THE finding the ornament benchmark exists to produce.

    A trill notated on a sixteenth realises to 2 notes, below
    TRILL_MIN_ALTERNATIONS = 4, so it is missed. The boundary sits between
    0.25q and 0.5q REGARDLESS of tempo, because realisation subdivides the
    written value rather than working in real time -- which is why the
    benchmark sweeps note values and not just tempi.

    This test pins the miss rather than the fix: whether 4 is the right
    minimum is a tuning question the scoreboard exists to inform, and tuning
    it here would be answering the question with the measurement.
    """
    from music21 import expressions

    short_notes, short_ref = realise_ornaments(
        _ornament_score(expressions.Trill, quarter_length=0.25), bpm=90.0)
    long_notes, _ = realise_ornaments(
        _ornament_score(expressions.Trill, quarter_length=0.5), bpm=90.0)

    assert short_ref["trill"], "sanity: the short trill is still ground truth"
    assert analysis.detect_trills(short_notes) == []
    assert len(analysis.detect_trills(long_notes)) == 1


def test_the_trill_reference_pitch_is_the_lower_of_the_pair():
    """`detect_trills` always reports the LOWER pitch as the written note.
    Ground truth recorded on the upper one would never match, and the failure
    would look like a recall problem instead of a units mismatch."""
    from music21 import expressions

    notes, reference = realise_ornaments(
        _ornament_score(expressions.Trill), bpm=90.0)
    detected = analysis.detect_trills(notes)

    assert reference["trill"][0][1] == detected[0].pitch


def test_an_unrealisable_ornament_records_no_reference():
    """A symbol music21 cannot expand is not ground truth. Scoring a detector
    against it would count a miss for a sound nobody can say the shape of."""
    from music21 import expressions

    class Unrealisable(expressions.Trill):
        def realize(self, *a, **k):
            raise ValueError("cannot realise")

    s = _ornament_score(Unrealisable)
    notes, reference = realise_ornaments(s, bpm=90.0)

    assert notes, "the plain note is still emitted"
    assert "trill" not in reference


# --- span matching --------------------------------------------------------

def test_spans_join_by_value_not_position():
    """REGRESSION in kind: `format_comparison` once zipped rows by index,
    which silently compared unrelated cases whenever the counts matched. A
    span matcher has the same failure mode with the same invisibility."""
    detected = [(2.0, 72), (0.0, 60)]
    reference = [(0.0, 60), (2.0, 72)]

    result = score_spans(detected, reference)

    assert (result.tp, result.fp, result.fn) == (2, 0, 0)


def test_a_detection_at_the_wrong_pitch_is_not_a_hit():
    result = score_spans([(0.0, 67)], [(0.0, 60)])
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)


def test_one_detection_cannot_claim_two_references():
    """Otherwise a detector that fires once on a cluster of notated ornaments
    would score perfect recall on all of them."""
    result = score_spans([(0.0, 60)], [(0.0, 60), (0.05, 60)])
    assert (result.tp, result.fp, result.fn) == (1, 0, 1)


def test_a_detection_outside_the_tolerance_is_a_miss_and_a_false_positive():
    result = score_spans([(9.0, 60)], [(0.0, 60)], tolerance=0.25)
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)


# --- the unscoreable rule -------------------------------------------------

def test_an_unscoreable_result_serialises_f1_as_none():
    """The `velocity_valid` rule. A number that cannot be interpreted is worse
    than an absent one: it prints in tables and gets quoted."""
    row = unscoreable("dynamics", "corpus is constant-velocity").as_row()

    assert row["f1"] is None
    assert row["precision"] is None
    assert row["recall"] is None
    assert row["valid"] is False
    assert row["invalid_reason"]


def test_nothing_detected_against_nothing_to_detect_is_not_a_zero():
    """A mordent correctly NOT called a trill has tp=fp=fn=0, and F1 is 0/0
    there. Printing 0.000 would file a perfect negative result in the column a
    reader scans for failures."""
    pooled = aggregate([DetectionResult(label="x", kind="mordent",
                                        tp=0, fp=0, fn=0)], kind="mordent")

    assert pooled.valid is False
    assert pooled.as_row()["f1"] is None


def test_a_real_zero_is_still_reported_as_zero():
    """The other side of the boundary: a detector that missed everything it
    was given must NOT be excused as unscoreable."""
    pooled = aggregate([DetectionResult(label="x", kind="trill",
                                        tp=0, fp=0, fn=5)], kind="trill")

    assert pooled.valid is True
    assert pooled.as_row()["f1"] == 0.0


def test_aggregate_sums_counts_rather_than_averaging_f1():
    """Averaging per-piece F1 would weight a score with one ornament the same
    as one with fifty."""
    pooled = aggregate([
        DetectionResult(label="a", kind="trill", tp=1, fp=0, fn=0),
        DetectionResult(label="b", kind="trill", tp=0, fp=0, fn=9),
    ], kind="trill")

    assert (pooled.tp, pooled.fn) == (1, 9)
    assert pooled.recall == pytest.approx(0.1)


def test_unscoreable_rows_are_not_averaged_into_the_pool():
    pooled = aggregate([
        DetectionResult(label="a", kind="trill", tp=3, fp=0, fn=1),
        unscoreable("trill", "no material", label="b"),
    ], kind="trill")

    assert (pooled.tp, pooled.fn) == (3, 1)


def test_the_table_marks_unscoreable_rows_instead_of_printing_a_number():
    out = format_detection_table([unscoreable("dynamics", "constant velocity")])
    assert "n/a" in out
    assert "0.000" not in out


# --- the tempo sweep ------------------------------------------------------
#
# WHY THESE EXIST. The committed real-repertoire trill F1 was a SINGLE point at
# 100 BPM, while the tool's own conclusion said the value was worth +/-0.05
# because a 60-140 BPM sweep moved it between 0.337 and 0.446. That sweep was
# run by hand and only its prose survived, so the error bar could not be
# reproduced -- and an error bar you cannot re-measure cannot tell you whether
# a change helped. These pin the sweep that replaces it.

def _sweep(monkeypatch, f1_by_bpm):
    """Run `_sweep_real_ornaments` against canned per-tempo results.

    The scoring pass is stubbed rather than driven through the corpus: what is
    under test is the summarising arithmetic, and making it parse Beethoven to
    check a mean would be slow and would fail for unrelated reasons.
    """
    from tools import benchmark_notation as B

    def fake_score(truths_and_scores, bpm, quiet):
        tp, fp, fn = f1_by_bpm[bpm]
        return [DetectionResult(label="x", kind="trill", tp=tp, fp=fp, fn=fn)]

    monkeypatch.setattr(B, "_score_real_ornaments", fake_score)
    return B._sweep_real_ornaments([], tuple(f1_by_bpm), quiet=True)


def test_the_sweep_reports_the_mean_and_the_range_not_one_tempo(monkeypatch):
    """The headline must be the mean, with the spread beside it.

    A single tempo is what made 0.337 look like a measurement when it was one
    sample from a spread of 0.109.
    """
    rows, summary = _sweep(monkeypatch, {
        60.0: (1, 1, 1),    # P .5  R .5  F1 .5
        100.0: (3, 1, 1),   # P .75 R .75 F1 .75
    })

    assert [r["bpm"] for r in rows] == [60.0, 100.0]
    assert summary["f1_mean"] == pytest.approx(0.625)
    assert summary["f1_min"] == pytest.approx(0.5)
    assert summary["f1_max"] == pytest.approx(0.75)
    assert summary["f1_range"] == pytest.approx(0.25)


def test_each_swept_tempo_is_pooled_by_counts_not_averaged(monkeypatch):
    """Per-tempo rows carry their own tp/fp/fn, so a later reader can re-derive
    the mean rather than trust it."""
    rows, _ = _sweep(monkeypatch, {60.0: (2, 1, 3)})

    assert (rows[0]["tp"], rows[0]["fp"], rows[0]["fn"]) == (2, 1, 3)
    assert rows[0]["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 3))


def test_the_sweep_scores_each_tempo_through_the_same_path(monkeypatch):
    """The swept tempo must reach `_score_real_ornaments` unchanged.

    The sweep exists to be comparable with the single-tempo pass -- verified
    against the real corpus, where the sweep's 100 BPM row reproduces the
    standalone run's tp/fp/fn exactly. If the sweep passed a different tempo
    (or silently reused one), it would measure something else while looking
    like the same number.
    """
    from tools import benchmark_notation as B

    seen = []

    def fake_score(truths_and_scores, bpm, quiet):
        seen.append(bpm)
        return [DetectionResult(label="x", kind="trill", tp=1, fp=0, fn=0)]

    monkeypatch.setattr(B, "_score_real_ornaments", fake_score)
    B._sweep_real_ornaments([], (60.0, 140.0), quiet=True)

    assert seen == [60.0, 140.0]


def test_a_sweep_with_nothing_scoreable_reports_none_not_zero(monkeypatch):
    """Same rule as `unscoreable`: a missing measurement is not a zero.

    If every tempo yields tp=fp=fn=0 the detector has not been shown to be bad,
    and averaging that to 0.000 would file a perfect negative result in the
    failure column.
    """
    _, summary = _sweep(monkeypatch, {60.0: (0, 0, 0), 100.0: (0, 0, 0)})

    assert summary["n_scoreable"] == 0
    assert summary["f1_mean"] is None
    assert summary["f1_range"] is None


# --- staccato scoring -----------------------------------------------------

def _articulated(n=6, staccato_every=2):
    """`n` quarter notes, every `staccato_every`-th marked staccato.

    `staccato_every=0` marks none -- note that `i % huge == 0` is still true
    for i=0, so "a number bigger than n" does NOT give a legato score.
    """
    from music21 import articulations, note, stream

    s = stream.Stream()
    for i in range(n):
        element = note.Note(60 + i, quarterLength=1.0)
        if staccato_every and i % staccato_every == 0:
            element.articulations.append(articulations.Staccato())
        s.append(element)
    return s


def test_the_rendered_performance_actually_plays_staccato_notes_short():
    """Without this step there is nothing to detect. Notation says WHETHER a
    note is staccato; the detector asks whether one was played short, so a
    benchmark run straight off notated durations would score the renderer's
    defaults rather than the detector."""
    from evaluation.notation import render_articulation

    notes, reference = render_articulation(_articulated(), bpm=100.0)

    assert len(reference) == 3
    short = [n.duration for n in notes[::2]]
    long = [n.duration for n in notes[1::2]]
    assert max(short) < min(long)


def test_staccato_scores_through_the_real_quantise_pipeline():
    """Hand-built QuantisedNotes are exactly what hid the Phase 21 bug: it
    lived in the interaction between quantisation and detection, so the score
    must go through `quantise_notes` rather than around it."""
    from evaluation.notation import score_staccato

    result = score_staccato(_articulated(), bpm=100.0, label="x")

    assert result.valid
    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)


def test_a_legato_score_yields_no_staccato_reference_and_is_unscoreable():
    from evaluation.notation import score_staccato

    result = score_staccato(_articulated(n=4, staccato_every=0), bpm=100.0)

    assert result.valid is False
    assert result.as_row()["f1"] is None


def test_truncation_past_the_staccato_is_not_scored_as_a_zero():
    """REGRESSION: the note cap filters the reference by onset, so a piece
    whose marks all fall past the prefix produced 0 tp / 0 fp / N fn and
    scored 0.000 -- filing a failure against a detector that correctly found
    nothing in material that contained nothing. Measured on a quartet movement
    whose staccato all falls past note 600."""
    from music21 import articulations, note, stream
    from evaluation.notation import score_staccato

    s = stream.Stream()
    for i in range(20):
        element = note.Note(60 + (i % 12), quarterLength=1.0)
        if i >= 15:
            element.articulations.append(articulations.Staccato())
        s.append(element)

    result = score_staccato(s, bpm=100.0, max_notes=5)

    assert result.valid is False
    assert "beyond the scored prefix" in result.invalid_reason


# --- key scoring ----------------------------------------------------------

def test_the_relative_minor_matches_the_signature_but_not_the_tonic():
    """THE most interpretable number this benchmark produces, and the reason
    signature and tonic are scored separately. D minor and F major share one
    flat: the reading spells every accidental correctly while naming the wrong
    tonic. Measured over 25 corpus scores, signature 0.80 against tonic 0.60,
    and this is the whole of the gap."""
    result = score_key(_Estimate("D", "minor"), truth_sharps=-1,
                       truth_tonic="F", label="x")

    assert result.signature_match is True
    assert result.tonic_match is False


def test_a_declined_key_reading_is_scored_as_a_miss_not_skipped():
    """`detect_key` returns None on material too chromatic to call. Printing
    no key signature is still a decision the reader lives with, so it is
    scored rather than excluded."""
    result = score_key(None, truth_sharps=2, truth_tonic="D", label="x")

    assert result.signature_match is False
    assert result.confident is False
    assert result.est_sharps is None


def test_tonic_accuracy_ignores_scores_whose_truth_names_no_tonic():
    """Most corpus scores carry a bare KeySignature, which counts accidentals
    without saying which of the two keys sharing them is meant. Scoring those
    as tonic misses would report the corpus's silence as the detector's
    error."""
    named = score_key(_Estimate("D", "major"), 2, "D", label="named")
    unnamed = score_key(_Estimate("D", "major"), 2, "", label="unnamed")

    stats = key_accuracy([named, unnamed])

    assert stats["n"] == 2
    assert stats["n_tonic_labelled"] == 1
    assert stats["tonic_accuracy"] == pytest.approx(1.0)


def test_key_accuracy_reports_strata_separately():
    """Palestrina is 71% of the music21 corpus, so a pooled figure describes
    repertoire this project does not target."""
    tonal = score_key(_Estimate("D", "major"), 2, "D", label="a",
                      stratum="tonal")
    modal = score_key(_Estimate("D", "major"), -1, "F", label="b",
                      stratum="modal")

    stats = key_accuracy([tonal, modal])

    assert stats["by_stratum"]["tonal"]["signature_accuracy"] == pytest.approx(1.0)
    assert stats["by_stratum"]["modal"]["signature_accuracy"] == pytest.approx(0.0)


def test_key_accuracy_on_no_results_is_none_not_zero():
    assert key_accuracy([])["signature_accuracy"] is None


# --- ground-truth extraction ---------------------------------------------

def test_ground_truth_notes_are_not_clamped():
    """`NoteEvent.__post_init__` lengthens sub-20ms notes when clamping is on.
    That is right for engine output and wrong for a reference: it rewrites the
    ground truth before it is scored. `read_midi` sets clamp=False for exactly
    this reason."""
    from music21 import note, stream

    s = stream.Stream()
    # 1/64th note at 600 BPM -> well under the 20ms clamp floor.
    s.append(note.Note("C5", quarterLength=0.0625))

    notes = notes_from_score(s, bpm=600.0)

    assert notes[0].clamp is False


def test_notes_outside_the_piano_range_are_dropped_not_clamped():
    """The corpus contains vocal and ensemble music. Transposing an out-of-
    range note into the piano's compass would invent evidence, and
    `NoteEvent` raises on it rather than allowing it."""
    from music21 import note, stream

    s = stream.Stream()
    s.append(note.Note("C0"))   # MIDI 12, below A0
    s.append(note.Note("C4"))

    notes = notes_from_score(s, bpm=100.0)

    assert [n.pitch for n in notes] == [60]


def test_a_score_that_will_not_parse_is_counted_not_dropped():
    """Silent exclusion is how a benchmark reports 0.95 on the eleven files
    that happened to work."""
    def boom(path):
        raise ValueError("bad file")

    truth, parsed = NC.load_truth(Path("nonexistent.mxl"), loader=boom)

    assert parsed is None
    assert truth.usable is False
    assert "parse failed" in truth.skipped_reason


def test_a_score_without_a_key_signature_is_skipped_with_a_reason():
    from music21 import note, stream

    s = stream.Stream()
    for _ in range(12):
        s.append(note.Note("C4"))

    truth, _ = NC.load_truth(Path("x.mxl"), loader=lambda p: s)

    assert truth.usable is False
    assert truth.skipped_reason == "no key signature"


def test_material_below_the_detector_minimum_is_skipped_not_scored():
    """`config.KEY_MIN_NOTES` makes `detect_key` return None below 8 notes.
    Scoring that would measure the guard, not the detector."""
    from music21 import key as m21key, note, stream

    s = stream.Stream()
    s.append(m21key.Key("D"))
    for _ in range(3):
        s.append(note.Note("D4"))

    truth, _ = NC.load_truth(Path("x.mxl"), loader=lambda p: s)

    assert truth.skipped_reason == "too few notes"


def test_the_summary_counts_every_skip_reason():
    truths = [
        NC.ScoreTruth(label="a", path="a", sharps=0),
        NC.ScoreTruth(label="b", path="b", skipped_reason="no key signature"),
        NC.ScoreTruth(label="c", path="c", skipped_reason="no key signature"),
        NC.ScoreTruth(label="d", path="d", skipped_reason="parse failed: OSError"),
    ]

    summary = NC.summarise(truths)

    assert summary["n_selected"] == 4
    assert summary["n_usable"] == 1
    assert summary["n_skipped"] == 3
    assert summary["skipped_reasons"]["no key signature"] == 2


# --- selection ------------------------------------------------------------

def test_selection_is_reproducible_for_a_seed():
    paths = [Path(f"corpus/bach/{i}.mxl") for i in range(50)]
    a = NC.select_scores(10, seed=13, paths=paths)
    b = NC.select_scores(10, seed=13, paths=paths)
    assert a == b


def test_selection_is_stratified_so_palestrina_cannot_swamp_the_sample():
    """Palestrina alone is 1,318 of 3,194 parseable corpus scores. A uniform
    sample drew 6 of 8 modal, and the headline signature accuracy read 0.500
    -- a fact about Renaissance polyphony, not about the detector."""
    paths = ([Path(f"corpus/palestrina/{i}.mxl") for i in range(90)]
             + [Path(f"corpus/bach/{i}.mxl") for i in range(10)])

    picked = NC.select_scores(10, seed=13, paths=paths)

    assert sum(NC.is_modal(p) for p in picked) == 5


def test_an_explicit_collection_filter_is_not_overridden_by_stratification():
    """A caller naming collections is saying what they want measured; adding
    modal scores back would silently re-include what they excluded."""
    paths = ([Path(f"corpus/palestrina/{i}.mxl") for i in range(50)]
             + [Path(f"corpus/bach/{i}.mxl") for i in range(50)])

    picked = NC.select_scores(10, seed=13, collections=("bach",), paths=paths)

    assert picked and not any(NC.is_modal(p) for p in picked)


def test_asking_for_more_scores_than_exist_returns_all_of_them():
    paths = [Path(f"corpus/bach/{i}.mxl") for i in range(3)]
    assert len(NC.select_scores(99, paths=paths)) == 3


# --- CLI wiring -----------------------------------------------------------

def test_bad_arguments_are_rejected_before_any_work():
    from tools.benchmark_notation import main

    assert main(["--n", "0"]) == 1
    assert main(["--bpm", "0"]) == 1


def test_an_unwritable_json_path_fails_before_the_run(tmp_path):
    """The run is minutes of parsing. Discovering the output path is bad
    afterwards wastes all of it."""
    from tools.benchmark_notation import main

    assert main(["--json", str(tmp_path), "--quiet", "--n", "2"]) == 1


def test_the_artifact_carries_its_own_provenance(tmp_path):
    """`tools/calibrate_frame_threshold.py` writes an artifact whose committed
    version has an environment block the tool cannot produce -- it was added
    by hand. Every number here must be regenerable from the artifact alone."""
    import json

    from tools.benchmark_notation import main

    out = tmp_path / "artifact.json"
    assert main(["--n", "2", "--quiet", "--json", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["schema"] == 1
    assert payload["generated"].endswith("Z")
    assert payload["environment"]["git_commit"]
    assert payload["source"]["seed"] == NC.SELECTION_SEED
    assert payload["conclusion"]


def test_the_artifact_says_why_dynamics_and_meter_are_unscored(tmp_path):
    """Both are absent for reasons, and an artifact that simply omitted them
    would read as an oversight."""
    import json

    from tools.benchmark_notation import main

    out = tmp_path / "artifact.json"
    main(["--n", "2", "--quiet", "--json", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["dynamics"]["scored"] is False
    assert payload["dynamics"]["reason"]
    assert payload["meter"]["scored"] is False
    assert "no meter detector exists" in payload["meter"]["reason"]
