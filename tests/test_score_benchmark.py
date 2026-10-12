"""Score-level benchmark: `evaluation/score_benchmark.py`.

THE POINT OF THIS FILE
----------------------
Phase 32 compares the PRINTED score against the published one, through a chain
of two matchings: our note -> performance note -> score note. Each column
(duration, rhythm, staff, meter) must fail for exactly its own reason. A
tempo-octave error, where we track eighths as beats, must show up in the meter
column and NOT zero every duration, or the table would blame the wrong half of
the pipeline. The cases below are small enough to work out by hand.
"""

import pytest

from evaluation.asap import BeatMap, ScoreNote
from evaluation.score_benchmark import (
    align_to_score,
    chain,
    duration_accuracy,
    evaluate_piece,
    meter_factor,
    rhythm_accuracy,
    staff_accuracy,
)
from notation.quantise import QuantisedNote
from transcriber.events import NoteEvent

# One beat per second throughout, so seconds == score beats.
PERF_MAP = BeatMap(tuple(float(t) for t in range(0, 12)))


def _score():
    # Treble C5 D5 E5 as quarter, quarter, half; bass C3 as a whole note.
    return [
        ScoreNote(72, 0.0, 1.0, "upper"), ScoreNote(74, 1.0, 1.0, "upper"),
        ScoreNote(76, 2.0, 2.0, "upper"), ScoreNote(48, 0.0, 4.0, "lower"),
    ]


def _perf():
    return [NoteEvent(72, 0.0, 0.9), NoteEvent(74, 1.0, 1.9),
            NoteEvent(76, 2.0, 3.9), NoteEvent(48, 0.0, 3.9)]


def _ours(beat_scale=1.0, lengths=(1.0, 1.0, 2.0, 4.0)):
    """Our quantised notes, each pointing at the performance note it came from.

    `beat_scale` 2.0 models a beat tracker locked onto eighths: every position
    and length is written in twice as many (half-size) beats.
    """
    perf = _perf()
    starts = (0.0, 1.0, 2.0, 0.0)
    return [QuantisedNote(p.pitch, s * beat_scale, l * beat_scale, source=p)
            for p, s, l in zip(perf, starts, lengths)]


STAFFS = ["upper", "upper", "upper", "lower"]


# --- alignment: performance -> score ---------------------------------------

def test_align_matches_same_pitch_at_the_same_beat():
    got = align_to_score(_perf(), PERF_MAP, _score())
    assert got == {0: 0, 1: 1, 2: 2, 3: 3}


def test_align_prefers_the_nearest_of_repeated_pitches():
    score = [ScoreNote(60, 0.0, 0.25, "upper"), ScoreNote(60, 0.25, 0.25, "upper")]
    perf = [NoteEvent(60, 0.27, 0.4), NoteEvent(60, 0.02, 0.2)]
    assert align_to_score(perf, PERF_MAP, score) == {0: 1, 1: 0}


def test_align_leaves_unmatched_what_is_too_far_or_the_wrong_pitch():
    # An ornament the score does not print, and a wrong note.
    perf = [NoteEvent(73, 0.0, 0.5), NoteEvent(72, 0.6, 0.9)]
    assert align_to_score(perf, PERF_MAP, _score()) == {}


def test_align_is_one_to_one():
    perf = [NoteEvent(72, 0.0, 0.5), NoteEvent(72, 0.05, 0.5)]
    assert len(align_to_score(perf, PERF_MAP, _score())) == 1


# --- meter ------------------------------------------------------------------

def test_meter_factor_is_one_when_we_track_the_written_beat():
    raw, snapped = meter_factor([0.0, 1.0, 2.0, 3.0], PERF_MAP)
    assert snapped == 1.0
    assert raw == pytest.approx(1.0)


def test_meter_factor_catches_a_tracker_locked_onto_eighths():
    raw, snapped = meter_factor([0.0, 0.5, 1.0, 1.5, 2.0], PERF_MAP)
    assert snapped == 0.5


def test_meter_factor_snaps_small_tempo_wobble():
    raw, snapped = meter_factor([0.0, 0.97, 1.95, 2.9], PERF_MAP)
    assert snapped == 1.0
    assert raw == pytest.approx(0.97, abs=0.01)


# --- the chain and its columns ----------------------------------------------

def _chain(ours=None, staffs=STAFFS):
    perf_to_score = align_to_score(_perf(), PERF_MAP, _score())
    return chain(ours or _ours(), staffs, _perf(), perf_to_score, _score())


def test_chain_links_every_note_in_a_perfect_case():
    assert len(_chain()) == 4


def test_chain_drops_our_invented_notes():
    extra = QuantisedNote(90, 0.0, 1.0, source=NoteEvent(90, 0.0, 0.5))
    assert len(_chain(_ours() + [extra], STAFFS + ["upper"])) == 4


def test_perfect_case_scores_full_marks():
    c = _chain()
    assert duration_accuracy(c, 1.0) == (4, 4)
    assert rhythm_accuracy(c, 1.0) == (2, 2)   # 3 onset groups -> 2 gaps
    assert staff_accuracy(c) == (4, 4)


def test_a_wrong_duration_costs_exactly_that_note():
    c = _chain(_ours(lengths=(1.0, 0.5, 2.0, 4.0)))
    assert duration_accuracy(c, 1.0) == (3, 4)
    assert rhythm_accuracy(c, 1.0) == (2, 2)   # onsets untouched


def test_tempo_octave_is_not_charged_to_durations_or_rhythm():
    c = _chain(_ours(beat_scale=2.0))
    assert duration_accuracy(c, 0.5) == (4, 4)
    assert rhythm_accuracy(c, 0.5) == (2, 2)


def test_a_misplaced_onset_shows_in_rhythm():
    ours = _ours()
    ours[1] = QuantisedNote(74, 1.5, 0.5, source=ours[1].source)  # D5 late
    c = _chain(ours)
    ok, total = rhythm_accuracy(c, 1.0)
    assert total == 2 and ok == 0          # both gaps around it are wrong


def test_a_wrong_staff_costs_exactly_that_note():
    c = _chain(staffs=["upper", "lower", "upper", "lower"])
    assert staff_accuracy(c) == (3, 4)


# --- one piece, end to end --------------------------------------------------

def test_evaluate_piece_reports_every_column():
    perf_anns_beats = [float(t) for t in range(0, 5)]
    r = evaluate_piece(
        label="demo", row="audio",
        ours=_ours(), our_staffs=STAFFS,
        grid_beats=[0.0, 1.0, 2.0, 3.0, 4.0], beats_per_bar=4,
        perf_notes=_perf(), perf_map=PERF_MAP,
        ref_beats=perf_anns_beats, ref_downbeats=[0.0, 4.0],
        score=_score(),
    )
    assert r.n_score == 4 and r.n_aligned == 4 and r.n_chained == 4
    assert r.factor == 1.0
    assert r.beat_f == pytest.approx(1.0)
    assert r.downbeat_f == pytest.approx(1.0)
    assert r.duration == (4, 4)
    assert r.rhythm == (2, 2)
    assert r.staff == (4, 4)


def test_chain_matches_one_pitch_at_a_time_like_the_full_matching():
    # Phase 33: mir_eval's matching builds all-notes x all-notes tables (~1.6 GB
    # on the Wanderer Fantasy). Only same-pitch notes can match, so matching
    # pitch by pitch must give exactly the pairs the full matching gives.
    import numpy as np
    import mir_eval.transcription as T

    from evaluation.score_benchmark import ONSET_TOL_SEC, _intervals

    rng = np.random.default_rng(3)
    perf = sorted((NoteEvent(int(p), float(t), float(t) + 0.3)
                   for p, t in zip(rng.integers(60, 66, 400), rng.uniform(0, 40, 400))),
                  key=lambda n: n.onset)
    ours = [QuantisedNote(n.pitch, k * 0.5, 0.5,
                          source=NoteEvent(n.pitch, n.onset + float(rng.normal(0, 0.03)),
                                           n.offset))
            for k, n in enumerate(perf) if rng.random() > 0.1]
    score = [ScoreNote(n.pitch, k * 0.5, 0.5, "upper") for k, n in enumerate(perf)]
    perf_to_score = {i: i for i in range(len(perf))}

    ref_iv, ref_hz = _intervals(perf)
    est_iv, est_hz = _intervals([q.source for q in ours])
    full = T.match_notes(ref_iv, ref_hz, est_iv, est_hz,
                         onset_tolerance=ONSET_TOL_SEC, offset_ratio=None)
    want = sorted((score[i].start_beat, ours[k].start_beats) for i, k in full)

    got = chain(ours, ["upper"] * len(ours), perf, perf_to_score, score)
    assert sorted((c.score.start_beat, c.ours.start_beats) for c in got) == want


def test_a_compound_piece_written_in_compound_time_scores_full_marks():
    # Phase 34: in 6/8 the score's beat is a dotted quarter, and so is ours,
    # but our positions are in QUARTER notes (an eighth = 0.5). Dividing by the
    # beat length puts them in our beats before the beat-level factor applies.
    eighth_beats = 1 / 3                                 # an eighth, in dotted-quarter beats
    score = [ScoreNote(60 + i % 5, i * eighth_beats, eighth_beats, "upper") for i in range(9)]
    perf = [NoteEvent(60 + i % 5, i * eighth_beats, i * eighth_beats + 0.3) for i in range(9)]
    ours = [QuantisedNote(p.pitch, i * 0.5, 0.5, source=p) for i, p in enumerate(perf)]
    common = dict(label="x", row="audio", ours=ours, our_staffs=["upper"] * 9,
                  grid_beats=[0.0, 1.0, 2.0, 3.0], beats_per_bar=2,
                  perf_notes=perf, perf_map=PERF_MAP, ref_beats=[0.0, 1.0, 2.0],
                  ref_downbeats=[0.0, 2.0], score=score)
    r = evaluate_piece(**common, beat_quarters=1.5)
    assert r.factor == 1.0
    assert r.rhythm == (8, 8) and r.duration == (9, 9)
    # Read as quarter beats, the same notes are wrong: what the old code did.
    assert evaluate_piece(**common).rhythm[0] == 0


def test_meter_factor_knows_the_dotted_relationship_in_every_octave():
    # 2026-10-12 (user decision): FACTORS had 2/3 and 3/2 (a quarter-note beat
    # against a dotted-quarter beat) and 3, 1/3, but not 4/3 and 3/4 -- the same
    # relationship one octave over -- so 12/16 pieces written with a quarter
    # beat snapped to 1.5 and scored ~0% rhythm.
    _, snapped = meter_factor([0.0, 0.75, 1.5, 2.25, 3.0], PERF_MAP)
    assert snapped == pytest.approx(3 / 4)
    _, snapped = meter_factor([0.0, 4 / 3, 8 / 3], PERF_MAP)
    assert snapped == pytest.approx(4 / 3)


# --- the bar (2026-10-12) -----------------------------------------------------
#
# The end-to-end check found the Schumann Toccata printed in 1/4 and Rachmaninoff
# 23/6 (4/4 in ASAP) in 3/4, and no column showed either. `bar_ok`: our bar,
# counted in score beats through the beat-level factor, equals the score's
# (beats between ASAP's annotated downbeats).

def _bar_case(beats_per_bar, grid_beats, ref_downbeats):
    return evaluate_piece(
        label="demo", row="audio", ours=_ours(), our_staffs=STAFFS,
        grid_beats=grid_beats, beats_per_bar=beats_per_bar,
        perf_notes=_perf(), perf_map=PERF_MAP,
        ref_beats=[float(t) for t in range(0, 9)], ref_downbeats=ref_downbeats,
        score=_score())


def test_the_right_bar_is_counted():
    assert _bar_case(4, [float(t) for t in range(9)], [0.0, 4.0, 8.0]).bar_ok


def test_a_wrong_bar_is_not():
    assert not _bar_case(3, [float(t) for t in range(9)], [0.0, 4.0, 8.0]).bar_ok


def test_the_bar_is_compared_through_the_beat_level():
    # Our beats at twice the score's rate (factor 1/2): 8 of ours = 4 of theirs.
    assert _bar_case(8, [t / 2 for t in range(17)], [0.0, 4.0, 8.0]).bar_ok
