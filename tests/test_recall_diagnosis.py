"""Which notes the model misses: `evaluation/recall_diagnosis.py` (Phase 27).

THE POINT OF THIS FILE
----------------------
The diagnosis exists to aim a GPU week. If it mislabels which notes were missed,
it aims that week at the wrong deficit -- and unlike a wrong benchmark number,
nothing downstream would contradict it, because there is no second measurement
of "which notes". So the matching itself is pinned here, on inputs whose right
answer is known by construction.

`test_missed_is_the_same_set_mir_eval_counts` is the load-bearing one: it asserts
this module and `metrics.score` disagree about nothing.
"""

import numpy as np
import pytest

from evaluation.metrics import score
from evaluation.recall_diagnosis import (
    BANDS,
    aggregate,
    band_of,
    format_profile,
    profile_track,
)
from transcriber.events import NoteEvent, Transcription


def _tr(notes):
    return Transcription(notes=list(notes), pedals=[],
                         duration=max([n.offset for n in notes], default=0.0))


def _n(pitch, onset, dur=0.3, vel=80):
    return NoteEvent(pitch, onset, onset + dur, vel)


# --- the contract ---------------------------------------------------------

def test_a_perfect_transcription_misses_nothing():
    notes = [_n(60, 0.0), _n(64, 0.5), _n(67, 1.0)]

    p = profile_track(_tr(notes), _tr(notes))

    assert p.n_missed == 0
    assert p.recall == 1.0
    assert p.invented == []


def test_a_note_the_estimate_lacks_is_missed():
    ref = [_n(60, 0.0), _n(64, 0.5), _n(67, 1.0)]
    est = [_n(60, 0.0), _n(67, 1.0)]

    p = profile_track(_tr(ref), _tr(est))

    assert p.n_missed == 1
    assert p.missed[0].pitch == 64


def test_a_note_the_reference_lacks_is_invented():
    ref = [_n(60, 0.0)]
    est = [_n(60, 0.0), _n(72, 0.5)]

    p = profile_track(_tr(ref), _tr(est))

    assert p.n_missed == 0
    assert [n.pitch for n in p.invented] == [72]


def test_an_empty_estimate_misses_everything():
    """A model that returns nothing must not read as perfect recall."""
    ref = [_n(60, 0.0), _n(64, 0.5)]

    p = profile_track(_tr(ref), Transcription(notes=[], pedals=[], duration=1.0))

    assert p.n_missed == 2
    assert p.recall == 0.0


# --- agreement with the published numbers ---------------------------------

def test_missed_is_the_same_set_mir_eval_counts():
    """THE test.

    `metrics.score` reports recall by counting mir_eval's matching; this module
    keeps the unmatched indices from the SAME matcher. If the two ever disagree,
    the diagnosis is describing a different set of notes than the recall figure
    everyone quotes, and every conclusion drawn from it is about a phantom.
    """
    rng = np.random.default_rng(7)
    ref_notes = [_n(int(p), float(t))
                 for p, t in zip(rng.integers(21, 108, 120),
                                 np.sort(rng.uniform(0, 60, 120)))]
    # Drop a third at random, and add some invented ones, to make both
    # directions non-trivial.
    keep = [n for i, n in enumerate(ref_notes) if i % 3]
    est_notes = keep + [_n(int(p), float(t))
                        for p, t in zip(rng.integers(21, 108, 20),
                                        rng.uniform(0, 60, 20))]

    p = profile_track(_tr(ref_notes), _tr(est_notes))
    s = score(_tr(ref_notes), _tr(est_notes))

    # Recall from the diagnosis must equal recall from the metric.
    assert p.recall == pytest.approx(s.onset_recall, abs=1e-9)
    # And the miss COUNT must be the deficit that recall implies.
    implied = s.n_reference - round(s.onset_recall * s.n_reference)
    assert p.n_missed == implied


def test_it_matches_on_onsets_not_offsets():
    """A note found at the right time with the wrong DURATION is found.

    The project's headline figures are onset F1. Folding offsets in here would
    count correctly-detected notes as missing and inflate the deficit this
    phase is trying to explain -- and PTify's durations are known to be
    miscalibrated (HANDOFF section on the frame head), so it would have
    inflated it a lot.
    """
    ref = [NoteEvent(60, 0.0, 2.0, 80)]
    est = [NoteEvent(60, 0.0, 0.2, 80)]

    p = profile_track(_tr(ref), _tr(est))

    assert p.n_missed == 0


# --- the breakdowns -------------------------------------------------------

def test_every_piano_pitch_lands_in_a_band():
    for pitch in range(21, 109):
        assert band_of(pitch) != "out-of-range"


def test_bands_do_not_overlap_or_gap():
    ordered = sorted(((lo, hi) for _, lo, hi in BANDS))
    assert ordered[0][0] == 21
    assert ordered[-1][1] == 108
    for (_, hi), (lo, _) in zip(ordered, ordered[1:]):
        assert lo == hi + 1


def test_the_breakdown_reports_rates_against_the_full_reference():
    """A band's denominator is every REFERENCE note in it, not every missed one.

    Rates are the whole point: the middle register holds most notes in most
    piano music, so it tops a raw count of misses while being the band the model
    handles best. A denominator bug here would invert the conclusion.
    """
    ref = [_n(30, 0.0), _n(31, 1.0),          # 2 in the contra band
           _n(60, 2.0), _n(62, 3.0), _n(64, 4.0)]  # 3 in the middle
    est = [_n(60, 2.0), _n(62, 3.0), _n(64, 4.0)]  # the contra ones are missed

    p = profile_track(_tr(ref), _tr(est))
    bands = p.by_band()

    assert bands["contra   A0-B1"] == (2, 2)     # all missed
    assert bands["middle   C4-B4"] == (0, 3)     # none missed


def test_velocity_buckets_count_the_quiet_ones():
    ref = [_n(60, 0.0, vel=20), _n(62, 1.0, vel=20), _n(64, 2.0, vel=100)]
    est = [_n(64, 2.0, vel=100)]

    p = profile_track(_tr(ref), _tr(est))

    assert p.by_velocity()["pp  <40"] == (2, 2)
    assert p.by_velocity()["f   80+"] == (0, 1)


def test_polyphony_counts_simultaneous_onsets():
    """A six-note chord is six notes at 6 voices, not one event."""
    chord = [_n(60 + i * 3, 0.0) for i in range(6)]
    single = [_n(72, 5.0)]

    p = profile_track(_tr(chord + single), _tr(single))

    assert p.by_polyphony()["5-6 voices"] == (6, 6)
    assert p.by_polyphony()["1-2 voices"] == (0, 1)


def test_polyphony_uses_onsets_not_sustained_overlap():
    """Notes still RINGING are not the crowd an onset head has to resolve.

    Under a pedal most of a bar overlaps, so counting sustained overlap would
    report near-maximal polyphony everywhere and the dimension would carry no
    information.
    """
    held = NoteEvent(36, 0.0, 10.0, 80)          # rings the whole time
    later = NoteEvent(72, 5.0, 5.2, 80)          # struck alone, much later
    p = profile_track(_tr([held, later]), _tr([held]))

    assert p.by_polyphony()["1-2 voices"] == (1, 2)


# --- aggregation ----------------------------------------------------------

def test_aggregate_pools_rather_than_averaging():
    """A 4,000-note track must not count the same as a 200-note one.

    Averaging per-track rates answers "where does the average track miss
    notes"; the question is where the CORPUS misses them.
    """
    big_ref = [_n(60, i * 0.5) for i in range(100)]
    big_est = big_ref[:]                       # perfect
    small_ref = [_n(30, i * 0.5) for i in range(4)]
    small_est = []                             # all missed

    a = profile_track(_tr(big_ref), _tr(big_est), label="big")
    b = profile_track(_tr(small_ref),
                      Transcription(notes=[], pedals=[], duration=2.0),
                      label="small")

    total = aggregate([a, b])

    assert total.n_reference == 104
    assert total.n_missed == 4
    # Pooled: 4/104, not the per-track mean of (0.0 + 1.0)/2.
    assert total.recall == pytest.approx(100 / 104, abs=1e-9)


def test_the_report_renders_without_a_zero_division():
    """Empty bands are common -- most pieces touch neither extreme of the
    keyboard -- and a report that raises is a report nobody reads."""
    p = profile_track(_tr([_n(60, 0.0)]), _tr([]))

    text = format_profile(p)

    assert "MISSED" in text
    assert "middle" in text


def test_a_constant_velocity_reference_reports_no_velocity_breakdown():
    """MAPS assigns every note velocity 80, and the first run of this tool
    rendered that as a single '15.3%' row that looked like a finding.

    A dimension the corpus cannot measure must be ABSENT and say so, not
    present and degenerate -- `ScoreResult.velocity_valid` exists for the same
    reason on the same corpus.
    """
    ref = [_n(60, 0.0, vel=80), _n(62, 1.0, vel=80), _n(64, 2.0, vel=80)]
    est = [_n(64, 2.0, vel=80)]

    p = profile_track(_tr(ref), _tr(est))

    assert p.velocity_valid is False
    assert p.by_velocity() == {}
    text = format_profile(p)
    assert "no dynamics" in text
    # The register breakdown still works -- only velocity is degenerate.
    assert "by register" in text


def test_a_reference_with_real_dynamics_still_reports_velocity():
    ref = [_n(60, 0.0, vel=30), _n(62, 1.0, vel=100)]
    est = [_n(62, 1.0, vel=100)]

    p = profile_track(_tr(ref), _tr(est))

    assert p.velocity_valid is True
    assert p.by_velocity()["pp  <40"] == (1, 1)
    assert "by velocity" in format_profile(p)
