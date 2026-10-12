"""Running the rhythm model on a transcription: `rhythm/infer.py` + wiring.

THE POINT OF THIS FILE
----------------------
Plumbing, not musical quality (an untrained model is used on purpose): every
note must come back exactly once, in a usable order, from pieces longer than
one window, and the score must still engrave. Quality is the benchmark's job.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from notation.quantise import grid_from_tempo  # noqa: E402
from notation.score import transcription_to_score  # noqa: E402
from rhythm.infer import predict, rhythm_quantise  # noqa: E402
from rhythm.model import RhythmTransformer  # noqa: E402
from transcriber.events import NoteEvent, Transcription  # noqa: E402


def _tiny(max_len=16):
    torch.manual_seed(0)
    return RhythmTransformer(d_model=32, n_layers=1, n_heads=2, max_len=max_len).eval()


def _notes(n, seed=0):
    rng = np.random.default_rng(seed)
    return [NoteEvent(int(rng.integers(40, 90)), 0.25 * i, 0.25 * i + 0.2,
                      int(rng.integers(30, 110))) for i in range(n)]


def test_predict_covers_every_note_once_across_windows():
    notes = _notes(50)                       # longer than one 16-note window
    p = predict(_tiny(), notes)
    assert sorted(p["order"].tolist()) == list(range(50))
    for k in ("delta", "duration_ticks", "downbeat_prob"):
        assert len(p[k]) == 50
    assert np.all(p["delta"] >= 0)
    assert np.all((p["downbeat_prob"] >= 0) & (p["downbeat_prob"] <= 1))


def test_predict_handles_unsorted_input_and_reports_the_order():
    notes = _notes(10)[::-1]
    p = predict(_tiny(), notes)
    onsets = [notes[i].onset for i in p["order"]]
    assert onsets == sorted(onsets)


def test_rhythm_quantise_returns_one_written_note_per_note():
    notes = _notes(30)
    grid, qnotes = rhythm_quantise(notes, _tiny())
    assert len(qnotes) == 30
    assert {id(q.source) for q in qnotes} == {id(n) for n in notes}
    assert all(q.length_beats > 0 for q in qnotes)
    assert len(grid.beats) >= 2
    assert all(b2 > b1 for b1, b2 in zip(grid.beats, grid.beats[1:]))


def test_score_engraves_with_the_model_rhythm():
    tr = Transcription(notes=_notes(40), duration=11.0, engine="t")
    sc, stats = transcription_to_score(tr, grid_from_tempo(120.0, 11.0),
                                       rhythm="model", rhythm_model=_tiny())
    assert stats.n_notes == 40
    assert stats.rhythm == "model"
    assert len(stats.grid_beats) >= 2
    assert sc is not None


def test_model_rhythm_without_a_model_is_refused():
    tr = Transcription(notes=_notes(5), duration=2.0, engine="t")
    with pytest.raises(ValueError):
        transcription_to_score(tr, grid_from_tempo(120.0, 2.0), rhythm="model")


def test_default_rhythm_is_unchanged():
    tr = Transcription(notes=_notes(8), duration=3.0, engine="t")
    _, stats = transcription_to_score(tr, grid_from_tempo(120.0, 3.0))
    assert stats.rhythm == "grid"


# --- hybrid: model rhythm on the pipeline's grid ------------------------------

def test_hybrid_places_notes_on_the_given_grid():
    notes = _notes(30)
    grid = grid_from_tempo(120.0, 9.0)                 # beats every 0.5s
    out_grid, qnotes = rhythm_quantise(notes, _tiny(), grid=grid, bars="grid")
    assert len(qnotes) == 30
    assert out_grid.beats_per_bar == grid.beats_per_bar
    # the pipeline's beats are kept, so bars and beat F come from the audio grid
    assert out_grid.beats[-len(grid.beats):] == grid.beats
    starts = [q.start_beats for q in sorted(qnotes, key=lambda q: q.source.onset)]
    assert all(b >= a for a, b in zip(starts, starts[1:]))


def test_hybrid_engraves_through_transcription_to_score():
    tr = Transcription(notes=_notes(40), duration=11.0, engine="t")
    _, stats = transcription_to_score(tr, grid_from_tempo(120.0, 11.0),
                                      rhythm="hybrid", rhythm_model=_tiny(),
                                      rhythm_bars="grid")
    assert stats.rhythm == "hybrid" and stats.n_notes == 40


def test_unknown_rhythm_bars_is_refused():
    tr = Transcription(notes=_notes(5), duration=2.0, engine="t")
    with pytest.raises(ValueError):
        transcription_to_score(tr, grid_from_tempo(120.0, 2.0), rhythm="hybrid",
                               rhythm_model=_tiny(), rhythm_bars="sometimes")


# --- choosing the beat grid by the model's own confidence ---------------------
#
# Schumann's Toccata: the beat tracked from the notes locked onto groups of
# three 16ths, and the model (correctly, relative to that pulse) wrote triplets:
# 0% rhythm where the audio grid gave 86-90%. So the model reads the piece on
# both grids and keeps the one it is more confident about.

def test_predict_reports_its_confidence():
    p = predict(_tiny(), _notes(20))
    assert 0.0 < p["confidence"] <= 1.0


def test_choose_grid_keeps_the_more_confident_reading(monkeypatch):
    import rhythm.infer as ri
    calls = []

    def fake_predict(model, notes, beats=None):
        calls.append(beats is None)
        out = predict(model, notes, beats)
        out["confidence"] = 0.4 if beats is None else 0.9   # audio grid wins
        return out

    monkeypatch.setattr(ri, "predict", fake_predict)
    grid = grid_from_tempo(120.0, 9.0)
    _, qnotes = rhythm_quantise(_notes(30), _tiny(), grid=grid, bars="model",
                                choose_grid=True)
    assert calls == [True, False]                       # both grids were read
    assert len(qnotes) == 30


# --- gaps that stay on the beat (Phase 33) --------------------------------------

def test_predict_exposes_what_the_phase_decoder_needs():
    from rhythm.features import N_DELTA_CLASSES
    p = predict(_tiny(), _notes(20))
    assert p["delta_logits"].shape == (20, N_DELTA_CLASSES)
    assert p["beat_pos"].shape == (20,)
    assert (p["delta"] == p["delta_logits"].argmax(-1)).all()


def test_the_model_path_decodes_gaps_on_the_beat(monkeypatch):
    import rhythm.infer as ri
    calls = []
    real = ri.phase_consistent_deltas
    monkeypatch.setattr(ri, "phase_consistent_deltas",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    rhythm_quantise(_notes(20), _tiny())
    assert calls


# --- the written scale, chosen at decode time (Phase 33, beat level) --------------
#
# The model writes in TRACKED beats, which were 2x-3x too fast in most benchmark
# pieces. A learned scale head did not generalise (run 7: 32.6% on validation).
# Real scores' most common gap is a 16th (49% of training windows, 44% of
# validation), so the scale is the one that makes the most common written gap a
# 16th: right in 44.5% of validation windows against 22.8% for no rescaling.

def test_scale_makes_the_most_common_gap_a_sixteenth():
    from rhythm.infer import written_scale_from_gaps
    assert written_scale_from_gaps(np.array([0, 12, 12, 12, 24, 0, 12])) == 0.5
    assert written_scale_from_gaps(np.array([0, 6, 6, 6, 12])) == 1.0
    assert written_scale_from_gaps(np.array([0, 24, 24, 48])) == 0.25


def test_chords_do_not_count_and_no_gaps_means_no_rescaling():
    from rhythm.infer import written_scale_from_gaps
    assert written_scale_from_gaps(np.array([0, 0, 0, 12, 0, 0, 12])) == 0.5
    assert written_scale_from_gaps(np.array([0, 0, 0])) == 1.0


def test_the_written_scale_multiplies_positions_values_and_beats(monkeypatch):
    import rhythm.infer as ri
    notes = _notes(40)
    def at(scale):
        monkeypatch.setattr(ri, "written_scale_from_gaps", lambda d: scale)
        return rhythm_quantise(notes, _tiny(), rescale=True)
    g1, q1 = at(1.0)
    g2, q2 = at(0.5)
    assert [q.start_beats * 0.5 for q in q1] == pytest.approx([q.start_beats for q in q2])
    assert [q.length_beats * 0.5 for q in q1] == pytest.approx([q.length_beats for q in q2])
    # Written beat k is tracked beat 2k: same times, half as many beats.
    assert g2.beats == pytest.approx(g1.beats[::2][:len(g2.beats)])
    assert len(g2.beats) == pytest.approx(len(g1.beats) / 2, abs=1)



def test_rescaling_is_off_by_default(monkeypatch):
    import rhythm.infer as ri
    monkeypatch.setattr(ri, "written_scale_from_gaps", lambda d: 0.5)
    notes = _notes(40)
    plain = rhythm_quantise(notes, _tiny())[1]
    assert [q.start_beats for q in plain] == [q.start_beats for q in
                                              rhythm_quantise(notes, _tiny(), rescale=False)[1]]
    assert [q.start_beats for q in plain] != [q.start_beats for q in
                                              rhythm_quantise(notes, _tiny(), rescale=True)[1]]


# --- compound meter (Phase 34) ----------------------------------------------------
#
# A dotted scale (0.75 / 1.5 quarters per tracked beat) means the beat IS a
# dotted value (12/16, 6/8). Writing it on a quarter beat put four compound
# pieces' meter wrong (benchmark factor 4/3, rhythm ~0%). So the beat stays on
# the tracked beat and the grid records its length in quarters.

def test_a_dotted_scale_keeps_the_beat_on_the_dotted_value(monkeypatch):
    import rhythm.infer as ri
    notes = _notes(40)
    g1, q1 = rhythm_quantise(notes, _tiny())
    monkeypatch.setattr(ri, "written_scale_from_gaps", lambda d: 1.5)
    g2, q2 = rhythm_quantise(notes, _tiny(), rescale=True)
    assert g2.beat_quarters == 1.5 and g1.beat_quarters == 1.0
    assert g2.beats == pytest.approx(g1.beats)                 # same beat times
    assert g2.beats_per_bar == g1.beats_per_bar               # bar in beats
    assert [q.start_beats * 1.5 for q in q1] == pytest.approx([q.start_beats for q in q2])


def test_a_simple_scale_still_puts_the_beat_on_quarters(monkeypatch):
    import rhythm.infer as ri
    monkeypatch.setattr(ri, "written_scale_from_gaps", lambda d: 0.5)
    g, _ = rhythm_quantise(_notes(40), _tiny(), rescale=True)
    assert g.beat_quarters == 1.0


# --- tempo candidates (Phase 35 follow-up) ----------------------------------------
#
# Validation: exact positions 33.8 -> 34.5% (ASAP, real), 66.0 -> 67.6% (PDMX
# held-out) when the most confident of 7 tempo candidates is kept.

def test_the_most_confident_tempo_candidate_is_kept(monkeypatch):
    import rhythm.infer as ri
    notes = _notes(40)
    cands = [np.arange(0, 12, 0.5 * f) for f in (1.0, 2.0, 0.5)]
    monkeypatch.setattr(ri, "track_beat_candidates", lambda on, ve: cands)
    real_predict, real_phase = ri.predict, ri.phase_consistent_deltas
    readings = {}
    def fake(model, ns, beats=None):
        p = real_predict(model, ns, beats=beats)
        which = next((k for k, c in enumerate(cands) if beats is not None
                      and len(beats) == len(c) and np.allclose(beats, c)), "audio")
        p["confidence"] = 0.9 if which == 2 else 0.5
        readings[id(p["beat_pos"])] = which
        return p
    used = []
    monkeypatch.setattr(ri, "predict", fake)
    monkeypatch.setattr(ri, "phase_consistent_deltas",
                        lambda logits, bpos: used.append(readings[id(bpos)]) or real_phase(logits, bpos))
    rhythm_quantise(notes, _tiny(), grid=grid_from_tempo(120.0, 12.0),
                    choose_grid=True, try_tempi=True)
    assert used == [2]
    assert sorted(v for v in readings.values() if v != "audio") == [0, 1, 2]


def test_tempo_candidates_need_grid_choice():
    with pytest.raises(ValueError):
        rhythm_quantise(_notes(20), _tiny(), try_tempi=True)
