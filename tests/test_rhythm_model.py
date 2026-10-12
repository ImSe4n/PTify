"""Rhythm model and training loop: `rhythm/model.py`, `rhythm/train.py`.

THE POINT OF THIS FILE
----------------------
A training run costs GPU quota and an hour of waiting, so anything that would
make it silently useless is caught here first: wrong tensor shapes, unlabelled
notes leaking into the loss, augmentation that corrupts targets, and a model
that cannot even memorise eight notes.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rhythm.features import (  # noqa: E402
    IGNORE,
    IGNORE_TICK,
    N_INPUT,
    N_DELTA_CLASSES,
    N_DURATION_CLASSES,
    deltas,
    encode_pair,
    pack,
)
from rhythm.model import RhythmTransformer, rhythm_loss  # noqa: E402
from rhythm.train import Windows, augment, collate  # noqa: E402


def _piece(n=40, seed=0):
    rng = np.random.default_rng(seed)
    notes, t, beat = [], 0.0, 0.0
    for i in range(n):
        step = [0.25, 0.5, 1.0][i % 3]
        notes.append({"pitch": int(rng.integers(40, 90)), "onset": t,
                      "offset": t + 0.4 * step, "velocity": int(rng.integers(30, 110)),
                      "target": {"start_beat": beat, "length_beats": step,
                                 "staff": "upper" if i % 2 else "lower"}})
        t += 0.5 * step
        beat += step
    beats = [float(b) for b in range(int(beat) + 2)]
    return {"notes": notes, "beats": [b * 0.5 for b in beats],
            "downbeats": [b * 0.5 for b in beats[::4]]}


def test_deltas_relabel_the_successor_of_a_dropped_note():
    ticks = np.array([0, 12, 24, 48], dtype=np.int64)
    assert deltas(ticks).tolist() == [IGNORE, 12, 12, 24]
    dropped = np.delete(ticks, 1)
    assert deltas(dropped).tolist() == [IGNORE, 24, 24]
    with_gap = np.array([0, IGNORE_TICK, 24], dtype=np.int64)
    assert deltas(with_gap).tolist() == [IGNORE, IGNORE, 24]


def test_model_output_shapes():
    m = RhythmTransformer(d_model=32, n_layers=1, n_heads=2, max_len=16)
    pitch = torch.zeros(2, 16, dtype=torch.long)
    cont = torch.zeros(2, 16, N_INPUT)
    pad = torch.zeros(2, 16, dtype=torch.bool)
    out = m(pitch, cont, pad)
    assert out["delta"].shape == (2, 16, N_DELTA_CLASSES)
    assert out["duration"].shape == (2, 16, N_DURATION_CLASSES)
    assert out["downbeat"].shape == (2, 16, 2)
    assert out["hand"].shape == (2, 16, 2)


def test_loss_ignores_unlabelled_notes_and_stays_finite_when_all_are():
    m = RhythmTransformer(d_model=32, n_layers=1, n_heads=2, max_len=8)
    batch = collate([Windows.slice(pack([encode_pair(_piece(8))]), 0, 8)])
    for k in ("delta", "duration", "downbeat", "hand"):
        batch[k][:] = IGNORE
    total, parts = rhythm_loss(m(batch["pitch"], batch["cont"], batch["pad"]), batch)
    assert torch.isfinite(total) and float(total) == 0.0


def test_windows_cover_every_note_and_never_cross_pieces():
    packed = pack([encode_pair(_piece(30)), encode_pair(_piece(10, seed=1))])
    w = Windows(packed, length=16, stride=12)
    covered = set()
    for start, end in w.spans:
        assert end - start <= 16
        piece = np.searchsorted(packed["offsets"], start, side="right") - 1
        assert end <= packed["offsets"][piece + 1]
        covered.update(range(start, end))
    assert covered == set(range(40))


def test_augment_keeps_targets_consistent():
    packed = pack([encode_pair(_piece(40))])
    rng = np.random.default_rng(0)
    win = augment(Windows.slice(packed, 0, 40), rng, drop_soft=0.5, drop_loud=0.2,
                  jitter_sec=0.01)
    assert len(win["pitch"]) < 40                       # something was dropped
    # Labels are recomputed from the window AFTER dropping, in tracked-beat
    # units (run 5): the gap from each surviving note's predecessor, rescaled.
    from rhythm.features import grid_scale
    r = grid_scale(win["start_tick"], win["beat_pos"])
    lab = win["start_tick"] != IGNORE_TICK
    scaled = win["start_tick"].copy()
    scaled[lab] = np.round(win["start_tick"][lab] / r).astype(np.int64)
    assert win["delta"].tolist() == deltas(scaled).tolist()
    assert np.all(np.diff(win["onset"]) >= 0)            # re-sorted after jitter


def test_a_tiny_model_can_memorise_a_tiny_piece():
    torch.manual_seed(0)
    m = RhythmTransformer(d_model=64, n_layers=2, n_heads=4, max_len=32, dropout=0.0)
    batch = collate([Windows.slice(pack([encode_pair(_piece(24))]), 0, 24)])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        total, _ = rhythm_loss(m(batch["pitch"], batch["cont"], batch["pad"]), batch)
        first = float(total) if first is None else first
        opt.zero_grad()
        total.backward()
        opt.step()
    assert float(total) < 0.2 * first


def test_window_labels_are_in_tracked_beat_units():
    # Written eighths (12 quarter-ticks apart) under a tracker that beats in HALF
    # notes: each note is a quarter of a tracked beat -> 6 ticks apart.
    piece = _piece(16)
    enc = encode_pair(piece)
    enc["start_tick"] = (np.arange(16) * 12).astype(np.int64)
    enc["length_tick"] = np.full(16, 12, dtype=np.int64)
    enc["beat_pos"] = np.arange(16) * 0.25
    win = Windows.slice(pack([enc]), 0, 16)
    assert win["delta"][1:].tolist() == [6] * 15
    from rhythm.features import DURATION_TICKS
    assert all(DURATION_TICKS[d] == 6 for d in win["duration"])


# --- downbeat class weight (Phase 33) --------------------------------------------
#
# Only ~16% of labelled notes are downbeats, so the plain loss leaves the head
# saying "no" almost everywhere (val F1 0.18 at argmax). A positive-class weight
# makes a missed downbeat cost more. Weight 1 must be the old loss exactly.

def _down_logits_and_batch(labels):
    logits = {"downbeat": torch.zeros(1, len(labels), 2)}
    logits["downbeat"][..., 0] = 1.0                   # always says "not a downbeat"
    return logits, {"downbeat": torch.tensor([labels])}


def test_downbeat_weight_one_is_the_old_loss():
    logits, batch = _down_logits_and_batch([1, 0, 0, 0])
    assert float(rhythm_loss(logits, batch)[0]) == pytest.approx(
        float(rhythm_loss(logits, batch, downbeat_pos_weight=1.0)[0]))


def test_downbeat_weight_makes_a_missed_downbeat_cost_more():
    logits, batch = _down_logits_and_batch([1, 0, 0, 0])
    plain = float(rhythm_loss(logits, batch)[0])
    assert float(rhythm_loss(logits, batch, downbeat_pos_weight=4.0)[0]) > plain


def test_downbeat_weight_leaves_the_other_heads_alone():
    logits = {"hand": torch.zeros(1, 4, 2)}
    logits["hand"][..., 0] = 1.0
    batch = {"hand": torch.tensor([[1, 0, 0, 0]])}
    assert float(rhythm_loss(logits, batch, downbeat_pos_weight=4.0)[0]) == pytest.approx(
        float(rhythm_loss(logits, batch)[0]))


def test_train_accepts_the_downbeat_weight():
    from rhythm.train import build_parser
    assert build_parser().parse_args(
        ["--data", "d", "--out", "o", "--downbeat-pos-weight", "4"]).downbeat_pos_weight == 4.0
    assert build_parser().parse_args(["--data", "d", "--out", "o"]).downbeat_pos_weight == 1.0


# --- extra training data (Phase 35, PDMX shards) ----------------------------------

def test_extra_shards_are_appended_without_crossing_pieces(tmp_path):
    from rhythm.train import load_training
    a = pack([encode_pair(_piece(20)), encode_pair(_piece(12, seed=1))])
    b = pack([encode_pair(_piece(16, seed=2))])
    np.savez_compressed(tmp_path / "train.npz", **a)
    np.savez_compressed(tmp_path / "pdmx-train-000.npz", **b)
    got = load_training(tmp_path, "pdmx-train-*.npz")
    assert got["offsets"].tolist() == [0, 20, 32, 48]
    assert got["pitch"].tolist() == a["pitch"].tolist() + b["pitch"].tolist()
    assert all(b2 <= 48 for _, b2 in Windows(got, length=8, stride=8).spans)
    assert load_training(tmp_path, None)["offsets"].tolist() == [0, 20, 32]


def test_train_accepts_an_extra_glob():
    from rhythm.train import build_parser
    a = build_parser().parse_args(["--data", "d", "--out", "o", "--extra-glob", "pdmx-train-*.npz"])
    assert a.extra_glob == "pdmx-train-*.npz"
    assert build_parser().parse_args(["--data", "d", "--out", "o"]).extra_glob is None


def test_training_data_out_of_range_fails_on_the_cpu(tmp_path):
    # A bad index reached the GPU as an unreadable CUDA assert (run 8). Check
    # every input and label range when the data is loaded instead.
    from rhythm.train import load_training
    bad = pack([encode_pair(_piece(20))])
    bad["pitch"][3] = 95
    np.savez_compressed(tmp_path / "train.npz", **bad)
    with pytest.raises(ValueError, match="pitch"):
        load_training(tmp_path, None)
