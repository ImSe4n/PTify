"""The training segment index — determinism, leakage, and lossless expansion.

Three properties matter here, and none of them fail loudly on their own:

  - **Determinism.** The index names which audio a checkpoint saw. If it
    drifts between runs, a training result stops being reproducible and
    nothing reports an error.
  - **No leakage.** Segments overlap by 90%, so one performance appearing in
    both train and validation would validate the model on audio it had
    memorised — inflating the Phase 16 dev gate that decides whether the whole
    training track continues.
  - **Lossless expansion.** The index stores tracks and regenerates segments
    to keep the artifact at 0.44MB instead of 231MB. If expansion disagreed
    with generation, training would read different data than the index claims.

No network, no audio, no model.
"""

import json

import pytest

from evaluation.corpus import TrackMeta, parse_metadata
from training.index import (
    HOP_SECONDS,
    SCHEMA,
    Segment,
    assert_no_track_overlap,
    build_index,
    build_segments,
    read_index,
    segment_starts,
    segments_from_index,
    summarise,
    write_index,
)
from training.targets import SEGMENT_SECONDS

HEADER = ",".join([
    "canonical_composer", "canonical_title", "split", "year",
    "midi_filename", "audio_filename", "duration",
])

# Real MAESTRO filename shapes: a long shared prefix differing only near the
# end. A fixture using `a.midi`/`b.midi` cannot catch truncation collisions.
_P = "MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1"

CSV = HEADER + "\n" + "\n".join([
    f'Joseph Haydn,Sonata Hob.XVI:41,train,2011,2011/{_P}_02_Track02.midi,'
    f'2011/{_P}_02_Track02.wav,240.8',
    f'Joseph Haydn,Sonata Hob.XVI:41,validation,2011,2011/{_P}_03_Track03.midi,'
    f'2011/{_P}_03_Track03.wav,128.6',
    'Frederic Chopin,Ballade No. 3,train,2013,2013/b.midi,2013/b.wav,35.0',
    'Claude Debussy,Estampes,validation,2015,2015/c.midi,2015/c.wav,19.5',
])


def _tracks():
    return parse_metadata(CSV)


# --- segment arithmetic ---------------------------------------------------

def test_segment_starts_are_whole_segments_only():
    """A 35s track at a 1s hop yields starts 0..25 — never 26, which would
    run 1s past the end of the audio."""
    starts = segment_starts(35.0, seconds=10.0, hop=1.0)

    assert starts[0] == 0.0
    assert starts[-1] == 25.0
    assert len(starts) == 26
    assert starts[-1] + 10.0 <= 35.0


def test_track_shorter_than_a_segment_yields_nothing():
    """Padding silence with no labels costs a full forward pass and teaches
    nothing, so short tracks are dropped rather than padded."""
    assert segment_starts(9.9, seconds=10.0, hop=1.0) == []


def test_track_exactly_one_segment_long_yields_one():
    assert segment_starts(10.0, seconds=10.0, hop=1.0) == [0.0]


def test_hop_larger_than_segment_is_allowed():
    """Non-overlapping segmentation is legitimate — Phase 14.5 may use it to
    cut epoch size — so it must not be rejected."""
    starts = segment_starts(100.0, seconds=10.0, hop=20.0)

    assert starts == [0.0, 20.0, 40.0, 60.0, 80.0]


def test_non_positive_hop_is_rejected():
    """A zero hop would generate an unbounded list."""
    with pytest.raises(ValueError, match="hop must be positive"):
        segment_starts(100.0, hop=0.0)


def test_starts_do_not_drift_with_float_accumulation():
    """Starts are written to JSON and compared for equality, so accumulated
    float error would break the index round-trip."""
    starts = segment_starts(1000.0, seconds=10.0, hop=0.1)

    assert starts[10] == pytest.approx(1.0, abs=1e-9)
    assert all(s == round(s, 6) for s in starts)


# --- determinism ----------------------------------------------------------

def test_index_is_deterministic():
    assert build_index(CSV) == build_index(CSV)


def test_segment_order_does_not_depend_on_csv_row_order():
    """Re-downloading the metadata must not be able to reshuffle the index."""
    rows = CSV.strip().split("\n")
    shuffled = "\n".join([rows[0], rows[4], rows[2], rows[1], rows[3]])

    assert build_segments(parse_metadata(shuffled)) == build_segments(_tracks())


def test_max_tracks_per_split_truncates_deterministically():
    """Phase 14.5's smoke run takes 20 tracks; the same 20 every time."""
    a = build_segments(_tracks(), max_tracks_per_split=1)
    b = build_segments(_tracks(), max_tracks_per_split=1)

    assert a == b
    assert len({s.track for s in a if s.split == "train"}) == 1


# --- leakage --------------------------------------------------------------

def test_splits_are_track_disjoint_on_the_fixture():
    assert_no_track_overlap(build_segments(_tracks()))


def test_overlapping_track_across_splits_is_rejected():
    """The check that caught a real bug: two Haydn performances collided into
    one stem, putting one name in both train and validation."""
    leaked = [
        Segment("same-track", "a.wav", "a.midi", "train", 0.0),
        Segment("same-track", "a.wav", "a.midi", "validation", 0.0),
    ]

    with pytest.raises(ValueError, match="appears in both"):
        assert_no_track_overlap(leaked)


def test_real_maestro_filenames_do_not_collide_across_splits():
    """Both Haydn rows are the same piece, same composer, same year, and
    differ only late in the filename. Before the `corpus.py` stem fix these
    produced one stem spanning train and validation."""
    segments = build_segments(_tracks())
    by_track = {}
    for seg in segments:
        by_track.setdefault(seg.track, set()).add(seg.split)

    assert all(len(splits) == 1 for splits in by_track.values())
    # Both Haydn performances survive as distinct tracks, one per split.
    haydn = [t for t in by_track if t.startswith("Joseph_Haydn")]
    assert len(haydn) == 2
    assert {next(iter(by_track[t])) for t in haydn} == {"train", "validation"}


def test_missing_split_fails_loudly():
    with pytest.raises(ValueError, match="No tracks in split"):
        build_segments(_tracks(), splits=("test",))


# --- the stored artifact --------------------------------------------------

def test_index_expands_to_exactly_the_generated_segments():
    """The compression that took the artifact from 231MB to 0.44MB is only
    safe if expansion reproduces generation exactly — same order, not just
    the same set."""
    index = build_index(CSV)

    assert segments_from_index(index) == build_segments(_tracks())


def test_index_can_be_filtered_by_split():
    index = build_index(CSV)
    validation = segments_from_index(index, split="validation")

    assert validation
    assert {s.split for s in validation} == {"validation"}


def test_inconsistent_segment_count_is_rejected():
    """If an index is hand-edited or written with different segmentation
    parameters than it records, every `start` it expands is wrong."""
    index = build_index(CSV)
    index["tracks"][0]["n_segments"] += 5

    with pytest.raises(ValueError, match="inconsistent"):
        segments_from_index(index)


def test_index_records_its_provenance():
    index = build_index(CSV)

    assert index["schema"] == SCHEMA
    assert index["segment_seconds"] == SEGMENT_SECONDS
    assert index["hop_seconds"] == HOP_SECONDS
    assert len(index["csv_sha256"]) == 64
    assert index["license"] == "CC BY-NC-SA 4.0"


def test_index_stores_no_absolute_paths():
    """The index is written here and resolved on Kaggle, so a local absolute
    path would make it unusable there."""
    index = build_index(CSV)

    for entry in index["tracks"]:
        assert not entry["audio_filename"].startswith(("/", "C:", "\\"))
        assert ":" not in entry["audio_filename"][:3]


def test_short_tracks_are_absent_from_the_index():
    """A 4s track cannot fill a 10s segment, so it must not appear at all —
    not as a padded partial example."""
    csv = HEADER + "\n" + 'X,Y,train,2015,2015/s.midi,2015/s.wav,4.0\n' + \
        'Frederic Chopin,Ballade,train,2013,2013/b.midi,2013/b.wav,35.0'
    index = build_index(csv, splits=("train",))

    assert all(entry["duration"] >= SEGMENT_SECONDS for entry in index["tracks"])


def test_write_and_read_round_trip(tmp_path):
    index = build_index(CSV)
    path = write_index(index, tmp_path / "nested" / "index.json")

    assert read_index(path) == index


def test_reading_a_future_schema_fails_loudly(tmp_path):
    """A silently-misread index would train on the wrong segments."""
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"schema": 99}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        read_index(path)


# --- summary --------------------------------------------------------------

def test_summary_counts_tracks_and_segments():
    summary = summarise(build_segments(_tracks()))

    assert summary["train"]["tracks"] == 2       # Haydn + Chopin
    assert summary["validation"]["tracks"] == 2  # Haydn + Debussy
    assert summary["train"]["segments"] > 0


def test_summary_hours_are_exposure_not_distinct_audio():
    """At a 1s hop each second of audio appears in ~10 segments, so exposure
    hours far exceed the corpus duration. The field name and the CLI both say
    so; this pins that it is not accidentally reporting distinct hours."""
    tracks = [TrackMeta("A", "B", "train", 2011, "a.midi", "a.wav", 100.0)]
    summary = summarise(build_segments(tracks, splits=("train",)))

    # 91 segments x 10s = 910s of exposure from 100s of audio.
    assert summary["train"]["segments"] == 91
    assert summary["train"]["segment_hours"] == pytest.approx(910 / 3600, abs=1e-2)
