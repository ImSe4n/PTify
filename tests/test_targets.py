"""Training targets — round-tripped through the REAL decoder.

WHY THESE TESTS LOOK LIKE THIS
------------------------------
Asserting on array shapes would pass for an encoding that trains happily and
decodes to garbage. The model's targets are a regression ramp whose only
contract is "`RegressionPostProcessor` can invert it", so the tests invert it
with the actual post-processor rather than a reimplementation. If ByteDance
ever changes that decoder, these fail — which is the correct outcome, because
a checkpoint trained against the old encoding would silently mis-decode.

No model, no network, no GPU: `RegressionPostProcessor` is plain numpy, so
this stays inside the suite's "pure functions" discipline.
"""

import numpy as np
import pytest

from piano_transcription_inference.utilities import RegressionPostProcessor

from training.targets import (
    BEGIN_NOTE,
    CLASSES_NUM,
    FRAMES_PER_SECOND,
    VELOCITY_SCALE,
    render_targets,
    segment_frames,
)
from transcriber.events import NoteEvent, PedalEvent

# The thresholds `PianoTranscription` uses at inference (inference.py:42-45).
# Decoding with anything else would test a configuration nobody runs.
ONSET_THRESHOLD = 0.3
OFFSET_THRESHOLD = 0.3
FRAME_THRESHOLD = 0.1
PEDAL_OFFSET_THRESHOLD = 0.2


def post_processor():
    return RegressionPostProcessor(
        frames_per_second=FRAMES_PER_SECOND,
        classes_num=CLASSES_NUM,
        onset_threshold=ONSET_THRESHOLD,
        offset_threshold=OFFSET_THRESHOLD,
        frame_threshold=FRAME_THRESHOLD,
        pedal_offset_threshold=PEDAL_OFFSET_THRESHOLD,
    )


def decode_notes(targets):
    """Run the real post-processor over rendered targets -> note events."""
    output = {
        "reg_onset_output": targets["reg_onset"],
        "reg_offset_output": targets["reg_offset"],
        "frame_output": targets["frame"],
        "velocity_output": targets["velocity"],
    }
    events, _ = post_processor().output_dict_to_midi_events(output)
    return events


def decode_onsets(targets, key):
    """Recovered onset times for one key, in seconds.

    Goes through `get_binarized_output_from_regression` directly rather than
    the full note decoder, so a failure points at the ramp encoding itself
    rather than at note assembly.
    """
    pp = post_processor()
    binary, shift = pp.get_binarized_output_from_regression(
        reg_output=targets["reg_onset"], threshold=ONSET_THRESHOLD, neighbour=2
    )
    frames = np.where(binary[:, key] == 1)[0]
    return [(n + shift[n, key]) / FRAMES_PER_SECOND for n in frames]


# --- the load-bearing test ------------------------------------------------

@pytest.mark.parametrize("onset", [
    0.500,    # exactly on a frame
    0.5049,   # just below the midpoint, rounds down
    0.5051,   # just above the midpoint, rounds up
    0.5070,
    0.5099,   # just below the next frame
    1.2345,   # arbitrary
    3.3333,
])
def test_onset_round_trips_to_sub_millisecond(onset):
    """The whole training track depends on this inverting exactly.

    A wrong ramp shape still trains and still decodes; it just decodes to the
    wrong time. Nothing else in the suite would catch that.
    """
    notes = [NoteEvent(pitch=60, onset=onset, offset=onset + 0.4, velocity=80)]
    targets = render_targets(notes)

    recovered = decode_onsets(targets, key=60 - BEGIN_NOTE)

    assert len(recovered) == 1, f"expected exactly one onset, got {recovered}"
    # The plan's gate is 3ms. The encoding is algebraically exact, so hold it
    # to 0.1ms and let the gate have headroom it does not need.
    assert recovered[0] == pytest.approx(onset, abs=1e-4)


def test_offset_round_trips():
    """Offsets decode with neighbour=4, a wider window than onsets."""
    notes = [NoteEvent(pitch=72, onset=1.0, offset=2.3456, velocity=80)]
    targets = render_targets(notes)

    pp = post_processor()
    binary, shift = pp.get_binarized_output_from_regression(
        reg_output=targets["reg_offset"], threshold=OFFSET_THRESHOLD, neighbour=4
    )
    key = 72 - BEGIN_NOTE
    frames = np.where(binary[:, key] == 1)[0]

    assert len(frames) == 1
    n = frames[0]
    assert (n + shift[n, key]) / FRAMES_PER_SECOND == pytest.approx(2.3456, abs=1e-4)


def test_full_decode_recovers_pitch_time_and_velocity():
    """End to end through the same call inference makes."""
    notes = [
        NoteEvent(pitch=60, onset=0.5, offset=1.0, velocity=64),
        NoteEvent(pitch=64, onset=1.5, offset=2.0, velocity=96),
        NoteEvent(pitch=67, onset=2.5, offset=3.5, velocity=112),
    ]
    events = decode_notes(render_targets(notes))

    assert len(events) == 3
    for expected, got in zip(notes, sorted(events, key=lambda e: e["onset_time"])):
        assert got["midi_note"] == expected.pitch
        assert got["onset_time"] == pytest.approx(expected.onset, abs=2e-3)
        # Velocity round-trips through /128 then *128, so it is exact only up
        # to integer truncation.
        assert abs(got["velocity"] - expected.velocity) <= 1


def test_chord_decodes_as_separate_notes():
    """Simultaneous onsets must not merge or mask each other."""
    notes = [
        NoteEvent(pitch=p, onset=1.0, offset=2.0, velocity=80)
        for p in (60, 64, 67)
    ]
    events = decode_notes(render_targets(notes))

    assert sorted(e["midi_note"] for e in events) == [60, 64, 67]


def test_fast_repeat_keeps_both_onsets():
    """Overlapping ramps must not flatten into a plateau.

    `_paint_ramp` uses np.maximum precisely for this: assignment would let the
    second ramp overwrite the first one's peak, and a plateau fails the
    decoder's strict-monotonicity check — losing BOTH onsets, not one.
    """
    notes = [
        NoteEvent(pitch=60, onset=1.00, offset=1.08, velocity=80),
        NoteEvent(pitch=60, onset=1.12, offset=1.30, velocity=80),
    ]
    recovered = decode_onsets(render_targets(notes), key=60 - BEGIN_NOTE)

    assert len(recovered) == 2
    assert recovered[0] == pytest.approx(1.00, abs=3e-3)
    assert recovered[1] == pytest.approx(1.12, abs=3e-3)


# --- segment boundaries ---------------------------------------------------

def test_note_before_segment_is_ignored():
    notes = [NoteEvent(pitch=60, onset=0.5, offset=1.0, velocity=80)]
    targets = render_targets(notes, segment_start=10.0)

    assert targets["reg_onset"].sum() == 0.0
    assert targets["frame"].sum() == 0.0


def test_note_after_segment_is_ignored():
    notes = [NoteEvent(pitch=60, onset=50.0, offset=51.0, velocity=80)]
    targets = render_targets(notes, segment_start=0.0)

    assert targets["reg_onset"].sum() == 0.0


def test_held_note_is_supervised_in_both_segments():
    """A note crossing a boundary must sound in both, with one onset.

    Teaching the model that a held note stops at an arbitrary segment edge
    would manufacture offsets that the audio does not contain.
    """
    notes = [NoteEvent(pitch=60, onset=8.0, offset=12.0, velocity=80)]
    key = 60 - BEGIN_NOTE

    first = render_targets(notes, segment_start=0.0)
    second = render_targets(notes, segment_start=10.0)

    # Sounds in both.
    assert first["frame"][:, key].sum() > 0
    assert second["frame"][:, key].sum() > 0

    # The onset belongs to the first segment only; the offset to the second.
    assert len(decode_onsets(first, key)) == 1
    assert len(decode_onsets(second, key)) == 0
    assert first["reg_offset"][:, key].sum() == 0.0
    assert second["reg_offset"][:, key].sum() > 0.0

    # The second segment sounds from its very first frame.
    assert second["frame"][0, key] == 1.0


def test_offset_relative_to_segment_start():
    notes = [NoteEvent(pitch=60, onset=10.25, offset=10.75, velocity=80)]
    recovered = decode_onsets(
        render_targets(notes, segment_start=10.0), key=60 - BEGIN_NOTE
    )

    assert recovered == pytest.approx([0.25], abs=1e-4)


# --- shapes, dtypes and the velocity mask ---------------------------------

def test_segment_frames_is_one_more_than_seconds_times_fps():
    """center=True emits a frame at t=0 AND at t=seconds. Measured: a 3.0s
    input returns 301 frames from a real forward pass, not 300."""
    assert segment_frames(10.0, 100) == 1001
    assert segment_frames(3.0, 100) == 301


def test_shapes_and_dtypes():
    targets = render_targets([NoteEvent(60, 1.0, 2.0, 80)])
    frames = segment_frames()

    for key in ("reg_onset", "reg_offset", "frame", "velocity", "mask"):
        assert targets[key].shape == (frames, CLASSES_NUM), key
        assert targets[key].dtype == np.float32, key

    assert targets["pedal_frame"].shape == (frames, 1)


def test_empty_segment_is_all_zeros():
    """Every TARGET is zero on silence.

    `onset_weight` is excluded and must be: it is a per-cell loss weight, not a
    target, and it is all ONES by construction. Zero weights would delete the
    negative space that teaches the model not to fire -- see
    `tests/test_soft_onset_weighting.py::test_silence_keeps_full_weight`.
    """
    targets = render_targets([])

    for key, array in targets.items():
        if key == "onset_weight":
            assert (array == 1.0).all(), key
            continue
        assert array.sum() == 0.0, key
        assert np.isfinite(array).all(), key


def test_mask_marks_exactly_the_onset_frames():
    """The velocity loss is masked to these frames.

    Unmasked, velocity is supervised as 0 everywhere no note starts — which
    is most of the array — so the model learns to predict silence and the
    term dominates the total loss.
    """
    notes = [
        NoteEvent(pitch=60, onset=1.00, offset=2.0, velocity=80),
        NoteEvent(pitch=64, onset=1.50, offset=2.0, velocity=80),
    ]
    mask = render_targets(notes)["mask"]

    assert mask.sum() == 2.0
    assert mask[100, 60 - BEGIN_NOTE] == 1.0
    assert mask[150, 64 - BEGIN_NOTE] == 1.0


def test_velocity_is_scaled_by_128():
    notes = [NoteEvent(pitch=60, onset=1.0, offset=2.0, velocity=64)]
    velocity = render_targets(notes)["velocity"]

    assert velocity[100, 60 - BEGIN_NOTE] == pytest.approx(64 / VELOCITY_SCALE)
    assert velocity.max() <= 1.0


def test_velocity_stays_within_unit_range():
    """127/128 is below 1.0, but the clamp guards a target the sigmoid could
    never reach if VELOCITY_SCALE were ever lowered."""
    notes = [NoteEvent(pitch=60, onset=1.0, offset=2.0, velocity=127)]
    assert render_targets(notes)["velocity"].max() <= 1.0


def test_ramp_peaks_at_one():
    """The decoder thresholds at 0.3; a ramp that never reaches it is invisible."""
    notes = [NoteEvent(pitch=60, onset=1.0, offset=2.0, velocity=80)]
    targets = render_targets(notes)

    assert targets["reg_onset"].max() == pytest.approx(1.0)
    assert targets["reg_offset"].max() == pytest.approx(1.0)


def test_ramp_is_strictly_monotonic_around_the_peak():
    """`is_monotonic_neighbour` rejects plateaus outright."""
    notes = [NoteEvent(pitch=60, onset=1.0, offset=2.0, velocity=80)]
    column = render_targets(notes)["reg_onset"][:, 60 - BEGIN_NOTE]
    peak = int(np.argmax(column))

    for i in range(1, 3):
        assert column[peak - i] < column[peak - i + 1]
        assert column[peak + i] < column[peak + i - 1]


def test_half_width_below_two_is_rejected():
    """At J=1 the ramp misses the +-2 neighbours the decoder reads and
    sub-frame error rises to 0.86ms. Fail loudly rather than degrade."""
    with pytest.raises(ValueError, match="half_width"):
        render_targets([NoteEvent(60, 1.0, 2.0, 80)], half_width=1)


# --- pedal ----------------------------------------------------------------

def test_pedal_frame_is_rendered():
    targets = render_targets(
        [], pedals=[PedalEvent(onset=1.0, offset=2.0)]
    )
    pedal = targets["pedal_frame"]

    assert pedal[100, 0] == 1.0
    assert pedal[200, 0] == 1.0
    assert pedal[50, 0] == 0.0


def test_pedal_outside_segment_is_ignored():
    targets = render_targets([], pedals=[PedalEvent(onset=50.0, offset=51.0)])
    assert targets["pedal_frame"].sum() == 0.0
