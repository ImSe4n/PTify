"""Notes -> the regression targets the ByteDance CRNN is trained against.

THIS IS THE MODULE MOST LIKELY TO FAIL SILENTLY
-----------------------------------------------
The model does not predict a binary piano roll. It predicts a *regression*
ramp around each onset and offset, and `RegressionPostProcessor` recovers
sub-frame timing from the ratio of neighbouring values. Encode the targets
with the wrong shape — a binary spike, a Gaussian, a plateau — and training
still runs, the loss still falls, and the decoder still returns notes. They
are just wrong. Nothing raises.

That is why `tests/test_targets.py` round-trips through the REAL decoder
rather than asserting on array shapes, and why it was written before any
other training code in this package.

THE ENCODING, DERIVED FROM THE DECODER
--------------------------------------
`get_binarized_output_from_regression` (utilities.py:299) accepts frame `n`
as an event when `x[n] > threshold` and the values are monotonic across
`n +- neighbour`. It then recovers the sub-frame shift as:

    if x[n-1] > x[n+1]:  shift = (x[n+1] - x[n-1]) / (x[n] - x[n+1]) / 2
    else:                shift = (x[n+1] - x[n-1]) / (x[n] - x[n-1]) / 2

That formula is the algebraic inverse of a **symmetric linear ramp** peaking
at the true event time. So the target is exactly that ramp:

    value(n) = max(0, 1 - |n - t*fps| / J)

Verified against the real decoder, not assumed: sub-frame onsets across a
whole frame recover with **0.000ms** error at J >= 2. At J = 1 the ramp does
not reach the +-2 neighbours the decoder inspects and error rises to 0.86ms.

RAMP_HALF_WIDTH = 5 is ByteDance's own value, confirmed by running the
pretrained checkpoint on a synthetic note and reading the ramp it emits:

    [0.007 0.042 0.178 0.359 0.536 0.690 0.721 0.622 0.495 0.349 0.183 ...]

i.e. a monotonic rise and fall spanning roughly +-5 frames. Any J >= 2
decodes exactly; matching the pretrained model's own scale matters because
this is a FINE-TUNE — targets that disagree with what the network already
emits would fight the initialisation instead of refining it.

TWO DECODER QUIRKS THAT CONSTRAIN THE ENCODING
----------------------------------------------
1. **A plateau is rejected.** `is_monotonic_neighbour` requires strict
   monotonicity, so two adjacent frames of equal value silently drop the
   event. A note landing exactly between two frames must still produce
   unequal neighbours — the linear ramp does, but a naive `round()` to the
   nearest frame followed by a fixed template would not.
2. **Velocity is read at the onset frame only** (`velocity_output[bgn]` in
   piano_vad.py:39). Everywhere else it is undefined, which is why
   `render_targets` also returns an onset `mask`: the velocity loss must be
   masked to those frames or it trains toward zero and dominates the total.

A THIRD QUIRK, NOT OURS TO FIX
------------------------------
`note_detection_with_onset_offset_regress` tests `if bgn:` rather than
`if bgn is not None:` (piano_vad.py:34,43), so an onset detected in frame 0
is falsy and skipped. It costs at most the first 10ms of a segment and lives
upstream; segments overlap during inference, so a note lost at one segment's
frame 0 is caught by the neighbouring segment. Noted here so a future reader
does not mistake it for a bug in this file.
"""

from __future__ import annotations

import numpy as np

from transcriber import config
from transcriber.events import NoteEvent, PedalEvent

#: Frames per second of the model's output. ByteDance's `config.py` — the
#: hop is derived from it as `sample_rate // frames_per_second`, so this is
#: not free to change without retraining from scratch.
FRAMES_PER_SECOND = 100

#: Model output classes: the 88 piano keys, A0 (21) to C8 (108).
CLASSES_NUM = config.NUM_KEYS

#: MIDI note of the lowest key, i.e. output bin 0. ByteDance `begin_note`.
BEGIN_NOTE = config.MIDI_LOWEST

#: Training segment length. ByteDance `segment_seconds`; also the window
#: `PianoTranscription` uses at inference, so training and inference see
#: identically-shaped inputs.
SEGMENT_SECONDS = 10.0

#: Half-width of the regression ramp, in frames. See the module docstring:
#: J >= 2 decodes exactly, and 5 matches the pretrained model's own output.
RAMP_HALF_WIDTH = 5

#: `RegressionPostProcessor` multiplies the velocity output by this to get a
#: MIDI velocity, so the target is velocity/128. ByteDance `velocity_scale`.
VELOCITY_SCALE = 128


def segment_frames(seconds: float = SEGMENT_SECONDS,
                   fps: int = FRAMES_PER_SECOND) -> int:
    """Frames the model emits for a segment of `seconds`.

    One MORE than seconds*fps, because the STFT runs with `center=True` and
    therefore emits a frame at t=0 as well as at t=seconds. Measured against
    a real forward pass: a 3.0s input returns 301 frames, not 300. Off-by-one
    here would silently misalign every label in the last frame.
    """
    return int(round(seconds * fps)) + 1


def _paint_ramp(target: np.ndarray, time_sec: float, key: int,
                fps: int, half_width: int) -> None:
    """Paint a symmetric linear ramp peaking at `time_sec` into `target`.

    `np.maximum` rather than assignment: two events of the same pitch close
    together (a fast repeat, or a note re-struck under pedal) overlap, and
    the stronger claim on a frame must win. Overwriting would let the second
    event flatten the first one's peak into a plateau, which the decoder's
    monotonicity check then rejects outright — losing BOTH events.
    """
    frames = target.shape[0]
    center = time_sec * fps

    lo = max(0, int(np.floor(center - half_width)) + 1)
    hi = min(frames - 1, int(np.ceil(center + half_width)) - 1)
    if hi < lo:
        return

    idx = np.arange(lo, hi + 1)
    values = 1.0 - np.abs(idx - center) / half_width
    np.maximum(target[lo:hi + 1, key], values, out=target[lo:hi + 1, key])


#: Per-note onset-loss weighting by velocity (Phase 28).
#:
#: WHY THIS IS NOT `--loss-weights`
#: --------------------------------
#: `losses.compute` scales four HEADS by four scalars. That cannot express
#: "care more about this note than that one", because `bce` averages uniformly
#: over ~88,000 cells and every note contributes equally. Phase 27 measured a
#: **16x** spread in the ONSET head's miss rate by velocity -- pp 38.3% against
#: f 2.4% over 52,478 MAESTRO notes, with pp+p accounting for 66.6% of all
#: missed notes from 33.4% of the reference. No existing knob addresses that.
#:
#: Phase 23's `--loss-weights velocity=0.1` is a DIFFERENT intervention on a
#: DIFFERENT head: it downweighted predicting how loud notes are. This changes
#: how hard the model is pushed to DETECT quiet ones.
#:
#: The weight is `1 + (SOFT_ONSET_BOOST - 1) * (1 - v/VELOCITY_SCALE)`, so a
#: silent note would score SOFT_ONSET_BOOST and a maximal one exactly 1.0 --
#: loud notes are never DOWNweighted, only soft ones lifted. A boost of 1.0
#: reproduces the unweighted target bit-for-bit, which is the default and what
#: every published checkpoint was trained with.
SOFT_ONSET_BOOST_OFF = 1.0


def render_targets(
    notes: list[NoteEvent],
    pedals: list[PedalEvent] | None = None,
    segment_start: float = 0.0,
    *,
    seconds: float = SEGMENT_SECONDS,
    fps: int = FRAMES_PER_SECOND,
    half_width: int = RAMP_HALF_WIDTH,
    soft_onset_boost: float = SOFT_ONSET_BOOST_OFF,
) -> dict[str, np.ndarray]:
    """Render one segment's training targets.

    Args:
      notes: ground-truth notes, in absolute track time.
      pedals: ground-truth pedal events, in absolute track time. Accepted and
        rendered so the data pipeline is complete, though Phase 15 freezes the
        pedal model and does not consume these.
      segment_start: where this segment begins in the track, in seconds.
      seconds: segment length.
      fps: model output frame rate.
      half_width: regression ramp half-width, in frames.

    Returns a dict of float32 arrays, each `(frames, 88)` unless noted:
      reg_onset   ramp peaking at each onset
      reg_offset  ramp peaking at each offset
      frame       1.0 while a note sounds
      velocity    velocity/128, held for the note's duration
      mask        1.0 at onset frames only — the velocity loss mask
      pedal_frame (frames, 1), 1.0 while the sustain pedal is down

    Times are relative to `segment_start`. Events wholly outside the segment
    are dropped; events that straddle a boundary are painted for the part
    that falls inside, so a note held across two segments is supervised in
    both rather than appearing to stop and restart.
    """
    if half_width < 2:
        # Below 2 the ramp cannot reach the +-2 neighbours the decoder
        # inspects, and sub-frame recovery degrades (measured 0.86ms at J=1).
        raise ValueError(
            f"half_width must be at least 2 to decode exactly, got {half_width}"
        )

    frames = segment_frames(seconds, fps)
    shape = (frames, CLASSES_NUM)

    reg_onset = np.zeros(shape, dtype=np.float32)
    reg_offset = np.zeros(shape, dtype=np.float32)
    frame = np.zeros(shape, dtype=np.float32)
    velocity = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    # Ones, not zeros: cells with no onset must keep their full weight, or the
    # loss would ignore the negative space that teaches the model NOT to fire.
    onset_weight = np.ones(shape, dtype=np.float32)
    pedal_frame = np.zeros((frames, 1), dtype=np.float32)

    end = segment_start + seconds

    for note in notes:
        # Reject only notes with no overlap at all. A note starting before
        # the segment still sounds inside it and must appear in `frame`, or
        # the model is taught that a held note stopped at the boundary.
        if note.offset <= segment_start or note.onset >= end:
            continue

        key = note.pitch - BEGIN_NOTE
        if not 0 <= key < CLASSES_NUM:
            # NoteEvent already rejects out-of-range pitches, so this only
            # fires if BEGIN_NOTE and config.MIDI_LOWEST ever diverge.
            continue

        onset_rel = note.onset - segment_start
        offset_rel = note.offset - segment_start

        # Regression ramps are painted only for events genuinely inside the
        # segment. Painting a ramp for an onset that happened earlier would
        # invent an onset the audio does not contain.
        if 0.0 <= onset_rel < seconds:
            _paint_ramp(reg_onset, onset_rel, key, fps, half_width)

            # Painted over the RAMP's footprint, not just the peak frame: the
            # onset loss is computed across the whole ramp, so weighting one
            # frame would leave most of a soft note's supervision unweighted.
            # `np.maximum` matches `_paint_ramp`'s own overlap rule -- where two
            # notes share frames, the stronger claim wins, so a soft note is
            # never quietly demoted by a loud neighbour.
            if soft_onset_boost != SOFT_ONSET_BOOST_OFF:
                softness = 1.0 - min(note.velocity / VELOCITY_SCALE, 1.0)
                w = 1.0 + (soft_onset_boost - 1.0) * softness
                lo_w = max(0, int(np.floor(onset_rel * fps - half_width)) + 1)
                hi_w = min(frames - 1,
                           int(np.ceil(onset_rel * fps + half_width)) - 1)
                if hi_w >= lo_w:
                    np.maximum(onset_weight[lo_w:hi_w + 1, key], w,
                               out=onset_weight[lo_w:hi_w + 1, key])

            onset_frame = int(round(onset_rel * fps))
            if 0 <= onset_frame < frames:
                mask[onset_frame, key] = 1.0

        if 0.0 <= offset_rel < seconds:
            _paint_ramp(reg_offset, offset_rel, key, fps, half_width)

        # `frame` and `velocity` are clipped to the segment, so a note
        # crossing a boundary is supervised on both sides.
        lo = max(0, int(round(max(onset_rel, 0.0) * fps)))
        hi = min(frames - 1, int(round(min(offset_rel, seconds) * fps)))
        if hi >= lo:
            frame[lo:hi + 1, key] = 1.0
            # Velocity is held across the note rather than placed only at the
            # onset. The decoder reads it at the onset frame, but a target
            # that is nonzero in exactly one frame per note is a far sparser
            # signal to learn, and holding it costs nothing at decode time.
            velocity[lo:hi + 1, key] = min(note.velocity / VELOCITY_SCALE, 1.0)

    for pedal in pedals or []:
        if pedal.offset <= segment_start or pedal.onset >= end:
            continue
        lo = max(0, int(round(max(pedal.onset - segment_start, 0.0) * fps)))
        hi = min(frames - 1,
                 int(round(min(pedal.offset - segment_start, seconds) * fps)))
        if hi >= lo:
            pedal_frame[lo:hi + 1, 0] = 1.0

    return {
        "reg_onset": reg_onset,
        "reg_offset": reg_offset,
        "frame": frame,
        "velocity": velocity,
        "mask": mask,
        "onset_weight": onset_weight,
        "pedal_frame": pedal_frame,
    }
