"""The training-time augmentation sampler.

Two properties matter more than anything else here, and both fail silently:

  1. **Labels follow the audio.** A detune that moves the sound but not the
     ground truth trains a systematic time offset into the model. The loss
     still falls. Nothing raises.
  2. **The same segment draws the same condition on a resume.** Otherwise a
     resumed run silently trains on a different distribution than the one it
     started on, and the checkpoint's `rng_state` cannot help — augmentation
     is hash-seeded precisely so it does not depend on stream position.
"""

import numpy as np
import pytest

from evaluation.augment import detune_ratio, detune_source_seconds
from training.augment import AugmentationSampler, AugmentPlan, segment_seed
from transcriber.events import NoteEvent, PedalEvent, Transcription

SR = 16000
SECONDS = 10.0
SAMPLES = int(SECONDS * SR)


def _tone(seconds=SECONDS, freq=440.0):
    t = np.arange(int(round(seconds * SR))) / SR
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def _labels(duration=SECONDS):
    tr = Transcription(duration=duration)
    tr.notes = [NoteEvent(60, 1.0, 2.0, 90), NoteEvent(64, 5.0, 6.0, 80)]
    tr.pedals = [PedalEvent(0.5, 3.0)]
    return tr


def _sampler(**kw):
    kw.setdefault("ir_bank_size", 4)
    return AugmentationSampler(SR, seconds=SECONDS, **kw)


# --- the seed -------------------------------------------------------------

def test_seed_is_stable_across_processes():
    """blake2b, not Python's hash(), which is salted per process and would
    give a different augmentation in every run and every worker."""
    import subprocess
    import sys

    expected = segment_seed(0, 0, 42)
    out = subprocess.run(
        [sys.executable, "-c",
         "from training.augment import segment_seed;"
         "print(segment_seed(0, 0, 42))"],
        capture_output=True, text=True, check=True,
    )
    assert int(out.stdout.strip()) == expected


def test_seed_varies_with_every_input():
    base = segment_seed(0, 0, 0)
    assert segment_seed(1, 0, 0) != base
    assert segment_seed(0, 1, 0) != base
    assert segment_seed(0, 0, 1) != base


def test_seeds_do_not_collide_across_a_realistic_index_range():
    seeds = {segment_seed(0, 0, i) for i in range(10000)}
    assert len(seeds) == 10000


# --- drawing --------------------------------------------------------------

def test_plan_is_deterministic():
    a, b = _sampler(seed=3), _sampler(seed=3)
    assert a.plan(17) == b.plan(17)


def test_plan_varies_across_segments():
    s = _sampler(seed=0, clean_prob=0.0)
    plans = [s.plan(i) for i in range(50)]
    assert len({p.cents for p in plans}) > 40


def test_plan_varies_across_epochs():
    a, b = _sampler(seed=0, clean_prob=0.0), _sampler(seed=0, clean_prob=0.0)
    b.set_epoch(1)
    assert a.plan(5) != b.plan(5)


def test_resume_draws_the_identical_condition():
    """The property `capture_rng_state` could not give: a fresh sampler with
    the same seed and epoch reproduces every plan exactly, regardless of what
    order segments were visited in before the crash."""
    before = [_sampler(seed=9).plan(i) for i in range(100)]
    after = [_sampler(seed=9).plan(i) for i in range(100)]
    assert before == after


def test_workers_drawing_disjoint_indices_do_not_duplicate():
    """Separate worker processes hold copies of the sampler and handle
    different indices; the draws must still differ."""
    s = _sampler(seed=0, clean_prob=0.0)
    worker_a = [s.plan(i).cents for i in range(0, 40, 2)]
    worker_b = [s.plan(i).cents for i in range(1, 40, 2)]
    assert not set(worker_a) & set(worker_b)


def test_clean_probability_is_respected():
    s = _sampler(seed=0, clean_prob=0.2)
    clean = sum(s.plan(i).clean for i in range(2000))
    assert 0.17 < clean / 2000 < 0.23


def test_clean_plan_reads_exactly_the_segment_length():
    s = _sampler(seed=0, clean_prob=1.0)
    assert s.plan(0).source_seconds == SECONDS


def test_detune_stays_within_the_configured_range():
    s = _sampler(seed=0, clean_prob=0.0, max_cents=50.0)
    cents = [s.plan(i).cents for i in range(500)]
    assert min(cents) >= -50.0 and max(cents) <= 50.0


def test_detune_is_centred_on_in_tune():
    """Triangular, not uniform: most pianos are close to in tune."""
    s = _sampler(seed=0, clean_prob=0.0, max_cents=50.0)
    cents = np.array([s.plan(i).cents for i in range(2000)])
    assert abs(cents.mean()) < 3.0
    # More mass near zero than a uniform draw would give (which would be 50%).
    assert (np.abs(cents) < 25.0).mean() > 0.6


def test_source_seconds_follows_the_detune():
    s = _sampler(seed=0, clean_prob=0.0)
    for i in range(20):
        plan = s.plan(i)
        assert plan.source_seconds == pytest.approx(
            detune_source_seconds(SECONDS, plan.cents)
        )


def test_upshift_is_clamped_when_the_track_has_no_tail():
    """315 of the 1099 indexed tracks have under 300ms of tail — less than a
    +50-cent over-read needs. Clamp rather than let fit_length pad silence."""
    s = _sampler(seed=0, clean_prob=0.0)
    # Find a segment that wants an upshift.
    index = next(i for i in range(200) if s.plan(i).cents > 10)
    plan = s.plan(index, available_seconds=SECONDS + 0.01)
    assert plan.source_seconds <= SECONDS + 0.01
    assert plan.cents_clamped


def test_a_downshift_is_never_clamped():
    """It consumes less source than it produces, so it is always safe."""
    s = _sampler(seed=0, clean_prob=0.0)
    index = next(i for i in range(200) if s.plan(i).cents < -10)
    plan = s.plan(index, available_seconds=SECONDS)
    assert not plan.cents_clamped
    assert plan.cents < 0


def test_no_tail_at_all_falls_back_to_no_detune():
    s = _sampler(seed=0, clean_prob=0.0)
    index = next(i for i in range(200) if s.plan(i).cents > 10)
    plan = s.plan(index, available_seconds=SECONDS)
    assert plan.cents == 0.0
    assert plan.source_seconds == SECONDS


def test_eq_is_off_by_default():
    """22.6ms — more than the rest of the chain combined — for a factor that
    is not in the measured degradation curve."""
    s = _sampler(seed=0, clean_prob=0.0)
    assert all(s.plan(i).low_gain_db == 0.0 for i in range(200))
    assert all(s.plan(i).high_gain_db == 0.0 for i in range(200))


def test_eq_can_be_switched_on():
    s = _sampler(seed=0, clean_prob=0.0, eq_prob=1.0)
    assert any(s.plan(i).low_gain_db != 0.0 for i in range(20))


def test_noise_probability_is_respected():
    s = _sampler(seed=0, clean_prob=0.0, noise_prob=0.5)
    noisy = sum(s.plan(i).snr_db is not None for i in range(2000))
    assert 0.45 < noisy / 2000 < 0.55


def test_rt60_is_reported_from_the_bank():
    s = _sampler(seed=0, clean_prob=0.0)
    plan = s.plan(0)
    assert plan.rt60 == pytest.approx(s.bank.rt60s[plan.ir_index])


# --- applying -------------------------------------------------------------

def test_apply_returns_the_exact_segment_length():
    s = _sampler(seed=0, clean_prob=0.0)
    for i in range(10):
        plan = s.plan(i)
        audio = _tone(plan.source_seconds)
        out, _ = s.apply(audio, _labels(), plan)
        assert len(out) == SAMPLES


def test_apply_is_bit_identical_for_the_same_seed():
    a, b = _sampler(seed=4), _sampler(seed=4)
    plan = a.plan(11)
    audio = _tone(plan.source_seconds)
    out_a, _ = a.apply(audio, _labels(), plan)
    out_b, _ = b.apply(audio, _labels(), plan)
    assert np.array_equal(out_a, out_b)


def test_apply_changes_the_audio():
    s = _sampler(seed=0, clean_prob=0.0)
    plan = s.plan(1)
    audio = _tone(plan.source_seconds)
    out, _ = s.apply(audio, _labels(), plan)
    assert not np.allclose(out, audio[:SAMPLES])


def test_clean_passthrough_leaves_labels_untouched():
    s = _sampler(seed=0, clean_prob=1.0)
    labels = _labels()
    out, out_labels = s.apply(_tone(), labels, s.plan(0))
    assert out_labels is labels
    assert len(out) == SAMPLES


def test_labels_are_rescaled_by_the_detune():
    s = _sampler(seed=0, clean_prob=0.0)
    plan = s.plan(2)
    labels = Transcription(duration=plan.source_seconds)
    labels.notes = [NoteEvent(60, 9.0, 9.5, 90, clamp=False)]

    _, out = s.apply(_tone(plan.source_seconds), labels, plan)

    assert out.notes[0].onset == pytest.approx(
        9.0 / detune_ratio(plan.cents), abs=0.002
    )


def test_output_is_float32():
    """A float64 promotion doubles memory and breaks AMP; `collate` enforces
    this downstream, but paying the conversion here is cheaper."""
    s = _sampler(seed=0, clean_prob=0.0)
    plan = s.plan(3)
    out, _ = s.apply(_tone(plan.source_seconds), _labels(), plan)
    assert out.dtype == np.float32


def test_output_never_clips():
    s = _sampler(seed=0, clean_prob=0.0)
    for i in range(30):
        plan = s.plan(i)
        out, _ = s.apply(_tone(plan.source_seconds), _labels(), plan)
        assert np.abs(out).max() <= 1.0


def test_quiet_segments_really_are_quiet():
    """The gain step is applied last and NOT renormalised — a quiet recording
    is a real failure mode, measured at a peak of 0.019 in Phase 1."""
    s = _sampler(seed=0, clean_prob=0.0, quiet_prob=1.0,
                 peak_range=(0.05, 0.06))
    plan = s.plan(0)
    out, _ = s.apply(_tone(plan.source_seconds), _labels(), plan)
    assert np.abs(out).max() < 0.07


def test_reverb_smears_energy_past_a_click():
    s = _sampler(seed=0, clean_prob=0.0, noise_prob=0.0, quiet_prob=0.0)
    plan = s.plan(0)
    audio = np.zeros(int(round(plan.source_seconds * SR)), dtype=np.float32)
    audio[:100] = 1.0
    out, _ = s.apply(audio, _labels(), plan)
    assert np.abs(out[SR:]).max() > 0


def test_call_draws_and_applies_in_one_go():
    s = _sampler(seed=0, clean_prob=0.0)
    out, _ = s(_tone(), _labels(), index=7)
    assert len(out) == SAMPLES


def test_empty_pedals_and_notes_survive():
    s = _sampler(seed=0, clean_prob=0.0)
    plan = s.plan(0)
    out, labels = s.apply(_tone(plan.source_seconds),
                          Transcription(duration=SECONDS), plan)
    assert len(out) == SAMPLES
    assert labels.notes == []


# --- cost -----------------------------------------------------------------

def test_augmentation_stays_inside_the_throughput_budget():
    """Budget, not measurement. The chain measures ~19.8ms median on this
    machine, but asserting that would be flaky in CI; what matters is that it
    stays far enough under the >=15 segments/sec/worker floor.

    Warms soxr first: the FIRST resample in a process costs 1.9-6.9 SECONDS
    because soxr initialises lazily, so an unwarmed timing measures startup.
    """
    import time

    s = _sampler(seed=0, clean_prob=0.0)
    plan = s.plan(0)
    audio = _tone(plan.source_seconds)
    s.apply(audio, _labels(), plan)          # warm soxr and the FFT plans

    times = []
    for i in range(20):
        plan = s.plan(i)
        audio = _tone(plan.source_seconds)
        started = time.perf_counter()
        s.apply(audio, _labels(), plan)
        times.append(time.perf_counter() - started)

    assert np.median(times) < 0.040


# --- the round trip that catches silent label corruption -----------------
#
# This is the test the whole phase turns on. Everything above proves the
# pieces behave; this proves the pieces COMPOSE — that a detuned segment's
# rendered targets still decode, through the REAL post-processor, to the
# times the audio actually contains. It is the only test here that would
# catch the 284.7ms drift, and drift is not a scoring error but a systematic
# time offset trained into the model, with a falling loss and no exception.

def test_detuned_targets_decode_to_the_shifted_truth():
    from tests.test_targets import decode_onsets
    from training.targets import render_targets

    s = _sampler(seed=0, clean_prob=0.0)
    # A segment with a real detune, not an accidental near-zero one.
    index = next(i for i in range(300) if abs(s.plan(i).cents) > 20)
    plan = s.plan(index)

    onsets = [0.5, 3.25, 9.0]           # including one near the segment END,
    labels = Transcription(duration=plan.source_seconds)   # the worst case
    labels.notes = [NoteEvent(60, t, t + 0.4, 90, clamp=False) for t in onsets]

    _, shifted = s.apply(_tone(plan.source_seconds), labels, plan)
    targets = render_targets(shifted.notes, shifted.pedals, 0.0,
                             seconds=SECONDS)

    recovered = decode_onsets(targets, 60 - 21)
    expected = [t / detune_ratio(plan.cents) for t in onsets]

    assert len(recovered) == len(expected)
    for got, want in zip(sorted(recovered), sorted(expected)):
        assert got == pytest.approx(want, abs=0.006)


def test_uncorrected_drift_would_fail_that_round_trip():
    """REGRESSION, stated as the counterfactual.

    Rendering the ORIGINAL labels against detuned audio puts the last onset
    143ms out at 25 cents — nearly 3x mir_eval's 50ms tolerance, and growing
    with t. Pinned so that removing `_rescale_times` fails loudly here rather
    than quietly degrading a training run nobody can debug.
    """
    from tests.test_targets import decode_onsets
    from training.targets import render_targets

    cents = 25.0
    labels = Transcription(duration=SECONDS)
    labels.notes = [NoteEvent(60, 9.0, 9.4, 90, clamp=False)]

    # What the targets would say if labels were carried through unchanged.
    naive = render_targets(labels.notes, labels.pedals, 0.0, seconds=SECONDS)
    true_time = 9.0 / detune_ratio(cents)

    recovered = decode_onsets(naive, 60 - 21)
    assert abs(recovered[0] - true_time) > 0.10
