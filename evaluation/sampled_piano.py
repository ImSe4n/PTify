"""Render MIDI through sampled acoustic pianos (Phase 41).

Our own minimal sample player for the VCSL pianos (`tools/fetch_vcsl.py`,
CC0): two uprights and two grands. MAESTRO is one Yamaha Disklavier grand;
this gives the model other instruments with EXACT labels, because the audio
is built from the labels.

What it models: per-note samples chosen by key (nearest sampled key, pitch
shifted -- the libraries sample every 1-3 keys) and velocity (layer by
velocity, gain by a velocity curve because several libraries normalised their
layers), round robins, damper release at note-off (or at the pedal release if
the sustain pedal is down), and the library's own release-noise samples.
What it does NOT model: sympathetic resonance, half-pedal, una corda, hammer
noise beyond what the samples contain. So it is a plausible piano, not a
physical one -- which is why Phase 41 judges it on REAL held-out pianos.

OCTAVES: the libraries disagree on octave numbering (Kawai "A#-1" is the
lowest A#), so names give pitch CLASS and a provisional octave, and each
bank's octave offset is MEASURED from the samples' own pitch (`calibrate`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from transcriber.events import Transcription

NOTE_RE = re.compile(r"(?<![A-Za-z])([A-G]#?)(-?\d)(?=_|\.|$)")
VEL_RE = re.compile(r"_v(?:l)?(\d+)")
RR_RE = re.compile(r"_rr(\d+)")
PC = {n: i for i, n in enumerate(
    ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"))}

#: Folder holding the main (dampers-normal) samples, and the release-noise one.
SUSTAIN_DIRS = ("Sustains", "NoSus")
RELEASE_DIRS = ("Releases", "Rel")

MAX_SAMPLE_SEC = 8.0      # longer tails are cut (with a fade) to bound memory
RELEASE_SEC = 0.18        # damper decay after note-off
RELEASE_NOISE_GAIN = 0.25
VELOCITY_EXP = 1.8        # gain = (velocity/127)**VELOCITY_EXP


#: Salamander Grand (Phase 43): `A0v12.flac` = note, octave, velocity layer;
#: `rel37.flac` = the key-release noise of key 20+37 (rel1 is A0, key 21).
SALAMANDER_RE = re.compile(r"^([A-G]#?)(-?\d)v(\d+)\.")
SALAMANDER_REL_RE = re.compile(r"^rel(\d+)\.")


def parse_name(name: str) -> tuple[int, int, int] | None:
    """(provisional MIDI note with C4=60, velocity layer, round robin)."""
    m = SALAMANDER_RE.match(name)
    if m:
        return 12 * (int(m.group(2)) + 1) + PC[m.group(1)], int(m.group(3)), 1
    m = SALAMANDER_REL_RE.match(name)
    if m:
        return 20 + int(m.group(1)), 1, 1
    m = NOTE_RE.search(name)
    if not m:
        return None
    note = 12 * (int(m.group(2)) + 1) + PC[m.group(1)]
    v = VEL_RE.search(name)
    rr = RR_RE.search(name)
    return note, int(v.group(1)) if v else 1, int(rr.group(1)) if rr else 1


def _load(path: Path, sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True,
                        duration=MAX_SAMPLE_SEC)
    peak = float(np.abs(y).max())
    if peak > 0:
        y = y / peak
    n = int(0.05 * sr)                     # fade the cut tail
    if len(y) > n and len(y) >= int(MAX_SAMPLE_SEC * sr) - 1:
        y[-n:] *= np.linspace(1, 0, n)
    return y.astype(np.float32)


@dataclass
class PianoBank:
    name: str
    sr: int
    #: root MIDI note -> sorted velocity layers -> list of round robins
    notes: dict[int, dict[int, list[np.ndarray]]] = field(default_factory=dict)
    releases: dict[int, list[np.ndarray]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Compact float16 npz at `self.sr` -- what ships to Kaggle."""
        arrays = {}
        for root, layers in self.notes.items():
            for vel, rrs in layers.items():
                for k, y in enumerate(rrs):
                    arrays[f"n_{root}_{vel}_{k}"] = y.astype(np.float16)
        for root, rrs in self.releases.items():
            for k, y in enumerate(rrs):
                arrays[f"r_{root}_{k}"] = y.astype(np.float16)
        np.savez_compressed(path, __sr=np.array(self.sr), **arrays)

    @classmethod
    def load_npz(cls, path: Path) -> "PianoBank":
        bank = None
        with np.load(path) as z:
            bank = cls(name=Path(path).stem, sr=int(z["__sr"]))
            for key in z.files:
                if key == "__sr":
                    continue
                y = z[key].astype(np.float32)
                parts = key.split("_")
                if parts[0] == "n":
                    root, vel = int(parts[1]), int(parts[2])
                    bank.notes.setdefault(root, {}).setdefault(vel, []).append(y)
                else:
                    bank.releases.setdefault(int(parts[1]), []).append(y)
        return bank

    @property
    def roots(self) -> np.ndarray:
        return np.array(sorted(self.notes))

    @classmethod
    def load(cls, folder: Path, sr: int = 16000, octave_shift: int | None = None):
        bank = cls(name=Path(folder).name, sr=sr)
        files = []
        for sub in SUSTAIN_DIRS + RELEASE_DIRS:
            for f in sorted((Path(folder) / sub).glob("*.wav")):
                parsed = parse_name(f.name)
                if parsed:
                    files.append((sub in RELEASE_DIRS, parsed, f))
        if not any(not rel for rel, _, _ in files):
            raise ValueError(f"no sustain samples parsed under {folder}")
        if octave_shift is None:
            octave_shift = calibrate([(p[0], f) for rel, p, f in files if not rel], sr)
        for rel, (note, vel, _), f in files:
            key = note + octave_shift
            y = _load(f, sr)
            if rel:
                bank.releases.setdefault(key, []).append(y)
            else:
                bank.notes.setdefault(key, {}).setdefault(vel, []).append(y)
        return bank

    def render(self, tr: Transcription, duration: float | None = None,
               seed: int = 0, timbre: "Timbre | None" = None) -> np.ndarray:
        """Audio for `tr` (notes + sustain pedal) at `self.sr`.

        `timbre` (Phase 43) turns this bank into a slightly different
        instrument per render; None renders the samples as recorded.
        """
        rng = np.random.default_rng(seed)
        tb = timbre or Timbre()
        dur = duration if duration is not None else (
            max((n.offset for n in tr.notes), default=0.0) + 2.0)
        out = np.zeros(int(dur * self.sr) + 1, dtype=np.float32)
        roots = self.roots
        pedals = sorted((p.onset, p.offset) for p in tr.pedals)
        for n in tr.notes:
            root = int(roots[np.argmin(np.abs(roots - n.pitch))])
            layers = self.notes[root]
            keys = sorted(layers)
            layer = keys[min(int(n.velocity / 128 * len(keys)), len(keys) - 1)]
            sample = layers[layer][int(rng.integers(len(layers[layer])))]
            ratio = 2.0 ** ((n.pitch - root) / 12.0)
            end = _damper_time(n.offset, pedals)
            length = int((end - n.onset + RELEASE_SEC) * self.sr)
            src = np.arange(min(length, int(len(sample) / ratio))) * ratio
            if len(src) == 0:
                continue
            y = np.interp(src, np.arange(len(sample)), sample).astype(np.float32)
            if tb.detune_cents > 0:
                # Two strings a few cents apart: the beating of an out-of-tune
                # unison, characteristic of older uprights.
                r2 = ratio * 2.0 ** (tb.detune_cents / 1200.0)
                src2 = np.arange(len(y)) * r2
                y2 = np.interp(src2, np.arange(len(sample)), sample, right=0.0)
                y = (0.5 * (y + y2)).astype(np.float32)
            if tb.decay_per_sec > 0:
                y *= np.exp(-tb.decay_per_sec * np.arange(len(y)) / self.sr).astype(np.float32)
            hold = int((end - n.onset) * self.sr)
            if hold < len(y):                      # dampers fall at `end`
                tail = len(y) - hold
                y[hold:] *= np.exp(-np.arange(tail) / (RELEASE_SEC * self.sr / 5))
            y *= (n.velocity / 127.0) ** VELOCITY_EXP
            _mix(out, y, int(n.onset * self.sr))
            if self.releases and end - n.onset < MAX_SAMPLE_SEC:
                rroots = np.array(sorted(self.releases))
                rr = int(rroots[np.argmin(np.abs(rroots - n.pitch))])
                rel = self.releases[rr][int(rng.integers(len(self.releases[rr])))]
                _mix(out, rel * RELEASE_NOISE_GAIN * (n.velocity / 127.0) ** VELOCITY_EXP,
                     int(end * self.sr))
        if tb.brightness:
            out = _tilt(out, self.sr, tb.brightness)
        peak = float(np.abs(out).max())
        return out / peak * 0.9 if peak > 0 else out


@dataclass(frozen=True)
class Timbre:
    """Per-render instrument variation (Phase 43). Defaults = as recorded.

    decay_per_sec   extra exponential decay: shorter sustain, as on uprights
    detune_cents    second copy of each note this many cents sharp (unison
                    mistuning, "honky-tonk")
    brightness      -1..1 spectral tilt around ~1.5 kHz (dark .. bright)
    """

    decay_per_sec: float = 0.0
    detune_cents: float = 0.0
    brightness: float = 0.0

    @classmethod
    def draw(cls, rng: np.random.Generator) -> "Timbre":
        return cls(decay_per_sec=float(rng.uniform(0.0, 1.5)),
                   detune_cents=float(rng.choice([0.0, rng.uniform(1.0, 8.0)])),
                   brightness=float(rng.uniform(-0.6, 0.8)))


def _tilt(x: np.ndarray, sr: int, amount: float, fc: float = 1500.0) -> np.ndarray:
    """Split at `fc` with a one-pole low-pass; boost or cut the highs."""
    from scipy.signal import lfilter
    a = np.exp(-2 * np.pi * fc / sr)
    low = lfilter([1 - a], [1, -a], x).astype(np.float32)
    high = x - low
    return (low * (1 - 0.5 * max(amount, 0)) + high * (1 + amount)).astype(np.float32)


def _damper_time(offset: float, pedals: list[tuple[float, float]]) -> float:
    """When the damper falls: at key release, or at the pedal release if the
    sustain pedal is down at that moment."""
    for on, off in pedals:
        if on <= offset < off:
            return off
    return offset


def _mix(out: np.ndarray, y: np.ndarray, start: int) -> None:
    """Add `y` at `start`. A NEGATIVE start (a note struck before a training
    segment begins, still sounding inside it) contributes its tail."""
    if start >= len(out):
        return
    if start < 0:
        y, start = y[-start:], 0
    n = min(len(y), len(out) - start)
    if n > 0:
        out[start:start + n] += y[:n]


def calibrate(named: list[tuple[int, Path]], sr: int) -> int:
    """Octave offset (a multiple of 12) between names and measured pitch."""
    import librosa
    mids = sorted((n, f) for n, f in named if 48 <= n <= 84)[:: max(1, len(named) // 12)]
    if not mids:
        mids = named[: 12]
    shifts = []
    for n, f in mids[:12]:
        y, _ = librosa.load(str(f), sr=sr, mono=True, offset=0.05, duration=0.6)
        f0 = librosa.yin(y, fmin=25, fmax=min(4000, sr / 2 - 1), sr=sr)
        hz = float(np.median(f0[np.isfinite(f0)])) if np.isfinite(f0).any() else 0
        if hz > 0:
            shifts.append(librosa.hz_to_midi(hz) - n)
    if not shifts:
        return 0
    return int(12 * round(float(np.median(shifts)) / 12))
