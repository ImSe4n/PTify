"""What happens to piano audio on its way to an upload (Phase 38, tier T3).

`augment.py` simulates the ROOM and the MIC. Real uploads also pass through
two stages it never modelled:

  1. LOUDNESS PROCESSING. Phone recorder apps, YouTube and TikTok all squash
     dynamics (automatic gain, compression, limiting). For a transcriber that
     matters most for QUIET notes, already its weakest case (Phase 27: pp
     notes missed 38% of the time against 2% for f).
  2. A LOSSY CODEC. YouTube serves Opus, TikTok and phone apps use AAC. Lossy
     codecs drop quiet high partials first -- exactly the cue that separates a
     note from its neighbour's overtone.

Both are simulated in-process with libsndfile (MP3, Opus, Vorbis), so no
ffmpeg is needed. MP3 stands in for AAC: libsndfile has no AAC encoder. Both
are perceptual transform codecs; the MP3 proxy at a given bitrate is, if
anything, a little harsher than AAC.

THE CODEC'S TIME SHIFT IS REMOVED, OR IT WOULD BE SCORED AS AN ENGINE ERROR
-------------------------------------------------------------------------
An MP3 round-trip returns ~1,500 extra samples of encoder delay (~30 ms at
48 kHz) -- most of the 50 ms onset tolerance. `codec_roundtrip` finds the lag
by cross-correlation and removes it, and returns exactly the input length.

BITRATES, measured 2026-10-16 on 30 s of MAESTRO piano (var/codec_probe.py)
-----------------------------------------------------------------------
  Opus  level 0.3 -> 183 kbps, 0.5 -> 133 (preset "youtube"), 0.7 -> 82,
        0.8 -> 57 (preset "tiktok", deliberately on the harsh side)
  MP3   level 0.3 -> 62 kbps (presets "phone", "phone_far" -- a voice-memo
        style AAC rate), 0.7 and above -> ~33

What this does NOT simulate: phone noise suppression / echo cancellation
(voice-call processing that eats sustained tones). Recorder apps usually turn
it off; when they do not, real phone audio is worse than these presets.

SIMULATED, AND LABELLED SO: these presets turn real MAESTRO recordings into
plausible uploads. They are not uploads. Tier T2 (MAPS, SMD) is the real-room
check; nothing here replaces a real phone recording with exact MIDI.
"""

from __future__ import annotations

import io

import numpy as np

#: Opus encodes only at these rates; anything else is resampled to 48 kHz.
OPUS_RATES = (8000, 12000, 16000, 24000, 48000)

#: How far the codec lag is searched, in seconds. Encoder delays are a few
#: milliseconds; a much larger "lag" would mean the correlation locked onto
#: a repeated passage, not the delay.
MAX_LAG_SEC = 0.1

#: Seconds of audio used to measure the lag (the start of the piece).
LAG_WINDOW_SEC = 20.0


def _envelope(x: np.ndarray, sr: int, tau_sec: float) -> np.ndarray:
    """One-pole smoothed |x|."""
    from scipy.signal import lfilter

    a = np.exp(-1.0 / max(tau_sec * sr, 1.0))
    return lfilter([1 - a], [1, -a], np.abs(x))


def compress(audio: np.ndarray, sr: int, threshold_db: float = -24.0,
             ratio: float = 4.0, attack_sec: float = 0.005,
             release_sec: float = 0.15, peak: float = 0.95) -> np.ndarray:
    """Downward compression, then make-up gain to `peak` (what makes uploads
    loud). An approximation of a real compressor's ballistics, vectorised:
    the gain reduction is smoothed twice, fast (`attack_sec`) and slow
    (`release_sec`), and the DEEPER of the two is applied -- so reduction
    clamps on within the attack time and lets go over the release time.
    Smoothing with the release time alone let every attack through
    uncompressed (measured in the test: the quiet/loud ratio got WORSE).
    """
    if len(audio) == 0:
        return audio
    level = _envelope(audio, sr, attack_sec)
    level_db = 20 * np.log10(np.maximum(level, 1e-9))
    over = np.maximum(level_db - threshold_db, 0.0)
    gain_db = -over * (1 - 1 / ratio)
    gain_db = np.minimum(_smooth(gain_db, sr, attack_sec),
                         _smooth(gain_db, sr, release_sec))
    gain = 10 ** (gain_db / 20)
    out = audio * gain
    top = np.abs(out).max()
    return out * (peak / top) if top > 0 else out


def _smooth(x: np.ndarray, sr: int, tau_sec: float) -> np.ndarray:
    from scipy.signal import lfilter

    a = np.exp(-1.0 / max(tau_sec * sr, 1.0))
    return lfilter([1 - a], [1, -a], x)


def measure_lag(reference: np.ndarray, decoded: np.ndarray, sr: int) -> int:
    """Samples by which `decoded` is delayed relative to `reference`."""
    from scipy.signal import correlate

    n = min(len(reference), len(decoded), int(LAG_WINDOW_SEC * sr))
    max_lag = int(MAX_LAG_SEC * sr)
    if n == 0:
        return 0
    corr = correlate(decoded[:n + max_lag], reference[:n], mode="full",
                     method="fft")
    zero = n - 1                                  # index of lag 0
    lo, hi = zero, zero + max_lag + 1             # delays only, never advances
    return int(np.argmax(corr[lo:hi]))


def codec_roundtrip(audio: np.ndarray, sr: int, fmt: str, subtype: str,
                    level: float) -> np.ndarray:
    """Encode and decode in memory; same length and timing as the input.

    `level` is libsndfile's compression_level: 0 = best quality, 1 = smallest.
    """
    import librosa
    import soundfile as sf

    if len(audio) == 0:
        return audio
    rate = sr
    if subtype == "OPUS" and sr not in OPUS_RATES:
        rate = 48000
    x = audio if rate == sr else librosa.resample(audio, orig_sr=sr, target_sr=rate)
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), rate, format=fmt, subtype=subtype,
             compression_level=level)
    buf.seek(0)
    y, got = sf.read(buf, dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if got != sr:
        y = librosa.resample(y, orig_sr=got, target_sr=sr)
    y = y[measure_lag(audio, y, sr):]
    if len(y) < len(audio):
        y = np.pad(y, (0, len(audio) - len(y)))
    return y[:len(audio)]
