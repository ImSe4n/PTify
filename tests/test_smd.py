"""SMD corpus fetcher: pairing, folder choice, idempotent extraction (no network)."""

import io
import zipfile

from evaluation.smd import build_corpus, list_pieces
from transcriber.events import NoteEvent, Transcription
from transcriber.midi import write_midi


def _midi_bytes(tmp_path) -> bytes:
    tr = Transcription(source_path="x", duration=2.0)
    tr.notes.append(NoteEvent(pitch=60, onset=0.5, offset=1.0, velocity=70))
    path = tmp_path / "ref.mid"
    write_midi(tr, path)
    return path.read_bytes()


def _zip(tmp_path) -> zipfile.ZipFile:
    buf = io.BytesIO()
    mid = _midi_bytes(tmp_path)
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("midi/", b"")
        zf.writestr("midi/A-SMD.mid", mid)
        zf.writestr("midi/B-SMD.mid", mid)           # no recording: skipped
        zf.writestr("midi/C-SMD.mid", mid)
        zf.writestr("wav_22050_mono/C-SMD.wav", b"recorded C")
        zf.writestr("wav_22050_mono/A-SMD.wav", b"recorded")
        zf.writestr("midi_wav_22050_mono/A-SMD.wav", b"SYNTHESISED")
        zf.writestr("wav_44100_stereo/A-SMD.wav", b"big")
        zf.writestr("csv/A-SMD.csv", b"Start;Duration")
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_lists_only_pieces_with_recording_and_midi(tmp_path):
    pieces = list_pieces(_zip(tmp_path))
    assert [p.stem for p in pieces] == ["A-SMD", "C-SMD"]
    assert pieces[0].wav_member == "wav_22050_mono/A-SMD.wav"   # never the synth
    # Each piece keeps its OWN name and files (a loop-variable bug once named
    # every piece after the last one).
    assert pieces[1].wav_member == "wav_22050_mono/C-SMD.wav"
    assert pieces[1].mid_member == "midi/C-SMD.mid"


def test_build_writes_flat_pairs_and_is_idempotent(tmp_path):
    zf = _zip(tmp_path)
    out = tmp_path / "smd"
    manifest = build_corpus(out, zf, progress=False)
    assert (out / "A-SMD.wav").read_bytes() == b"recorded"
    assert (out / "A-SMD.mid").exists()
    assert (out / "C-SMD.wav").read_bytes() == b"recorded C"
    assert manifest["n"] == 2 and manifest["tracks"][0]["n_notes"] == 1
    assert manifest["velocity_metric_valid"] is True

    (out / "A-SMD.wav").write_bytes(b"kept")       # an existing file is not refetched
    build_corpus(out, zf, progress=False)
    assert (out / "A-SMD.wav").read_bytes() == b"kept"
