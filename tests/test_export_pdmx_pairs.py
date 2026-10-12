"""`tools/export_pdmx_pairs.py`: PDMX scores -> synthetic training shards (Phase 35)."""

import csv
import json

import numpy as np
import pretty_midi

from tools.export_pdmx_pairs import is_heldout, main


def _write_midi(path, program=0, n=240):
    m = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    ins = pretty_midi.Instrument(program=program)
    for i in range(n):
        ins.notes.append(pretty_midi.Note(70, 60 + i % 12, i * 0.25, i * 0.25 + 0.2))
    m.instruments.append(ins)
    m.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    path.parent.mkdir(parents=True, exist_ok=True)
    m.write(str(path))


def _corpus(tmp_path, names):
    root = tmp_path / "extract"
    rows = []
    for name, program in names:
        _write_midi(root / "mid" / f"{name}.mid", program)
        # The CSV says piano; the file may disagree (it happens in PDMX).
        rows.append([f"./mid/{name}.mid", "0", "240", "True", "True"])
    path = tmp_path / "PDMX.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mid", "tracks", "n_notes", "subset:deduplicated",
                    "subset:no_license_conflict"])
        w.writerows(rows)
    return path, root


def test_exports_shards_a_summary_and_counts_failures(tmp_path):
    names = [(f"p{i}", 0) for i in range(6)] + [("violin", 40)]
    csv_path, root = _corpus(tmp_path, names)
    out = tmp_path / "pairs"
    assert main(["--csv", str(csv_path), "--midi-root", str(root), "--out", str(out),
                 "--cap", "10", "--workers", "1", "--shard", "2"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["failed"] == {"not a piano-only score": 1}
    n_train = sum(1 for n, _ in names[:6] if not is_heldout(f"./mid/{n}.mid"))
    assert summary["pieces"]["train"] == n_train
    shards = sorted(out.glob("pdmx-train-*.npz"))
    assert len(shards) == (n_train + 1) // 2
    with np.load(shards[0]) as z:
        assert z["offsets"][-1] == len(z["pitch"]) and (z["hand"] == -100).all()


def test_the_split_is_by_score_and_stable():
    assert is_heldout("./mid/a.mid") == is_heldout("./mid/a.mid")
    share = np.mean([is_heldout(f"./mid/{i}.mid") for i in range(2000)])
    assert 0.07 < share < 0.13


def test_mxl_source_labels_values_and_hands_on_the_same_split(tmp_path):
    # Phase 36: MusicXML gives written values and staves. The split must stay
    # the one the MIDI export used (hash of the MIDI path), so no score moves
    # between train and held-out across runs.
    from music21 import clef, meter, note, stream

    root = tmp_path / "extract"
    rows = []
    for i in range(4):
        rh, lh = stream.PartStaff(), stream.PartStaff()
        for part, c, p in ((rh, clef.TrebleClef(), "C5"), (lh, clef.BassClef(), "C3")):
            for b in range(60):
                m = stream.Measure(number=b + 1)
                if b == 0:
                    m.append([c, meter.TimeSignature("4/4")])
                m.append([note.Note(p, quarterLength=1.0) for _ in range(4)])
                part.append(m)
        path = root / "mxl" / f"s{i}.mxl"
        path.parent.mkdir(parents=True, exist_ok=True)
        stream.Score([rh, lh]).write("mxl", fp=str(path))
        rows.append([f"./mid/s{i}.mid", f"./mxl/s{i}.mxl", "0-0", "480", "True", "True"])
    csv_path = tmp_path / "PDMX.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mid", "mxl", "tracks", "n_notes", "subset:deduplicated",
                    "subset:no_license_conflict"])
        w.writerows(rows)
    out = tmp_path / "pairs"
    assert main(["--csv", str(csv_path), "--midi-root", str(root), "--out", str(out),
                 "--cap", "10", "--workers", "1", "--shard", "10", "--source", "mxl"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["source"].endswith("mxl")
    n_train = sum(1 for i in range(4) if not is_heldout(f"./mid/s{i}.mid"))
    assert summary["pieces"].get("train", 0) == n_train
    if n_train:
        with np.load(sorted(out.glob("pdmx-train-*.npz"))[0]) as z:
            assert (z["hand"] != -100).all() and (z["duration"] != -100).all()
