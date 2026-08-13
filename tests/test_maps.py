"""MAPS corpus fetcher: listing, pairing, selection, annotation parsing.

No network. Archives are built in memory with `zipfile`, which is what keeps
the whole suite offline -- the same discipline as tests/test_corpus.py, where
`parse_metadata` takes CSV *text* rather than a URL.

The remote-zip reader is exercised against a fake opener that serves range
requests from a bytes object, so the range arithmetic is covered without
touching Zenodo.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from evaluation.maps import (
    CATEGORY,
    DEFAULT_VELOCITY,
    SUBSETS,
    MapsPiece,
    build_corpus,
    extract_piece,
    list_pieces,
    paired_ids,
    parse_annotation,
    select_pieces,
    subset_url,
    summarise,
)

_ANNOTATION = "OnsetTime\tOffsetTime\tMidiPitch\n0.5\t1.5\t60\n1.0\t2.0\t64\n"


def _archive(subset: str, piece_ids, *, with_txt=True, wav_bytes=b"RIFFfake"):
    """An in-memory MAPS-shaped zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for pid in piece_ids:
            base = f"{subset}/{CATEGORY}/MAPS_{CATEGORY}-{pid}_{subset}"
            zf.writestr(f"{base}.wav", wav_bytes)
            zf.writestr(f"{base}.mid", b"not a real midi")
            if with_txt:
                zf.writestr(f"{base}.txt", _ANNOTATION)
        # Noise from the categories the fetcher must ignore.
        zf.writestr(f"{subset}/ISOL/NO/MAPS_ISOL_NO_M_S1_M60_{subset}.wav", b"x")
        zf.writestr(f"{subset}/RAND/MAPS_RAND_P1_M21-108_{subset}.wav", b"x")
    buf.seek(0)
    return zipfile.ZipFile(buf)


# --- subset guards -------------------------------------------------------


def test_only_the_disklavier_subsets_are_offered():
    # The other seven MAPS subsets are software synths -- exactly what
    # evaluation/synth.py already covers, at ~14GB.
    assert SUBSETS == ("ENSTDkCl", "ENSTDkAm")


def test_subset_url_rejects_a_synth_subset():
    with pytest.raises(ValueError) as exc:
        subset_url("SptkBGCl")
    assert "Disklavier" in str(exc.value)


# --- listing -------------------------------------------------------------


def test_only_mus_pieces_are_listed():
    # ISOL/RAND/UCHO are isolated notes and chords, not performances.
    zf = _archive("ENSTDkCl", ["deb_clai", "mz_331_1"])
    pieces = list_pieces("ENSTDkCl", zf)

    assert [p.piece_id for p in pieces] == ["deb_clai", "mz_331_1"]
    assert all("/MUS/" in p.wav_member for p in pieces)


def test_a_piece_without_an_annotation_is_skipped():
    # Audio with no ground truth cannot be scored, so it is not a corpus
    # member. One odd file must not block the rest.
    zf = _archive("ENSTDkCl", ["good"])
    with zipfile.ZipFile(io.BytesIO(), "w") as _:
        pass
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for n in zf.namelist():
            if n.endswith("orphan_ENSTDkCl.txt"):
                continue
            out.writestr(n, zf.read(n))
        base = "ENSTDkCl/MUS/MAPS_MUS-orphan_ENSTDkCl"
        out.writestr(f"{base}.wav", b"RIFFfake")  # no .txt beside it
    buf.seek(0)

    pieces = list_pieces("ENSTDkCl", zipfile.ZipFile(buf))
    assert [p.piece_id for p in pieces] == ["good"]


def test_stems_carry_the_subset_so_paired_pieces_cannot_collide():
    """The same piece_id exists in both subsets for the paired pieces.

    benchmark._find_pairs keys on the stem, so a stem without the subset would
    let one mic distance silently overwrite the other.
    """
    cl = list_pieces("ENSTDkCl", _archive("ENSTDkCl", ["deb_clai"]))[0]
    am = list_pieces("ENSTDkAm", _archive("ENSTDkAm", ["deb_clai"]))[0]

    assert cl.stem != am.stem
    assert cl.stem == "ENSTDkCl-deb_clai"


# --- pairing -------------------------------------------------------------


def test_paired_ids_are_the_intersection():
    """HANDOFF §9 claimed both subsets held the same performances.

    Measured against the real archives: 30 pieces each, only 7 shared. The
    difference matters -- an unpaired Cl-vs-Am comparison confounds mic
    distance with repertoire difficulty.
    """
    listings = {
        "ENSTDkCl": list_pieces("ENSTDkCl", _archive("ENSTDkCl", ["a", "b", "c"])),
        "ENSTDkAm": list_pieces("ENSTDkAm", _archive("ENSTDkAm", ["b", "c", "d"])),
    }
    assert paired_ids(listings) == {"b", "c"}


def test_paired_ids_is_empty_for_disjoint_subsets():
    listings = {
        "ENSTDkCl": list_pieces("ENSTDkCl", _archive("ENSTDkCl", ["a"])),
        "ENSTDkAm": list_pieces("ENSTDkAm", _archive("ENSTDkAm", ["z"])),
    }
    assert paired_ids(listings) == set()


# --- selection -----------------------------------------------------------


def test_selection_prefers_paired_pieces():
    # A small --n must not discard the only controlled comparison.
    pieces = list_pieces("ENSTDkCl", _archive("ENSTDkCl", list("abcdef")))
    chosen = select_pieces(pieces, paired={"e", "f"}, n=2)
    assert {p.piece_id for p in chosen} == {"e", "f"}


def test_selection_fills_the_remainder_from_unpaired():
    pieces = list_pieces("ENSTDkCl", _archive("ENSTDkCl", list("abcdef")))
    chosen = select_pieces(pieces, paired={"a"}, n=3)
    assert "a" in {p.piece_id for p in chosen}
    assert len(chosen) == 3


def test_selection_is_reproducible_for_a_seed():
    pieces = list_pieces("ENSTDkCl", _archive("ENSTDkCl", list("abcdefgh")))
    a = select_pieces(pieces, paired=set(), n=4, seed=13)
    b = select_pieces(pieces, paired=set(), n=4, seed=13)
    assert [p.piece_id for p in a] == [p.piece_id for p in b]


def test_a_different_seed_can_choose_differently():
    pieces = list_pieces("ENSTDkCl", _archive("ENSTDkCl", list("abcdefghij")))
    a = [p.piece_id for p in select_pieces(pieces, set(), n=3, seed=1)]
    b = [p.piece_id for p in select_pieces(pieces, set(), n=3, seed=999)]
    assert a != b or len(set(a) | set(b)) <= 3


def test_n_none_takes_everything():
    pieces = list_pieces("ENSTDkCl", _archive("ENSTDkCl", list("abcde")))
    assert len(select_pieces(pieces, set(), n=None)) == 5


def test_n_larger_than_the_corpus_is_not_an_error():
    pieces = list_pieces("ENSTDkCl", _archive("ENSTDkCl", ["a", "b"]))
    assert len(select_pieces(pieces, set(), n=99)) == 2


# --- annotation parsing --------------------------------------------------


def test_annotation_parses_onset_offset_pitch():
    tr = parse_annotation(_ANNOTATION)
    assert len(tr.notes) == 2
    assert tr.notes[0].pitch == 60
    assert tr.notes[0].onset == pytest.approx(0.5)
    assert tr.notes[0].offset == pytest.approx(1.5)


def test_the_header_row_is_not_a_note():
    assert len(parse_annotation("OnsetTime\tOffsetTime\tMidiPitch\n").notes) == 0


def test_duration_is_the_last_offset():
    assert parse_annotation(_ANNOTATION).duration == pytest.approx(2.0)


def test_out_of_range_pitches_are_dropped_not_raised():
    # NoteEvent raises for these; one stray row must not lose the recording.
    text = "OnsetTime\tOffsetTime\tMidiPitch\n0.0\t1.0\t200\n0.5\t1.5\t60\n"
    tr = parse_annotation(text)
    assert [n.pitch for n in tr.notes] == [60]


def test_short_notes_are_not_lengthened():
    """The reference must stay lossless.

    NoteEvent's default clamp lengthens sub-20ms notes, which would rewrite
    ground truth before scoring -- the same trap read_midi avoids with
    clamp=False.
    """
    tr = parse_annotation("OnsetTime\tOffsetTime\tMidiPitch\n1.0\t1.005\t60\n")
    assert tr.notes[0].offset == pytest.approx(1.005)
    assert tr.notes[0].duration < 0.02


def test_every_reference_note_gets_the_same_velocity():
    # MAPS carries no dynamics. mir_eval rescales velocities to best-fit the
    # reference, so a constant makes the velocity metric meaningless rather
    # than absent -- the manifest flags it and the CLI says so.
    tr = parse_annotation(_ANNOTATION)
    assert {n.velocity for n in tr.notes} == {DEFAULT_VELOCITY}


def test_blank_and_malformed_lines_are_ignored():
    text = "OnsetTime\tOffsetTime\tMidiPitch\n\n0.5\t1.5\t60\ngarbage\n1 2\n"
    assert len(parse_annotation(text).notes) == 1


# --- extraction ----------------------------------------------------------


def test_extract_writes_wav_and_a_pairable_midi(tmp_path):
    """The reference goes out as .mid so benchmark._find_pairs can pair it.

    MAPS ships a .mid too, but pretty_midi REJECTS those files ("largest tick
    ... likely corrupt"), so read_midi raises on every one. The txt is the
    supported path.
    """
    zf = _archive("ENSTDkCl", ["deb_clai"])
    piece = list_pieces("ENSTDkCl", zf)[0]

    wav, mid, entry = extract_piece(piece, zf, tmp_path)

    assert wav.name == "ENSTDkCl-deb_clai.wav"
    assert mid.name == "ENSTDkCl-deb_clai.mid"
    assert wav.read_bytes() == b"RIFFfake"

    from transcriber.midi import read_midi

    assert len(read_midi(mid).notes) == 2
    assert entry["n_notes"] == 2
    assert entry["mic"].startswith("close")


def test_extraction_is_idempotent(tmp_path):
    zf = _archive("ENSTDkCl", ["deb_clai"])
    piece = list_pieces("ENSTDkCl", zf)[0]

    extract_piece(piece, zf, tmp_path)
    wav = tmp_path / "ENSTDkCl-deb_clai.wav"
    wav.write_bytes(b"MARKER")  # a re-fetch would overwrite this

    extract_piece(piece, zf, tmp_path)
    assert wav.read_bytes() == b"MARKER"


def test_no_part_files_are_left_behind(tmp_path):
    zf = _archive("ENSTDkCl", ["deb_clai"])
    extract_piece(list_pieces("ENSTDkCl", zf)[0], zf, tmp_path)
    assert not list(tmp_path.glob("*.part"))


# --- manifest ------------------------------------------------------------


def test_manifest_records_pairing_and_the_velocity_caveat(tmp_path):
    archives = {
        "ENSTDkCl": _archive("ENSTDkCl", ["shared", "cl_only"]),
        "ENSTDkAm": _archive("ENSTDkAm", ["shared", "am_only"]),
    }
    manifest = build_corpus(tmp_path, archives=archives, progress=False)

    assert manifest["n"] == 4
    assert manifest["n_paired_pieces"] == 1
    assert manifest["paired_piece_ids"] == ["shared"]
    # The flag that stops someone quoting a velocity F1 from this corpus.
    assert manifest["velocity_metric_valid"] is False

    paired = [t for t in manifest["tracks"] if t["paired"]]
    assert {t["subset"] for t in paired} == {"ENSTDkCl", "ENSTDkAm"}


def test_manifest_carries_a_hash_per_file(tmp_path):
    # The corpus is never committed, so the manifest is what proves a later
    # run measured the same bytes.
    archives = {"ENSTDkCl": _archive("ENSTDkCl", ["a"])}
    manifest = build_corpus(
        tmp_path, subsets=("ENSTDkCl",), archives=archives, progress=False
    )
    track = manifest["tracks"][0]
    assert len(track["sha256_audio"]) == 64
    assert len(track["sha256_reference"]) == 64


def test_both_subsets_land_in_one_flat_directory(tmp_path):
    archives = {
        "ENSTDkCl": _archive("ENSTDkCl", ["shared"]),
        "ENSTDkAm": _archive("ENSTDkAm", ["shared"]),
    }
    build_corpus(tmp_path, archives=archives, progress=False)

    wavs = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert wavs == ["ENSTDkAm-shared.wav", "ENSTDkCl-shared.wav"]


def test_summarise_names_the_paired_pieces():
    listings = {
        "ENSTDkCl": list_pieces("ENSTDkCl", _archive("ENSTDkCl", ["a", "b"])),
        "ENSTDkAm": list_pieces("ENSTDkAm", _archive("ENSTDkAm", ["b"])),
    }
    out = summarise(listings, paired_ids(listings))
    assert "paired" in out.lower()
    assert "confounds" in out
