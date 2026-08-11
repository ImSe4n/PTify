"""MAESTRO corpus metadata parsing and track selection.

The download path is not tested here — it is exercised through an injected
fake so the suite keeps its no-network, no-model guarantee. What IS tested is
everything that decides *which* tracks get measured, because a selection bug
produces a corpus that looks fine and measures the wrong thing. That is
exactly how Phase 12 lost two rounds of conclusions.
"""

from pathlib import Path

import pytest

from evaluation.corpus import (
    REQUIRED_COLUMNS,
    SELECTION_SEED,
    TrackMeta,
    build_corpus,
    fetch_metadata,
    fetch_track,
    parse_metadata,
    select_tracks,
    summarise_selection,
    write_manifest,
)

# A miniature MAESTRO CSV: the real header, and rows chosen to exercise the
# things that actually break — a quoted title containing commas, several
# composers with differing track counts, and both splits.
HEADER = ",".join(REQUIRED_COLUMNS)

CSV = HEADER + "\n" + "\n".join([
    'Frederic Chopin,Ballade No. 3,test,2011,2011/a.midi,2011/a.wav,290.5',
    'Frederic Chopin,Nocturne Op. 27,test,2013,2013/b.midi,2013/b.wav,310.0',
    'Frederic Chopin,Etude Op. 10,test,2015,2015/c.midi,2015/c.wav,150.25',
    'Johann Sebastian Bach,Prelude and Fugue,test,2009,2009/d.midi,2009/d.wav,420.0',
    'Johann Sebastian Bach,Partita No. 2,test,2017,2017/e.midi,2017/e.wav,505.5',
    '"Joseph Haydn","Sonata E-Flat Major, Hob. XVI:49, First Movement",'
    'test,2008,2008/f.midi,2008/f.wav,480.75',
    'Ludwig van Beethoven,Sonata No. 3,test,2018,2018/g.midi,2018/g.wav,600.0',
    'Franz Liszt,Hungarian Rhapsody,test,2011,2011/h.midi,2011/h.wav,650.0',
    'Claude Debussy,Pagodes,train,2015,2015/i.midi,2015/i.wav,349.0',
    'Domenico Scarlatti,Sonata K466,validation,2009,2009/j.midi,2009/j.wav,240.0',
])


def _tracks():
    return parse_metadata(CSV)


# --- parsing --------------------------------------------------------------

def test_parses_every_row():
    assert len(_tracks()) == 10


def test_parses_types():
    t = _tracks()[0]
    assert isinstance(t.year, int)
    assert isinstance(t.duration, float)
    assert t.duration == pytest.approx(290.5)


def test_quoted_title_with_commas_survives():
    """MAESTRO titles genuinely contain commas. Splitting on ',' instead of
    using csv.DictReader would shift every column after the title."""
    haydn = [t for t in _tracks() if t.composer == "Joseph Haydn"][0]
    assert haydn.title == "Sonata E-Flat Major, Hob. XVI:49, First Movement"
    assert haydn.audio_filename == "2008/f.wav"
    assert haydn.duration == pytest.approx(480.75)


def test_missing_columns_raise_loudly():
    """An upstream schema change must fail here, not silently select nothing."""
    with pytest.raises(ValueError, match="missing columns"):
        parse_metadata("composer,title\nChopin,Ballade\n")


def test_empty_csv_has_no_tracks():
    assert parse_metadata(HEADER + "\n") == []


# --- stem -----------------------------------------------------------------

def test_stem_is_filesystem_safe():
    haydn = [t for t in _tracks() if t.composer == "Joseph Haydn"][0]
    for char in '<>:"/\\|?*':
        assert char not in haydn.stem
    assert " " not in haydn.stem


def test_accented_composer_survives_the_stem(tmp_path):
    """MAESTRO is UTF-8 and really does contain "Frederic" with accents. The
    name renders as mojibake in a cp1252 console, which invites "fixing" the
    decode to latin-1 — that would be the actual corruption. This pins that
    accented stems are preserved and usable as real filenames."""
    t = TrackMeta(
        composer="Frédéric Chopin", title="Étude Op. 10",
        split="test", year=2015, midi_filename="2015/x.midi",
        audio_filename="2015/x.wav", duration=160.0,
    )
    assert "é" in t.stem
    path = tmp_path / f"{t.stem}.wav"
    path.write_bytes(b"x")
    assert next(tmp_path.iterdir()).stem == t.stem


def test_stems_are_unique_across_the_corpus():
    """composer+title is NOT unique in MAESTRO — the same piece recurs across
    competition years — so the stem carries the year and source filename."""
    stems = [t.stem for t in _tracks()]
    assert len(set(stems)) == len(stems)


# --- selection ------------------------------------------------------------

def test_selects_only_the_requested_split():
    for t in select_tracks(_tracks(), n=8):
        assert t.split == "test"


def test_selection_is_deterministic():
    a = select_tracks(_tracks(), n=6)
    b = select_tracks(_tracks(), n=6)
    assert [t.stem for t in a] == [t.stem for t in b]


def test_different_seed_gives_a_different_draw():
    a = select_tracks(_tracks(), n=6, seed=SELECTION_SEED)
    b = select_tracks(_tracks(), n=6, seed=SELECTION_SEED + 1)
    assert [t.stem for t in a] != [t.stem for t in b]


def test_selection_is_stable_under_input_row_order():
    """Selection must not depend on CSV row order, or re-downloading the
    metadata could silently reshuffle the corpus and invalidate a baseline
    while every other input looked identical."""
    forward = _tracks()
    shuffled = list(reversed(forward))
    assert ([t.stem for t in select_tracks(forward, n=6)]
            == [t.stem for t in select_tracks(shuffled, n=6)])


def test_round_robin_maximises_composer_diversity():
    """Chopin has 3 test tracks and Bach 2; a plain sample could return three
    Chopins. Round-robin must spend its first picks on distinct composers."""
    picked = select_tracks(_tracks(), n=5)
    assert len({t.composer for t in picked}) == 5


def test_every_composer_appears_before_any_repeats():
    """The fixture's test split has 5 composers (Chopin x3, Bach x2, and three
    singletons). Asking for 6 must cover all 5 before drawing a second track
    from anyone — that is the property round-robin exists to guarantee, and it
    is what keeps a 12-track mean from being four Chopin performances."""
    picked = select_tracks(_tracks(), n=6)
    assert len({t.composer for t in picked[:5]}) == 5
    assert len({t.composer for t in picked}) == 5


def test_n_larger_than_the_pool_returns_the_pool():
    """Must not raise, loop forever, or pad with duplicates."""
    picked = select_tracks(_tracks(), n=999)
    assert len(picked) == 8  # every test-split row in the fixture
    assert len({t.stem for t in picked}) == 8


def test_no_duplicates_when_composers_are_exhausted_unevenly():
    picked = select_tracks(_tracks(), n=8)
    assert len({t.stem for t in picked}) == 8


def test_zero_or_negative_n_raises():
    with pytest.raises(ValueError):
        select_tracks(_tracks(), n=0)
    with pytest.raises(ValueError):
        select_tracks(_tracks(), n=-1)


def test_unknown_split_raises_and_names_what_exists():
    with pytest.raises(ValueError, match="validation"):
        select_tracks(_tracks(), n=2, split="nonexistent")


# --- reporting ------------------------------------------------------------

def test_summary_reports_counts():
    out = summarise_selection(select_tracks(_tracks(), n=5))
    assert "5 tracks" in out
    assert "5 composers" in out


def test_summary_handles_empty():
    assert summarise_selection([]) == "(no tracks selected)"


def test_summary_has_no_nan():
    assert "nan" not in summarise_selection(select_tracks(_tracks(), n=3))


# --- fetching (Phase 13c) -------------------------------------------------
#
# Every one of these injects a fake opener/downloader. Nothing here touches
# the network, and importing evaluation.corpus must never import a network
# client either.

def _fake_pair_writer():
    """A downloader that writes a real (tiny) WAV or MIDI, recording calls."""
    import soundfile as sf

    from evaluation.cases import load
    from evaluation.synth import DEFAULT_SAMPLE_RATE, render
    from transcriber.midi import write_midi

    calls = []

    def download(url, dest):
        calls.append(url)
        tr = load("triads")
        if str(dest).endswith(".wav"):
            sf.write(str(dest), render(tr, sr=DEFAULT_SAMPLE_RATE),
                     DEFAULT_SAMPLE_RATE)
        else:
            write_midi(tr, dest)

    return download, calls


def test_fetch_metadata_uses_the_cache_without_calling_out(tmp_path):
    cache = tmp_path / "maestro.csv"
    cache.write_text(CSV, encoding="utf-8")

    def explode(url):
        raise AssertionError("cache hit must not reach the network")

    assert fetch_metadata(cache, opener=explode) == CSV


def test_fetch_metadata_writes_the_cache(tmp_path):
    cache = tmp_path / "nested" / "maestro.csv"
    text = fetch_metadata(cache, opener=lambda url: CSV.encode("utf-8"))
    assert text == CSV
    assert cache.read_text(encoding="utf-8") == CSV


def test_fetch_metadata_decodes_as_utf8():
    """MAESTRO really is UTF-8. Decoding as latin-1 would turn every accented
    composer into mojibake, and that name reaches filenames and the manifest."""
    body = 'canonical_composer\nFrédéric Chopin\n'.encode("utf-8")
    assert "é" in fetch_metadata(None, opener=lambda url: body)


def test_fetch_track_writes_dot_midi(tmp_path):
    """The corpus writes .midi on purpose, so the real data exercises the
    pairing fix rather than leaving it covered only by unit tests."""
    download, _ = _fake_pair_writer()
    meta = _tracks()[0]
    audio, midi = fetch_track(meta, tmp_path, downloader=download)

    assert audio.suffix == ".wav" and midi.suffix == ".midi"
    assert audio.exists() and midi.exists()

    from evaluation.benchmark import _find_pairs
    assert len(_find_pairs(tmp_path)) == 1


def test_fetch_track_is_idempotent(tmp_path):
    """Re-running after an interruption must resume, not re-download."""
    download, calls = _fake_pair_writer()
    meta = _tracks()[0]

    fetch_track(meta, tmp_path, downloader=download)
    assert len(calls) == 2
    fetch_track(meta, tmp_path, downloader=download)
    assert len(calls) == 2, "existing files were re-downloaded"


def test_default_downloader_cleans_up_a_failed_partial(tmp_path):
    """REGRESSION GUARD: a truncated WAV left behind by an interrupted
    download reads fine in librosa and would score as a mysteriously bad
    recording rather than as a broken one."""
    from evaluation.corpus import _default_downloader

    dest = tmp_path / "x.wav"

    def boom(url, filename, reporthook=None):
        Path(filename).write_bytes(b"partial")
        raise KeyboardInterrupt

    import evaluation.corpus as corpus_mod
    original = corpus_mod.urllib.request.urlretrieve
    corpus_mod.urllib.request.urlretrieve = boom
    try:
        with pytest.raises(KeyboardInterrupt):
            _default_downloader("http://example/x.wav", dest)
    finally:
        corpus_mod.urllib.request.urlretrieve = original

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == [], "a .part file was left behind"


def test_build_corpus_produces_a_complete_manifest(tmp_path):
    download, _ = _fake_pair_writer()
    manifest = build_corpus(
        n=3, out_dir=tmp_path / "recs", progress=False,
        opener=lambda url: CSV.encode("utf-8"), downloader=download,
    )

    assert manifest["n"] == 3
    assert manifest["license"] == "CC BY-NC-SA 4.0"
    assert len(manifest["tracks"]) == 3
    for entry in manifest["tracks"]:
        assert entry["n_notes"] > 0
        assert len(entry["sha256_audio"]) == 64
        assert len(entry["sha256_midi"]) == 64
        assert entry["composer"] and entry["title"]


def test_build_corpus_is_reproducible(tmp_path):
    """Two builds of the same corpus must agree byte for byte, or a baseline
    measured against one of them is not comparable to the other."""
    download, _ = _fake_pair_writer()
    kwargs = dict(n=3, progress=False,
                  opener=lambda url: CSV.encode("utf-8"), downloader=download)

    a = build_corpus(out_dir=tmp_path / "a", **kwargs)
    b = build_corpus(out_dir=tmp_path / "b", **kwargs)

    assert [t["stem"] for t in a["tracks"]] == [t["stem"] for t in b["tracks"]]
    assert ([t["sha256_audio"] for t in a["tracks"]]
            == [t["sha256_audio"] for t in b["tracks"]])
    assert a["csv_sha256"] == b["csv_sha256"]


def test_manifest_round_trips_as_json(tmp_path):
    import json

    download, _ = _fake_pair_writer()
    manifest = build_corpus(
        n=2, out_dir=tmp_path / "recs", progress=False,
        opener=lambda url: CSV.encode("utf-8"), downloader=download,
    )
    path = write_manifest(manifest, tmp_path / "out" / "manifest.json")
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def test_importing_corpus_pulls_in_no_network_client():
    """The suite's no-network guarantee is structural: the module must not
    import a heavyweight HTTP client at import time."""
    import sys
    assert "huggingface_hub" not in sys.modules
