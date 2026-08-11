"""MAESTRO corpus metadata parsing and track selection.

The download path is not tested here — it is exercised through an injected
fake so the suite keeps its no-network, no-model guarantee. What IS tested is
everything that decides *which* tracks get measured, because a selection bug
produces a corpus that looks fine and measures the wrong thing. That is
exactly how Phase 12 lost two rounds of conclusions.
"""

import pytest

from evaluation.corpus import (
    REQUIRED_COLUMNS,
    SELECTION_SEED,
    TrackMeta,
    parse_metadata,
    select_tracks,
    summarise_selection,
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
