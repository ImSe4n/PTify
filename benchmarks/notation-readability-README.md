# Phase 29 — making the printed page readable

The user transcribed their own recording, compared it against a published
arrangement of the same music, and said the output was "messy and dirty" with
"a lot of random black squares". Every defect below was found by **looking at
the rendered page**, and none of them would have been caught by the test suite:
`test_svg_is_not_an_empty_page` bounds density from BELOW.

## What the reference proved first

The user supplied images of a published arrangement. Measured against them, the
transcription was already substantially right:

| | reference | PTify |
|---|---|---|
| key | A-flat major / F minor | **F minor** |
| tempo | 122 BPM | **120.2** |
| notes inside the key | — | **94.2%** |
| bass roots | Db → C → F → Ab | **same, bar for bar** |

So "the notes are all wrong" was not what the data said. The music was being
found; it was being **written down** badly. That reframed the whole phase.

## The defects, in the order they were found

| # | defect | before | after |
|---|---|---|---|
| 1 | spurious 16th rests | 68 | 28 |
| 2 | accidentals | 334 (57% of notes) | 38 (7%) |
| 3 | whole-measure rests | 110 | 8 |
| 4 | chord symbols | none | 52 |
| 5 | chord exact-match | 5/14 | **11/14** (14/14 roots) |

### 1. Hairline gaps became rests

`makeNotation` fills every gap between quantised notes however short, so a note
ending one subdivision before the next onset left a 16th rest nobody played.

Justified by measurement rather than taste: of 123 same-staff gaps, **55
(44.7%) were exactly one subdivision** and the next size up appeared 4 times. A
single spike at the grid's own resolution is a quantisation artifact; real
rhythm does not distribute like that.

### 2. Every black key printed as a sharp

`music21.note.Note(61)` is C#4 whatever the key signature says. In F minor that
prints every black key as a sharp AND forces a natural on the white key after
it. Two independent bugs, both entirely in spelling — **the key signature in
the file was already correct**:

- 142 sharps, generating 151 naturals to cancel them
- `Pitch(60)` carries an explicit `natural` whose `displayStatus` is False;
  music21 knows not to print it, the MusicXML exporter writes it anyway, and
  Verovio engraves it. **125 naturals on C, F and G** — steps a 4-flat
  signature does not touch, so not one could ever be needed.

### 3. The black squares — chord symbols, twice over

**First**, inserting `ChordSymbol` into the flat part before `makeNotation`
made music21 lay the staff out as two parallel VOICES, and every measure whose
second voice held no notes engraved a whole-measure rest — a solid black bar.
The threshold is sharp: one symbol is harmless, **five trigger all 57**.
Placing them into the measures *after* `makeNotation` avoids it.

**Second**, Verovio writes a symbol's flat as `<tspan font-family="Leipzig">`
holding U+EA64, a Private Use Area codepoint. The PDF path is SVG → svglib →
reportlab, which has no such font and substitutes a missing-glyph rectangle.

Two false fixes preceded the real one, and both are worth recording:

- Substituting **U+266D**, the proper Unicode flat, changes nothing: reportlab
  renders the base-14 faces through **WinAnsiEncoding**, which stops at 255,
  and U+266D is 9837. One unavailable glyph for another.
- Registering a TrueType face with `pdfmetrics` alone changes nothing either:
  svglib resolves `font-family` through its **own** map and silently falls back
  to Helvetica. The finished PDF had only base-14 fonts and the boxes were
  back.

The fix is `svglib.fonts.get_global_font_map().register_font()`, with an ASCII
fallback (`Db`) when no font on the machine carries the glyph. Verified: the
PDF embeds `AAAAAA+SegoeUISymbol`.

**Third**, even correctly spelled, symbols rendered wrong. Verovio splits `Db`
across nested tspans — root letter in one, accidental in another with no x/y
because SVG flows it after the first. svglib does not implement that flow: it
places every tspan at the text origin, so the accidental landed ON TOP of the
root and, at 720px against the root's 405px, hid it. `Db` rendered as a lone
flat; `DbMaj7` as a flat then `Maj7`.

### 4-5. Chord symbols, and what they cost

Naming a bar from every sounding pitch gives `Fm7addB-` and `C7addC#`, because
a melody has more attacks than the harmony under it. Raising the weight
threshold trades those for `Cpower` and `Fpower`. Neither is a symbol any
arrangement prints.

Template matching asks the right question — *which actual chord best explains
this bar* — scored by weighted duration against ten qualities a pop/jazz
arrangement actually uses. Extensions past the triad additionally have to earn
their place: measured, an Eb at 12.5% was renaming a clean F minor triad to
`Fm7`, where the arrangement prints `Fm`.

## Generalisation — the check this phase could have skipped

Every constant above was tuned on **14 bars of one recording**. That is the
shape of the Phase 24 rate floor, which looked perfect at one tempo and was
rejected at nine.

Measured on ground-truth MIDI from **12 MAESTRO pieces** the detector was never
tuned on — Bach, Beethoven, Brahms, Chopin, Debussy, Haydn, Liszt,
Mendelssohn, Scarlatti, Schubert, Scriabin:

    mean coverage       96.9%
    degenerate figures  0 of 2,389 named
    mean support        0.70 - 0.84

Ground-truth MIDI rather than transcriptions, deliberately: running on PTify's
output would mix naming error with transcription error, and the question is
whether the NAMER generalises.

Root concentration behaves musically too. Scarlatti K.525 in F major: C 39%,
F 15%, G 12%, Bb 12% — top three are 67% of bars. Scriabin's Sonata No. 9
measures 36.1%, which is correct: it is the "Black Mass", built on a synthetic
chord and deliberately near-atonal. `tools/chord_generalisation.py` regenerates
this.

## What this phase does NOT fix

**The notes themselves.** The transcription is at the model's measured ceiling
(0.8502 MAPS onset F1), and Phases 27 and 28 closed the loss-side routes to
improving it as negative results. A cleaner page is a cleaner rendering of the
same notes.

**The remaining chord misses are one extension off**, not wrong chords: `Db` vs
`Dbmaj7`, `Fm7` vs `Fm9`. All 14 roots are correct.

## The process lesson

Four wrong diagnoses preceded the black-square fix: accidentals (a real bug,
not that one), a phantom "103 overlaps" that turned out to be chord members
sharing an onset, overlap-trimming code written for a problem that did not
exist, and a chord duration that was already 0.0. Every one came from reasoning
over MusicXML element counts. **The bisect that found it took one command**,
and the user's screenshot showed it immediately — the boxes were sitting inside
the chord symbol text, which no element count could reveal.
