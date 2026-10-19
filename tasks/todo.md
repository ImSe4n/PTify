# NEXT (planned 2026-10-18, branch phase-42-production-beta) -- awaiting user pick

Live: https://ptify.vercel.app, invite-only beta, model p39. Done this branch:
sign-in + legal accuracy copy updated to p39 figures (SMD 0.955, far phone +1.9,
MAPS 0.844); redeployed.

## A. DONE 2026-10-19 (live, verified)
- api/ops.py + api/routes/ops.py: POST /v1/jobs/{id}/feedback (owner only;
  files copied to /data/feedback/<id>/ ONLY with keep_audio opt-in);
  GET /v1/admin/feedback, /v1/admin/stats (admins; 404 for everyone else);
  job_log written on every job end via InProcessQueue.on_finish (no audio).
- DELETE /v1/auth/me: account + jobs + files + feedback (incl. kept audio).
- Frontend: "Something wrong?" form on the result page (keep box unticked by
  default); "Delete account" (confirm) beside Sign out.
- Privacy copy CORRECTED: in-app LegalScreen said audio stays on this
  server / default engine sends nothing (false in production) and that the
  model trained on MAPS (test only); terms.html claimed a history delete
  button that does not exist. Both fixed; opt-in feedback + job log described.
- Tests: tests/test_api_ops.py (5); 318 API tests pass; tsc + vite build OK.
- LIVE SMOKE: signup -> job succeeded -> feedback kept=True -> admin routes
  404 for a normal user -> delete account 204 -> login 401 -> volume
  feedback/ empty. Old smoketest@ptify.invalid account deleted the same way.
- Kept recordings: `modal volume get ptify-data feedback/<id> <dir>`; list
  via GET /v1/admin/feedback with the admin account.
- Still running: score benchmark through the fixed remote path
  (benchmarks/score-benchmark-rhythm-run8-p39.json).

## (original A plan)
We have NO real upload with known-correct notes; every number is a proxy.
- "This transcription is wrong" button on the result page: user can attach a
  note and, with explicit opt-in consent, keep the audio + our output for
  evaluation (default: deleted after 1 h as now). Admin-only listing.
- Ops basics found while deploying: delete-account route (privacy page
  promises deletion on request), job-failure log/alert, re-run the score
  benchmark through the FIXED remote path (old audio rows used ByteDance
  thresholds, so the notation baseline is slightly stale).
- Success: feedback stored and listable by admin; delete-account tested.

## B. Phase 43 -- IN PROGRESS 2026-10-19
Built: Salamander Grand v3 fetch (CC BY 3.0, attribution in
var/pianos/Grand Piano, Salamander/LICENSE.txt; `tools/fetch_vcsl.py
--salamander`), Salamander name parsing, `Timbre` per-render variation
(decay 0-1.5/s, unison detune 0 or 1-8 cents, brightness -0.6..0.8;
`--augment-synth-timbre`), 15 sampled_piano tests + sampler test. No clean-
licence uprights beyond VCSL's two (Pianobook/K18 terms unclear) -> timbre
randomisation stands in for instrument variety.
Run design: continue from p41 (INITS["p41"], sha e87c30ee...) 8,000 steps
with Knight + Kawai + Steinway + Salamander, synth 0.5, timbre on; Yamaha
upright still HELD OUT. = 16k piano-diverse steps total. Judged vs p39
(shipped) with the same rule: MAPS >= +1.0; SMD/maestro/uploads >= p39-0.2.
Scoring kernel now ships the p39 baseline in its own code dataset.

### (original B plan)
p41: MAPS +0.54 (12/14), held-out upright +2.95 (12/12), grands -0.1. Bar was +1.0.
- More instruments: Salamander Grand (CC BY 3.0), Sofia's grands (CC BY),
  more uprights (check K18 and Pianobook licences); keep one sampled upright
  held out.
- Longer run: 16k steps across two Kaggle sessions (`--resume auto`), synth
  share 0.5; try both T4s (Kaggle gave 2).
- Criteria (pre-registered): MAPS >= +1.0 vs p39; held-out upright up; SMD,
  MAESTRO clean and uploads not below p39 - 0.2.

## C. Phase 44 -- IN PROGRESS: how the duration head is wrong (ASAP val, run 8,
`tools/duration_breakdown.py` ratio bands, predicted/written):
  exact 36.7% | x2/3-x3/4 ("lost a dot"/triplet) 22.3% | x4/3-x3/2 8.9% |
  x1/2 8.0% | x2 8.2% | <=x1/4 3.5% | >=x4 1.7% | other short 6.3% | long 4.5%
-> the largest error is a systematic ~2/3-3/4 SHORTENING. Next: which
written values (var/dur_confusion.py), then pick the fix.
WHICH VALUES (var/dur-confusion.log): top confusion written 1/3 beat ->
predicted 1/4 (4,381), but 84.5% of it is ONE piece (Islamey), 98% in three.
-> not a local head weakness: a PIECE-LEVEL metric misreading.
SCALE ORACLE (var/dur_scale_oracle.py, var/dur-scale-oracle.log): head
accuracy 38.04% as is -> 54.44% with the best single factor per piece
(answer key picks it). 8 of 20 pieces at the wrong scale (x0.5, x2/3, x4/3,
x1.5). So: up to +16 pt is a per-piece METER/BEAT-LEVEL decision -- the same
problem HANDOFF parked (no chooser beat model confidence: 29-31/89 vs 64).
The other ~46% of errors are per-note. DECISION PENDING: do not retry a
chooser without a NEW signal; candidates for that signal are listed in the
user report.
SCORE BENCHMARK THROUGH THE FIXED PATH (p39, onset 0.6 / frame 0.01 = what
production runs; benchmarks/score-benchmark-rhythm-run8-p39.json), audio
row vs the old (16b decoded at ByteDance 0.7/0.05): chained 83.9 -> 86.1
(+2.2) BUT rhythm 76.9 -> 73.0, durations 58.8 -> 56.2, beat level 19 -> 17,
beat F 0.50 -> 0.43. Perfect-MIDI row identical (sanity). Hypothesis: the
rhythm model reads PLAYED length; frame 0.05 ends notes far better (offset
F1 0.49 -> 0.69, 2026-10-17 probe). RUNNING: onset 0.6 + frame 0.05
(var/score_bench_f005.py -> benchmarks/score-benchmark-rhythm-run8-p39-f005.json).
If it restores rhythm while keeping recall, change PTIFY_FRAME_THRESHOLD
(check MAPS/SMD onset F1 too -- frame was found inert for onsets in Phase 23).
Phase 43 TRAINING on Kaggle (v3, from p41, 4 pianos + timbre), started ~23:25.

### (original C plan)
Phase 37: durations 38% right; the duration HEAD is the bottleneck (+28.8 pt
ceiling), not engraving. Retrain/re-decode the rhythm model's duration head;
judge on ASAP validation, then the score benchmark once.

## D. Songscription bar (user's stop rule)
Needs: can the free tier export MIDI? If yes: 3-4 clips with known notes,
score both through the same metric.

VERCEL FIX (2026-10-18): Git build of this branch failed (repo root ->
Python detected -> torch has no cp314 wheel). Project now: root dir
`frontend`, framework vite, VITE_API_BASE in project env; root
`.vercelignore` limits CLI uploads to frontend/. Preview + prod rebuilt from
the root without --build-env: API URL and new copy in the live bundle.

Recommended order: A (1 session) -> B on Kaggle in the background while C is
built locally. D whenever the MIDI question is answered.

---

# Phase 10 -- production readiness (user: "resolve all of these", 2026-10-18)

Target: PRIVATE BETA. Frontend on Vercel; backend as a second Modal app
(ASGI) with a persistent Volume for SQLite + job files; transcription stays
on the existing GPU host. Nothing goes live without the user's OK.

- [x] 1. Remote path thresholds FIXED: RemoteEngine asks for
      PTIFY_REMOTE_ENGINE (default ptify) with its thresholds, and raises if
      the host serves another model. 67 remote tests pass (3 new). NOTE:
      score-benchmark audio rows (benchmarks/score-benchmark-*) were produced
      through the old default (ByteDance thresholds on ptify), so they
      understate ptify slightly.
- [x] 2. Security review DONE. Already sound: PBKDF2 600k + per-user salt,
      compare_digest everywhere, JWT alg pinned + 24 h expiry, upload suffix
      allow-list (client filename never a path), size cap, other users' jobs
      404. FIXED (tests/test_api_production.py, 8 tests):
        F1 anonymous could start GPU jobs -> PTIFY_REQUIRE_ACCOUNT=1
           (enforced at job creation; /me stays 200 so the app shows sign-in)
        F2 login/signup unlimited (PBKDF2 = CPU flood) -> per-IP limit,
           PTIFY_AUTH_RATE_LIMIT_PER_MINUTE (10); X-Forwarded-For only with
           PTIFY_TRUST_PROXY=1
        F3 open sign-up -> PTIFY_SIGNUP_CODE invite code (+ frontend field)
        F4 no daily cap -> PTIFY_DAILY_JOB_LIMIT (rolling 24 h, in-process)
        F5 weak JWT secret -> >=32 chars required in production mode
      Not fixed, accepted: ffmpeg decodes untrusted uploads (keep the image
      updated; it runs inside Modal's sandbox); in-process limits reset on
      restart (single instance).
- [x] 3. Cost: account required + daily cap (above).
      USER: set a spending limit in the Modal dashboard.
      ALSO FOR USER: AuthScreen shows v1-era claims ("0.840 onset F1 on MAPS",
      "+5.3 over ByteDance"); update before going public.
- [~] 4. hosting/modal/api_app.py WRITTEN (not deployed): one container,
      Volume ptify-data (/data: SQLite + job files), secret `ptify-api`,
      remote-only engine, production settings. SQLite WAL on a Modal Volume
      MEASURED OK (throwaway probe, 200/200 rows, probe volume deleted).
      Also added PTIFY_ALLOWED_ENGINES (local engines = CPU abuse) + engine
      list filtered; 313 API tests pass. DEPLOYED 2026-10-18 (user: "go"):
      https://imse4n--ptify-api-api.us-east.modal.run (Modal app ptify-api,
      secret ptify-api: generated JWT secret, host token, invite code in
      var/invite-code.txt). Admin: PTIFY_ADMIN_EMAILS=snie7638@gmail.com skips
      rate/concurrency/daily limits (sign-up with that email + invite code).
      First deploy failed: numpy upgraded to 2.x by the 2nd pip step -> torch
      "Numpy is not available"; pinned again, redeployed. SMOKE (live): engines
      ['remote']; signup w/o invite 403; anonymous job 401; 25 s clip ->
      succeeded in 45 s, MIDI/MusicXML/PDF served. Test account
      smoketest@ptify.invalid left in the prod DB (no delete-account route).
      CORS: redeployed with PTIFY_CORS_ORIGINS=https://ptify.vercel.app
      (preflight 200, other origins get no ACAO). REDEPLOY THE API WITH THAT
      ENV SET, or CORS falls back to localhost.
- [x] 5. frontend/public/terms.html (AI-drafted, NOT legal advice; user to
      review), linked from sign-up. Uploads deleted ~1 h after a job finishes
      (job_ttl_seconds janitor). Frontend builds.
- [x] 6. Frontend LIVE: https://ptify.vercel.app (Vercel project pt-ify/ptify,
      `npx vercel deploy --prod --yes --build-env VITE_API_BASE=<api url>`
      from frontend/). terms.html served.

---

# RESUME HERE (2026-10-18 ~12:10)

Shipped: p39 is the live model (host redeployed, crosscheck PASS). Release
asset model-v2 NOT published yet (user, via GitHub web UI).

Running on Kaggle: `ptify-score` v3 = p39 vs p41 (Phase 41, sampled pianos).
(v2 scored ONLY maestro3min: one cache per model across sets tripped the
bind_cache guard; fixed to one cache per model+set, test added.)
v2 maestro3min (`benchmarks/wild/p41/`), p39 -> p41 onset F1: clean 0.9562 ->
0.9550 (-0.13) | phone -0.06 | youtube -0.12 | tiktok -0.16 | phone_far
-0.16; p41 better on only 2-5 of 12 tracks. p39 numbers reproduce exactly.
1. `python -m kaggle kernels status imse4n/ptify-score` -> COMPLETE?
2. `python -m tools.kaggle_score fetch` -> var/score-run/ (OVERWRITES the
   p39-vs-16b JSONs there; those are already saved in benchmarks/wild/p39/).
3. Compare p39-* vs p41-* per set (same script as Phase 39 step 4).
   Criteria (pre-registered): MAPS +1.0 pt or more; SMD, maestro clean and
   upload presets not below p39 - 0.2. Also report held-out vcsl_yamaha.
4. If it passes: ship p41 like p39 (PTIFY_SHIPPED_* + host pin, redeploy
   with PTIFY_HOST_ENGINE=ptify, crosscheck). If not: p39 stays.
User's stop rule: "if this doesn't make us as good as Songscription, give
up." Open question to user: can Songscription's free tier export MIDI?

---

# Phase 41 -- many pianos: train on sampled uprights/grands (STARTED 2026-10-17)

## Why
- Biggest measured gap is the INSTRUMENT: MAPS close mic (upright Disklavier,
  confirmed by DAFx 2019) 0.88-0.90 even delay-corrected vs 0.955 on MAESTRO
  grands. Caveat: MAPS ENSTDkCl annotations have known errors (803 unplayed,
  342 deviations, 1,613 slow starts), so part of that gap is the answer key.
- Users mostly play home uprights / digital pianos. MAESTRO is one Yamaha
  Disklavier grand model. No room/codec augmentation changes the instrument.

## User's stop rule (2026-10-17)
"If this really doesn't make us as good as Songscription, give up."
Needs a measurable comparison: ASK whether Songscription's free tier exports
MIDI (then score both on the same clips with known notes). Otherwise the
comparison is screenshots only.

## Pianos (all CC0, VCSL, github.com/sgossner/VCSL)
Upright Knight (420 MB), Upright Yamaha (248 MB), Grand Kawai (460 MB),
Grand Steinway B (1.3 GB). Raw WAV, per-instrument naming; sparse sampling
-> our own sample player (no SFZ/fluidsynth), pitch-shift to nearest sample.

## Steps
- [x] 1. `tools/fetch_vcsl.py` -> var/pianos/ (2.4 GB WAV).
- [x] 2. `evaluation/sampled_piano.py` (+ tests): octave offset MEASURED
      (Kawai/Knight/Yamaha +12, Steinway 0), velocity curve, damper/pedal,
      release noise, negative-start tails; banks -> var/piano-banks/*.npz.
- [x] 3. Probe A, p39 on test MIDI rendered (`benchmarks/wild/vcsl-*`):
        real MAESTRO 0.956 | Steinway 0.930 | Yamaha upright 0.907 |
        Kawai 0.895 | Knight upright 0.834   (precision 0.80-0.93 vs 0.99)
      -> piano-bound, mostly INVENTED notes, as on MAPS. Caveat: renderer is
      simpler than a real piano. (First run scored all four 0.8950: cache
      keyed by stem reused piano 1 for all -- fixed, `bind_cache` guard.)
- [x] 4. Probe B SCORED 2026-10-18 (`benchmarks/wild/p41/`), p39 -> p41 onset F1:
        MAPS 0.8437 -> 0.8492 (+0.54, 12/14) | held-out Yamaha upright (render)
        0.9074 -> 0.9369 (+2.95, 12/12) | Knight (trained) +7.20 | SMD -0.04 |
        maestro clean -0.13, phone -0.06, youtube -0.12, tiktok -0.16, far -0.16
      Pre-registered rule (MAPS >= +1.0): FAILS. p39 stays live. Gains are
      precision (MAPS P 0.866 -> 0.880). Caveat: held-out piano uses OUR
      renderer. Recommendation: bigger run (more/varied uprights, longer)
      rather than shipping a near-miss; user may choose to ship p41.
      Was: Probe B RUNNING (Kaggle v2, from p39, p39 recipe + synth 0.35 on
      Knight/Kawai/Steinway; Yamaha HELD OUT), started 2026-10-18 00:00.
      (A first prepare swallowed `--init p39` into train args; caught before
      the kernel was pushed; launcher now refuses it.)
      Scoring kernel retargeted: p39 vs p41 + rendered yamaha/knight sets.
      TRAINED (done by 11:52 2026-10-18): var/transcriber-run-p41/ (sha
      e87c30ee1fee, deployable, differs from p39). Val step 500 -> 8000:
      onset -0.8%, frame -4.4%, offset -1.8%; aug onset -2.6%, aug frame
      -5.0%. Console log came back empty (download glitch); synth was on:
      train.py refuses synth without banks, and aug val started higher than
      p39's (0.0312 vs 0.0290). Scoring kernel v2 pushed.
      Probe B (~10 h Kaggle): train from p39 with rendered segments mixed
      in (on-the-fly in the dataloader). Criteria: MAPS +1.0 pt or more,
      SMD and uploads not below p39 - 0.2.

---

# Phase 40 probe -- are missed notes half-heard? (2026-10-17)

`tools/chord_decode.py`, clean maestro_test12_3min, 16b activations cached in
var/activations/ (`benchmarks/wild/chord-decode-maestro3min.json`):
- Re-decode reproduces the engine exactly (base 0.9552 = the wild run).
- CHORD-AWARE decode FAILS: never beats a global lower threshold
  (g0.3 0.9630 vs c0.3 0.9615; g0.4 0.9620 vs c0.4 0.9607). Dropped.
- MISSED notes' peak onset activation (+-50 ms), 2,071 misses at 0.6:
  <0.05 35.2% (unheard) | 0.05-0.2 20.0% | 0.2-0.6 43.7% (half-heard) | >=0.6 1.0%
  -> almost half the misses are nearly heard. A global threshold recovers
  them on MAESTRO/SMD but adds false notes on MAPS (Phase 38 A).
- NEXT: is a candidate-note RESCORER learnable? Quick check: classifier on
  candidates from 6 excerpts, test on the other 6 (and MAPS), must beat the
  best global threshold held-out. If yes -> plan a proper second-stage model.
- RESCORER (`tools/rescore_probe.py`, HistGradientBoosting on 14 candidate
  features), 2-fold by track on maestro 3-min, cuts chosen on train tracks:
    fold 1: base 0.9563 | global 0.3 0.9647 | rescorer 0.9638
    fold 2: base 0.9542 | global 0.3 0.9612 | rescorer 0.9582
  -> does NOT beat a global threshold on MAESTRO.
  MAESTRO -> MAPS: base 0.8503 | global 0.3 0.8398 | rescorer@0.45 0.8457
  -> FAILS both tests. With 16b's activations and 12 tracks of training, a
  rescorer does not beat plain decoding. (More training data untested.)
- MAPS probe (`benchmarks/wild/chord-decode-maps.json`): base 0.8503 matches;
  every lower threshold hurts. Misses: <0.05 33% | 0.05-0.2 11% | 0.2-0.6
  31% | >=0.6 25%. The >=0.6 group (1,191): 1,181 have a same-pitch note
  decoded 50-150 ms away (median 58 ms) = TIMING misses, vs 1% on MAESTRO.
  Per-track signed offset (model onset - annotation, matched within 150 ms):
  CLOSE mic +4..+9 ms typical (scn16_4 +21); AMBIENT mic (3-4 m) +15..+29 ms,
  on the SAME performances with the SAME annotations -> acoustic, not an
  annotation error: ~10 ms flight time + reverb-smeared attacks. Far mic =
  systematic late onsets -> notes cross the 50 ms window and count as a miss
  AND an extra. Harmless for sheet music (constant), but it inflates far-mic
  error counts. Fixable without training: estimate per-recording delay and
  shift. Oracle upper bound (true median shift per track) running.

---

# Phase 39 -- retrain with UPLOAD-CHAIN augmentation (PLAN 2026-10-17)

## Why (Phase 38 evidence)
- Errors are acoustic, not musical: implausible extras <= 2.2% everywhere.
- Upload conditions cost: far phone -7.2 pt, tiktok -3.4, phone -1.7; harmonic
  extras triple under tiktok-style compression + codec.
- The 16b training sampler (`training/augment.py`) has room, detune, noise and
  level -- but mic EQ is OFF (`eq_prob=0.0`) and there is NO compression and NO
  codec. The model has never heard an upload.
- Phase 23: continuing 16b on the SAME recipe made it worse ("change the
  recipe, not the schedule"). This changes ONE thing: the acoustic chain.
- A lower onset threshold helped SMD/uploads but hurt MAPS (Phase 38 A), so
  no decode setting fixes this; the model has to hear better.

## Requirements
- Training sampler gains three stages, drawn per segment like the others:
  mic EQ (turn on), compression (threshold/ratio ranges), codec round-trip
  (Opus/MP3, bitrate range), reusing `evaluation/upload_chain.py`.
- Benchmark presets untouched (they are the yardstick, not the distribution).
- Throughput stays >= 15 segments/s/worker (the existing budget), measured.
- "It bites" measured on REAL audio, as the module requires.

## Success criteria (judged at onset threshold 0.6 vs 16b at 0.6)
- Upload set: far phone and tiktok mean onset F1 each +1.0 pt or more.
- No regression beyond noise: SMD >= 0.9539 - 0.002, MAPS paired >= 0.8502
  - 0.002, MAESTRO 3-min clean >= 0.9552 - 0.002.
- Otherwise 16b stays the shipped checkpoint.

## Files
- `training/augment.py` (+ tests): new plan fields + apply stages.
- `training/train.py` / Kaggle notebook: flags for the new stages.
- Scoring: `tools/wild_benchmark.py --checkpoint` (local) or a temporary host
  deploy -- NOT the live host without asking.

## Not doing
No architecture change, no loss change, no threshold change, no new data.

## Steps
- [x] 1. Sampler stages + tests (`training/augment.py`, flags
      `--augment-compress-prob/--augment-codec-prob`); defaults reproduce the
      16b recipe exactly. Cost 106 ms/segment at .5/.5/.5 -- accepted: GPU
      consumes ~2.2 seg/s, 2 workers supply ~19.
- [x] 2. It bites (`var/it-bites.log`, 69 x 10 s segments, 16b via GPU host):
      clean 0.9545 | old recipe 0.9245 | new (.5/.5/.5) 0.9185. Per stage
      (new-old): none 0.000 (old params identical, as designed); codec alone
      ~0; compress -0.6 pt; eq+compress -1.7 pt. -> run with eq .7,
      compress .7, codec .5. Expectation: targets the UPLOAD gap; it will not
      fix clean-audio masked chord misses (45% of errors).
- [x] 3. TRAINED 2026-10-17 11:40-19:30 (8,000 steps, 0.28 steps/s, T4):
      var/transcriber-run/checkpoints/ptify-note-pedal.pth (sha 6687be39bdb0,
      deployable, differs from 16b). Per-head val, step 500 -> 8000:
      onset -1.1%, frame -0.1%, offset -2.2%; aug onset -1.8%, aug frame
      -0.2% -- inside noise, but onset did NOT rise (Phase 23's did).
      Scoring kernel `tools/kaggle_score.py` (16b vs p39, same audio) pushed.
      Was: Kaggle run via `tools/kaggle_transcriber.py` (ships tested code in a
      private dataset; no commit needed), from 16b, Phase 23 recipe
      (velocity=0.1, batch 4 x accum 2, clean 0.2), 8,000 steps (~8.2 h; a
      script kernel has a 12 h cap).
- [x] 4. SCORED on Kaggle (`tools/kaggle_score.py`, both models, same 16 kHz
      audio, threshold 0.6; `benchmarks/wild/p39/`). Onset F1 16b -> p39:
        far phone 0.8603 -> 0.8788 (+1.85, 12/0) | tiktok 0.9301 -> 0.9401
        (+1.00, 12/0) | phone +0.78 (11/1) | youtube +0.05 | clean +0.11 |
        SMD 0.9538 -> 0.9553 (+0.15, 33/16) | MAPS 0.8503 -> 0.8437 (-0.66, 3/11)
      PRE-REGISTERED RULE: FAILS on MAPS (allowance -0.2). 16b stays unless
      the user decides otherwise.
      Post-hoc (NOT the rule): MAPS drop is mostly far-mic timing. p39 onsets
      ~1 ms later; far-mic tracks already sit ~24 ms late (sound travel,
      `tools/latency_probe.py`), so more notes cross 50 ms. Note counts barely
      change. Delay-corrected MAPS: 0.8924 -> 0.8906 (-0.18, inside noise).
      Per-recording delay is NOT estimable from audio (the audio is late too):
      it is a key-press-vs-sound artifact, harmless for sheet music.
      DECISION (user, 2026-10-17): SHIP p39. DONE:
        - transcriber/ptify.py: PTIFY_SHIPPED_* = p39 (16b constants kept as
          the pin for old reports); doctor/CLI use SHIPPED.
        - checkpoints/ptify-p39-step8000.pth (sha 6687be39...).
        - Host: pins p39, bakes it FROM THIS MACHINE (add_local_file) because
          the model-v2 release asset is NOT PUBLISHED (no gh CLI here).
          Redeployed with PTIFY_HOST_ENGINE=ptify; crosscheck PASS
          (var/crosscheck-p39.json: 286/286 notes, onset F1 1.000000; 16b gave
          284 on this clip). Tests: 53 weight/host tests pass.
        - TODO (user): publish release model-v2 with
          checkpoints/ptify-p39-step8000.pth, or fresh clones cannot fetch it.
        - Rollback: point PTIFY_SHIPPED_* (and the host pin) back at 16b.
        - var/wild-estimates/ptify caches are 16b output under the label
          `ptify`; use a new --cache for p39 runs.

---

# FUTURE GOALS (user, 2026-10-16) -- not now, transcription first

- **Accompaniment**: a song with vocals + piano -> isolate the background
  piano and transcribe it as an accompaniment part.
- **Arrangement**: a song with no piano (or piano + vocals) -> our own model
  writes a piano arrangement. `docs/arbitrary-audio.md` has the costed
  assessment (separation stage, then arrangement, which has no metric yet).
- Both depend on the transcriber being right on real uploads first.

---

# Phase 38 -- transcription on REAL uploads: measure, then classify the errors (PLAN, awaiting approval 2026-10-16)

## Why
Users upload YouTube videos, TikToks, phone recordings of themselves and of
others. Our numbers come from MAESTRO (one competition hall, Disklavier, good
mics), where 16b is at 0.963 onset F1 -- near the ceiling and not the target.
On MAPS (different piano/room) it is 0.850; nothing measures phone or
YouTube/TikTok audio. User's hypothesis: the model has no musical knowledge,
which is why it overlaps notes and does weird things. That is testable: an
error taxonomy says what share of errors a musical prior could fix.
Goal (user): isolate and transcribe exactly what the piano plays. 100% is the
direction, not a reachable number; every step is judged on note F1 on the
real-upload set.

## Requirements
1. A REAL-UPLOAD test set in three tiers, each reported separately:
   T1 gold: real home recordings with exact MIDI -- user's digital piano
      (MIDI out) + phone mic at the same time. NEEDS USER (keyboard?).
   T2 real room, exact MIDI, public: MAPS ENSTDkAm/Cl (have), SMD
      (Saarland Music Data, Disklavier, CC BY-NC-SA 3.0 -- NC is OK here).
   T3 simulated upload chains on MAESTRO test audio: phone-mic EQ + room +
      noise floor, YouTube/TikTok-like lossy codec + loudness limiting,
      detune. Labelled SIMULATED; uses/extends `evaluation/augment.py`.
      (no ffmpeg on PATH: codec step needs imageio-ffmpeg or similar).
   Real YouTube/TikTok clips have no exact answer -> listening only, never a metric.
2. Error taxonomy over T1-T3 for 16b, every error in exactly one class:
   extra: harmonic (octave/12th/2 octaves above a played note), re-onset of
   a held note (split/duplicate), short fragment, out-of-key isolated, other;
   missed: by velocity band, masked by a louder simultaneous note, register;
   timing: right note, onset off by > 50 ms.
   Builds on `tools/recall_diagnosis.py` / `tools/precision_review.py`.

## Success criteria (this phase decides the next one)
- T2 + T3 numbers for 16b (and ByteDance as yardstick only), per tier.
- Taxonomy shares sum to 100% of errors per tier.
- Decision written now:
  * "musically implausible" classes (out-of-key isolated, duplicates,
    fragments) >= 30% of errors -> next phase is a note-level MUSICAL PRIOR
    (our own small Transformer over note sequences, trained on lots of MIDI,
    rescoring the acoustic model's candidate notes).
  * harmonics/quiet/masked dominate -> acoustic: train with upload-chain
    augmentation (recipe change, not a 4th continuation of 16b).
  * both large -> report, recommend, user picks.

## Files
- new `evaluation/upload_chain.py` (T3 degradations) + test
- new `evaluation/smd.py` (fetch/pair SMD, like `evaluation/maps.py`) + test
- new `tools/error_taxonomy.py` + test
- `benchmarks/wild-*.json` results
- NO change to the transcriber, the host, or the app.

## Not doing
No training, no deploy, no notation work, no separation/arrangement.

## Steps
- [x] 1. SMD fetch (`evaluation/smd.py`, 50 pieces, 282.4 min, 151,206 notes;
      range reads now retried in `maps._RemoteFile`). First piece, clean:
      onset F1 0.9965 -- studio Disklavier is EASY; real-room != hard.
- [~] 2. Upload presets youtube/phone/phone_far/tiktok (`evaluation/upload_chain.py`,
      codec delay removed, bitrates measured). Running on 3-min excerpts of
      maestro_test12 (`tools/cut_excerpts.py` -> recordings/maestro_test12_3min)
      via `tools/wild_benchmark.py` (caches notes in var/wild-estimates).
      CPU is ~2.8x slower than real time: ~8 h for 12 x 3 min x 5 presets.
- [x] 3. Error taxonomy tool (`tools/error_taxonomy.py`, tested).
- [~] 4. Run taxonomy, report + apply the decision rule.
      CLEAN maestro_test12_3min (22,757 played, 2,218 errors,
      `benchmarks/wild/taxonomy-maestro3min-clean.json`): MISSES are 92%:
      masked-in-chord 45.4%, other 20.7%, repeat 14.7%, quiet 11.6%; extras
      harmonic 4.0%, other 1.7%; timing 1.0%; musically implausible 0.9%.
      RESULTS 2026-10-17 (GPU host, `benchmarks/wild/`), mean onset F1:
        maestro 3-min: clean 0.9552 | youtube 0.9555 | phone 0.9382 |
                       tiktok 0.9212 | phone_far 0.8830;  SMD (50) clean 0.9539
      Taxonomy: musically implausible <= 2.2% everywhere (rule bar 30%) ->
      ACOUSTIC. Misses dominate (masked chord notes 34-45%); harmonic extras
      4% clean -> 12.6% tiktok; far mic costs most (-7.2 pt).
      (A) onset-threshold sweep, maestro 3-min (CHOSEN ON these excerpts):
        clean   0.6 0.9552 | 0.5 0.9598 | 0.4 0.9620 | 0.3 0.9629
        tiktok  0.6 0.9212 | 0.5 0.9257 | 0.4 0.9256 | 0.3 0.9233
        far     0.6 0.8830 | 0.5 0.8912 | 0.4 0.8925 | 0.3 0.8882
      0.4 best balance. NOT shipped: validating on SMD (50) and MAPS paired
      (0.6 was chosen on MAPS in Phase 22) before changing
      config.PTIFY_ONSET_THRESHOLD. Ship only if SMD and MAPS do not fall.
      VALIDATION -> NOT SHIPPED (rule: neither may fall):
        SMD (50)     0.6 0.9539 -> 0.4 0.9596  (better 46, worse 2)
        MAPS paired  0.6 0.8503 -> 0.4 0.8445  (better 2, worse 12; P 0.875->0.851)
      (0.8503 reproduces the committed 0.8502 through this harness.) MAPS
      pianos already make the model invent notes, so a lower global threshold
      adds more. No single threshold fits both -> the fix is the model (B).
      Untried cheap idea: a per-recording threshold from the onset-probability
      distribution. (B) PLAN a retrain with upload-chain
      augmentation targeting masked chord notes + harmonics (recipe change).
      2026-10-17: PC crash left 4 stray spaces at line 1 of
      evaluation/estimate_cache.py (IndentationError); removed, diff now empty.
- [-] T1 impossible: user has no MIDI keyboard (2026-10-16).

FINDING 2026-10-17 (not acted on): `get_engine("remote")` builds
`RemoteEngine()` with ByteDance thresholds (frame 0.05 / onset 0.7) while the
host serves ptify (wants 0.01 / 0.6) and applies client thresholds verbatim.
So the web app's remote option and `score_benchmark --engine remote` decode
ptify with the wrong thresholds. But it is NOT simply worse
(`var/threshold-probe.log`, local CPU, 2 MAESTRO test tracks):
  Scarlatti K525  ours: onset 0.9572 offset 0.4903 | remote-default: onset 0.9508 offset 0.6935
  Chopin 10/12    ours: onset 0.9688 offset 0.2662 | remote-default: onset 0.9640 offset 0.4432
Onset -0.5 pt, offsets +17-20 pt (frame 0.05 ends notes better here). Choose
thresholds on the real-upload set, then make remote pass them explicitly.

---

# Phase 37 -- durations: diagnosis DONE 2026-10-16, then PARKED (user: transcription first)

RESULT (ASAP val, run 8, `var/duration-breakdown.log`, 1560 tests pass):
  head 38.04% | head+B 38.74% (= val column, check passed) | oracle+B 67.53% | ceiling 92.40%
Decision rule -> oracle+B - head+B = +28.8 points: the DURATION HEAD is the
bottleneck, not single-voice engraving (which costs only 7.6 points even on
the written score). The rest of the oracle's gap to the ceiling is our onsets
/ staves. When resumed: retrain or re-decode the duration head.
PARKED because the user wants transcription accuracy first (2026-10-16).

(original plan below)

## Why
Durations are the weakest per-note column (test 58.7%, perfect MIDI 59.5% --
so not transcription). Two post-rules failed (legato fill, Phase 30-31 rules).
The printed value passes through TWO stages, and the column mixes them:
  A. the duration head: argmax over DURATION_TICKS (`rhythm/infer.py:95`)
  B. engraving: `_trim_staff_overlaps` cuts every note at the next onset on
     its staff and gives each chord its longest member's value
     (`notation/score.py:300-330`) -- ONE voice per staff. A whole note held
     under moving eighths cannot survive B, however right A is.
Retraining A is wasted if B caps the column. Measure first.

## Requirements
- A per-stage breakdown on ASAP VALIDATION (not the test set): head argmax
  vs written value, after B, and an ORACLE (true written values pushed
  through B on our own onsets) = the ceiling B allows.
- Same note linking and DURATION_TOL as the benchmark's `duration_accuracy`.

## Success criteria (for the diagnosis; it decides the next step)
- Numbers for: head-only, head+B (should reproduce the val durations
  column), oracle+B, and oracle+B with TRUE onsets.
- Decision rule, written now so it cannot drift:
  * oracle+B - head+B >= 10 points  -> the HEAD is worth fixing: plan a
    retrain / decode change (e.g. decode to "until next onset in this voice"
    classes, or expected-value decode), judged on val then test once.
  * oracle+B <= head+B + 5          -> B is the ceiling: the fix is engraving
    (a second voice for held notes), which changes what the page LOOKS like
    -- your decision, I will bring options with before/after renders.
  * between                         -> report both, recommend, you choose.

## Files
- new `tools/duration_breakdown.py` (reuses `tools.score_validation`
  `val_rows`/`run_val_piece` path and `evaluation.score_benchmark` linking).
- new `tests/test_duration_breakdown.py` (oracle path on a hand-made piece:
  held bass under moving treble loses exactly the held notes under B).
- NO change to `notation/`, `rhythm/` or the shipped app in this step.

## Not doing
- No retraining, no engraving change, no test-set run until the diagnosis is
  read and the next step agreed.

## Verification
- head+B number matches `tools.score_validation` durations to within 0.1.
- full pytest suite still passes.

## Steps
- [ ] 1. Write the test, then `tools/duration_breakdown.py`.
- [ ] 2. Run on ASAP val (run 8, web-app flags); log to var/duration-breakdown.log.
- [ ] 3. Report the four numbers + per-piece worst cases; apply the decision rule.

---

# RESUME HERE (stopped 2026-10-14, night)

State: all 1556 tests pass; app behaviour = the shipped bar fix (bar right
15/38) and nothing else new. Two changes are BUILT BUT OFF:

- `notation.analysis.LOCAL_SPELLING = False`. Local-key spelling FAILED the
  test set: 89.38% -> 89.21% (worse 20, better 16; Bach F#/G# pieces -20 to
  -24 from enharmonic twins). Fix written: `nearest_enharmonic` (test pinned).
  Validation WITH the fix (var/spelling-val3.log): 84.83% -> 86.07% (8 beats;
  4: 86.02, 16: 85.87, 32: 85.63) -- smaller than the 87.4% without it, so
  part of the earlier gain was the same enharmonic flips.
  RESULT 2026-10-16 -- FAILED AGAIN, stays off: test set spelling 89.38% ->
  88.90% audio (perfect MIDI 89.11 -> 88.50); every other column identical,
  key 32/38 both. Better on 25 pieces, worse on 10, but the losses are large
  on long chromatic pieces (Chopin Polonaise 53 -8.3, Wanderer -5.7, Liszt
  S145/2 -5.3) and outweigh gains (Liszt TE3 +8.4, Beethoven 16-2 +6.1).
  `benchmarks/score-benchmark-rhythm-run8-localkey2.json`, compare with
  `var/cmp_bench.py`. Untried idea: switch key only where the local key is
  confidently different from the global one. Command used (flag set at runtime):
      python -m tools.score_benchmark --engine remote --served ptify         --rhythm-model var/rhythm-models/run8.pt --rhythm-choose-grid         --rhythm-try-tempi --rhythm-halve-fast-eighths         --json benchmarks/score-benchmark-rhythm-run8-localkey2.json
      baseline: benchmarks/score-benchmark-rhythm-run8-spelling.json (89.38%)
  (3) keep only if spelling rises and no other column falls; else revert.
- `notation.score.LEGATO_FILL = False`: TRIED AND FAILED (keep off).
  Diagnosis said 37% of values too short, 95% of those written legato, but
  filling to the next onset on the staff does not hit the written value:
  validation durations off 38.74%, theta 0.75 38.68%, 0.6 38.85%, 0.5 36.98%,
  0.35 36.82% (`var/legato-sweep.log`). Next idea for durations, untried:
  the duration head itself (retrain targets / decode), not a post-rule.

Waiting on the USER: Songscription uploads (item 3 below).
New tools today: `tools/score_validation.py` (full pipeline on ASAP val,
per-note columns comparable, beat columns not), `tools/spelling_validation.py`,
`tools/beat_level_validation.py`, `tools/fit_beat_chooser.py`,
`tools/bar_validation.py`. `_fetch_asap` now reads `var/asap-repo` first
(works offline).

---

# Toward Songscription: the five-item plan (agreed 2026-10-13, "do all")

1. [x] **Spelling + key signature column** (`evaluation/spelling.py`).
   Test set (web app): spelling 89.4% on 90% of linked notes (the rest are 2
   pieces ASAP gives no bar map), key signature 32/38. Pieces with the key
   right: 92.6%; key wrong: 81.0%. (Bug found on the way: ASAP's bar map is a
   0-based measure POSITION, not the printed number.)
   FOLLOW-UP [~]: spell by the LOCAL key (`analysis.spelling_keys`, 8-beat
   window; signature stays global). ASAP validation (`tools/spelling_validation.py`,
   20 scores): 84.8% -> 87.4% (4 beats 87.8, 16 87.1, 32 86.3); better on 14,
   worse on 3 (worst -2.0). 1548 tests pass. Test-set run in progress
   (`score-benchmark-rhythm-run8-localkey.json`).
2. [x] **Beat level** (19/38): PARKED. ASAP val (`tools/beat_level_validation.py`):
   most-confident candidate right 29/89, some candidate right 64/89, but no
   chooser finds it: tempo prior / most-common-gap prior (train split) best
   30/89; bar strength <= 28/89; logistic regression on all those features
   reaches only 31/89 even IN-SAMPLE (`tools/fit_beat_chooser.py`), so the
   notes do not carry the signal. Agrees with run 7 (a learned written-scale
   head failed on val). ASAP's beat is a notational convention.
3. [ ] **Songscription row**. USER uploads `var/songscription-upload/*.wav`
   (7 benchmark segments, 14.0 min, inside the free 15 min/month) and exports
   MusicXML to `var/songscription-out/` with the same file names.
   Their XML has no performance timing, so build a SCORE-TO-SCORE path:
   align their written notes to ASAP's score by note order + pitch, and score
   rhythm (gaps, global factor), durations, spelling, key, bar, staff. Score
   PTify through the SAME path for a fair row. Caveat (user accepted):
   MAESTRO is non-commercial and this uploads it to a third party.
4. [ ] **Model**: notes reaching the score (84% audio vs 92% perfect MIDI, quiet
   notes, Phase 27); durations 58.7%; staff 89.1%.
5. [ ] **Product**: notation editor (largest user-facing gap). YouTube input
   needs a terms-of-service decision from the user. More instruments is a
   separate project.

---

# Meter (bar) -- the weakest column (found 2026-10-12)

End-to-end check through `api.pipeline.run` (saved Modal transcriptions):
Toccata printed 1/4 (fixed: `_written_bar`), Rachmaninoff 23/6 (4/4) printed
3/4. New benchmark column `bar_ok`: web app 10/38, old rules 11/38 (5 both, 5
only model, 6 only rules). Old rules re-scored under the new FACTORS: rhythm
55.5%, durations 42.2% (not 58.3 / 43.5). Next candidate: the bar length.
The model's bar comes from `_bar` (downbeat head > 0.5, most common spacing);
soft decoding failed on downbeat F before -- measure on `bar_ok` this time,
and choose on validation (ASAP val has time signatures in val.jsonl; PDMX
held-out has them too).

DONE 2026-10-13: bar = (length, phase) whose downbeats stand out most (Welch's t).
Validation 9 -> 25/89; test (web app) bar right 10 -> 15/38, downbeat F 0.218 ->
0.233, nothing else moved, no piece worse. PDMX held-out NOT used: its shards are
encoded model inputs, not performances (needs its own decode path); the ASAP
result was clear-cut. Remaining bar errors are mostly BEAT LEVEL: on validation
the true bar is 8 or 12 tracked beats in 25/89 (the tracker counts eighths).
Next candidate: beat level (19/38 right).

---

# Phase 36 -- PDMX MusicXML: written note values and hands (DONE 2026-10-12: criteria NOT met, run 8 stays)

## Why

After Phase 35 the weakest columns are durations (58.7%) and staff (89.1%,
unchanged all along). PDMX's MIDI could label neither: no staves, and note ends
are PLAYED lengths (0.6% on a value grid). Even unlabelled, the synthetic data
lifted ASAP-validation durations 50.6 -> 57.7%. PDMX's MusicXML has both written
values and staves for the same scores.

## Requirements

- Synthetic pairs whose targets include WRITTEN note values (ties merged) and
  the STAFF of every note, from the same 10,000 selected scores and the same
  train/held-out split (by the score's path).
- Everything Phase 35 labels (positions, bar lines, pickups, meter) still
  labelled, now from the score itself (measures, not MIDI bar-length events).
- Grace notes and other zero-length notes get no position/value target.
- Performance simulation unchanged (`rhythm.synth`); repeats NOT unfolded.

## Success criteria

- ASAP validation (real playing), run 9 vs run 8: durations above 57.7%, hand
  above 83.7%, gaps not more than 1 point below 83.5%.
- Score benchmark, web-app settings: durations above 58.7% and staff above
  89.1%, rhythm not below 76.9%, beat level not below 19/38.
- If both hold, run 9 becomes the default.

## Files

- new `evaluation/pdmx_mxl.py`: MusicXML -> `WrittenScore` with
  `known_length=True` and a staff per note (music21: `stripTies`, PartStaff).
- `rhythm/synth.py`: `WrittenNote.staff` -> target "upper"/"lower".
- `tools/export_pdmx_pairs.py`: `--source mxl` (default stays `mid`).
- Tests first, on hand-built MusicXML (ties, two staves, pickup, grace note).

## Decisions for the USER

1. Download `mxl.tar.gz` (1.9 GB) into `var/pdmx/` (gitignored).
2. Parser: music21 (installed). Step 1 measures its speed on 50 files; if the
   10,000-score export would take more than ~4 h with 4 workers, I stop and ask
   before adding another dependency.

## Not doing

- No change to decoding or defaults until run 9 is benchmarked.
- No dynamics/articulation labels (the model has no head for them).

## Steps

- [x] 1. mxl.tar.gz downloaded (md5 49ffd75e... matches Zenodo), extracted to
      var/pdmx/extract/mxl. 50/50 parse; 37/50 two staves; 96.1% of notes with
      a written length; 0.56 s/score.
- [x] 2. `evaluation/pdmx_mxl.py` (5 tests); bar logic shared with the MIDI
      reader (`pdmx.beats_from_bars`; usual bar = most common length).
- [~] 3. synth staff + `--source mxl` done (tested). Export to
      var/pdmx-mxl-pairs was STOPPED by Claude Code's low-memory reaper at
      7,500/10,000: 12 complete train shards (6,000 pieces, 3.0M notes) + 1
      held-out shard, all range-checked. User is restarting Claude Code with
      CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1. To resume (full re-export,
      ~1 h; it overwrites the partial shards):
          rm var/pdmx-mxl-pairs/pdmx-*.npz
          python -m tools.export_pdmx_pairs --source mxl --out var/pdmx-mxl-pairs --workers 3
      then run 9:
          python -m tools.kaggle_rhythm --extra-dir var/pdmx-mxl-pairs prepare --             --d-model 192 --layers 3 --dropout 0.2 --weight-decay 0.05 --epochs 15             --downbeat-pos-weight 4 --extra-glob "pdmx-train-*.npz"
      Note: the local uvicorn server (api.app) predates today's changes; restart
      it to serve run 8 + tempo candidates + halving + start phase.
- [x] 4. Run 9 = run 8 settings + 6,000 MusicXML pieces (the partial export;
      written values + hands labelled). ASAP val vs run 8: durations 57.7 ->
      57.6% (FAIL: must beat 57.7), hand 83.7 -> 84.0%, gaps 83.5 -> 83.4%,
      downbeat F1 0.388 -> 0.402. Not benchmarked; run 8 stays the default.
      Likely why: `rhythm.synth` holds every note for one random share of its
      written value per piece, so "played length -> written value" in synthetic
      data is a mapping real playing (phrasing, pedal) does not follow. Run 8's
      duration gain (50.6 -> 57.7%) came without length labels at all.
      Next idea if revisited: per-note articulation in synth (staccato/legato
      by context, pedal-extended offsets) before more labels.
- [ ] 5. Not run (criteria failed at step 4).

---

# Beat placement (2026-10-12): start phase, ON

Within the right beat level, beat F was comparable to the pipeline (0.520 vs
0.533) except for collapses (Schubert 664-1 0.06). Cause: positions started at
0 while the Viterbi's gaps assumed its own start phase. Fix: decode from the
returned start phase, with START_PENALTY 4.0 (validation 28.1 -> 50.1%).
Benchmark: beat F 0.418 -> 0.495 (pipeline 0.489), rhythm/downbeat unchanged,
durations 58.8 -> 58.7. Open: Bach fugue 891 0.98 -> 0.01 (tracker on its
off-beats; the old bug cancelled it).

---

# Printed level: halve fast 8ths (2026-10-12): ON in the web app

Held-out: a fixed "most common gap -> 16th" rule helps ASAP (12.6 -> 19.0%) but
hurts PDMX (47.4 -> 32.0%): 8th-mode readings are 16ths-at-double-speed in
ASAP (median 191 bpm) but real 8ths in PDMX (111 bpm). Tempo separates them:
halve only above 160 bpm -> 19.1% / 48.7% (both up). Benchmark first showed
rhythm 76.9 -> 71.1% -- all from three compound pieces whose factor became 4/3,
missing from FACTORS. User added 3/4 and 4/3 (dotted relationship in every
octave); re-scored: rhythm 76.9 / durations 58.8 unchanged, beat level 10 ->
19/38, downbeat F 0.226 -> 0.215. Enabled (`rhythm_halve_fast_eighths=True`).

---

# Tempo candidates (2026-10-12): ON in the web app

Grid choice now also tries 6 multiples of the note tracker's tempo
(`rhythm.beats.track_beat_candidates`, `rhythm_quantise(try_tempi=True)`).
Held-out first: exact positions 33.8 -> 34.5% (ASAP val), 66.0 -> 67.6% (PDMX).
Benchmark (run 8): rhythm 67.1 -> 76.9%, durations 49.5 -> 58.8%, Toccata
0 -> 95%; beat level 17 -> 10/38, beat F 0.461 -> 0.429. ~+2 s per 8-min job.

Next: beat level. The readings are right up to x2 / x1/2, so the open problem
is only WHICH power-of-2 level to print. A cheap candidate: after choosing the
reading, pick the dyadic level (x1/2, x1, x2) whose most common written value
is a 16th or 8th -- the `written_scale_from_gaps` idea restricted to powers of
2 (validation earlier: 33.9% exact scale, vs 22.8% unscaled), which cannot
create the 4/3 meter failures that dotted scales did.

---

# Meter cue on PDMX held-out (2026-10-12): FAILED, rescale stays off

Run 8 on the 965 held-out PDMX pieces (synthetic performances, known time
signatures): the rescale rule picks a dotted scale on 94; 45 of those (48%) are
really compound. Cue = best integer-bar contrast of the model's downbeat
log-odds on whole tracked beats. AUC 0.56; best threshold 53% vs 48% for
"always compound". Why: "not compound" mixes TWO cases -- Toccata-like (tracker
on 3-sixteenth groups; downbeats NOT periodic in tracked beats; fix = rescale
to quarters) and simple-meter pieces full of triplet figures (tracked beat IS
the beat; downbeats periodic; fix = keep triplets, no rescale). Gaps and
downbeats make the triplet case look like compound. Not pursued further.
Script: job scratch `heldout_cue.py` (not in the repo).

---

# Phase 35 -- synthetic performances from PDMX (DONE 2026-10-12: run 8 is the default)

## Why

Two rhythm problems need data ASAP cannot give: (1) the Schumann Toccata's even
16th runs read as triplets (0% rhythm), and (2) whether a dotted pulse means
compound meter -- ASAP validation has only 4 scores with a dotted pick, too few
to choose a cue. PDMX (250K public-domain MuseScore scores, Zenodo v9) supplies
thousands of piano scores with known rhythms and time signatures. User chose it
2026-10-12 (non-commercial project; NC data acceptable).

## Requirements

- Training pairs in the existing format (`evaluation.rhythm_pairs` /
  `rhythm.features.encode_pair`): notes with onset/offset/velocity and written
  targets (start/length in quarters and in beats, downbeats, time signatures).
- Performances simulate what the model sees at inference: tempo drift, timing
  jitter, chord spread, velocity accents, varied articulation; transcription
  misses are already simulated by `train.augment` (dropped notes).
- Hand/staff unlabelled for synthetic pairs (MIDI loses staves): the hand head
  trains on ASAP only.
- PDMX scores split train / held-out BY SCORE; nothing PDMX touches the score
  benchmark (real recordings) or ASAP validation, which stay the judges.

## Success criteria

- Run 8 (real + synthetic) on the score benchmark, from audio, same decoding as
  the default: rhythm >= 64.3% (run 6) AND durations >= 47.7%, staff unchanged.
- Schumann Toccata rhythm > 0% (both performances).
- ASAP validation gaps/durations not more than 1 point below run 6
  (79.7% / 50.1%): synthetic data must not pull the model away from real playing.
- PDMX held-out gives >= 200 pieces with a dotted pick, enough to choose a
  compound-meter cue (Phase 34 follow-up) -- reported, not required to pass.

## Files

- new `evaluation/pdmx.py`: CSV filter + MIDI -> written notes / beats /
  downbeats / time signatures (pretty_midi; quarter = MIDI beat).
- new `rhythm/synth.py`: score -> synthetic performance (seeded).
- new `tools/export_pdmx_pairs.py`: download check, filter, split, write
  `var/pdmx-pairs/{train,heldout}.jsonl` + `.npz` (same packer).
- `rhythm/features.py`: `staff` may be None -> hand IGNORE.
- `rhythm/train.py`: `--extra-data DIR` (more train.npz to concatenate).
- `tools/kaggle_rhythm.py`: ship the extra npz in the private dataset.
- Tests for each new module first (hand-built MIDI with known answers).

## Decisions for the USER (recommendations first)

1. Download: `PDMX.csv` (225 MB) + `mid.tar.gz` (214 MB) into `var/pdmx/`
   (gitignored), NOT the 1.9 GB MusicXML. MuseScore's MIDI keeps exact written
   positions and time signatures; staves are lost (hand head unaffected).
2. Filter: one part whose track name is piano, `subset:deduplicated` AND
   `subset:no_license_conflict`, 200-20,000 notes, a time signature present;
   cap 10,000 scores (seeded sample). About 4x ASAP's notes; Kaggle T4 run
   ~2 h instead of ~30 min.
3. Mix: all of ASAP train + all synthetic in one training set; model selection
   stays on ASAP validation (real performances).

## Not doing

- No MusicXML parsing, no audio rendering (the model reads notes, not audio).
- No change to decoding or defaults until run 8 is benchmarked.
- No ornament/staccato work (the old PDMX deferral reason still applies there).

## Verification

Unit tests on hand-built MIDI; a sample of converted pairs spot-checked
against their MIDI; one Kaggle run; one score-benchmark run. Level: replayed
against real recordings and published scores.

## Steps

- [x] 1. Downloaded PDMX.csv + mid.tar.gz (md5 match Zenodo); filter: 254,077
      -> deduplicated 102,635 -> no licence conflict 77,354 -> piano-only 1-2
      tracks 39,066 -> 200-20,000 notes 21,438; seeded sample 10,000.
- [x] 2. `evaluation/pdmx.py` (7 tests). MEASURED: MuseScore writes each bar's
      actual length as a time signature (pickups "1/4" then "4/4"); note ends
      are PLAYED lengths (0.6% on a value grid) -> lengths never claimed;
      97.8% of starts on a 16th/triplet grid -> those are claimed.
- [x] 3. `rhythm/synth.py` (10 tests). Fixed: staff None was labelled "lower".
- [x] 4. Export (`tools/export_pdmx_pairs.py`, sharded, 3 workers): train
      8,884 pieces / 5.51M notes, held-out 965 / 0.63M; 151 failed (145 not
      piano in the file). Labels vs ASAP: 97% vs 94% labelled, downbeat share
      0.20 vs 0.16, simpler gaps (more 8ths/quarters). `--extra-glob` in train;
      Kaggle prepare ships shards only when asked.
- [x] 5. Run 8 (first attempt crashed: 269 notes outside the 88 keys; fixed with
      `read_score` filtering + `train.check_ranges`). ASAP val: gaps 80.1 ->
      83.5%, durations 50.6 -> 57.7%, downbeat F1 0.397 -> 0.388, hand 84.0 ->
      83.7%. Passed.
- [x] 6. Benchmark: rhythm 67.1% (>= 64.3 PASS), durations 49.5% (>= 47.7
      PASS), Toccata 0% (FAIL). 23 better / 7 worse. USER chose to switch:
      run 8 is the default (`checkpoints/rhythm-run8.pt`).

---

# Phase 34 -- compound meter (BUILT 2026-10-12; criteria NOT met, rescale stays off)

## Why

Decode-time rescaling (`--rhythm-rescale`) gets the beat level right in 21/38
pieces (pipeline 19, current default 16), lifts durations 47.7 -> 49.4% and fixes
the Schumann Toccata (0% -> 68/75%). It is held back by four compound pieces
(Bach fugue 873, preludes 858/873, Liszt TE 3, Beethoven 18-4): their beat is a
DOTTED value, rescaling writes the right note values but on a QUARTER beat, so
meter, bar lines and beaming are wrong (benchmark factor 4/3 -> rhythm ~0%).

## Requirements

- When the decoded scale is dotted (0.75 or 1.5 quarters per tracked beat), the
  score keeps the beat on the dotted value: compound time signature, bars of
  whole dotted beats, tempo mark referring to the dotted beat.
- Simple-meter output is unchanged (identical notes, bars, signature, tempo).
- Positions and note values stay in quarter notes (music21's unit) everywhere.

## Success criteria (benchmark, from audio, run 6 + on-beat + grid choice + rescale)

- The four collapsed pieces recover to within 2 points of their unscaled rhythm.
- Overall rhythm >= 64.3% (current default) and beat level >= 21/38.
- Every simple-meter piece's row identical to `run6-rescale`.
- A 12/16 and a 6/8 case round-trip through MusicXML with that signature.
- If all hold, rescaling becomes the web app default.

## Files touched

- `notation/quantise.py`: `BeatGrid.beat_quarters: float = 1.0` (quarters per
  beat). Default keeps every existing grid as it is.
- `rhythm/infer.py`: `_written_scale` -- for a dotted scale, keep the tracked
  beat times, set `beat_quarters = scale`, keep the bar in beats.
- `notation/score.py`: time signature from (beats per bar, beat_quarters);
  MetronomeMark with a dotted referent; chord-symbol bar index from bar length
  in quarters; `ScoreStats.time_signature`.
- `evaluation/score_benchmark.py` (+ `tools/score_benchmark.py`): our positions
  converted quarters -> our beats (divide by beat_quarters) before the factor,
  since the factor is score beats per OUR beat.
- `api/pipeline.py`: `rhythm_rescale=True` only if the criteria hold.
- Tests in `tests/test_quantise*.py`, `tests/test_notation.py`,
  `tests/test_rhythm_infer.py`, `tests/test_score_benchmark.py`.

## Decisions for the USER

1. Dotted-8th beats (scale 0.75, e.g. the Toccata-like 3-sixteenth groups):
   print as x/16 (12/16, 6/16) with 16ths, or double everything to x/8 with
   8ths? (Recommend x/16: it keeps the note values the rule chose.)
2. Tempo mark for compound time: dotted-beat referent ("dotted quarter = 80"),
   as published scores do? (Recommend yes.)

## Not doing

- No compound detection for the rule pipeline (`grid` mode) or the CLI.
- No new meters beyond what the scale gives (no 5/8, 7/8, no 3/4-vs-6/8
  disambiguation by accent).
- No retraining.

## Verification

Unit tests first; then one full benchmark with rescale on; MusicXML round-trip
of two compound cases. Level: replayed against real recordings and scores.

## Steps

- [x] 1. BeatGrid.beat_quarters (default 1.0) + compound `_written_scale`.
- [x] 2. score.py: `time_signature_for`, dotted tempo mark, chord bar index,
      ScoreStats.beats_per_bar/beat_quarters; MusicXML round-trip of 6/8, 12/16.
      Also fixed: MIDI export and staccato used sec-per-BEAT on quarter-note
      positions (would be 1.5x slow in 6/8) -> `BeatGrid.sec_per_quarter`.
- [x] 3. Benchmark: positions / beat_quarters before the factor; tool takes
      beats per bar from stats, not the signature numerator.
- [x] 4. 1484 tests pass. Benchmark `run6-compound`: rhythm 62.2% (< 64.3%),
      durations 46.1%, beat level 21/38. Bach fugue 873 0 -> 96.7%, Beethoven
      18-4 0 -> 85.6%, but Schumann Toccata 68/75% -> 0% (2/4; the dotted pulse
      is the tracker's) and preludes 858/873, Liszt TE 3 never go compound.
- [ ] 5. NOT DONE: criteria fail, rescale stays off.

## Results / next

A dotted pulse is compound meter in some pieces and a tracker artefact in
others; gaps alone cannot tell. Next would need a separate cue (e.g. do
predicted downbeats / accents recur every 3 tracked sub-beats?), chosen on
validation.

---

# Phase 33 -- rhythm on the page: a baseline, then our own model

## Why

Phase 32: the page is wrong from PERFECT notes as often as from audio
(durations 39.6%, rhythm 52.9%, beat level right in 19/38). The fixed 0.25-beat
grid cannot write triplets or compound meter (0-4% rhythm on those pieces).
All three gaps (division, beat level, note values) are one problem: turning
performed timing into written rhythm. The user wants a home-built model; this
is where one can move the page. The rule-based baseline comes first, so the
model has a number to beat.

## Requirements

- Part A (baseline): choose the beat division per piece, four or three per
  beat, from the notes' own timing. Only what exists at inference.
- Part B (data): training pairs from ASAP performances that are NOT among the
  39 benchmark scores, split BY SCORE. Eval-only data never enters training.
- Part C (model): a small Transformer, notes + tracked beats in, written
  rhythm out. Design reviewed by the user BEFORE any GPU spend.

## Success criteria

- A: on `tools/score_benchmark.py`, rhythm improves on the 5 triple-division
  pieces, and pooled rhythm and durations do not drop on the 33 others (each
  within 0.5 points). Otherwise it is not wired in.
- B: pair counts reported. Zero overlap with benchmark scores, asserted in a test.
- C: beats the better of baseline and current on pooled rhythm AND durations.

## Steps

- [x] A1. Tests first: `estimate_subdivision` on hand-built duple, triplet and
      6/8 note sets, plus too few notes (keeps 0.25).
- [x] A2. Measured: rhythm 52.9% -> 58.3%, durations 39.6% -> 43.5% (audio);
      3 pieces improved (bwv_873 0.0->95.6%), 35 identical, 0 worse. Fixed run
      reproduced Phase 32 exactly (determinism).
- [x] A3. Wired into `transcription_to_score(division="auto")`; `ScoreStats.subdivision`
      reports it. Wired run == measured run on all 76 rows. Full suite 1390 passed.
- [x] B1. `evaluation/rhythm_pairs.py` + `tests/test_rhythm_pairs.py` (5 tests;
      written with the code, not RED-first). Counts from ASAP metadata:
      held out 39 scores / 128 perfs (every perf of a benchmark score);
      **train 848 perfs / 176 scores, val 91 perfs / 20 scores**. Before
      beat-count refusals (ASAP: 29 perfs skip beats).
- [x] C1. `docs/phase33-rhythm-model.md`. **STOPPED for user review** (GPU spend,
      licence).

## Files (expected)

- `notation/quantise.py`, `tests/test_quantise.py` (A)
- `tools/score_benchmark.py` (a `--subdivision` switch for before/after)
- new `evaluation/rhythm_pairs.py` + test (B)

## Not doing

- No GPU training before C1 is reviewed.
- No meter/downbeat work in A (that's the model's job, or a later rule).
- No third-party engines.

## Decisions (2026-10-10)

- PTify is non-commercial / personal: training on ASAP (CC BY-NC-SA) is fine.
- GPU: Kaggle, driven from the shell (`tools/kaggle_rhythm.py`).

## C progress

- [x] C2. ASAP cloned to `var/asap-repo` (commit afc815c, 460 MB, gitignored).
      `tools/export_rhythm_pairs.py`: train 785 perfs / 2.50M notes (94.5%
      labelled), val 89 / 287k; 65 skipped with reasons (39 beat counts, 18
      staff count, 18 out-of-range pitch). Packed: train.npz 41 MB, val.npz 5 MB.
- [x] C3. `rhythm/` (features, model, train) + 19 tests incl. a tiny-overfit
      check. Local CPU smoke run on real data OK (6,636 train windows).
- [~] C4. Private dataset `imse4n/ptify-rhythm-pairs`; kernel `imse4n/ptify-rhythm-train`.
      **Run 1 (T4, 23 min, 30 epochs): did not generalise.** Val vs
      always-majority: gap 60.9% vs 49.9%, value 24.0% vs 21.3%, hand 83.7% vs
      56.5%, downbeat F1 0.23. Train loss kept falling (value CE 2.28 -> 0.87)
      while val stalled after epoch 1: memorising.
      **Cause (measured): targets were in ANNOTATED beats, a per-piece convention.**
      Quarters per annotated beat across 235 ASAP scores: 1.0 x159, 1.5 x41,
      2.0 x18, 0.5 x10, 0.75 x5, 3.0 x2. The same rhythm got contradictory
      labels. Fix: positions and values in QUARTER NOTES from the score MIDI's
      tick clock (`ScoreNote.start_quarter`/`length_quarters`, tests added);
      downbeats stay annotation-based. Val labels now: gap 6 ticks (16th) most
      common after chords, value 6 ticks 32.6%.
      Also fixed: the Kaggle CLI on Windows silently failed the big file in
      `datasets version` with a relative `-p` path; the helper now runs inside the
      folder and decides create vs version from the dataset list.
      **Run 2 (quarter targets): gap improved, values still at guessing.**
      Val gap 63.9% (majority 49.9%), value 31.9% (majority 32.6%). Best ckpt =
      epoch 0; later epochs memorised (train up, val down). Per-performance val
      value accuracy 0.04-0.72, median 0.20.
      Diagnosis: absolute-seconds inputs make the same written value look
      different at every tempo; 176 scores is too few to learn tempo-normalising.
      **Run 3 (pushed 00:00):** + relative timing inputs (gap / local pulse,
      chord spread excluded; tempo-invariance tested), smaller + more regularised
      (d192, 3 layers, dropout 0.2, wd 0.05, 20 epochs), train accuracy logged per
      epoch, final weights saved too.
      **Run 3 result: memorises.** Train gap 83.1% / value 63.8% vs val 63.0% /
      28.0%. Val real (non-chord) gaps only 34.6% (chords 97%). Benchmark
      (38 pieces, audio): rhythm 39.5%, durations 31.5% vs pipeline 58.3% /
      43.5%. NOT wired in.
      Re-plan: the rule pipeline wins by snapping to a beat grid, which the
      model was not given. Beat tracking from NOTES measured beat F 0.48 on val
      (audio 0.49), so training MIDI and transcriptions get the same input.
      **Run 4 (pushed 01:00):** + per-note beat inputs (`rhythm/beats.py`: phase
      sin/cos, log beat length, log gap in beats; tracked once per piece from
      clean notes, recomputed per window after drops/jitter). Same settings as
      run 3, so the only change is the beat grid. Export now takes 39 min
      (beat tracking 874 performances); worth caching if re-exported often.
      **Run 4 result: same ceiling** (val gap 63.0%, value 28.3%; train 83%/65%).
      Four different setups all plateau at val gap ~63% / value ~28-30%:
      a label problem, not a model one. **Confirmed:** written scale is a
      notational choice (2/4 vs 2/2 sound identical). Rescoring run 4 val with one
      scale factor per piece: real gaps 34.1% -> 58.5%, values 33.6% -> 48.9%.
      **Run 5 (pushed 01:21):** labels re-expressed per window in TRACKED-beat
      units (`features.grid_scale`: written quarters per tracked beat, snapped),
      so every piece shares the model-visible convention. Same settings.
      Caveat: oracle-scale gap accuracy (58.5%) only matches the rule pipeline's
      rhythm (58.3%); values (48.9%) look like the model's real edge. If run 5
      only ties on rhythm -> hybrid: rules place onsets, model supplies note
      values + downbeats. Option 2 (synthetic data) after that.
      **Run 5 RESULT (2026-10-10 01:49): it WORKS and BEATS the pipeline.**
      Val tracks train (gap 80.0/81.0%, value 50.8/60.0%); real gaps 64.4%
      (majority 37.6%), values 51.5% (majority 30.5%). Benchmark, 38 pieces,
      audio: rhythm 58.3% -> **63.3%**, durations 43.5% -> **47.0%**, staff
      89.1% = 89.1%; better on 28/38 pieces (rhythm and durations). BUT beat
      level 19 -> 16/38, beat F 0.49 -> 0.30, downbeat F 0.17 -> 0.14.
      Weights: `var/rhythm-models/run5.pt` (gitignored). NOT made the default.
- [x] C6. Benchmarked (above). Meets the stated bar (rhythm AND durations up,
      staff not lower).

## 2026-10-10 morning: hybrid tried, model made the default

- Hybrid (model gaps anchored to the audio grid, `rhythm/decode.anchor_positions`):
  rhythm 54.5%, durations 42.7% with either bar source: worse than both. Not
  used. The model's downbeats on the audio grid gave downbeat F 0.23 (best).
- **Default switched to the model** (`rhythm.weights.DEFAULT_MODE="model"`,
  `api/pipeline._rhythm_choice`). Falls back to rules without weights; a
  misconfigured env var is an `engine_unavailable` job error. Weights in
  `checkpoints/rhythm-run5.pt` (gitignored, sha256 pinned). `tests/conftest.py`
  pins grid for hermetic tests. Full suite: 1449 passed.
- Cost: 8-min piece 9.5 s -> 23.7 s (steady state); 21-min piece 90 s -> 276 s.
- The notation CLI (`notation/__main__.py`) still uses the rules; only the web app
  pipeline switched.

## 2026-10-10 late morning

- Speed: note beat tracking now uses scipy's FFT (scoped, restored): 36 s ->
  22.6 s on the 21-min piece, beats IDENTICAL on all 22 cached pieces checked
  (test pins it); benchmark rows identical to run 5. 8-min piece steady state
  23.7 s -> 21.6 s. Profile: model network 0.4 s, beat tracking ~8 s, music21
  engraving ~35 s (613k Fraction ops: triplet-type positions take music21's
  slow exact path). Engraving speed left for later.
- Bar lines: tried choosing (bar, offset) from SOFT downbeat evidence
  (log-odds contrast). REVERTED (2026-10-11): downbeat F 0.142 -> 0.136, worse
  on 21 pieces, better on 11; rhythm/durations/staff unchanged
  (`var/sb-model-softbars.json`). `_bar` is back to the >0.5 rule.
- Benchmark memory: `chain` matches notes one pitch at a time instead of
  mir_eval's all-notes x all-notes tables. Same pairs (test pins it; 10/10
  rows identical to `sb-wired` on 5 pieces).
- Model's losses: the Schumann Toccata (x2) 86-90% -> 0%. The note tracker locked
  onto 3-sixteenth groups, so the model wrote 16ths as triplets (really wrong
  notation, so the benchmark was NOT loosened). Other losses small (Chopin 53
  -14, Beethoven 16-1 -11, rest < 6). Biggest wins: Beethoven 3-2 13.5 -> 88.4%,
  Schubert 664-3 24.7 -> 89.9%, Rachmaninoff 23/6 24.2 -> 86.6%.
- `choose_grid`: read the piece on the note-tracked AND the audio grid, keep
  the more confident reading. Benchmark (2026-10-11): rhythm 63.3 -> 64.0%,
  durations 47.0 -> 47.5%, 6 better / 1 worse. ON in the web app when the grid
  is tracked from audio (`api/pipeline._build_score`, 2 tests). Did NOT fix the
  Toccata: confidence ties (0.922 vs 0.922) and the model scores 11% even
  anchored to the audio grid, so it misreads the 16th runs itself (data gap).

## 2026-10-11 evening

- On-the-beat decoder (`phase_consistent_deltas`): Viterbi over position in the
  beat, tuned on validation (36.1 -> 51.0% phase right). Fixed positions drifting
  by a triplet slip forever. Fraction-valued notes 2082 -> 108 on one piece.
- Run 6 (downbeat loss weight 4): val downbeat F1 0.18 -> 0.40.
- Benchmark, run 6 + on-beat + grid choice: rhythm 64.3%, durations 47.7%,
  downbeat F 0.207 (pipeline 0.175), beat F 0.43. NOW THE DEFAULT
  (`checkpoints/rhythm-run6.pt`).
- Engraving speed did NOT improve from fewer fractions (7.0 s either way).

- Memory/speed: note beat tracking computes librosa's tempo from a CHUNKED
  tempogram (`beats._tempo_in_chunks`): peak ~3 GB -> ~210 MB on 20+ minute
  pieces (likely cause of the low-memory kills), beats IDENTICAL on 37/37
  pieces checked against the training-time path (the 5 longest not compared:
  the old path needs the 3 GB). Wanderer Fantasy model path 60.9 -> 53.2 s.
- What is left of the cost is BEAT LEVEL: on the Wanderer the model writes 1375
  bars to the pipeline's 526 (beat ~2.6x too fast), and music21's time scales
  with bars (makeNotation 30.3 s vs 6.7 s). Cause: since run 5 labels are in
  TRACKED-beat units (`grid_scale`), so the model never learns the written
  scale. Benchmark factors: mostly 0.5 / 0.33 for both pipeline and model.

## Next

1. Beat level. TRIED (2026-10-11, run 7): a `scale` head predicting written
   quarters per tracked beat from the `grid_scale` label. Val scale accuracy
   peaked at 32.6% (always guessing 0.5 scores 30%) while train reached 79%:
   it memorises each piece's notation and does not generalise; gaps 79.7 ->
   79.0%, durations 50.1 -> 48.4% (best epoch). NOT benchmarked, NOT kept.
   Patch: `var/patches/2026-10-11-scale-head.patch`; log
   `benchmarks/training/rhythm-run7.jsonl`.
   THEN (2026-10-12) decode-time scale: most common gap -> 16th. Opt-in
   (`rescale=True`, `--rhythm-rescale`). Beat level 16 -> 21/38, durations
   47.7 -> 49.4%, Toccata fixed; but rhythm 64.3 -> 63.9%, downbeat F 0.207 ->
   0.177, four compound pieces collapse to ~0%. USER to weigh the trade-off.
   Open: a compound-aware rule that validation supports (the obvious 8th
   target for triple pulses scored LOWER on validation: 31.8% vs 41.4%).
   FOUND (2026-10-12): the four collapses all have raw factor 4/3 (1.32-1.34),
   which FACTORS lacks, so it snaps to 1.5 and every gap fails. They are
   dotted-beat pieces (12/16, 6/8-type): rescaling writes the right note values
   but on a QUARTER beat, so meter, bar lines and beaming are wrong. On
   validation dotted scales are often right (65/140 dotted picks are truly
   0.75), so the fix is NOT to forbid them but to write COMPOUND METER: when
   the scale is 0.75/1.5 keep the beat on the dotted value and print x/8 or
   x/16. Touches notation (BeatGrid / build_score / time signature, all
   quarter-based today) and the benchmark's beat units. Needs a plan + USER
   decisions on compound notation before building.
2. Bar lines: soft decoding failed; next is retraining with the downbeat loss
   weighted.
3. The Toccata: the model writes even 16th runs as triplets. Needs training
   data with long even runs (synthetic performances, item 4).
4. More data (synthetic performances) for the next model.

## (older) Next (for the user to look at first)

1. Decide whether `rhythm="model"` becomes the product default. Needed: load
   the weights in `api/pipeline.py` and decide where weights live (repo or
   download). Rhythm and durations are up, bar lines slightly down.
2. Fix the beat/bar-line regression cheaply: the model's downbeat head
   (val F1 ~0.18) is weak. Try a hybrid that keeps the rule pipeline's
   meter/bars and uses the model's positions and values, or weight the
   downbeat loss. Benchmark both.
3. Look at the 9 pieces where the model loses rhythm.
4. Then option 2 (synthetic performances) for more data.
      Known mismatch for the AUDIO row: training offsets are key releases;
      transcription offsets are sounding (pedal-extended). Augment for it.
- [x] C5. `rhythm/decode.py` (7 tests), `rhythm/infer.py` + wiring
      `transcription_to_score(rhythm="model", rhythm_model=...)` (6 tests;
      default stays "grid"). `ScoreStats.rhythm`/`grid_beats`. Benchmark
      `--rhythm-model PATH` scores the grid actually used. Default path
      re-verified identical; model path smoke-tested end to end.
- [ ] C6. Benchmark vs 58.3% rhythm / 43.5% durations.

---

# Phase 32 -- score-level benchmark: real recordings vs the real printed score

## Why

MAESTRO scores NOTES, and onset F1 (0.956) is near its ceiling, so a month of
MAESTRO work moved nothing a user sees. Users judge the PAGE. No benchmark in
the repo compares the printed output against a real published score from a real
recording (`benchmark_notation` uses scores synthesised from music21). This is
the scoreboard any home-built model (Phase 33) has to move.

## Requirements

- For MAESTRO **test-split** performances that ASAP aligns to a score, run the
  full pipeline audio -> score and compare against the published score, note by
  note, through the alignment chain:
  our note <-> performance note (onset match) <-> score note (ASAP alignment).
- Per matched note, SEPARATE columns: printed duration (note value), staff/hand,
  pitch spelling, beat position in the bar. Per piece: key signature, time
  signature, note recall/precision.
- Two rows per piece: **from audio** and **from the perfect performance MIDI**.
  The difference says which half of the pipeline (transcription vs notation)
  loses each column, which tells Phase 33 what to build.
- Zero train-split leakage (the model trained on MAESTRO train). ASAP is
  CC BY-NC-SA: evaluation only, nothing from it committed -- manifest + hashes
  only, like `maestro_test12.json`.

## Success criteria

- Runs on >= 10 test-split pieces, deterministic across two runs.
- Reports the CURRENT pipeline's baseline on every column. No pass bar: this is
  the first measurement.
- Each column has a hand-built test where the right answer is known.

## Steps

- [x] 1. Inventory (2026-10-09). ASAP metadata in `var/asap-meta/` (gitignored):
      **74 performances / 39 distinct scores / 9.6 h** are MAESTRO **test**
      split (399 train, 46 validation, 548 no MAESTRO audio). Beethoven 26,
      Bach 17, Schubert 10, Liszt 9, Chopin 8, Schumann 2, Debussy 1,
      Rachmaninoff 1.
      - Link back: `QuantisedNote.source` -> input `NoteEvent`, so the chain
        works up to quantisation. The final music21 score (voices/ties) loses
        it: spelling must be matched by pitch + position.
      - Note alignment: NOT using the external CPJKU alignments (repo not
        found). Build our own: ASAP's performance + score beat/downbeat
        annotations map performance seconds -> score beats; then match pitches.
      - ASAP `start`/`end` columns: some performances are a segment of the
        MAESTRO recording. Respect them.
      - Leakage note: other performances of the same 39 scores may be in
        MAESTRO train. Fine for testing transcription (different audio);
        any Phase 33 model trained on scores must split BY SCORE.
      - Design risk for step 3: our beat may be the score's beat x2 or /2
        (tempo octave), or bar 1 may be offset by a pickup. Report meter
        errors as their own column so they don't silently zero the rest.
- [x] 2. Tests first: `tests/test_asap.py`, `tests/test_score_benchmark.py`,
      `tests/test_score_benchmark_tool.py` (28 tests; RED confirmed, then GREEN).
- [x] 3. `evaluation/asap.py`, `evaluation/score_benchmark.py`,
      `tools/score_benchmark.py`. Spelling column deferred (needs MusicXML).
- [x] 4. Full run: 38/39 pieces (Liszt TE 11 skipped: 1% note match to
      MAESTRO, unresolved). `benchmarks/score-benchmark-ptify-asap.json`.
- [x] 5. `benchmarks/score-benchmark-README.md`. HANDOFF section 1 updated.

## Results (2026-10-09)

| row | reached | meter ok | beatF | dbF | dur | rhythm | staff |
|---|---|---|---|---|---|---|---|
| audio | 83.9% | 19/38 | 0.49 | 0.17 | 39.6% | 52.9% | 89.1% |
| perfect MIDI | 92.4% | 19/38 | 0.49 | 0.17 | 39.3% | 52.2% | 89.6% |

- Notation, not transcription, breaks the page (audio == perfect MIDI on
  every per-note column).
- The 0.25-beat grid cannot write triple division: the 5 worst rhythm scores
  (0-4%) are all compound or triplet music, even with the beat tracked right.
- Beat level wrong in 19/38; downbeats F 0.17; the pipeline assumes 4/4.

Candidate Phase 33 targets, from the table (USER to choose):
  a. beat-division detection (duple vs triple grid)
  b. beat-level / meter / downbeat estimation
  c. note durations (about 40% even where rhythm is right)

## Not doing

- No training (that is Phase 33, chosen from this table).
- No Transkun/third-party engines (user prefers building our own).
- No changes to notation behaviour.

## Verification level

Replayed against real recordings and real published scores.

---


## Phase 31 conclusion (2026-10-09): NOT wired in

`notation.quantise.notated_offsets` (shipped before Phase 31) already ends notes
at min(own offset, pedal lift, next onset same staff), and it beats the winner
on the same 12 tracks: cond 51.0% vs 45.3%, sustain 40.4% vs 34.0%, pedal-up
70.7% vs 66.4%, offset F1 0.507 vs 0.459. Wiring the rule in would make the page
worse. Steps 5-6 cancelled. `evaluation/sustain_rules.py` stays only as the
measurement that showed this (user to decide whether to commit it).

---

# Phase 31 — note durations under sustain

## The finding this plan rests on

`benchmarks/offset-diagnosis-ptify-maestro.json` (4 MAESTRO tracks, 14,360
onset-matched notes): of notes whose onset already matched, **29%** end in
tolerance. 10,075 are too long, 76 too short. **Under sustain 20%, pedal up
70%.**

Under sustain a key release is acoustically silent (dampers are up), so the
model is reporting the sound correctly and the notated key-release is not in
the audio. The model cannot learn it, so this is a decoding problem: infer the
key release from musical context. The pedal-up 30% is observable model error
and is NOT this phase.

## Requirements

- A note released under the (estimated) sustain pedal gets a duration that
  matches where the player lifted the key, inferred from context.
- Notes outside sustain are unchanged. Onsets, pitches and note counts are
  unchanged by construction (this phase edits offsets only).
- The rule uses only what exists at inference: the model's OWN pedal
  estimate, never the reference CC64.

## Decisions (2026-09-19)

- Fix applies to **transcriber output** (MIDI, roll, score, benchmark). MIDI
  keeps CC64.
- **No fixed bar**: measure all four rules, decide from the numbers.
- User commits; Claude only stages.
- (2026-10-09) Winner: **next_onset_same_hand**. Layer: **all** (transcriber
  output: MIDI, roll, score, benchmark). Open: user is weighing scrapping
  rule-based work for a Transformer; step 5 waits on that call.

## Success criteria

- Conditional accuracy **under sustain**: report 0.20 -> each rule's value, on
  all 12 MAESTRO tracks, not just 4. Winner chosen by the user from the table.
- Conditional accuracy **pedal up**: stays >= 0.70 (no regression).
- `onset_f1` identical to 4 decimals (proves offsets-only).
- `offset_f1` reported alongside, since it is the published number.

## Steps

- [x] 0. Staged the pending diagnosis JSON, `calibrate_frame_threshold.py`
      and HANDOFF.md changes for the user to commit.
- [x] 1. (12 tracks cached in var/estimates/ptify/;
      benchmarks/offset-diagnosis-ptify-maestro12{,-estpedal}.json)
      Cache engine output once per track (notes + estimated pedals) so each
      decode candidate is scored in seconds, not ~1.9x realtime. Use the Modal
      host (9.3x) for the 12-track run.
- [x] 2. (code done; the confusion table prints in the step-1 run) Add `--pedal-source {reference,estimated}` to the diagnosis: measure
      how well the model's untrained, frozen pedal head matches CC64. If it is
      poor, every rule built on it is capped, and that is the first finding.
- [x] 3. Score candidate rules for notes ending under estimated sustain:
      a. baseline (as-is)
      b. end at the next onset in the same hand (reuses the hand model)
      c. end at the next bass/harmony change (reuses chord detection)
      d. end at the pedal change (re-pedal point)
- [ ] 4. Pick the winner by the criteria above. Report all four, not just the
      winner.
- [ ] 5. Wire it in at the agreed layer (see open question), with tests
      written first against small hand-built cases.
- [ ] 6. Update HANDOFF.md §1 and the benchmark README.

## Files touched (expected)

- `evaluation/offset_diagnosis.py`, `tools/offset_diagnosis.py` — pedal
  source flag, cached-estimate input
- new `transcriber/sustain.py` OR `notation/` module — the decode rule
- `tests/` — new tests for the rule and the flag
- `benchmarks/` — 12-track before/after reports

## Not doing

- No training, no new checkpoint, no GPU training quota.
- Not the pedal-up 30% (separate phase, possibly a frame-head question).
- Not the onset/recall deficit.

## Verification level expected

Replayed against real MAESTRO recordings with ground truth. Not listened to by
a musician; readability on the page is a separate check.

## Open question for a human

Where does the corrected duration live?
- **Transcriber output**: MIDI, roll, score and benchmark all change. The MIDI
  keeps CC64 so it still sounds pedalled, but roll playback (`summary.notes`)
  may sound choppier unless it honours pedal. Unchecked.
- **Notation only**: printed durations change; MIDI/roll stay as heard; the
  published `offset_f1` does not move.

## Results (2026-10-09)

**Correction to "The finding":** the 4-track 20% / 70% split came from a bug in
`aggregate()` that pooled pedal spans across tracks (fixed, with a test).
Fixed 12-track baseline: under sustain **7.4%** (2311/31285), pedal up **61.1%**.

Model pedal vs reference CC64 at matched releases: agree 96.3%
(both 63.0%, neither 33.3%), so rules built on the estimated pedal are not capped by it.

Step 3, `benchmarks/sustain-rules-ptify-maestro12.json` (12 tracks; the split is judged on
reference CC64, and the rules see only the estimated pedal):

| rule | cond | sustain | pedal up | onset F1 | offset F1 | shortened |
|---|---|---|---|---|---|---|
| baseline | 26.1% | 7.4% | 61.1% | 0.9562 | 0.2801 | 0 |
| next_onset_same_hand | 45.3% | 34.0% | 66.4% | 0.9562 | 0.4591 | 24732 |
| next_bass_change | 39.0% | 25.4% | 64.3% | 0.9562 | 0.4032 | 21956 |
| pedal_change | 28.5% | 9.8% | 63.5% | 0.9562 | 0.2985 | 15651 |

The baseline row reproduces the diagnosis exactly, and onset F1 is identical to 4 decimals,
so the rules change offsets only. Step 4 (pick the winner) is the user's call.

**Scoring-convention finding (2026-10-09, scratch script, not in repo):** published
MAESTRO offset F1 (Onsets & Frames, Kong, PianoVAM) extends REFERENCE offsets
through the sustain pedal. Re-scored that way on the same 12 cached tracks:

| | raw key-release ref | pedal-extended ref |
|---|---|---|
| baseline | 0.2801 | **0.5579** |
| next_onset_same_hand | 0.4591 | 0.4625 |

So the rule helps PRINTED durations (key release) but hurts SOUNDING durations,
which is what the standard number measures. This reopens the "layer = all" decision
(notation-only would fit the evidence). Step 5 is on hold pending the user.
