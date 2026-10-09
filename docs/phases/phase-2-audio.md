# Phase 2 — Audio features

Status: **APPROVED 2026-10-09.** Decisions D1–D8 resolved. Step 0
environment spike passed (§16). Implementation plan: docs/phases/phase-2-audio-plan.md
(prototype findings in §17). Next: implementation, then the pilot (§11).
Last updated: 2026-10-09

## 0. Summary

Turn every unique song in the Phase 1 dataset into a cached, versioned set of
conditioning features on **one fixed time grid** that lines up exactly with
map times (`time_s`) and replay times (BSOR `frame.time`):

- **Spectral features** for the full mix: log-mel, onset strength, loudness,
  chroma.
- **Stems** (drums, bass, other, vocals) from Demucs, reduced to per-stem
  log-mel, onset strength and loudness. The stem audio itself is not kept
  (D3).
- **Beats and downbeats** estimated from audio, plus how well they agree with
  each map's own BPM grid.
- **Structure**: segment boundaries and repetition labels (A/B/A/C…).
- **Alignment utilities and checks**: sample any feature at map beat-grid
  times or replay frame times, and verify per map that the notes line up with
  the audio.

Everything below that says "measured" was read from
`E:\kata-data\processed\maps.parquet` / `tempo_maps.parquet` or from this
machine on 2026-10-09. Anything not verified is marked **[verify in pilot]**
or **[verify at setup]**.

### Decisions (all resolved 2026-10-09)

| # | Decision | Outcome | Rejected alternative |
|---|----------|---------|----------------------|
| D1 | Feature frame rate | **100 fps** (hop 441 @ 44.1 kHz = exactly 10 ms) | 50 fps: half the disk, 20 ms resolution |
| D2 | Stem model | **`htdemucs`** (4 stems, single model) | `htdemucs_ft` (~4× slower, ~+0.3 dB SDR); RoFormer models (better vocals, heavier deps) |
| D3 | Keep stem audio? | **No.** Keep per-stem features only; re-running Demucs costs < 1 GPU-night | Yes: Ogg Vorbis, ~70 GB |
| D4 | Beat/downbeat tracker | **Beat This!** (CPJKU, ISMIR 2024; pure PyTorch) | madmom (hard to install on Py 3.13), All-In-One (needs madmom + NATTEN) |
| D5 | Structure | **Unsupervised repetition segmentation** (boundaries + A/B/C labels). Functional labels (verse/chorus) deferred to Phase 6 | All-In-One functional labels via WSL2 |
| D6 | Near-duplicate audio | **Skipped.** Song groups stay as Phase 1 built them (title + exact audio hash); see §8 | Chromaprint fingerprints + group merging |
| D7 | Disk budget | **≤ 100 GB** under `$KATA_DATA/features/` | — |
| D8 | Split changes | **None.** Moot after D6: Phase 2 does not touch song groups or splits | — |

---

## 1. Goals and non-goals

**Goals**
1. Precomputed features for 100 % of unique audio files (or a recorded
   failure reason), on one documented time grid.
2. Exact time alignment: frame *i* of every feature is centred at
   `t = i / 100` s, on the same clock as map `time_s` and replay
   `frame.time`. This is verified, not assumed (§7).
3. Beats, downbeats, tempo and structure per song, usable at inference time
   when no map exists yet.
4. A per-map measure of how well the authored map is synced to the audio, and
   how its BPM grid relates to the audio beats (same / double / half …).
5. Fast random-window reads (Phase 4 trains on 8-bar windows).
6. A report regenerated on each build, like Phase 1.

**Non-goals (later phases)**
- Learned audio embeddings (MERT, MusicFM, CLAP, Jukebox). One 768-d layer at
  75 Hz would be ~125 GB in fp16. They are a Phase 4 option if the
  hand-crafted conditioning underperforms.
- Functional structure labels (intro/verse/chorus) → Phase 6 (D5).
- Lyrics, vocal transcription, pitch tracking.
- Near-duplicate audio detection and any change to song groups or splits
  (D6, D8).
- Any normalisation that depends on a model's needs (windowing, beat-relative
  resampling). Phase 2 stores features on the time grid. Converting them to a
  beat grid is a *function* (§7.2), not a stored table.
- Replays. They are still deferred (phase-1-data.md D9). Phase 2 only needs
  their clock convention, which Phase 1 already verified.

---

## 2. Inputs from Phase 1 (measured)

| Item | Value |
|------|-------|
| Selected maps | 5,437 (A 2,786 · B 2,651) |
| Unique audio files (`audio_sha256`) | **5,347** ← Phase 2's unit of work |
| Song groups | 4,832 |
| Total audio | **302.5 h** (train 274.0 h · val 14.0 h · test 14.5 h) |
| Duration | mean 200 s, median 197 s, max 710 s |
| Container | Ogg Vorbis (`.egg`) for all 5,437 |
| Sample rate | 44.1 kHz: 4,435 · 48 kHz: 960 · other: 42 (192 kHz ×16, 96 kHz ×14, 32 kHz ×7, and one each of 44,099 / 88.2k / 176.4k / 22.05k / 38k Hz) |
| Map BPM (metadata) | p1 88 · p25 128 · median 160 · p75 191 · p99 350 |
| Maps with a BPM change | 526 (1,495 difficulties) |
| Maps whose difficulties have *different* tempo maps | 77 |
| `_songTimeOffset ≠ 0` | 0 maps |

Phase 1 established that `time_s` is measured from the first sample of the
decoded audio, and that it matches replay `spawnTime` to < 0.04 ms. Phase 2
anchors its features to that same first sample.

Features are keyed by `audio_sha256`, not by map hash. Maps that share audio
share features (5,437 maps → 5,347 files).

**Hardware (measured)**: RTX 5080 16 GB (Blackwell, needs a CUDA ≥ 12.8
PyTorch build), Ryzen 7 7800X3D (8C/16T), 31 GB RAM. E: has 336 GB free.

---

## 3. Time conventions

These are the contract every later phase relies on.

- **Canonical audio**: decoded with `soundfile` (libsndfile, the same decoder
  Phase 1 used for validation), float32, resampled to **44,100 Hz** with
  `soxr` (linear-phase, delay-compensated), kept stereo for Demucs and
  down-mixed to mono (mean of channels) for mix features. Mono sources are
  duplicated to stereo for Demucs.
- **Frames**: `SR = 44100`, `HOP = 441`, `FPS = 100`. STFTs are *centred*
  (reflect padding), so frame *i* is centred on sample `i · 441`, i.e.
  `t_i = i / 100` s exactly.
- **Frame count**: `n_frames = n_samples // 441 + 1`, where `n_samples` is
  the length after resampling to 44.1 kHz. This is asserted for every file.
- **Events** (beats, downbeats, segment boundaries) are stored in **seconds**
  (float64), on the same clock.
- **Resampling from 48 kHz and other rates** must not shift time. The test in
  §11 puts an impulse at 1.000 s and requires it within ±1 sample after
  resampling, for every source rate in §2.
- **Vorbis decode start**: we assume libsndfile and the game decode the same
  first sample. Phase 1's replay alignment cannot test this (it compares map
  times with replay times, not with audio). The sync check in §7.3 tests it:
  a decoder offset shows up as a *systematic* lag across all maps.

---

## 4. Feature specification

### 4.1 Mix features (per frame, 100 fps)

| Array | Shape | dtype | Definition |
|-------|-------|-------|------------|
| `mel` | (T, 128) | fp16 | Power mel spectrogram, `n_fft = 2048` (46 ms Hann), 128 Slaney mel bands, 30 Hz – 16 kHz, `10·log10(max(S, 1e-10))`, clamped to ≥ −100 dB. No per-song normalisation |
| `chroma` | (T, 12) | fp16 | STFT chroma (`n_fft = 4096` for low-frequency resolution, same hop), L∞-normalised per frame. Used by structure; also a cheap harmony signal |
| `curves[:, onset_mix]` | (T,) | fp16 | Spectral flux on `mel`: `Σ_bins max(0, mel[t] − maxfilter₃(mel[t−1]))` (SuperFlux-style, lag 1, 3-bin max filter) |
| `curves[:, rms_mix]` | (T,) | fp16 | Frame RMS in dBFS (window 2048) |

All spectral code runs on the GPU in PyTorch (STFT + a mel/chroma
filterbank built once). This keeps it fast and avoids a hard dependency on
librosa/numba, whose Py 3.13 + NumPy 2.5 support is uncertain. librosa is
not part of the stack; tests use it as a reference only when it happens to
be installed.

### 4.2 Stems (D2, D3)

`htdemucs` separates each song into **drums, bass, other, vocals** (always
in this order). Settings: `split=True`, `overlap=0.25`, **`shifts=0`**.
`shifts=1` (the CLI default) applies a random time shift, and two runs on
the same audio then differ by up to 0.1 in sample value (§16). `shifts=0` is
bit-identical across runs, which idempotent stages need; it costs ≤ 0.2 dB
SDR. Demucs runs on the canonical 44.1 kHz stereo audio, so its output is
already on the frame grid.

The stems are reduced to features in memory and then discarded:

| Array | Shape | dtype | Definition |
|-------|-------|-------|------------|
| `stem_mel` | (T, 4, 64) | fp16 | As `mel`, but 64 bands (stems need less frequency detail). Time-major like every other array, so one window slice serves all arrays |
| `curves[:, onset_<stem>]` | (T,) × 4 | fp16 | Spectral flux on the 64-band stem mel |
| `curves[:, rms_<stem>]` | (T,) × 4 | fp16 | Stem RMS in dBFS |

Why stems matter here: mappers follow different instruments at different
times (drums in a drop, vocals in a verse). Per-stem onsets let the model and
the Phase 3 baseline see which layer an onset comes from.

### 4.3 Beat activations (D4)

Beat This! produces frame-wise beat and downbeat logits at 50 fps from
22.05 kHz input. We store them interpolated to 100 fps:

| Array | Shape | dtype | Definition |
|-------|-------|-------|------------|
| `curves[:, beat_act]` | (T,) | fp16 | Beat probability (sigmoid of logit) |
| `curves[:, downbeat_act]` | (T,) | fp16 | Downbeat probability |

**Verified (§16):** Beat This frame *j* is centred at `j · 20 ms`; no
offset correction is needed. Its beat *events* are frame peaks, so they have
20 ms resolution. On off-grid click tracks they land +4 to +15 ms late on
average (up to 33 ms worst case), and parabolic peak interpolation does not
improve this. Beat events therefore feed only the coarse uses (bar position,
tempo, structure, ±70 ms grid agreement). Timing-critical work (the sync
check, §7.3) uses the 100 fps onset curves instead.

### 4.4 `curves` columns

`curves` is one (T, 12) fp16 array. Its column order is fixed and also
written to the manifest:

`onset_mix, onset_drums, onset_bass, onset_other, onset_vocals,
rms_mix, rms_drums, rms_bass, rms_other, rms_vocals, beat_act, downbeat_act`

### 4.5 Per-song scalars (in `audio_features`)

Peak dBFS, integrated RMS loudness (dBFS), source sample rate, source
channels, `bandlimited` flag (source rate < 44.1 kHz: 10 files), clipping
ratio (fraction of samples at |x| ≥ 0.999).

**Normalisation** is not applied to the stored features. `stats` (§10) writes
per-bin / per-column mean and std over the **train split only**. Phase 4
applies them at load time.

---

## 5. Beats, downbeats and tempo

### 5.1 Events from audio

Beat This! (`dbn=False`, its default minimal peak picking) gives beat and
downbeat times. Per song we derive:

- `beat_in_bar` for each beat (1 = downbeat), counting from the previous
  downbeat;
- `est_bpm` = 60 / median inter-beat interval;
- `est_meter` = the most common number of beats per bar;
- `tempo_cv` = median absolute deviation of the inter-beat intervals divided
  by their median (a robust "is the tempo steady?" flag; a plain coefficient
  of variation was ~0.28 on steady songs because of a few missed beats).

Beat Saber maps carry no bar or time-signature information. Audio downbeats
are therefore the only bar-phase signal we have.

### 5.2 Agreement with the map grid

For every difficulty, the map's beat grid is `TempoMap.to_seconds(k)` for
integer `k` (reusing `kata.data.mapformat.tempo`), restricted to the span
from the first to the last note.

We compare it with the audio beats at metrical levels
**L ∈ {¼, ½, 1, 2, 4}** (for L < 1, every 1/L-th grid beat at the best
phase). The score is the F-measure with a ±70 ms tolerance window (the
standard beat-tracking measure, implemented in-house; ~20 lines, no
`mir_eval` dependency).

Stored per difficulty: `F` at each level, `best_level`, `best_f`.

This answers three questions later phases depend on:
1. How often mappers use a doubled or halved BPM (median map BPM is 160,
   which suggests doubling is common).
2. Whether audio beats can replace the map grid at inference time.
3. Which maps have an unusual or drifting grid (low `best_f` at every level).

---

## 6. Structure (D5)

Unsupervised, following McFee & Ellis (2014, "Analyzing song structure with
spectral clustering"):

1. Beat-synchronous features: mean `chroma` and 20 MFCCs (DCT of `mel`) per
   audio beat.
2. Recurrence matrix on chroma stacked with its previous and next beat
   (a *centred* delay embedding; a past-only one put boundaries about a bar
   late), mutual k-NN with k = ⌈2√n⌉, path-enhanced with a diagonal median
   filter; local adjacency from MFCC distances between neighbouring beats.
3. Laplacian embedding of the combined graph; *k*-means for
   `k ∈ {2 … 8}`. *k* is tried in eigengap order, and the first *k* whose
   merged segmentation still has ≥ 2 labels wins (the plain eigengap
   sometimes spent clusters on a few isolated beats, which then merged away).
4. Boundaries are the label changes, snapped to the nearest audio downbeat.
   Segments shorter than 8 beats are merged into a neighbour.

Output per song: `seg_idx, start_s, end_s, label` (an integer; equal labels
mean "this section repeats"), plus `n_labels`.

Phase 6's "motif reuse on repeated sections" needs to know *which sections
repeat*, which this gives. Functional names (chorus, verse) do not change
that. All-In-One gives functional names but needs NATTEN and madmom, neither
of which installs cleanly on Windows + Py 3.13. It can be revisited under
WSL2 when Phase 6 actually consumes structure.

Quality check without ground truth: the maps themselves. Note density
usually changes at section boundaries, and repeated sections often reuse
patterns. The report shows how often a segment boundary lies within one bar
of a large note-density change (Tier A).

---

## 7. Alignment

### 7.1 Clocks

All three clocks are the same clock:

| Source | Time value | Origin |
|--------|------------|--------|
| Features | `i / 100` | first decoded sample |
| Map objects | `time_s` | first decoded sample (Phase 1 §5.1) |
| Replays | `frame.time`, `spawnTime` | game song time, verified equal to `time_s` (Phase 1 §12.1) |

So alignment is interpolation, never a search.

### 7.2 Utilities (`kata.audio.frames`)

- `frame_index(t_s) -> float`: fractional frame index, `t_s · FPS`.
- `sample(arr, times_s) -> ndarray`: linear interpolation along time. Times
  outside the audio return the array's *silence row* (−100 dB for mel,
  0 for onset/activation curves, −100 dBFS for RMS), never an edge value
  repeated, so a model never sees fake sustained sound after the song ends.
- `beat_grid_times(tempo_map, start_beat, end_beat, subdiv) -> ndarray`:
  grid times in seconds (e.g. `subdiv = 4` for 1/4-beat steps).
- `window(sha, start_s, end_s) -> dict[str, ndarray]`: memory-mapped read of
  all arrays for a time window. This is the Phase 4 data-loader primitive.

Sampling at replay frames is `sample(arr, frame_times)`. Sampling on the map
grid is `sample(arr, beat_grid_times(...))`.

### 7.3 Map ↔ audio sync check (per difficulty)

This catches both pipeline bugs and badly synced maps.

1. Build a note impulse train at 100 fps from the unique note times
   (`time_s`, both colours), smoothed with a 20 ms Gaussian.
2. Cross-correlate it with `onset_mix` and with `onset_drums` over lags
   −200 … +200 ms. Refine the peak with parabolic interpolation (~1 ms
   precision).
3. Subtract the onset detector's own bias, measured once on the synthetic
   click track (§11).
4. Store `lag_ms`, `peak_corr`, and `peak_ratio` (peak / median correlation;
   low means "no clear sync", e.g. sparse or ambient maps).

**Expected result**: Tier A lags cluster tightly around 0. A *systematic*
offset across all maps (for example +20 ms everywhere) means our decoder
start or resampler is off. That would be a Phase 2 bug to fix, not a
property of the maps. Individual maps with |lag| > 50 ms are flagged for
later phases, not rejected here.

---

## 8. Near-duplicate audio — skipped (D6)

Phase 1 (§3.4) left near-duplicate detection to Phase 2. **We skip it.**

The gap: the same song uploaded with a different encode, trim or master gets
a different `audio_sha256`. If its title is also spelled differently, Phase 1
puts it in a different song group, and it can land in a different split.

We accept this as a small leakage risk:
- Phase 1 already groups by exact audio hash and by normalised
  (author, title). That catches most re-uploads.
- Val and test are 5 % each (249 and 262 maps). The expected number of
  undetected cross-split duplicates is small, and it can only make val/test
  scores slightly optimistic.

Song groups and splits stay exactly as Phase 1 built them (D8). If a later
phase's evaluation needs a stricter guarantee, fingerprinting can be added
then. The feature store does not depend on it.

---

## 9. Storage layout and size

```
$KATA_DATA/
  features/
    v1/                                   # FEATURE_VERSION; params change ⇒ new dir
      config.json                         # every parameter in §3–§6 + model ids/versions
      stats.json                          # train-split mean/std per bin and column
      {sha[:2]}/{sha}/
        mel.npy          (T,128)   fp16
        stem_mel.npy     (T,4,64)  fp16
        chroma.npy       (T,12)    fp16
        curves.npy       (T,12)    fp16
  processed/
    audio_features.parquet                # one row per sha (§4.5 + status, n_frames, version)
    audio_beats.parquet                   # sha, idx, time_s, beat_in_bar, is_downbeat
    audio_tempo.parquet                   # sha, est_bpm, est_meter, tempo_cv
    audio_segments.parquet                # sha, seg_idx, start_s, end_s, label, n_labels
    map_audio_sync.parquet                # hash, characteristic, difficulty, lag_ms, peak_corr, peak_ratio, src
    beat_agreement.parquet                # hash, characteristic, difficulty, f_l025 … f_l4, best_level, best_f
  reports/phase2-YYYY-MM-DD.md
```

Principles carried over from Phase 1:
- Features are rebuildable from `raw/` alone, without network access.
- `.npy` files are written via a temporary file + rename, so an interrupted
  run never leaves a half-written array that looks complete.
- One plain `.npy` per array, so that `np.load(mmap_mode="r")` gives
  zero-copy window reads. HDF5/Zarr would add a dependency without a benefit
  at this scale (21k files).
- `kata data build` adds the new Parquet tables as DuckDB views
  automatically, because it already globs `processed/*.parquet`.

**Size** (108.9 M frames = 302.5 h × 360,000):

| Array | Bytes/frame | Total |
|-------|-------------|-------|
| `mel` | 256 | 27.9 GB |
| `stem_mel` | 512 | 55.8 GB |
| `chroma` | 24 | 2.6 GB |
| `curves` | 24 | 2.6 GB |
| Tables | — | < 1 GB |
| **Total** | **816** | **~89 GB** |

At 50 fps (D1 alternative) this halves to ~45 GB. Keeping stem audio (D3
alternative) adds ~70 GB.

**Disk after Phase 2**: 336 − 89 ≈ 247 GB free. Phase 4 replays need
~64 GB (phase-1-data.md §13), which leaves ~183 GB. The stage stops below
50 GB free (Phase 1's `MIN_FREE_BYTES`).

---

## 10. Compute budget

| Step | Device | Estimate (302.5 h of audio) |
|------|--------|-----------------------------|
| Decode + resample | CPU, 8 workers | ~0.3 s/file ⇒ ~5 min |
| Mix + stem spectral features | GPU | minutes |
| Demucs `htdemucs` | GPU | **70–87× real time (measured) ⇒ ~3.5–4.5 h** |
| Beat This! | GPU | **~750× real time (measured) ⇒ ~25 min** |
| Structure | CPU | < 1 s/song ⇒ ~10 min |
| Sync + beat agreement (15k diffs) | CPU | minutes |

**Budget: one overnight run (≤ 12 h).** Demucs dominates. `htdemucs_ft`
would be ~8–24 h, which is why D2 picks the single model.

The `features` stage pipelines the work: CPU workers decode and resample the
next songs while the GPU separates the current one. Longest song (710 s):
Demucs processes it in overlapping ~8 s segments, so GPU memory stays
bounded: peak 1.14 GiB on a 204 s song (measured), and segment-bound, so not
expected to grow with length **[confirm on the 710 s song in the pilot]**.

---

## 11. Implementation plan (after approval)

**Stack** (added to Phase 1's, as installed in Step 0): `torch` 2.11.0 +
`torchaudio` 2.11.0 (CUDA 13.0 builds), `demucs` 4.1.0, `beat-this` 1.1.0,
`soxr` 1.1.0, `scipy` 1.18.1 (structure clustering).

The GPU stack is the optional `audio` extra (`uv sync --extra audio`), so
the Phase 1 pipeline keeps installing without it. Torch comes from the
PyTorch `cu130` index through `[tool.uv.sources]`. torch is pinned to 2.11
because `beat-this` requires `torchaudio`, whose last release is 2.11. The
lock is limited to Windows and Linux (`[tool.uv] environments`): demucs pins
`numpy<2` on Intel macOS, which conflicts with Phase 1's NumPy 2.5.

**Step 0 — environment spike (gate). ✅ Passed 2026-10-09**, see §16.

**Code layout**

```
src/kata/audio/
  frames.py       # SR/HOP/FPS, n_frames, frame_index, sample, beat_grid_times  (§3, §7.2)
  decode.py       # decode (soundfile) → resample (soxr) → stereo/mono canonical audio
  spectral.py     # torch STFT, mel & chroma filterbanks, log-mel, flux, RMS  (§4.1, §4.2)
  models.py       # Demucs + Beat This wrappers  (§4.2, §4.3)
  extract.py      # one song → all arrays, events and scalars  (§4)
  beats.py        # beat_in_bar, tempo stats, F-measure, grid agreement  (§5)
  structure.py    # beat-sync features, recurrence, spectral clustering  (§6)
  sync.py         # map ↔ audio lag estimation, onset-bias calibration  (§7.3)
  store.py        # feature dir layout, atomic .npy write, mmap read, window()
  pipeline.py     # stages below, using kata.data.state.State for resumability
  report.py
cli.py            # new `kata audio <stage>` group
```

Phase 1 changes: `Paths` gains `features`. Nothing else in Phase 1 changes.

**Stages.** Each one is idempotent and resumable through
`state/pipeline.duckdb`, keyed by `audio_sha256`:

1. `features` (GPU) → `features/v1/…`, `audio_features`, `audio_beats`,
   `audio_tempo`.
2. `structure` → `audio_segments`.
3. `sync` → `map_audio_sync`, `beat_agreement`.
4. `stats` → `stats.json` (train split only, from `maps.split` as Phase 1
   assigned it).
5. `report` → `reports/phase2-YYYY-MM-DD.md`.

After `report`, `kata data build` picks up the new Parquet tables as DuckDB
views.

**Pilot first**, as in Phase 1: run every stage on the Phase 1 pilot subset
(`pilot = true`, ~200 maps). The pilot:
- measures throughput, VRAM and real sizes;
- resolves every **[verify …]** item;
- tunes the sync and beat-agreement thresholds;
- shows the sync-lag histogram (the decoder check in §3).

The full run starts only after you've seen the pilot report.

**Tests** (TDD; synthetic fixtures only, real-data tests skip when
`KATA_DATA` is absent, as in Phase 1; GPU tests skip without CUDA):
- `frames`: `n_frames` formula; `frame_index` ↔ time round trip; `sample`
  interpolation and silence rows outside the audio; `beat_grid_times` on a
  multi-BPM `TempoMap` matches `to_seconds`.
- `io`: impulse at 1.000 s survives resampling from every source rate in §2
  within ±1 sample; an impulse written to Ogg Vorbis by `soundfile` and
  decoded again stays within ±1 ms; mono → stereo duplication; stereo → mono
  downmix.
- `spectral`: a 440 Hz sine peaks in the right mel bin; a click train gives
  onset peaks within ±1 frame of each click (this also measures the detector
  bias used in §7.3); silence gives the −100 dB floor; RMS of a full-scale
  sine is −3.01 dBFS.
- `beats`: F-measure on hand-made beat lists (exact, shifted 50 ms, shifted
  100 ms); grid agreement picks L = 2 for a doubled grid and L = ½ for a
  halved one, at either phase.
- `structure`: a synthetic A-B-A-B feature sequence gives labels
  `[0, 1, 0, 1]` with the right boundaries.
- `sync`: a note train shifted by +23 ms against its own onset curve gives
  23 ± 2 ms.
- `store`: atomic write + mmap read round trip; `window()` across the end of
  the song returns silence rows; a changed config refuses to write into an
  existing version dir.
- `pipeline`: re-running `features` on finished songs is a no-op.
- GPU integration: Demucs on a 10 s synthetic mix returns (4, 2, N);
  Beat This on a 120 BPM click track returns beats every 0.5 s ± 20 ms.

---

## 12. Acceptance criteria (definition of done)

1. ≥ 99.5 % of the 5,347 unique audio files have features. Every failure has
   a recorded reason.
2. `n_frames == n_samples // 441 + 1` for every file; every array of a song
   has the same `T`.
3. All timing unit tests pass (resampling, Vorbis round trip, click-track
   onsets).
4. **Sync**: over Tier A difficulties, the median `lag_ms` lies within
   ±5 ms of 0, and ≥ 90 % lie within ±30 ms **[thresholds tuned in pilot]**.
   A systematic offset fails this criterion, even if each map looks fine.
5. **Beats**: the report shows the `best_level` distribution and `best_f`
   per tier. Sanity bar: ≥ 85 % of Tier A difficulties reach `best_f ≥ 0.8`
   at some level **[tuned in pilot]**.
6. A random 12 s window of all four arrays reads in < 5 ms from mmap (warm
   cache). This is the Phase 4 loader budget.
7. `stats.json` is computed from train-split songs only (automated check).
   Song groups and splits are byte-identical to Phase 1's (D8).
8. Re-running any stage with no new input is a no-op.
9. Feature disk usage ≤ 100 GB (D7).
10. The report shows funnel counts, timings, sizes, the sync-lag histogram,
    beat-agreement tables, estimated vs. map BPM, segment counts and
    lengths, and the boundary ↔ note-density check.

---

## 13. Usage posture

Phase 1's usage posture (phase-1-data.md D6, §10) applies unchanged:
- Features, stems and derived tables are derived from
  copyrighted audio. They are **not republished**, and the per-song feature
  files are never committed. Aggregate statistics in reports are fine.
- Model licences: Demucs (MIT), Beat This! (MIT code; its checkpoints were
  trained on academic beat datasets **[check checkpoint licence at
  setup]**). Both are consistent with non-commercial use.

---

## 14. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| ~~GPU stack on Windows + Py 3.13 + Blackwell (sm_120)~~ | — | **Resolved in Step 0** (§16) |
| ~~`demucs` package pins / unmaintained deps~~ | — | **Resolved in Step 0**: demucs 4.1.0 installs with normal deps (§16) |
| libsndfile vs. game Vorbis decode start differ | Every feature is shifted | Systematic-lag check (§7.3, criterion 4) |
| Beat This beat events are coarse (20 ms frames, late on off-grid clicks) | Beat times off by up to ~30 ms | Convention verified exact (§16); events used only where ±70 ms is enough; sync uses onset curves (§4.3) |
| Mappers' doubled/halved BPM | Grid ≠ audio beats | Metrical levels in §5.2; documented, not "fixed" |
| Undetected near-duplicate songs across splits (D6) | Slightly optimistic val/test scores | Accepted; fingerprinting can be added later without touching the feature store (§8) |
| 89 GB of features | Disk | 50 GB free-space stop; 50 fps (D1 alternative) would halve it |

---

## 15. Decision log

| Date | # | Decision |
|------|---|----------|
| 2026-10-09 | — | Draft written with proposals P1–P8 |
| 2026-10-09 | D1 | 100 fps feature grid |
| 2026-10-09 | D2 | `htdemucs` single model |
| 2026-10-09 | D3 | Stem audio not kept; per-stem features only |
| 2026-10-09 | D4 | Beat This! for beats and downbeats |
| 2026-10-09 | D5 | Unsupervised repetition segmentation; functional labels deferred to Phase 6 |
| 2026-10-09 | D6 | Near-duplicate audio detection skipped |
| 2026-10-09 | D7 | Feature disk budget ≤ 100 GB |
| 2026-10-09 | D8 | No split changes (moot after D6) |
| 2026-10-09 | — | Step 0 passed. Stack pinned (torch 2.11.0+cu130). Demucs `shifts=0` for determinism (§16) |
| 2026-10-10 | — | Implementation plan written; prototype corrections to §4.2, §5.1, §6 (§17) |

---

## 16. Step 0 findings (2026-10-09)

Environment spike on the dev machine. Real-audio checks used two Tier A
pilot songs: one at 44.1 kHz (203 s, map BPM 154) and one at 48 kHz (204 s,
map BPM 170).

### 16.1 Install

`uv sync --extra audio` installs cleanly on Windows + Python 3.13: torch
2.11.0+cu130, torchaudio 2.11.0+cu130, demucs 4.1.0, beat-this 1.1.0,
soxr 1.1.0, scipy 1.18.1. Every native package has a cp313 Windows wheel.
demucs 4.1.0 no longer pins torchaudio or dora-search, so the `--no-deps`
fallback was not needed. The Phase 1 test suite still passes (33 tests).
Model weights download on first use: Beat This `final0` (77 MB, to the
torch hub cache) and `htdemucs` (from the Hugging Face Hub).

### 16.2 Results

| Check | Result |
|-------|--------|
| CUDA | Available. RTX 5080 reports capability (12, 0), and the torch build includes `sm_120`. fp32/fp16/bf16 matmuls agree with an fp64 reference |
| `soxr` resampling (VHQ) | An impulse at 1.000 s lands on sample 44,100 **exactly** (±0) from all nine source rates in §2, and output lengths are exact |
| Ogg Vorbis round trip | A click written by `soundfile` and decoded again starts on the exact sample, at both 44.1 and 48 kHz, and the length is unchanged. This shows libsndfile adds no decoder delay. It does not prove the game decodes identically; the sync check (§7.3) still covers that |
| Beat This convention | Clicks exactly on the 20 ms frame grid come back with **0.0 ms** error, so frame *j* is centred at `j · 20 ms` |
| Beat This precision | Off-grid clicks: mean error +4 to +15 ms, worst case 33 ms. Parabolic peak interpolation does not help (§4.3) |
| Beat This on real songs | ~750× real time. 44.1 kHz song: 150.0 BPM estimated vs. 154 in the map. 48 kHz song: 85.7 vs. 170, a half-time reading; this is exactly the metrical-level case §5.2 measures. Both songs: 4 beats per bar |
| Demucs `htdemucs` | 70× and 87× real time. Peak VRAM 1.14 GiB. Output (4, 2, N) in the order drums, bass, other, vocals. The stems sum back to the mix within 2.6–3.2 % RMS |
| Demucs determinism | `shifts=1`: two runs on the same 20 s clip differ by up to 0.097. `shifts=0`: identical (0.0). Hence `shifts=0` (§4.2) |

### 16.3 Consequences for the plan

- Demucs at ~75× real time means ~4 h for the full 302.5 h. With Beat This
  and the spectral features, the `features` stage fits comfortably in one
  overnight run (§10).
- VRAM is not a constraint (1.1 of 16 GiB). The pilot can try running two
  songs at once if the GPU turns out to be underused.
- The 154 vs. 150 BPM gap on the first song is a first hint that map grids
  and audio beats can disagree beyond simple double/half. §5.2 measures this
  across the dataset in the pilot.

---

## 17. Prototype findings (2026-10-10)

Every module in the implementation plan was prototyped and its tests run
(99 passing, including the Phase 1 suite and the GPU model tests) before the
plan was written. The full stage chain also ran on 4 real Tier A pilot maps
(12 difficulties). Corrections made to this doc as a result:

- `stem_mel` is stored time-major, (T, 4, 64) (§4.2, §9).
- `tempo_cv` is MAD / median (§5.1).
- Structure uses a centred delay embedding and a guarded choice of *k* (§6).

Measurements on the 4 real maps (not yet a sample worth drawing conclusions
from; the pilot covers ~200):

| Item | Result |
|------|--------|
| `features` per song | 2.1–2.8 s for ~200 s of audio (~80× real time end to end) |
| Feature size | 816 B/frame, as estimated in §9 |
| 12 s window read | median 2.0 ms, p95 2.2 ms (criterion 6: < 5 ms) |
| Onset detector bias (click track) | −11.3 ms: the spectral-flux peak sits about one frame before the click |
| Sync lag, mix onsets | median −7.5 ms (range −11.5 … −0.6) |
| Sync lag, drum onsets | median −1.5 ms (range −6.2 … +4.7) |
| Lag spread across one map's difficulties | ≤ 0.4 ms (sync is a property of the map, as expected) |
| Map grid vs. audio beats | all 12 difficulties `best_f` ≥ 0.9; one map at level ½ (half-time) |
| Structure | 12–23 segments per song, 3–8 labels; repeated section sequences are recovered |
| Boundary ↔ note-density check | hit rate 0.59 vs. 0.52 chance |

Open for the pilot:
- **Which onset source criterion 4 uses.** Mix onsets sit ~6 ms earlier than
  drum onsets after the click-track correction, so with mix onsets
  criterion 4 (median within ±5 ms) would fail on these 4 maps while drums
  pass. The pilot decides whether criterion 4 uses drum onsets or a wider
  tolerance.
- **Whether structure is good enough.** The boundary ↔ density signal is
  weak on 4 songs. The pilot shows whether it holds up across ~200.
