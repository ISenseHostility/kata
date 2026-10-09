# Phase 1 — Data acquisition & normalization

Status: **APPROVED 2026-10-09.** Pilot done (§12). Full map run uses D8/D9 (§13).
Replays are **deferred to Phase 4** (D9).
Last updated: 2026-10-09

## 0. Summary

Build a reproducible, resumable pipeline that turns BeatSaver maps and
BeatLeader replays into a local, queryable dataset:

- **Maps**: the full BeatSaver metadata catalogue, filtered locally into
  quality tiers, then download the selected maps. Parse v2/v3/v4 difficulty
  files into one canonical schema with every object timed in both beats and
  seconds.
- **Replays**: for each selected difficulty that has a BeatLeader
  leaderboard, download a small, filtered, player-diverse sample of BSOR
  replays. Parse them into head and hand tracks in the game's native frame,
  and verify their alignment against the map.
- **Storage**: Parquet files are the source of truth. A DuckDB file sits on
  top as a rebuildable query layer.

Everything below that says "verified" was checked against live API responses
or a parsed BSOR file on 2026-10-09. Anything not verified is marked
**[verify in pilot]**.

### Decisions (all resolved 2026-10-09)

| # | Decision | Outcome |
|---|----------|---------|
| D1 | Data location and budget | **`E:\kata-data`** (`KATA_DATA`). E: has 392 GB free; Phase 1 is ~105 GB |
| D2 | Dataset scope | **Tier A (ranked) + Tier B (curated / well-rated)**, labelled by tier |
| D3 | Replays per difficulty | **K = 3**, raw BSOR kept (applies when replays are fetched; see D9) |
| D4 | Characteristics | **`Standard` only** |
| D5 | HTTP `User-Agent` contact | **`kata-dataset/0.1 (+https://github.com/ISenseHostility/kata)`** |
| D6 | Usage posture | **No republishing of site content, no commercial use.** Not limited to local research (see §10) |
| D7 | Difficulties | **All, Easy → Expert+** |
| D8 | Tier B rule | **Curated AND rating > 0.95 AND upvotes > 100** (2,651 maps) |
| D9 | Replays | **Deferred to Phase 4.** v1 is maps + audio only; Tier A replays are fetched before Phase 4 starts |

---

## 1. Goals and non-goals

**Goals**
1. A clean, deduplicated, quality-tiered set of map difficulties in one
   format-independent schema.
2. Audio files stored next to the maps, hashed and validated (decodable, with
   a duration that matches the metadata).
3. Replay motion (head + both hands, 6-DoF, timestamped) linked to the exact
   map version that was played. Alignment is verified per replay.
4. Song-level train/val/test splits fixed in Phase 1 so that no later phase
   can leak data across them.
5. A data report (counts at every filter step, plus distributions) that is
   regenerated on each build.

**Non-goals (later phases)**
- Audio feature extraction, stems, beat tracking → Phase 2.
- Playability rules beyond the BeatSaver-provided parity summary → Phase 3.
- Motion normalisation (head-relative frames, height scaling, resampling,
  6D rotations) → Phase 4. Phase 1 stores **raw** motion with enough metadata
  to do this later.
- Lighting events, Chroma/Noodle/Mapping Extensions content, 360/90°
  maps.

---

## 2. Sources (verified)

### 2.1 BeatSaver (`https://api.beatsaver.com`)

| Use | Endpoint | Notes |
|-----|----------|-------|
| Full catalogue crawl | `GET /maps/latest?pageSize=100&sort=CREATED&automapper=true&before=<ISO ts>` | 100 maps/page, paginate backwards by `createdAt`. ~150k maps ⇒ ~1.5k requests. `automapper=true` means "include automapped maps"; we filter locally |
| Single map refresh | `GET /maps/hash/{hash}`, `GET /maps/id/{id}` | |
| Map download | `versions[].downloadURL` (`r2cdn.beatsaver.com/{hash}.zip`) | |

Fields we rely on (verified present):
- `id`, `name`, `uploader.{id,name}`, `metadata.{bpm,duration,songName,songSubName,songAuthorName,levelAuthorName}`
- `stats.{upvotes,downvotes,score}` (`score` is BeatSaver's rating, 0–1)
- `uploaded`, `createdAt`, `updatedAt`, `lastPublishedAt`, `curator`, `curatedAt`, `tags`
- `automapper`, `declaredAi` (`None` / `Uploader` / `SageScore`)
- `ranked`, `qualified` (ScoreSaber), `blRanked`, `blQualified`
- `versions[].{hash,state,sageScore,downloadURL}`
- `versions[].diffs[].{characteristic,difficulty,njs,offset,notes,bombs,obstacles,nps,seconds,length,events,chroma,me,ne,cinema,paritySummary{errors,warns,resets},stars,blStars,maxScore,environment}`

**Server-side search filters are unreliable for counting.** For example,
`automapper=true` returns *more* results (144,725) than an unfiltered search
(123,596). So we never select maps through search. We crawl everything once
and filter in DuckDB. Search totals are useful as a sanity check only:
~2.6k maps ranked on either leaderboard (BL ~1.6k, SS ~1.9k), ~13k curated.

**Rate limits**: no limit is published and the API sends no rate-limit
headers. The community convention is ≤ 10 req/s. **We self-cap at 4 req/s
for the API and 4 concurrent CDN downloads**, honour `429`/`Retry-After`, and
use exponential backoff with jitter.

### 2.2 BeatLeader (`https://api.beatleader.com`)

| Use | Endpoint | Notes |
|-----|----------|-------|
| Leaderboards for a map version | `GET /leaderboards/hash/{hash}` | Returns `{song, leaderboards[]}`, one per characteristic × difficulty. The leaderboard id looks like `517e7xxxxxxxxxx91` |
| Scores | `GET /leaderboard/{id}?page=&count=&sortBy=rank&order=desc` | Per-score: `id, playerId, accuracy, modifiers, pauses, fullCombo, badCuts, missedNotes, bombCuts, wallsHit, platform, hmd, controller, timeset`; `player.{bot,temporary}`. **`replay` is null in this listing.** Max `count` **[verify in pilot]** |
| Replay URL fallback | `GET /score/{id}` | `replay` is populated here, plus `offsets` (byte offsets of BSOR sections) |
| Replay download | `cdn.replays.beatleader.com/{scoreId}-{playerId}-{Difficulty}-{Mode}-{HASH}.bsor` | Deterministic pattern verified on one score; fall back to `/score/{id}` on 404 |

Leaderboard objects also carry `stars`, `predictedAcc`, and
pass/acc/tech ratings. We store these; they are useful difficulty labels
later.

**Rate limits (verified, from response headers)**: `x-rate-limit-limit: 10s`,
50 requests per 10 s window. **We self-cap at 3 req/s for the API and 4
concurrent CDN downloads**, and read `x-rate-limit-remaining` to back off.

### 2.3 Terms

- BeatSaver ToS prohibits republishing other users' content and
  "commercializing content available on the site". It says nothing about API
  automation or ML use.
- BeatLeader ToS: not yet reviewed **[verify before full crawl]**.
- Consequences are covered in §10.

---

## 3. Selection: quality tiers

Selection is a pure function of the crawled metadata, implemented as a DuckDB
query and versioned in code. Each step's counts go to the data report.

### 3.1 Map-level hard filters (all tiers)

1. `automapper = false` and `declaredAi = 'None'`.
2. Latest version `state = 'Published'`. We only ever use the latest
   version's hash.
3. `metadata.duration` between 30 s and 900 s.
4. At least one difficulty that passes §3.2.

### 3.2 Difficulty-level hard filters

1. `characteristic = 'Standard'` (D4).
2. `me = false`, `ne = false`. Chroma and Cinema are allowed, because they
   are cosmetic.
3. `notes ≥ 50` and `seconds ≥ 30`.
4. Parsing succeeded and the parsed object counts match BeatSaver's counts,
   ±0 for notes, bombs and obstacles (a cross-check of our parser).

### 3.3 Tiers

| Tier | Definition | Expected size |
|------|------------|---------------|
| **A — ranked** | `blRanked OR ranked` (ScoreSaber); only the ranked difficulties of these maps count as Tier A, the rest fall to B/C | ~2.6k maps, ~5–6k diffs |
| **B — endorsed** | not A, and `curatedAt IS NOT NULL` AND `stats.score > 0.95` AND `upvotes > 100` (D8) | 2,651 maps |
| C — rest | everything else that passes hard filters | metadata only; **not downloaded** in v1 |

Rationale:
- Tier A passed human ranking criteria (parity, vision, readability) and
  has the most replays. It is the gold set for validation and calibration in
  Phase 3.
- Tier B adds stylistic breadth.
- Tier C stays in the metadata tables so we can widen the dataset later
  without re-crawling.

The `0.80 / 30` thresholds are placeholders. After the metadata crawl I'll
show the rating × votes distribution and we tune them so that Tier B fits
the storage budget.

### 3.4 Deduplication and splits

- **Song groups**: many maps share the same song. Group key, in priority
  order:
  1. Exact audio SHA-256 after download.
  2. Otherwise normalised `(songAuthorName, songName)`: lowercase, strip
     brackets/feat./punctuation.

  Near-duplicate audio (same song, different encode) is caught in Phase 2 by
  audio fingerprinting. Phase 1 records `song_group_id` so that this can be
  refined later.
- **Splits**: assigned per song group as
  `hash(song_group_id) mod 100 → train < 90 ≤ val < 95 ≤ test`. The split is
  deterministic and stored in the `maps` table. It is stable when new maps
  are added, because the assignment depends only on the group key.
- **Mapper leakage**: noted, not solved. We store `level_author` so that
  later phases can evaluate on held-out mappers if style conditioning needs
  it.

---

## 4. Replay sampling

Only for selected difficulties that have a BeatLeader leaderboard. In
practice that means Tier A plus the Tier B maps that people played.

### 4.1 Score filters

- `modifiers = ''`. This removes speed modifiers (FS/SF/SS), which change
  the song-time ↔ wall-clock relationship, and gameplay modifiers
  (NF/NA/NB/…). Purely visual modifiers (GN, DA, SMC) could be allowed later
  if replay counts turn out to be thin.
- `pauses = 0`, `player.bot = false`, `player.temporary = false`.
- `missedNotes + badCuts ≤ max(3, 1 % of notes)`. We want competent
  motion, not only FCs, which skew towards a few top players.
- BSOR `leftHanded = false` in v1. Left-handed replays are counted and kept
  out of v1 for now; mirroring is a Phase 4 decision.
- Replay map hash = our downloaded version hash (guaranteed, because we
  query leaderboards by hash).

### 4.2 Sampling per difficulty (K = 3, see D3)

From the scores that pass the filters:
- 1 replay = the highest-accuracy score;
- K−1 replays = uniform random from those ranked in the top 50 %
  (seeded by leaderboard id), skipping players already chosen for this
  difficulty.

**Player diversity cap**: no player may exceed 1 % of all replays. Selection
runs as a global pass: when a player hits the cap, the next eligible score is
taken instead.

Top-only sampling would teach the motion model a handful of elite players'
minimal-movement style. This scheme keeps quality high while spreading the
data across players. Phase 4 can re-weight by player if needed.

### 4.3 Scale estimate

~6k leaderboards × K = 3 ⇒ ~18k replays.
At ~1.7 MB per 3 min replay (measured) ⇒ **~30 GB raw BSOR** and ~15 GB of
parsed float32 Parquet. Exact numbers will come from the pilot.

---

## 5. Unified map schema

### 5.1 Canonical conventions

- **Time**: every object stores `beat` (as authored) **and** `time_s`
  (seconds from the first sample of the decoded audio file). `time_s` is the
  join key for audio and replays.
- **Grid**: `x ∈ {0..3}` left→right, `y ∈ {0..2}` bottom→top.
- **Colour**: `0 = left/red`, `1 = right/blue`.
- **Cut direction**: `0 up, 1 down, 2 left, 3 right, 4 up-left, 5 up-right,
  6 down-left, 7 down-right, 8 any/dot`. `angle_offset` is in degrees
  (CCW, 0 in v2).

### 5.2 Format mapping

| Concept | v2 (`_version 2.x`) | v3 (`version 3.x`) | v4 (`version 4.x`) |
|---------|------|------|------|
| Note | `_notes[{_time,_lineIndex,_lineLayer,_type∈{0,1},_cutDirection}]` | `colorNotes[{b,x,y,c,d,a}]` | `colorNotes[{b,r,i}]` → `colorNotesData[i]{x,y,c,d,a}` |
| Bomb | `_notes` with `_type=3` | `bombNotes[{b,x,y}]` | `bombNotes[{b,r,i}]` → `bombNotesData[i]{x,y}` |
| Wall | `_obstacles[{_time,_lineIndex,_type,_duration,_width}]` (type 0 = full ⇒ y=0,h=5; type 1 = crouch ⇒ y=2,h=3) | `obstacles[{b,x,y,d,w,h}]` | `obstacles[{b,r,i}]` → `obstaclesData[i]{d,x,y,w,h}` |
| Arc | `_sliders` (v2.6 only) | `sliders[{c,b,x,y,d,mu,tb,tx,ty,tc,tmu,m}]` | `arcs[{hb,tb,hr,tr,hi,ti,ai}]` → head/tail via `colorNotesData`, `arcsData[ai]{m,tm,a}` |
| Chain | — | `burstSliders[{c,b,x,y,d,tb,tx,ty,sc,s}]` | `chains[{hb,tb,hr,tr,i,ci}]` → `chainsData[ci]{tx,ty,c,s}` |
| BPM changes | `_events` type 100 (`_floatValue`); legacy `_customData._BPMChanges` | `bpmEvents[{b,m}]` | `AudioData.dat` `bpmData` regions (sample index ↔ beat) **[verify key names in pilot]** |
| Info file | Info.dat v2 (`_beatsPerMinute`, `_songFilename`, `_difficultyBeatmapSets`) — used by v2 **and** v3 maps | same | Info.dat v4 (`audio.{songFilename,bpm,audioDataFilename}`, `difficultyBeatmaps[]`) |

Rejection and flagging rules:
- v2 Mapping Extensions precision values (`_lineIndex ≥ 1000` or `< 0`, or
  `_lineLayer` outside 0–2) → reject the difficulty.
- Out-of-grid values in v3/v4 → reject.
- Unknown cut direction → reject.
- Non-zero `_songTimeOffset` → flag; it is applied to `time_s`.
- v4 `r` (rotation lane) ≠ 0 → reject. This should never happen in
  `Standard`.
- Fake / `customData` objects (Noodle) → excluded by §3.2. If they appear
  anyway, the count check in §3.2 catches them.

### 5.3 Beat → seconds

Build a piecewise-linear `tempo_map` per difficulty from the BPM changes
(v2/v3) or the AudioData regions (v4), seeded by the Info BPM at beat 0.
Then `time_s = tempo_map(beat) + song_time_offset`.

This is checked in two ways:
1. Unit tests with hand-built multi-BPM maps.
2. **Against replays**: every BSOR note event carries `spawnTime`, the
   game's own computation of the note time in seconds (§6.3).

### 5.4 Tables (Parquet)

`maps` (one row per selected map, latest version)
: `map_id, hash, name, song_name, song_sub_name, song_author, level_author,
uploader_id, uploader_name, meta_bpm, meta_duration_s, uploaded_at,
curated_at, upvotes, downvotes, rating, ranked_ss, ranked_bl, qualified_ss,
qualified_bl, declared_ai, sage_score, tags[], tier, info_format_version,
audio_file, audio_sha256, audio_duration_s, audio_sample_rate,
song_group_id, split, download_status`

`difficulties`
: `hash, characteristic, difficulty, difficulty_rank, format_version,
njs, njs_offset, n_notes, n_bombs, n_obstacles, n_arcs, n_chains, nps,
duration_s, bs_parity_errors, bs_parity_warns, bs_parity_resets, ss_stars,
bl_stars, bl_pass_rating, bl_acc_rating, bl_tech_rating, chroma, cinema,
environment, parse_status, parse_warnings[]`

`notes`, `bombs`, `obstacles`, `arcs`, `chains`
: Long tables keyed by `(hash, characteristic, difficulty)` plus a stable
`idx` (sorted by beat, then x, then y). Columns follow §5.2 in canonical
names, with `beat` and `time_s` (and `tail_beat`/`tail_time_s` where
relevant). Partitioned by `split`.

`tempo_maps`
: `hash, characteristic, difficulty, seg_idx, start_beat, start_s, bpm`

`metadata_catalog` (all crawled maps, all tiers incl. C; raw JSON kept)
: the full BeatSaver map record flattened, plus `crawled_at`.

---

## 6. Replays

### 6.1 BSOR format (verified on a real v1 file, 1.7 MB, 179 s)

The file starts with magic `0x442d3d69`, then version byte `1`. Sections
follow, each introduced by an id byte:

| Id | Section | Content |
|----|---------|---------|
| 0 | Info | strings + scalars: mod/game version, timestamp, player id/name, platform, tracking system, HMD, controller, map hash, song, mapper, difficulty, score, mode, environment, modifiers, jumpDistance, leftHanded, height, startTime, failTime, speed |
| 1 | Frames | `n` × (time f32, fps i32, head, left, right), each as `pos xyz + quat xyzw` (7 × f32) |
| 2 | Notes | `noteID, eventTime, spawnTime, eventType`. Cut info follows for good/bad cuts: speedOK, directionOK, saberTypeOK, wasCutTooSoon, saberSpeed, saberDir, saberType, timeDeviation, cutDirDeviation, cutPoint, cutNormal, cutDistanceToCenter, cutAngle, beforeCutRating, afterCutRating |
| 3 | Walls | **wall hits only**, not the map's walls |
| 4 | Heights | auto-height changes. The Info `height` field was 0 in the sample, so we take height from here |
| 5 | Pauses | |
| 6 | Saber offsets | per-hand controller offset (pos + rot) |
| 7 | Custom data | |

`noteID = scoringType·10000 + lineIndex·1000 + lineLayer·100 + colorType·10 + cutDirection`.
Verified: `32018` → scoring 3, lane 2, layer 0, blue, dot.

### 6.2 Coordinate frame and timing (as stored)

- **Frame**: Unity world space as the game reports it. Left-handed axes,
  **+y up, +z forward (towards the incoming notes), +x right**. Units are
  metres. Quaternions are `(x, y, z, w)`. Sanity check from the sample: head
  mean ≈ (0.05, 1.48, 0.09) m; hands at y ≈ 0.5–1.7 m, x ≈ ±0.8 m.
- **Hands** are the saber/controller transforms as recorded. The section 6
  offsets are stored separately and **not applied**. Whether to apply them
  is a Phase 4 decision.
- **Time**: `frame.time` is song time in seconds, on the same clock as note
  `spawnTime`. The sample rate follows the headset: 90 Hz median on a
  Quest 3, with dt between 3.9 and 22.9 ms. Frames are stored with their raw
  timestamps; **no resampling in Phase 1**.
- We **do not convert axes in storage**. Any conversion (right-handed,
  head-relative, height-normalised) is a documented transform in Phase 4
  code. Storing the raw frame means a wrong convention guess can never
  corrupt the dataset.

### 6.3 Alignment check (per replay)

1. Decode each note event's `noteID` → `(x, y, color, cut_dir)`.
2. Match it to the map note with the same `(x, y, color, cut_dir)` whose
   `time_s` is nearest the event's `spawnTime`.
3. Record the residual `spawnTime − time_s` for every matched event.
4. Accept the replay if ≥ 98 % of events match and the p95 |residual| is
   ≤ 2 ms **[threshold tuned in pilot]**.
5. Rejected replays keep their row in the table with a reason. Most likely
   reasons: tempo-map bugs, a v2 BPM edge case, or `_songTimeOffset`.

This check doubles as the end-to-end test of §5.3. Replays get per-note links
to the map for free (`matched_note_idx`), which Phase 5 needs for swing →
note extraction.

### 6.4 Tables

`replays`
: `score_id, player_id, hash, characteristic, difficulty, leaderboard_id,
accuracy, rank, modifiers, pauses, full_combo, missed, bad_cuts,
platform, hmd, controller, tracking_system, game_version, bsor_version,
left_handed, jump_distance, height_median, n_frames, fps_median,
align_match_rate, align_p50_ms, align_p95_ms, status, reject_reason,
split (inherited from the map)`

`replay_frames` (one Parquet file per replay, under `hash=…/`)
: `t, fps, head_px..head_qw, left_px..left_qw, right_px..right_qw` (f32)

`replay_note_events`
: `score_id, note_id, event_type, event_time, spawn_time,
matched_note_idx, residual_ms`, plus cut info columns

`replay_aux`
: heights, saber offsets, wall hits

---

## 7. Storage layout

The data root comes from the `KATA_DATA` env var, which defaults to
**`E:\kata-data`** (D1). It is outside the repo. Any `data/` path inside the
repo is git-ignored as a safety net.

```
$KATA_DATA/
  raw/
    beatsaver/catalog/YYYY-MM-DD/page-*.jsonl.zst   # raw API pages, immutable
    maps/{hash}/Info.dat, *.dat, AudioData.dat, <song>.egg  # zip extracted; covers/videos dropped
    beatleader/leaderboards/{hash}.json.zst
    beatleader/scores/{leaderboard_id}.jsonl.zst
    replays/{hash}/{score_id}.bsor
  processed/
    metadata_catalog.parquet
    maps.parquet  difficulties.parquet  tempo_maps.parquet
    notes/ bombs/ obstacles/ arcs/ chains/        # partitioned by split
    replays.parquet  replay_note_events/  replay_aux/
    replay_frames/hash=*/{score_id}.parquet
  state/pipeline.duckdb                           # download/parse status per item, attempts, errors
  kata.duckdb                                     # views over processed/; disposable
  reports/phase1-YYYY-MM-DD.html|md
```

Principles:
- `raw/` is immutable and append-only.
- `processed/` can be rebuilt entirely from `raw/`, without network access.
- `kata.duckdb` holds only views.

Zips are extracted and then deleted. The zip's SHA is recorded, and the
`hash` (the BeatSaver version hash) still identifies the content.

**Size estimate (v1, K = 3)**

| Item | Estimate |
|------|----------|
| Maps (Tier A + B), ~14k × ~4 MB (audio dominates) | ~55 GB |
| Replays | ~30 GB |
| Processed data | ~20 GB |
| **Total** | **~105 GB** |

E: has 392 GB free, so Phase 1 fits with room to spare. Phase 2 (stems) is
the larger consumer, and its budget is set in the Phase 2 doc. The pipeline
checks free space before each download stage and stops below 50 GB free.

---

## 8. Implementation plan (after approval)

**Stack**: Python 3.13 (uv-managed; uv 0.12 is installed), `httpx` (async),
`duckdb`, `pyarrow`, `polars`, `pydantic` (raw-format models), `numpy`,
`soundfile` (Ogg/Vorbis decode for audio validation), `zstandard`, `typer`
(CLI), `pytest`.

Phase 0 is marked done, but the repo has no `pyproject.toml`, package
skeleton, `.gitignore` for data, or `CLAUDE.md` yet. That scaffolding is
step 1 here.

```
src/kata/
  data/
    http.py          # rate-limited client: token bucket per host, 429/Retry-After, backoff, UA
    beatsaver.py     # catalogue crawl, map download/extract
    beatleader.py    # leaderboards, scores, replay download
    select.py        # tier + replay selection SQL
    mapformat/       # v2.py v3.py v4.py → canonical.py; tempo.py
    bsor.py          # parser (streaming, numpy-backed frames)
    align.py         # replay ↔ map alignment
    build.py         # processed tables, splits, duckdb views
    report.py
  cli.py             # `kata data <stage>`
tests/
```

**Stages.** Every stage is idempotent and resumable through
`state/pipeline.duckdb`:

1. `crawl-catalog` — full BeatSaver crawl (~1.5k requests, ~10 min).
2. `select-maps` — tier assignment and report. **Checkpoint: review the
   counts and thresholds with you.**
3. `download-maps` — zips → `raw/maps/` (~14k files, ~1–2 h at 4 concurrent).
4. `parse-maps` — canonical tables, BeatSaver count cross-check, audio
   validation.
5. `crawl-leaderboards` — BL leaderboards and score pages for selected
   difficulties (~6k+ requests at 3 req/s, ~1 h).
6. `select-replays` → `download-replays` (~18k files, a few hours) →
   `parse-replays` → `align`.
7. `build` — splits, DuckDB views, `report`.

**Pilot first.** Before the full run, execute every stage on **100 Tier A +
100 Tier B maps** and their replays. The pilot:
- checks every **[verify in pilot]** item;
- measures real sizes and timings;
- tunes the alignment thresholds;
- surfaces format edge cases.

The full run starts only after you've seen the pilot report.

**Tests**
- Golden fixtures: the same small pattern hand-written in v2, v3 and v4 →
  byte-identical canonical output. Separate fixtures for multi-BPM v2 events,
  v3 `bpmEvents`, v4 AudioData, ME rejection and `_songTimeOffset`.
- BSOR: a small synthetic file built by a test helper (writer mirrors the
  reader) plus one real replay checked in (~1.7 MB, or truncated). The test
  parses all sections to EOF and checks the noteID decode.
- Alignment: on the real fixture, the match rate is 100 % and the p95
  residual is under the threshold.
- HTTP client: rate limiter and 429 handling against a mocked transport.
  No live-network tests in CI.

---

## 9. Acceptance criteria (definition of done)

1. The full catalogue is crawled and stored raw. `metadata_catalog` row count
   is within 1 % of BeatSaver's map count.
2. ≥ 99 % of selected difficulties parse. Every failure has a recorded
   reason, and the format split (v2/v3/v4) appears in the report.
3. Parsed object counts match BeatSaver's counts for 100 % of accepted
   difficulties. Anything that doesn't match is rejected, not accepted.
4. 100 % of stored audio decodes, and its duration is within 2 s of the
   metadata (or flagged).
5. *(Met in the pilot; for the full set it is checked when replays are fetched for Phase 4, see D9.)*
   ≥ 90 % of downloaded replays pass the alignment check. Rejections are
   broken down by reason.
6. No `song_group_id` appears in more than one split (automated check).
7. The report shows funnel counts per stage, plus distributions: NPS, NJS,
   stars, BPM, duration, format version, replays per player, HMD, and fps.
8. Example DuckDB queries run in under 5 s. One example: "ExpertPlus Tier A
   notes with their matched replay right-hand position at cut time".
9. Re-running any stage with no new input is a no-op (idempotency test).

---

## 10. Usage posture (D6)

- **No republishing**: the collected data (audio, map files, replays, and
  the raw API dumps) is not republished. This is consistent with the
  BeatSaver ToS ban on republishing users' content. Code, docs, trained
  models and generated maps are not affected by this rule.
- **No commercial use**: consistent with the BeatSaver ToS ban on
  commercialising site content. Phase 7 ("Product") is therefore read as
  free and non-commercial.
- **Generated maps** uploaded to BeatSaver must be declared as AI-generated
  (`declaredAi`), as the BeatSaver ToS requires.
- **Access**: we are polite clients. We stay well under the observed
  limits, send an identifying User-Agent with the repo URL (D5), and cache
  everything so we never re-fetch needlessly.
- **Players**: BeatLeader player ids and names are pseudonymous public data.
  We store only the ids that are needed for diversity capping and joins, and
  no names in processed tables.

---

## 11. Decision log

| Date | # | Decision |
|------|---|----------|
| 2026-10-09 | D1 | Data on `E:\kata-data` (392 GB free) |
| 2026-10-09 | D2 | Tier A + Tier B, labelled |
| 2026-10-09 | D3 | K = 3 replays per difficulty, raw BSOR kept |
| 2026-10-09 | D4 | `Standard` characteristic only |
| 2026-10-09 | D5 | User-Agent contact is the repo URL |
| 2026-10-09 | D6 | No republishing of site content, no commercial use; not restricted to local research |
| 2026-10-09 | D7 | All difficulties (Easy → Expert+) |
| 2026-10-09 | D8 | Tier B = curated AND rating > 0.95 AND upvotes > 100 (2,651 maps). Replaces the `rating ≥ 0.80 AND upvotes ≥ 30` OR curated placeholder (37k maps) |
| 2026-10-09 | D9 | Replays deferred to Phase 4; v1 has no replays |
| 2026-10-09 | — | Pilot items resolved (§12.2) |

---

## 12. Pilot findings (2026-10-09)

The pilot covered 100 Tier A + 100 Tier B maps, 454 difficulties and 1,340
replays. Full report: `E:\kata-data\reports\pilot-2026-10-09.md`.

### 12.1 Results against the acceptance criteria

| # | Criterion | Pilot result |
|---|-----------|--------------|
| 1 | Full catalogue crawled | 144,726 maps, 1,448 pages, raw 39 MB |
| 2 | ≥ 99 % of diffs parse | 454 / 454 (v2: 363, v3: 89, v4: 2) |
| 3 | Counts match BeatSaver | 454 / 454 exact |
| 4 | Audio decodes, duration within 2 s | 200 / 200, 0 mismatches |
| 5 | ≥ 90 % of replays pass | 1,256 / 1,340 = 93.7 % |
| 6 | No song group in two splits | Holds (asserted in `build`) |
| 9 | Idempotent | Re-runs skip completed items. `parse-*` and `build` rebuild deterministically |

**Alignment is exact.** Every right-handed replay matched 100 % of its notes.
The worst single residual was 0.039 ms; that is float precision, not a timing
error. This holds for every timing source:
- v2/v3 BPM events;
- v4 AudioData regions;
- 170 replays on diffs whose legacy `_customData._BPMChanges` we ignore.

That confirms the game ignores the legacy field. `_songTimeOffset` was 0 on
all pilot maps, so the "not applied" rule is still untested.

### 12.2 Resolved "[verify in pilot]" items

- **BeatLeader `count`**: returns > 100 per page. We use 100.
- **v4 AudioData keys**: `songFrequency` and `bpmData[{si, ei, sb, eb}]` are
  correct. Two v4 diffs aligned exactly, one of them with a BPM change.
- **Alignment thresholds**: 98 % match rate and p95 ≤ 2 ms are kept. Real
  residuals are about 1000× below the threshold, so it only catches gross
  errors.
- **Replay URLs**: the CDN pattern works for 57 %. The rest live at
  `api.beatleader.com/replays-storage/…`, and older ones have no score id
  prefix. The `/score/{id}` fallback handles them, at a cost of two extra API
  requests per replay.

### 12.3 Changes made during the pilot

- **Catalogue crawl**: pages can come back short (97 of 100) in the middle of
  the catalogue, so only an *empty* page ends the crawl. The first crawl
  stopped at 2021-02-22 because of this.
- **BSOR strings**: recorders before ~0.6 wrote string lengths as UTF-16
  unit counts but the bytes as UTF-8. Any non-ASCII player name, song or
  mapper misparses (24 / 1,340 replays). The reader now falls back to reading
  by UTF-16 units when the byte-length reading leaves an implausible next
  field. A test covers both encodings.
- **Pauses**: 129 of 135 replay pause entries fall within the first or last
  second; BeatLeader's `pauses = 0` doesn't count them. Only **mid-song**
  pauses (> 1 s from either end) reject a replay: 7 / 1,340.
- **Left-handed replays** (72 / 1,340 = 5.4 %) are mirrored, so their note
  ids never match the map. They stay rejected for v1. Mirrored alignment
  could recover them later.
- **Hash mismatches** (5 / 1,340): BeatLeader carried these scores over from
  another map version. They stay rejected.
- **Score pages**: per leaderboard we fetch the top half of scores, capped at
  500.
- **Table layout**: one Parquet file per table, not partitioned by split. The
  split lives on `maps` and other tables join to it. At this scale DuckDB
  doesn't need partitions.
- **Tests**: committed fixtures are synthetic. Real replays and map files are
  user content, so per D6 they are not committed. Real-data tests read from
  `KATA_DATA` and skip when it's absent.
- **Never-op filter**: `declaredAi <> 'None'` removed nothing beyond
  `automapper = false`. It is kept as a safeguard.

### 12.4 Scale and timing findings

These change the plan:

- **BeatLeader has a leaderboard for every map**, not just ranked ones.
  Tier B therefore gets replays too: 5.2 per map (Tier A: 8.3). At K = 3 on
  every selected diff, the full run is **~213k replays ≈ 330 GB raw**.
- **Tier B is 37k maps, not 10–14k.** BeatSaver ratings skew high; 21k
  non-ranked maps rate ≥ 0.90.
- **Map size**: Tier A averages 9.4 MB, Tier B 6.3 MB. All selected maps
  would take ≈ 255 GB.
- **Per replay**: BSOR 1.57 MB, parsed frames 1.24 MB.
- **Throughput**:
  - leaderboards + scores: ~7 requests per map, 3 req/s ⇒ ~2.3 s per map;
  - replays: ~0.23 s each;
  - maps: ~0.16 s each.

These findings led to decisions D8 and D9; see §13.

---

## 13. v1 scope after the pilot (D8, D9)

**D8 — Tier B rule.** A map is Tier B when it is curated **and** rated above
0.95 **and** has more than 100 upvotes. That yields 2,651 maps, against
36,973 under the placeholder rule. The rule drops about three quarters of
curated maps, mostly less-popular ones. We accept that in exchange for a
small, very clean v1. Widening Tier B later only means re-running `select`
and downloading the extra maps.

**D9 — Replays are deferred to Phase 4.** The v1 dataset is maps + audio
only:
- Tier A: 2,786 maps, ~26 GB.
- Tier B: 2,651 maps, ~16 GB.
- Total: **~5.4k maps, ~42 GB**, plus 1–2 GB of processed tables.

The replay stages (`crawl-leaderboards` → `select-replays` →
`download-replays` → `parse-replays`) are built, tested and piloted (§12).
They are simply not run for the full set yet. **Before Phase 4 starts**,
fetch Tier A replays with K = 3: ~23k replays, ~36 GB raw + ~28 GB parsed,
about 3–4 hours at our rate limits. At that point:
- decide whether Tier B gets replays too;
- decide whether to recover left-handed replays by mirroring (§12.3).

The pilot's replay tables (`*_pilot.parquet`) stay in `processed/` as a
reference sample.

Phases 2 and 3 need no replays. Phase 4 cannot start without them.

### 13.1 v1 build results (2026-10-09)

Report: `E:\kata-data\reports\phase1-v1-2026-10-09.md`.

**Coverage**
- **Maps**: 5,437 (A 2,786 · B 2,651). Every Info.dat and audio file is OK,
  and no audio duration differs from the metadata by more than 2 s.
- **Difficulties**: 15,182 / 15,188 accepted (99.96 %).
  - The 6 rejects are 5 undeclared Mapping Extensions diffs and 1 diff
    missing from its Info.dat.
  - Format mix: v2 10,488 · v3 4,615 · v4 79. 1,495 diffs have BPM
    changes.
- **Objects**: 13.5M notes, 0.73M bombs, 2.9M walls.
- **Splits** (by song group): train 4,926 · val 249 · test 262 maps.
  There are 4,832 song groups.
- **Disk**: 52 GB raw maps (above the 42 GB estimate; Tier A audio is
  larger than the pilot sample suggested), 1.8 GB processed. E: has
  336 GB free.

**Fixes found by the full run**
- **Extraction** now keeps every non-image, non-video file. One map names
  its difficulty files `*.ben`, and one names its song `.egg`, which has no
  suffix.
- **Arc and chain endpoints** may lie off the grid. Ranked maps do this, so
  it is vanilla. Notes stay strictly on-grid.
- **BeatSaver's obstacle count** sometimes repeats the note count, or
  notes + bombs. We accept the diff (with a warning) when notes and bombs
  match.
- **Audio fallback**: when Info.dat names a missing audio file, we use the
  folder's only audio file.

`_songTimeOffset` is 0 on all 5,437 maps, so the "not applied" rule never
matters in v1.

The 93 pilot Tier B maps that D8 deselected are still in `raw/maps`
(~0.6 GB). `parse-maps` ignores them.
