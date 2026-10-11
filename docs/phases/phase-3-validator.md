# Phase 3 — Validator + rule-based baseline

Status: **APPROVED 2026-10-10.** Prototype corrections in §17; pilot findings in §18.
Implementation plan: docs/phases/phase-3-validator-plan.md.
Last updated: 2026-10-11

## 0. Summary

Build two things and the harness that proves the first one works:

- **Validator**: a pure function that turns a map difficulty (notes, bombs,
  walls, chains) into a **pass/fail gate**, a **0–100 playability score**, a
  per-rule breakdown, a score-over-time curve, and a list of **per-swing
  issues**. At its core is a per-hand swing model: notes are grouped into
  swings, and each swing gets a forehand/backhand orientation chosen by
  dynamic programming. Every rule (parity, vision blocks, resets, swing
  angles, speed) reads that one swing table.
- **Baseline mapper**: audio features in (Phase 2), a playable notes-only map
  out, for all five difficulties. It uses onset picking, a tempo grid fitted
  to the audio beats, and parity-correct alternating patterns built on the
  validator's own swing model. Exported as v3 map folders and zips that
  ArcViewer and the game can load. It is the bar that the Phase 4/5 models
  must beat.
- **Calibration**: show that the score means something. Ranked maps (Tier A)
  must score clearly above a new set of low-rated maps (Tier D), **within
  strata** of upload year, difficulty and density. Each rule must find the
  issues injected into otherwise clean ranked maps.

Everything below that says "measured" was read from
`E:\kata-data\processed\*.parquet` on 2026-10-10. Rule definitions cite the
community tool or ranking criterion they come from (§16). Anything not yet
verified is marked **[verify in pilot]**; numbers to be tuned are marked
**[tune in pilot]**.

### Decisions (agreed 2026-10-10)

| # | Decision | Outcome | Rejected alternative |
|---|----------|---------|----------------------|
| D1 | Calibration negatives | **Tier D (low-rated, ~2,000 maps, audio dropped) + synthetic corruptions of Tier A maps** | Low-rated only (noisy); synthetic only (never meets real bad maps); Tier A vs. B (measures style, not playability) |
| D2 | Reference for load rules | **Tier A train, per stratum = difficulty label × 2-NPS band** | Star bands (generated maps have no stars); absolute physical limits (too lax for Easy or too strict for Expert+) |
| D3 | Baseline scope | **Colour notes only, Easy → Expert+** | Expert+ only; notes + heuristic bombs/walls |
| D4 | Validator architecture | **Swing model first, rules on top** | Independent per-rule checks on raw notes; learned scorer (deferred to Phase 6 as a critic) |
| D5 | Score | **Weighted geometric mean of per-rule sub-scores, equal weights; weights change only by your approval** | Arithmetic mean (one broken aspect hides behind good ones); automatically fitted weights (would learn the era confound) |
| D6 | Hard gate | **Separate `pass` flag; the score is still computed for gated maps** | Gated maps score 0 (Phase 5 couldn't rank failed attempts) |
| D7 | Orientation | **Charitable dynamic programming per hand** (the way an expert would play it) | Greedy state machine like BeatSaver's checker (false errors on dots) |
| D8 | Export | **v3.2.0 difficulty files + v2.1.0 `Info.dat`, folder + zip** | v4 (ScoreSaber can't rank v4 yet; fewer tools read it); v2 (no `bpmEvents`) |

---

## 1. Goals and non-goals

**Goals**
1. A deterministic, fast validator usable in-process by Phase 5 (validator
   in the loop) and as a feature source for a Phase 6 critic.
2. Every judgement explainable: each lost point traces back to named rules
   and specific swings.
3. A calibrated score: ranked maps score clearly above low-rated ones within
   comparable strata, and each rule demonstrably detects what it claims to.
4. A baseline mapper that produces playable maps for all five difficulties
   from audio alone, previewable in ArcViewer.
5. A report regenerated on each build, like Phases 1 and 2.

**Non-goals (v1)**
- Rules for pre-swing path collisions and hand crossovers (BL R3B), hitbox
  overlap (R3C), wall dodge timing (R8), intro/outro lengths, and lighting.
  The baseline produces none of these objects and Phase 5 starts with notes.
  They become a v2 rule set if Phase 5 playtests show they matter.
- Arcs (cosmetic, ignored). Chain links (only the head counts as a swing).
- A learned scorer (D4) → Phase 6.
- Bombs, walls, dots and deliberate resets in the baseline (D3).
- Replays. They are still deferred to Phase 4 (phase-1-data.md D9). Real
  hand speeds from replays could later calibrate the speed rule in physical
  units; v1 uses grid units.
- Uploading generated maps anywhere (§13).

---

## 2. Inputs (measured)

### 2.1 Existing data

| Item | Value |
|------|-------|
| Parsed difficulties | 15,182 (Tier A 7,463 · B 7,459 · C 260). "C" here means the unranked difficulties of ranked maps |
| Format | v2 10,488 · v3 4,615 · v4 79 |
| Notes / bombs / walls | 13.5M / 0.73M / 2.9M |
| Audio features | 5,347 songs (Phase 2 v1): onset curves per stem, Beat This beats/downbeats, tempo, segments |

**Tier A NPS by difficulty label** (BeatSaver `nps`; p10 / median / p90; median NJS):

| Label | n | p10 | median | p90 | NJS |
|-------|---|-----|--------|-----|-----|
| Easy | 652 | 1.29 | 2.40 | 4.02 | 12 |
| Normal | 947 | 1.87 | 3.45 | 5.38 | 14 |
| Hard | 1,480 | 2.69 | 4.37 | 6.99 | 16 |
| Expert | 1,889 | 3.60 | 5.90 | 8.95 | 18 |
| Expert+ | 2,495 | 4.81 | 7.66 | 10.97 | 20 |

Tier B is much sparser at the same label (Expert median 3.98, Expert+ 5.00).
This is why load rules are referenced by NPS band, not label alone (D2).

### 2.2 BeatSaver's parity summary is v2-only

BeatSaver computes `paritySummary` only for v2 difficulties; v3 and v4 always
report zeros (§16 [1–3]). Split by format:

| Group | Format | Diffs | Parity errors > 0 | Resets > 0 |
|-------|--------|-------|-------------------|------------|
| Tier A | v2 | 5,277 | 24.6 % | 55 % |
| Tier A | v3/v4 | 2,186 | (not computed) | (not computed) |
| Tier B | v2 | 4,971 | 44.8 % | 29 % |
| Low-rated (< 0.5, ≥ 20 votes) | mostly v2 | 5,167 | 93 % | 38 % |

Consequences:
- A quarter of ranked v2 maps have at least one BeatSaver parity "error", so
  a strict, greedy parity rule would fail maps that humans ranked. Our
  parity model reads maps charitably (D7) and its tolerances are calibrated
  on Tier A.
- BeatSaver agreement is a **diagnostic on v2 maps only**, never a target.

### 2.3 Low-rated pool (for Tier D)

Catalogue maps that pass Phase 1's map-level hard filters, are neither ranked
nor curated, are not already selected, and have ≥ 20 votes:

| Rating | Maps | Standard diffs passing §3.2 filters | Median votes | Median upload year |
|--------|------|-------------------------------------|--------------|--------------------|
| < 0.5 | 4,059 | 5,167 | 52 | 2019 |
| 0.5–0.6 | 1,617 | 2,099 | 37 | 2020 |

Tier A's median upload year is **2022**. The low-rated maps are older, so era
(and with it format and conventions) is a confound that §4 controls for.

---

## 3. Swing model

All rules read the swing table; the baseline mapper builds on the same code.

### 3.1 Objects and directions

- **Read**: colour notes, bombs, walls, chain heads (chain links ignored).
  Arcs are ignored.
- **Cut angle** of a note: `cut_dir` mapped to degrees, plus `angle_offset`.
  We use JoshaParity's convention (§16 [5–7]): for a right-hand forehand,
  down = 0°, left = −90°, right = +90°, up = −180°; the left hand mirrors.
- Dots carry no angle until §3.3 assigns one.

### 3.2 Swing formation (per colour)

Notes are sorted by `time_s`. A note joins the current swing if it arrives
within **70 ms** of the previous note of that colour (MapCheck's window;
JoshaParity uses 59 ms), **unless** it comes more than 5 ms later with an
arrow at least 90° away from one already in the swing. That is a fast stream,
not a slider, and starts a new swing (MapCheck's split rule; §17).

Same-time arrows (within 5 ms) always stay in one swing, so a stack or
window whose arrows disagree is a rule issue (H3/E3), never two swings.

Each swing records:

| Field | Meaning |
|-------|---------|
| `t_start`, `t_end` | first and last note time |
| `kind` | `single`, `stack` (same time, adjacent cells), `window` (same time, gap > 1 cell), `slider` (different times), `chain` (JoshaParity's classes) |
| `angle` | mean arrow angle of the swing (circular mean); undefined if all dots |
| `spread` | largest pairwise arrow angle difference within the swing |
| `entry`, `exit` | first and last cell in path order (sorted along the cut direction) |
| `note_idx[]` | indices of the member notes in the `notes` table |

### 3.3 Orientation by dynamic programming (per hand)

Each swing gets a **parity** (forehand or backhand) and, for all-dot swings,
a **direction** from the 8 options. Together these give the wrist **rotation**
ρ (the cut angle measured from that parity's neutral). The assignment
minimises total cost over the map with a Viterbi pass (≤ 16 states per
swing):

- **Strain**: `(ρ / 180)²` per swing.
- **Wrist limits**: rotations outside the hand's range are infeasible. Red
  −155° … +195°, blue −195° … +155° (MapCheck, §16 [12]; consistent with
  ScoreSaber's pronation 90–100° / supination 180–210°, §16 [14]).
- **Rotation change**: `w_rot · (Δρ / 180)²` between consecutive swings,
  `w_rot = 1`. A change of 180° or more also costs `w_unjust`.
- **Resets**: two consecutive swings of one hand with the same parity cost
  `w_unjust = 10` unless justified; a justified reset costs `w_slow = 0.5`
  (slow) or 0 (bomb) **[tune in pilot]**.
- **Bombs** (bs-parity and MapCheck rule, §16 [2, 4, 12]): the last bomb in
  the hand's own middle column (x = 1 for red, x = 2 for blue), top or bottom
  row, **between the hand's previous swing and this one**, forces this
  swing's orientation. A bottom bomb lifts the hand over it, so the next
  swing is a forehand (downward); a top bomb forces a backhand. (bs-parity's
  ¼ beat is a skip condition, not the window; §17.)

This is the approach of BeatLeader's analyser (§16 [19]), not BeatSaver's
greedy checker. It reads a map **charitably**, the way an expert would play
it, so an ambiguous dot never creates a false error.

**Reset labels.** After the DP, each same-parity pair is labelled:
- `bomb`: forced by a bomb (legitimate);
- `slow`: the gap from the previous swing's end is at least `T_reset`;
- `unjustified`: anything else. This is a parity error (E1).

`T_reset` depends on the difficulty label: the 5th percentile, over all Tier
A difficulties, of the gap between consecutive same-hand down-arrows
(measured 2026-10-10, §17):

| Easy | Normal | Hard | Expert | Expert+ |
|------|--------|------|--------|---------|
| 0.60 s | 0.47 s | 0.36 s | 0.30 s | 0.20 s |

### 3.4 Reaction time

The vision rule needs the game's half jump duration (HJD). Computed exactly as
the game does, from the Info BPM, NJS and NJS offset:

```
hjd = 4
while njs * (60 / bpm) * hjd > 17.999: hjd /= 2
hjd = max(hjd + njs_offset, 0.25)          # beats
reaction_time_s = hjd * 60 / bpm
```

---

## 4. Rules

### 4.1 Rule set

| Id | Rule | Definition | Kind |
|----|------|------------|------|
| H1 | Overlap | In one cell: a note and another object (note or bomb) whose depth separation `Δt · NJS` is < 0.5 m (BL R3A, §16 [15]); or two bombs less than 20 ms apart (ScoreSaber; bomb tunnels are legal, §17). NJS 0 means the game's default for the label | **hard** |
| H2 | Note in wall | A note whose cell and time fall inside a wall's span (cells `x … x+w−1`, rows `y … y+h−1` clipped to 0–2, time `[t, t+duration]`). Walls with width or duration ≤ 0 are ignored (BL R3A, SS) | **hard** |
| H3 | Impossible swing | One swing with `spread ≥ 90°` | **hard** |
| E1 | Parity break | An `unjustified` reset; or a rotation change of 180° or more between consecutive swings (BL AutoModder's parity-warning angle, §16 [9]); or an inverted swing, \|ρ\| > 135° (AutoModder, §16 [9]). See §17 for why this replaces JoshaParity's > 270° | error |
| E2 | Vision block | **Blocker**: a note or bomb at x ∈ {1, 2}, y = 1. **Blocked**: a later note on the blocker's half (x ≤ 1 for a blocker at x = 1, x ≥ 2 for x = 2), not in the same swing, arriving within `(t_min, t_max]` of the blocker. `t_max = min(reaction_time_s, cap)`. Bomb blockers use `t_min = 200 ms` (BL AutoModder, §16 [9, 16]) | error |
| E3 | Swing spread | `45° < spread < 90°` within one swing (SS same-snap rule, BL R4) | error |
| L1 | Speed | Per swing, `v = (1 + d) / Δt`, where `d` is the grid distance from the previous same-hand swing's exit to this swing's entry and `Δt` is start to start. Metric: the peak, over the map, of mean `v` across any 8 consecutive swings of one hand (a sustained-burst measure, after BL's pass rating, §16 [19]) | load |
| L2 | Angle strain | Per swing, `(Δρ / 180)²`, where Δρ is the rotation change from that hand's previous swing (BL tech rating, §16 [19]). Metric: the mean of the top 5 % of swings (at least one). A p95 collapsed onto the 45° steps, §17 | load |
| L3 | Reset density | `bomb` + `slow` resets per minute of mapped time | load |

**E2 windows by difficulty** (seconds, MapCheck, §16 [10]):

| Difficulty | `t_min` | `cap` |
|------------|---------|-------|
| Easy | 0.025 | 1.2 |
| Normal | 0.05 | 1.0 |
| Hard | 0.08 | 0.75 |
| Expert | 0.1 | 0.625 |
| Expert+ | 0.1 | 0.5 |

Hard rules feed the gate (§5.4). Error and load rules feed the score.

### 4.2 Per-map metrics

- **Error rules** (E1–E3): issues per 100 swings. These are absolute: a parity
  break is a parity break at any difficulty. Only the tolerance is learned,
  from all Tier A train maps pooled.
- **Load rules** (L1–L3): the tail statistic in the table, referenced to Tier
  A train **in the same stratum**. Load rules are **one-sided**: a map that is
  easier than ranked maps is not penalised.

### 4.3 Strata

- **NPS** for stratification is computed by us, identically for every map:
  `n_notes / (t_last_note − t_first_note)`. BeatSaver's `nps` is not used,
  because generated maps don't have one.
- **Stratum** = difficulty label × NPS band, bands 2 NPS wide (0–2, 2–4, …).
- A cell with fewer than **100** Tier A train maps merges with its nearest
  neighbouring band in the same label. If the label as a whole has fewer than
  100, the stratum falls back to the label alone.
- The final strata and their sizes are written to `reference.json` and the
  report.

---

## 5. Scoring

### 5.1 Sub-scores

Each scored rule *r* gets `s_r ∈ [0, 1]`, piecewise linear in its metric `m`,
anchored on the rule's reference quantiles (§4.2):

| Metric `m` | `s_r` |
|------------|-------|
| ≤ q90 | 1.0 (typical for ranked maps) |
| = q99 | 0.5 (as bad as the worst 1 % of ranked maps) |
| ≥ q99 + 2 · (q99 − q90) | 0.0 (extrapolated, so terrible maps still rank below merely bad ones) |

Linear in between **[tune anchors in pilot]**.

For error rules most ranked maps have zero issues, so q90 and q99 can
collapse. Anchors therefore have minimum spacing: q90 ≥ 0.2 and
q99 ≥ q90 + 1.0 issues per 100 swings **[tune in pilot]**.

### 5.2 Combination

```
score = 100 · Π_r max(s_r, 0.01) ^ w_r,   Σ w_r = 1
```

A weighted **geometric** mean: one badly broken aspect can't hide behind
perfect ones. Weights start **equal** (1/6 each for E1–E3, L1–L3). The pilot
and the report show each rule's own AUC (Tier A vs. Tier D). A weight change
is proposed to you, never fitted automatically (D5).

### 5.3 Local scores

The same sub-score maths runs on sliding windows of **16 map beats** (4 bars
of 4/4 in the map's own grid), stepped by 4 beats, giving a score-over-time
curve. Windows with fewer than 8 swings are skipped. Phase 5 uses this to
regenerate only the bad sections.

Windows use map beats, not audio bars, because Tier D maps have no audio
features (§6.1). For baseline output the two coincide (§8.1).

### 5.4 Gate

`pass = false` if any hard rule (H1–H3) fires. The score is computed
regardless (D6), so Phase 5 can rank failed attempts.

### 5.5 Public interface

```python
validate(MapInput(difficulty, njs, njs_offset, bpm, notes, bombs, walls, chains),
         reference) -> Result
# Result: passed, score, sub_scores{rule: s}, metrics{rule: m}, stratum,
#         swings (DataFrame), issues (DataFrame), local (DataFrame)
```

Pure and deterministic; it takes canonical Phase 1 frames and does no I/O.
Phase 5 calls it in-process. The pipeline stages (§10) are thin wrappers
around it.

---

## 6. Calibration

### 6.1 Tier D — low-rated maps (D1)

- **Rule**: rating < 0.6, ≥ 20 votes, `curated_at IS NULL`, not ranked on
  either leaderboard, passes Phase 1's map hard filters (§3.1 there), has at
  least one difficulty passing its difficulty filters (§3.2 there), and is
  not already selected. Pool: 5,676 maps (§2.3).
- **Sample**: 2,000 maps, seeded. Each upload year gets
  `2000 × (Tier A's share of maps from that year)` slots. Years where the
  pool is short contribute everything they have, and the leftover slots are
  shared proportionally among the other years. The report shows the
  achieved year mix next to Tier A's.
- **Download and parse** reuse Phase 1 (`download_map`, `extract_map`,
  `_parse_one_map`). After extraction, **audio files are deleted**; only
  `Info.dat`, difficulty files and `AudioData.dat` (v4 timing) are kept.
  `audio_status = 'dropped'`. The audio duration check is skipped for
  Tier D. Expected size: well under 1 GB.
- **Storage**: separate Parquet files under `processed/lowrated/` (`maps`,
  `difficulties`, `notes`, `bombs`, `obstacles`, `arcs`, `chains`,
  `tempo_maps`), written by the same parser code. No existing consumer
  (Phase 2, a future Phase 4 loader) can see Tier D by accident.
  `kata data build` adds them as `lowrated_*` views.
- **Halves**: song group key = Phase 1's `normalise_title` of
  (author, title). `hash(key) mod 2` gives `fit` (used for pilot tuning) and
  `holdout` (used only for the reported results).
- Tier D is **calibration only**. It is never used for training and never
  gets audio features.

### 6.2 What is fitted where

| Data | Used for |
|------|----------|
| Tier A **train** | `reference.json` quantiles; `T_reset`; baseline NPS and gap targets |
| Tier D **fit** half | pilot tuning (merge window, anchors, DP costs) |
| Tier A **val + test**, Tier D **holdout** | reported AUCs and acceptance criteria |
| Tier A **val + test** | synthetic corruptions |
| Tier B | reported only (a "good but unranked" reference point) |

Nothing reported is tuned on.

### 6.3 Synthetic corruptions

Applied to Tier A val/test difficulties. Each corruption is injected at 5
random swings per difficulty (seeded), one at a time, so each corrupted copy
has exactly one known defect.

| Id | Corruption | Expected rule |
|----|------------|---------------|
| C1 | Rotate the arrow of a single-note swing by 180°, where the gaps to the hand's neighbouring swings are both < `T_reset` | E1 |
| C2 | Insert a bomb at (1 or 2, 1), middle row, 250 ms before a note on the same half. Middle-row bombs do not affect parity (§3.3) | E2 |
| C3 | Duplicate an outer-lane note as a bomb in the same cell, 10 ms later | H1 |
| C4 | Time-compress a random 8-swing run of one hand until its mean speed is 1.5× the stratum's L1 q99. Runs that are already that fast, or whose gaps would shrink below 75 ms (slider range), are not eligible | L1 |
| C5 | Zig-zag the arrows of 4 consecutive single-arrow swings of one hand: +90°, −90°, +90°, −90° | L2 |
| C6 | Rotate one arrow of a same-time stack or window by 90° | H3 |
| C7 | Turn one arrow of a same-time stack or window by 60° (`angle_offset`) | E3 |

C4 is sized from the reference because L1 only claims to flag runs beyond
ranked maps' q99; a fixed 2× often stays inside that range (§17). C6/C7 use
same-time swings only: under §3.2 a turned slider note becomes a separate
swing.

**Detection**:
- **Hard and error rules**: an issue of the expected rule, on the same hand,
  within ±1 swing of the injection (for C4 and C5: up to the end of the
  changed run).
- **Load rules**: that rule's local sub-score (§5.3) drops below 0.5 in a
  window containing the injection. For C5, L3 there or an E1 issue nearby
  also counts: the DP reads a fast zig-zag as rotation strain or a parity
  break, and a slower one as resets (§17).

**False alarms**: issues in the corrupted copy that are absent from the clean
original and not within ±1 swing of the injection, divided by the number of
swings.

---

## 7. Success criteria at a glance

The full list is in §11. In short: the gate passes ranked maps, the score
separates Tier A from Tier D *within strata*, every rule catches its injected
defects, and the baseline is playable.

---

## 8. Baseline mapper

Inputs are audio features only (Phase 2) plus a target difficulty. It never
sees a human map, matching what Phase 5 has at inference time.

### 8.1 Tempo grid

1. Take Beat This beats `b_j` for the song. Assign integer beat indices
   step by step: `k_j = k_{j−1} + max(1, round((b_j − b_{j−1}) / median_IBI))`,
   so a missed beat leaves a gap. (Rounding `(b_j − b_0) / median_IBI`
   instead drifts by whole beats over a long song; §17.)
2. Fit `t = t0 + k · T` by least squares.
3. If the p95 |residual| exceeds **30 ms** (unsteady tempo), fit
   piecewise, one line per structure segment (`audio_segments`; segments
   with fewer than 4 beats merge into the next), with a `bpmEvent` wherever
   the tempo changes. On 300 songs, 44 % are steady; Beat This's 20 ms
   frames alone give 15–30 ms residuals on steady songs (§17).
4. Map beat 0 is always audio time 0 in v3. Map beat 1 is the first fitted
   audio beat at least half a period after time 0, so the **lead-in** (map
   beats 0 → 1) lasts 0.5–1.5 periods and its BPM stays within ×0.67–×2 of
   the main tempo. Every later fitted audio beat lands on an integer map
   beat.
5. The Info BPM is the fitted main tempo. Whether Beat This picked the half
   or double tempo does not matter for placement, but it sets the NJS/offset
   arithmetic, so it is recorded.

### 8.2 Note times

1. **Candidate slots**: every ¼ fitted beat (every ⅛ when the fitted tempo is
   below 110 BPM, so slow songs can reach Expert+ density) from the first to
   the last slot whose strength is at least 0.2 × the maximum, at least 1.5 s
   after the start and 2 s before the end (BL setup rules, §16 [15]).
2. **Slot strength**: the per-song normalised (÷ p99) onset curves, combined
   as `1.0·drums + 0.5·mix + 0.3·(bass + other + vocals)`, max-pooled over
   ±30 ms around the slot **[tune in pilot]**. Multiplied by a loudness boost
   `clip(1 + 0.5 · z_rms, 0.5, 1.5)`, where `z_rms` is the section's mean
   `rms_mix` z-scored within the song, so drops get denser.
3. **Target count**: Tier A train's median NPS (§4.3 definition) for the
   requested label, times the mapped duration.
4. **Selection**: greedy by strength, subject to a minimum gap between
   consecutive notes of one hand: the median, over Tier A train maps in the
   target stratum, of each map's 5th-percentile same-hand swing interval.
   Since hands alternate, consecutive selected notes are at least half that
   gap apart. The baseline is therefore never faster than a typical ranked
   map.

### 8.3 Hands, directions, positions

- **Hands** alternate by default. A slot within 30 ms of an audio downbeat
  whose strength is at least 2× the median strength of the selected slots
  becomes a **double** (both hands). Doubles happen on Hard and above only,
  only if both hands are due the same parity, and only if both hands have
  been free for the minimum gap and the next note is that far away. Doubles
  never use crossovers, so the hands never share a cell.
- **Directions**: each swing takes the parity opposite to that hand's
  previous swing (forehand ↔ backhand), via the validator's swing model. The
  angle is drawn from a small vocabulary: the straight direction, or a
  diagonal within ±45° of it. Easy and Normal use straight directions only.
- **Positions**: a flow table.
  - Forehand (downward) swings land in row 0, or row 1 with probability 0.3.
  - Backhand (upward) swings land in row 2, or row 1 with probability 0.3.
  - Red uses lanes 0–1, blue lanes 2–3. Row 1 uses only the outer lane
    (red 0, blue 3), so the baseline never places a vision blocker.
  - Crossovers (red into lane 2, blue into lane 1, never in row 1) only on
    Expert and up, with probability 0.05.
- **Self-check**: the finished difficulty is validated. The notes of every
  swing with a hard or error issue are dropped and the difficulty is
  re-validated, up to 3 rounds. Alternation, the direction vocabulary and the
  flow table make issues rare by construction, so a post-hoc repair replaces
  the per-placement retry of the draft (§17).
- **Deterministic**: seeded by `audio_sha256` and difficulty.
- **No dots, bombs, walls, arcs, chains or deliberate resets** (D3).

### 8.4 NJS and offset

NJS is Tier A train's median for the label (§2.1). The NJS offset is chosen
so that the reaction time (§3.4) equals Tier A train's median reaction time
for the label.

### 8.5 Export (D8)

```
$KATA_DATA/generated/baseline/v1/{sha}/
  Info.dat               # v2.1.0
  song.egg               # copy of the source audio
  cover.png              # 256×256 solid colour, written with the stdlib (zlib + struct)
  EasyStandard.dat … ExpertPlusStandard.dat   # v3.2.0
$KATA_DATA/generated/baseline/v1/{sha}.zip
```

- **`Info.dat`** carries every field BeatSaver's v2 schema requires (§16 [25]):
  - `_version`, `_songName`, `_songSubName`, `_songAuthorName`;
  - `_levelAuthorName = "kata-baseline v1"`;
  - `_beatsPerMinute`, `_songTimeOffset = 0`, `_shuffle`, `_shufflePeriod`;
  - `_previewStartTime` (30 % of the duration), `_previewDuration`;
  - `_songFilename`, `_coverImageFilename`;
  - `_environmentName = "DefaultEnvironment"`;
  - one `Standard` set with five difficulties, each with
    `_difficultyRank` 1/3/5/7/9, `_beatmapFilename`, `_noteJumpMovementSpeed`
    and `_noteJumpStartBeatOffset`.

  Song metadata comes from the Phase 1 `maps` row for that audio.
- **Difficulty files** write **every** top-level v3.2 key that BeatSaver's
  validator requires (§16 [26]). They are empty except `colorNotes` and
  `bpmEvents`; `useNormalEventsAsCompatibleEvents` is `true`. 3.2.0 avoids
  3.3's unverified fx-collection key spelling.
- **Round trip**: every export is re-parsed by Phase 1's parser. The
  canonical notes (beat, x, y, colour, direction) must be identical, and
  `time_s` must be within 0.1 ms of the intended time.
- **Preview**: the web version of ArcViewer loads **zips** through its file
  picker; drag-and-drop is not supported as far as its source shows. The
  desktop build also opens unzipped folders and reloads them with R
  (§16 [22, 23]).

### 8.6 Evaluation

Run on every **val + test** song (the song-group splits from Phase 1), all
five difficulties:

- **Gate and score** against the Tier A maps of the same songs, compared
  within strata.
- **Timing agreement**: F-measure at ±30 ms between baseline note times
  (unique times, both colours) and the Tier A map's note times for the same
  song and label. No target; it is the number Phase 5 must beat ("does it
  hit what humans hit").
- **Manual preview**: 5 songs across genres and tempos, picked in the
  pilot, loaded in ArcViewer by you, and in-game if you want.

The baseline's self-check uses the validator, so its validator score is
**not** independent evidence of quality (§14). Timing agreement and your
preview are the independent checks.

---

## 9. Storage layout

```
$KATA_DATA/
  raw/maps/{hash}/                         # Tier D maps too (audio deleted)
  processed/
    lowrated/{maps,difficulties,notes,bombs,obstacles,arcs,chains,tempo_maps}.parquet
    validation_metrics.parquet             # one row per difficulty: tier, split, year, status, every rule metric, nps, njs, reaction time
    validation_swings.parquet              # one row per swing (§3.2 fields + parity, rotation, reset label, speed, strain)
    validation_issues.parquet              # hash, characteristic, difficulty, rule, severity, hand, swing_idx, time_s, value, threshold
    validation_local_metrics.parquet       # per window: hash, characteristic, difficulty, window_start_beat, n_swings, rule metrics
    validation.parquet                     # validation_metrics + stratum, s_<rule>, score, pass, validator_version (§5.5)
    validation_local.parquet               # validation_local_metrics + stratum, s_<rule>, score
    corruption_results.parquet             # hash, characteristic, difficulty, corruption, rule, i, eligible, detected, false_alarms, n_swings
    baseline_eval.parquet                  # audio_sha256, split, difficulty, n_notes, nps, pass, score, stratum, s_<rule>, timing_f, grid_bpm, piecewise, round_trip
  validator/
    v1/
      config.json                          # every parameter in §3–§5, plus rule list and weights
      reference.json                       # Tier A train quantiles per rule and stratum, stratum sizes
  generated/baseline/v1/{sha}/ , {sha}.zip
  reports/phase3-YYYY-MM-DD.md
```

`validation*.parquet` carry `validator_version`. As in Phase 2, a changed
parameter means a new version directory: the stage refuses to write a
different config into an existing one, so scores never silently change
meaning.

**Size**: the swing table is ~10M rows (13.5M notes plus Tier D, minus
merges), well under 1 GB. Generated maps are a few MB each, mostly the copied
audio: ~511 songs ≈ 3 GB **[verify in pilot]**.

---

## 10. Implementation plan (after approval)

**Stack**: no new dependencies. NumPy for the DP and rule maths, Polars and
DuckDB for tables, the stdlib for the PNG cover. All CPU.

**Code layout**

```
src/kata/validate/
  swings.py      # notes → per-hand swings; DP orientation (§3)
  rules.py       # MapInput, analyse(): swing table, issues, per-map and per-window metrics (§4, §5.3)
  reference.py   # strata, Tier A train quantiles, reference.json, pinned config.json (§4.2–4.3)
  score.py       # sub-scores, geometric mean, gate, validate() (§5)
  corrupt.py     # C1–C7 with injected-issue ground truth (§6.3)
  pipeline.py    # stages below
  report.py
src/kata/baseline/
  grid.py        # tempo fit → bpmEvents, lead-in, slots (§8.1)
  onsets.py      # slot strengths, greedy selection (§8.2)
  patterns.py    # hands, directions, positions (§8.3)
  generate.py    # one song → five validated difficulties, repair (§8)
  export.py      # Info.dat, v3 .dat, cover, zip, round trip (§8.5)
  pipeline.py    # `generate` stage (§8.6)
src/kata/data/lowrated.py   # Tier D select / download / parse (§6.1)
cli.py           # `kata validate <stage>`, `kata baseline generate`, `kata data *-lowrated`
```

Phase 1 changes: extraction can drop audio; the catalogue filter and the map
parser are factored out (`_filter_catalog`, `_parse_selection`) so Tier D
reuses them with identical output for Tiers A–C (checked on real maps, §17);
`build` adds `lowrated_*` views. `Paths` gains `lowrated`, `validator` and
`generated`. Nothing else in Phases 1 and 2 changes.

**Stages** (each idempotent: download is resumable through
`state/pipeline.duckdb`, the others rebuild their tables deterministically):

1. `kata data select-lowrated` → `download-lowrated` → `parse-lowrated`.
   **Checkpoint: show you the Tier D year mix and parse results.**
2. `kata validate analyse` → `validation_metrics`, `validation_swings`,
   `validation_issues`, `validation_local_metrics` for Tiers A–D (85 s for
   all 15.4k difficulties, §17).
3. `kata validate reference` → `validator/v1/reference.json` (Tier A train
   only).
4. `kata validate score` → `validation`, `validation_local`.
5. `kata validate corrupt` → `corruption_results`.
6. `kata baseline generate` → map folders and zips for val + test songs;
   `baseline_eval`.
7. `kata validate report` → `reports/phase3-YYYY-MM-DD.md`.

**Pilot first**, as in Phases 1 and 2: `kata validate pilot` runs the whole
chain with 200 Tier D maps, corruptions on 100 Tier A val/test difficulties
and the baseline on the pilot's val/test songs. Analysis always covers every
parsed difficulty, because it is cheap. The pilot:
- resolves every **[verify in pilot]** item and tunes every
  **[tune in pilot]** value, on Tier A train and Tier D fit only;
- shows per-rule AUCs and the score distributions per tier;
- shows the 5 baseline songs for your ArcViewer check.

The full run starts only after you've seen the pilot report.

**Tests** (TDD; synthetic fixtures only; real-data tests skip without
`KATA_DATA`, as before):
- `swings`: hand-written patterns give the expected grouping and `kind`
  (stack, window, slider, chain head); the 70 ms boundary; dots get a
  direction that continues the flow; a bottom bomb forces a backhand; a
  same-parity pair is labelled `bomb`, `slow` or `unjustified` correctly;
  the DP respects the wrist limits.
- `rules`: one minimal positive and one minimal negative fixture per rule
  (H1–H3, E1–E3, L1–L3), including E2's same-half rule, the bomb-blocker
  minimum and the reaction-time cap.
- `reference` / `score`: anchor maths (1.0 at q90, 0.5 at q99, 0 at the
  extrapolated point); collapsed-anchor spacing; the 0.01 floor; stratum
  merging and fallback; local windows skip below 8 swings.
- `corrupt`: each of C1–C7 on a clean fixture is detected at the injected
  swing, with zero false alarms.
- `baseline`: the tempo fit recovers a synthetic click track's BPM and
  phase within 1 ms, including the lead-in; the minimum-gap rule holds;
  output passes the gate; the v3 export round-trips through the Phase 1
  parser.
- `pipeline`: re-running a stage with no new input is a no-op; a changed
  config refuses to write into an existing version directory.

**Compute**: CPU only. ~18k difficulties; the DP is O(swings × 16²) per
hand. Expected minutes per stage with 8 workers **[verify in pilot]**.

---

## 11. Acceptance criteria (definition of done)

1. **Tier D**: ≥ 1,900 of the 2,000 sampled maps downloaded and parsed, with
   ≥ 99 % of their difficulties parsing (Phase 1's bar). Audio deleted;
   Tier D under 1 GB. Year mix shown next to Tier A's.
2. **Coverage**: swings, scores and issues for 100 % of parsed Tier A, B, C
   and D difficulties, or a recorded failure reason.
3. **Gate**: ≥ 99 % of Tier A val + test difficulties pass.
4. **Separation**: score AUC, Tier A val + test vs. Tier D holdout,
   ≥ **0.85** overall. Also reported **per stratum** of upload year (≤ 2019,
   2020–21, 2022–23, ≥ 2024) × NPS (0–4, 4–6, 6–8, 8+), for every cell with
   ≥ 20 maps on each side; the size-weighted mean of those per-cell AUCs must
   be ≥ **0.80** **[tune in pilot]**. Era and density cannot fake this
   criterion. (The difficulty label is not a stratum dimension: with ~700
   Tier A val/test difficulties, cells would be too thin, and load rules are
   already label-referenced. Per-label AUC is reported separately; §17.)
5. **Corruptions**: each of C1–C7 is detected in ≥ 95 % of eligible
   injections, with false alarms ≤ 1 % of swings. The prototype measured
   C1 at 92 % and C5 at 82 % (§17); the pilot either finds the cause or
   proposes a different bar to you.
6. **Baseline**:
   - 100 % of generated difficulties pass the gate;
   - the median score per difficulty is at or above Tier A's q10 for the
     same strata;
   - 100 % of exports round-trip;
   - timing agreement is reported (no bar).
7. **Preview**: the 5 pilot baseline zips load and play in ArcViewer (web)
   without errors. This is a manual check by you.
8. **Speed**: `validate()` on one Expert+ difficulty takes < 100 ms median
   in-process (Phase 5 budget) **[verify in pilot]**.
9. **No leakage**: `reference.json` is built from Tier A train only, and
   pilot tuning used only Tier A train and Tier D fit (automated checks).
10. **Idempotency**: re-running any stage with no new input is a no-op, or
    rebuilds identical tables.
11. **Report** shows:
    - funnel counts and the Tier D year mix;
    - score distributions per tier and per stratum;
    - overall and per-stratum AUC, and per-rule AUCs;
    - corruption recall and false alarms;
    - baseline vs. Tier A, and timing agreement;
    - BeatSaver parity agreement (v2 maps only, diagnostic);
    - timings.

---

## 12. What the validator does not promise

- **A high score means "plays like a typical ranked map", not "fun".**
  Musicality, flow beyond parity, and readability beyond vision blocks are
  not measured. Those are Phase 5 playtests and the Phase 6 critic.
- **Novel but playable patterns can lose points** on load rules if no
  ranked map in that stratum does the same. The one-sided load rules and the
  charitable DP limit this, but it is inherent to calibrating on Tier A.

---

## 13. Usage posture

Phase 1's posture (phase-1-data.md D6, §10) applies unchanged:
- Tier D map files are user content: not republished, not committed.
- Generated maps contain copyrighted audio: they are kept locally for
  preview and **never uploaded**. If a later phase ever publishes generated
  maps, BeatSaver requires them to be declared AI-generated (`declaredAi`).
- Rule definitions are derived from public ranking criteria and open-source
  tools (§16). No code is copied from them; they are cited as sources of the
  definitions and thresholds.

---

## 14. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| A low rating reflects the song, lighting or hype, not playability | Tier D separation understates or overstates validity | Synthetic corruptions test each rule directly (§6.3); per-rule AUCs show which rules carry the separation |
| Era and format confound (Tier D median 2019, Tier A 2022) | AUC partly measures "old vs. new" | Year-matched sampling (§6.1); per-stratum AUC with a floor (criterion 4) |
| Validator encodes Tier A style | Novel playable patterns penalised; Phase 5/6 pushed towards conformity | Load rules one-sided; error rules limited to clear errors; charitable DP; §12 states the limit |
| Goodhart: the baseline uses the validator in its own loop | Baseline score is not independent evidence | Timing agreement and manual preview are the independent checks (§8.6) |
| Thin strata (rare label × NPS combinations) | Noisy reference quantiles | Merge rule with a 100-map minimum and a label fallback (§4.3) |
| 70 ms merge window wrong for very fast streams | Real swings merged, or splits missed | Measured: same-colour note gaps have a valley at 50–70 ms in Tier A; fast streams split by the 90° rule (§3.2, §17) |
| Older Tier D maps hit new format edge cases | Parse failures | Phase 1's parser already rejects with reasons; criterion 1 allows 1 % |
| Beat This picks half or double tempo | Odd Info BPM in baseline maps | Placement is unaffected (§8.1 step 5); NJS/offset use the recorded level |

---

## 15. Decision log

| Date | # | Decision |
|------|---|----------|
| 2026-10-10 | D1 | Calibration negatives: Tier D low-rated sample + synthetic corruptions |
| 2026-10-10 | D2 | Load rules referenced to Tier A train per difficulty label × NPS band |
| 2026-10-10 | D3 | Baseline: colour notes only, all five difficulties |
| 2026-10-10 | D4 | Validator architecture: swing model first, rules on top |
| 2026-10-10 | D5 | Score: geometric mean of piecewise-linear sub-scores; equal weights, changed only by approval |
| 2026-10-10 | D6 | Hard gate separate from the score; the score is still computed for gated maps |
| 2026-10-10 | D7 | Orientation by charitable DP (BeatLeader analyser style), not a greedy state machine |
| 2026-10-10 | D8 | Export v3.2.0 difficulties + v2.1.0 Info.dat, folder + zip |
| 2026-10-10 | — | Local score windows use 16 map beats, not 4 audio bars: Tier D has no audio features |
| 2026-10-10 | — | Per-stratum AUC gets a floor (size-weighted mean ≥ 0.80) in addition to the overall 0.85 |
| 2026-10-10 | — | Spec approved. Prototype corrections to §3.2, §3.3, §4.1, §6.3, §8.1–8.3, §9–§11 (§17) |

---

## 16. Sources (read 2026-10-10)

1. BeatSaver API map stats: https://github.com/beatmaps-io/beatsaver-main/blob/master/modules/shared/src/commonMain/kotlin/io/beatmaps/api/maps.kt
2. BeatSaver parity port (v2 only): https://github.com/beatmaps-io/beatsaver-common-mp/blob/master/src/commonMain/kotlin/io/beatmaps/common/bs-parity.kt
3. BeatSaver zip stats: https://github.com/beatmaps-io/beatsaver-common-mp/blob/master/src/jvmMain/kotlin/io/beatmaps/common/zip/stats.kt
4. GalaxyMaster2, bs-parity: https://github.com/GalaxyMaster2/bs-parity
5. JoshaParity swing classifier: https://github.com/Joshabi/JoshaParity/blob/master/JoshaParity/Core/MapSwingClassifier.cs
6. JoshaParity parity check: https://github.com/Joshabi/JoshaParity/blob/master/JoshaParity/GenericParityCheck.cs
7. JoshaParity utilities: https://github.com/Joshabi/JoshaParity/blob/master/JoshaParity/Core/ParityUtils.cs
8. BeatLeader AutoModder project: https://github.com/BeatLeader/ChroMapper-AutoModder/blob/main/BLMapCheck/BLMapCheck.csproj
9. AutoModder difficulty checks: https://github.com/BeatLeader/ChroMapper-AutoModder/tree/main/BLMapCheck/BeatmapScanner/CriteriaCheck/Difficulty
10. KivalEvan MapCheck checks: https://github.com/KivalEvan/BeatSaber-MapCheck/tree/main/src/ts/checks
11. MapCheck presets: https://github.com/KivalEvan/BeatSaber-MapCheck/tree/main/src/ts/checks/presets
12. bsmap parity: https://github.com/KivalEvan/BeatSaber-JSMap/blob/main/src/extensions/parity/parity.ts
13. bsmap swing: https://github.com/KivalEvan/BeatSaber-JSMap/blob/main/src/extensions/swing/swing.ts
14. ScoreSaber mapping criteria: https://wiki.scoresaber.com/ranking/criteria
15. BeatLeader ranking criteria: https://beatleader.wiki/en/ranking/rules/criteria
16. AutoModder config: https://github.com/BeatLeader/ChroMapper-AutoModder/blob/main/BLMapCheck/Configs/Config.cs
17. ScoreSaber technical limits: https://github.com/scoresaber/wiki/blob/main/content/docs/ranking/criteria/technical-limitations-criteria.md
18. BeatLeader PP and ratings: https://beatleader.wiki/en/ranking/PP
19. BeatLeader analyser: https://github.com/BeatLeader/beatleader-analyzer/tree/main/beatleader-analyzer/BeatmapScanner
20. BeatLeader rating API: https://github.com/BeatLeader/RatingAPI/blob/master/Controllers/RatingsController.cs
21. BeatLeader acc rating: https://github.com/BeatLeader/RatingAPI/blob/master/Controllers/AccRating.cs
22. ArcViewer: https://github.com/AllPoland/ArcViewer
23. ArcViewer URL arguments and FAQ: https://github.com/AllPoland/ArcViewer/wiki/URL-Argument-Docs, https://github.com/AllPoland/ArcViewer/wiki/FAQ
24. BSMG map format: https://github.com/bsmg/wiki/blob/master/wiki/mapping/map-format.md
25. BeatSaver Info v2 schema: https://github.com/beatmaps-io/beatsaver-common-mp/blob/master/src/jvmMain/kotlin/io/beatmaps/common/beatsaber/info/infoSchema2.kt
26. BeatSaver v3 validation: https://github.com/beatmaps-io/beatsaver-common-mp/blob/master/src/jvmMain/kotlin/io/beatmaps/common/beatsaber/map/mapValidation3.kt

---

## 17. Prototype findings (2026-10-10)

Every module in the implementation plan was prototyped in a throwaway
worktree and its tests run (134 passing, including the Phase 1 and 2 suites)
before the plan was written. The whole stage chain then ran on the real
dataset in a scratch data root, with Tier C standing in for Tier D (no
download). Corrections made to this doc as a result:

**Swing model and rules**
- **Merge window (§3.2).** Same-colour note gaps in Tier A have a clear
  valley at 50–70 ms (sliders below, separate swings above), so 70 ms
  stays. About 4.7k Tier A note pairs are 60–69 ms apart with *different*
  arrows: fast streams. An unconditional merge flagged them as H3, hence
  MapCheck's 90° split rule.
- **Reset time (§3.3).** One 1.0 s threshold made the DP read Normal-level
  down-down pairs 0.96 s apart as upside-down hits. Tier A's 5th percentile
  of same-hand down-arrow gaps runs from 0.60 s (Easy) to 0.20 s (Expert+),
  so `T_reset` is per label.
- **Bomb resets (§3.3).** bs-parity's ¼ beat is a skip condition, not a
  window: a reset bomb usually sits half a beat before the swing.
- **H1 (§4.1).** 98.7 % of 600 Tier A difficulties passed the gate; every
  failure was a bomb–bomb pair in a bomb tunnel. With ScoreSaber's 20 ms
  rule for bomb pairs: 99.7 %.
- **E1 (§4.1).** With only "unjustified reset or > 270°", flipped diagonal
  and horizontal arrows went undetected: the charitable DP explains them as
  upside-down hits. Rotation changes ≥ 180° are 0.055 % of Tier A swings
  and 0.72 % of Tier C swings (13×); |ρ| > 135° is AutoModder's
  inverted swing. Both now count. E1's AUC (Tier A vs. C) rose from 0.52 to
  0.67, and C1 recall from 66 % to 92 %.
- **L2 (§4.1).** Strain values sit on 45° steps, so a p95 gave q90 = q99 =
  0.25 and collapsed anchors. The mean of the top 5 % is continuous.

**Calibration (§6.3, §11)**
- **C4** at a fixed 2× was detected 33 % of the time: often the doubled run
  is still within ranked maps' range, which L1 does not claim to flag.
  Sized from the reference (1.5× the stratum's q99): 100 %, but only
  ~20 % of runs are eligible (the rest would shrink into slider range).
- **C5** (a +90° turn of 4 swings) was detected 11 % of the time. As a
  ±90° zig-zag: 60 %. Counting L3 and E1 as well: 72–82 %. The DP reads a
  slower zig-zag as three resets: defensible, and visible in L3, not L2.
- **Per-stratum AUC (criterion 4).** With label × year × NPS, almost no cell
  reaches 20 Tier A val/test maps. The label is dropped from the cells and
  reported on its own.

**Baseline (§8)**
- **Beat indices (§8.1).** Rounding `(b − b0) / IBI` gave a median p95
  residual of 200 ms on 300 songs: the indices drift by whole beats. Step-by-
  step rounding: 44 % of songs fit one line within 30 ms; the rest have real
  tempo changes or half-time sections and are fitted per segment.
- **Slow songs (§8.2).** At 80 BPM, ¼-beat slots cap Expert+ at 5.3 NPS
  (Expert and Expert+ came out identical). ⅛ slots below 110 BPM fix it.
- **Doubles (§8.3).** A neighbour median that includes the slot itself can
  never be exceeded 2× with one neighbour; a song-level median works. A
  double combined with a crossover put both hands in one cell (H1).

**Measurements**

| Item | Result |
|------|--------|
| `analyse` per difficulty | median 26 ms, max 68 ms (30 Tier A diffs; criterion 8: < 100 ms) |
| `analyse`, all 15,442 difficulties | 85 s, 0 errors (8 workers) |
| Gate, Tier A | 99.5 % (val+test 99.7 %); Tier B 98.9 %; Tier C 91.9 % |
| Score AUC, Tier A val+test vs. Tier C | 0.78–0.80; per rule E1 0.67, E2 0.60, E3 0.50, L1 0.56–0.58, L2 0.64–0.68, L3 0.69–0.73 |
| Corruption recall (100 Tier A val/test diffs) | C1 92 %, C2 100 %, C3 100 %, C4 100 %, C5 82 %, C6 100 %, C7 100 %; false alarms ≤ 0.01 % of swings |
| BeatSaver parity agreement (v2) | Tier A: BeatSaver errors 24.6 %, our E1 5.7 %, agreement 80 % |
| Tier D selection | pool 5,676 maps → 2,000 sampled, 2,568 difficulties; 2022–2025 fully used, so the sample stays older than Tier A |
| Baseline | ~0.3 s per song for 5 difficulties; pilot val/test: 15 songs, 100 % pass and round trip; median timing F 0.25 (Easy) – 0.50 (Expert+) |
| Refactored Phase 1 parser | byte-identical output on 22 real maps |

Tier C is not Tier D: it is the unranked difficulties of ranked maps. Real
Tier D numbers come from the pilot.

## 18. Pilot findings (2026-10-11)

`kata validate pilot` downloaded the 200 Tier D pilot maps and ran every
stage on all parsed difficulties (Tiers A–C plus Tier D pilot). It ran
twice: once on the plan's code, and once after the final-review fixes. The
fixes: corruption detection counts only new issues on the injected hand; the
config pin covers every score-changing parameter and is checked wherever the
reference is loaded; and the pilot report compares the **tuning halves**
(Tier A train vs. Tier D fit) instead of the held-out ones. The numbers below
are from the second run. Report: `E:\kata-data\reports\phase3-pilot-2026-10-11.md`.

| Item | Result |
|------|--------|
| Tier D download | 200 / 200 maps, 0 errors, 41 s; 0 audio files left |
| Tier D parse (criterion 1) | 258 / 259 difficulties ok (**99.6 %**; bar 99 %); 1 rejected |
| Tier D size | 1.8 MB tables + 39 MB raw `.dat` for 200 maps → ~0.4 GB for 2,000 (bar 1 GB) |
| Tier D year mix (2,000 sample) | 2018 309 · 2019 125 · 2020 377 · 2021 604 · 2022 352 · 2023 133 · 2024 74 · 2025 26. 2022–2025 are used in full: 29 % of the sample vs. 55 % of Tier A |
| `analyse` coverage (criterion 2) | 15,440 difficulties (A 7,463, B 7,459, C 260, D 258), 0 errors, 92 s |
| Gate (criterion 3) | Tier A train **99.5 %**, B 98.9 %, C 91.9 %, Tier D fit 91.5 %. Tier A val+test goes in the final report |
| Separation, Tier A train vs. Tier D fit (criterion 4) | overall AUC **0.897** (6,766 vs. 117); per-stratum **0.870**, but only 1 cell (≤ 2019 × 0–4 NPS, 335 vs. 40) has ≥ 20 maps per side in the pilot |
| Per-rule AUC (sub-scores) | E1 0.79, E2 0.81, E3 0.50, L1 0.54, L2 0.72, L3 0.79 |
| Per-label AUC | Easy 0.92 (5 D), Normal 0.83 (8), Hard 0.95 (19), Expert 0.985 (47), Expert+ 0.77 (38) |
| Corruptions, 100 Tier A val/test diffs (criterion 5) | C1 **92.1 %**, C2 98.8 %, C3 100 %, C4 100 % (90 eligible), C5 **80.6 %**, C6 100 %, C7 100 %; false alarms ≤ 0.008 % of swings |
| Baseline, 15 pilot songs × 5 (criterion 6) | 100 % pass, 100 % round trip, median score 100 in every label (Tier A q10 92.5–95.7); median timing F Easy 0.25, Normal 0.39, Hard 0.45, Expert 0.33, Expert+ 0.50; 2.5 s in total |
| `validate()` speed, 60 Tier A Expert+ diffs (criterion 8) | median **40–45 ms**, p90 79–85 ms, max 105–121 ms (two runs) |
| BeatSaver parity agreement, v2 (diagnostic) | Tier A train 80 %, B 60 %, C 63 %, D fit 76 % |

**C1 (92 %, bar 95 %): mostly the rule boundary.** 37 of 470 injections were
missed. In 26 of them, the flipped arrow becomes a swing with a wrist rotation
of exactly |ρ| = 135°. E1's inverted-swing test is |ρ| > 135° (AutoModder), so
these are not flagged. Moving the bound to ≥ 135° is not a free fix:
|ρ| = 135° is 0.65 % of Tier A train swings (19× the rate of |ρ| > 135°) and
1.6 % of Tier D fit swings, only 2.5× more. That would put the average ranked
map above E1's q90 anchor floor (0.2 issues per 100 swings). In the other 11
misses, the DP explains the flip with a rotation change below 180° and no
unjustified reset. The rule is behaving as designed; the corruption sometimes
creates a swing that ranked maps also contain.

**C5 (81 %, bar 95 %): the corruption, at slow tempos.** If the zig-zag's
swings are less than `T_reset` apart, it is detected in 211 of 212 cases
(99.5 %). Slower zig-zags are detected in 192 of 288 cases (67 %). The DP
reads them as slow resets, which are legal and show up in L3, not L2. Three
slow resets in a 16-beat window rarely push s_L3 below 0.5. A slow ±90°
zig-zag is playable, so missing it is the right call. The review fix (only
new issues and newly failing windows count) lowered C5 from 82.4 % to 80.6 %,
and C2 from 100 % to 98.8 %.

**Other observations.**
- E3's sub-score is 1.0 on average in every tier, so it adds nothing to the
  separation (AUC 0.50). Under the geometric mean it costs nothing either.
- L1 separates weakly (0.54).
- Expert+ is the weakest label (0.77).

**Leakage note (criterion 9).** The first run's report still compared Tier A
val+test with Tier D holdout, and the executor read it: overall AUC 0.915,
per-stratum 0.877 over one cell. Nothing was tuned on it, and the proposals
below use the tuning halves only.

**ArcViewer check (criterion 7, manual).** Five pilot zips under
`E:\kata-data\generated\baseline\v1\`, covering slow, fast and unsteady songs:

| Song | Grid BPM | Piecewise | Zip |
|------|----------|-----------|-----|
| Persona 3 — Burn My Dread (Short) | 97.5 | no | `3c16f76c2cb68c679475714788c65cee923dca33f0a605c155372cb89c0cee01.zip` |
| bbno$ — BPOT | 103.0 | no | `61d78d94d232bd19db4d84b280e791db01c3f124091e4fc2b86752adb297523f.zip` |
| Olivia Rodrigo — Brutal | 122.8 | yes | `d02ad8b5ec39d3742312d38cfb73ba492c282003f7867ba7fb4c3b671e3b5bfb.zip` |
| S3RL — What is a DJ? | 176.1 | yes | `7582d4adbd7c7261aef0a0686d09d60f8645fa1b6e9bf76478a8e92c27f202dd.zip` |
| Ashrount — LaμreLs | 201.2 | yes | `234e8af9e46c64ca17c5e7dc2614086637959cccdc2a3cf877ebb4589bc02c19.zip` |

**Proposed changes (awaiting approval; none applied).**
1. **C5 eligibility:** only zig-zags whose swings are less than `T_reset[label]`
   apart, which measures 99.5 % against the 95 % bar. Slow zig-zags are
   playable, so they are not a defect to detect. Changes the corruption only,
   not the validator, so no version bump.
2. **C1 bar:** lower it to 90 % and keep E1's |ρ| > 135°. Raising E1 to
   ≥ 135° would also reach 95 %, but would flag 0.65 % of ranked swings;
   not recommended.
3. **No threshold or weight changes.** On the tuning halves, 0.897 overall and
   0.870 per stratum already clear the 0.85 / 0.80 bars. The per-stratum
   criterion can only be judged on the full Tier D sample (Task 11).
