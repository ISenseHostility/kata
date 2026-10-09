# kata — Roadmap
# Rough plan. Every phase needs a detailed design doc
# (docs/phases/) reviewed BEFORE implementation starts.

## Phase 0 — Setup ✅ (this session)

## Phase 1 — Data acquisition & normalization
Goal: a clean, queryable dataset of maps, audio and replays.
- BeatSaver scraper with quality filters (ranked / curated / rating)
- Unified map schema covering v2, v3 and v4 formats
- BeatLeader replay download + BSOR parsing to head/hand tracks
- Metadata in DuckDB/Parquet
  ⚠️ Detailed plan needed: quality criteria, dataset size target, storage
  layout, API terms and rate limits, replay sampling (which players/scores),
  coordinate frames.

## Phase 2 — Audio features
Goal: precomputed, cached conditioning features per song.
- Mel spectrograms, Demucs stems, beat/downbeat grid, structure labels
- Time alignment with map BPM grids and replay timestamps
  📝 Design doc: docs/phases/phase-2-audio.md (approved 2026-10-09).

## Phase 3 — Validator + rule-based baseline
Goal: a playability scorer and a baseline mapper to beat.
- Parity, vision-block, reset, swing-angle and speed checks
- Simple onset + parity-aware pattern generator
- Export to map files, preview in ArcViewer
  ⚠️ Detailed plan needed: which rules, scoring scheme, calibration against
  ranked vs. poorly rated maps.

## Phase 4 — Motion model
Goal: generate plausible hand trajectories from music.
- Prerequisite: fetch BeatLeader replays (Tier A, K = 3). This was
  deferred from Phase 1; see docs/phases/phase-1-data.md §13.
- Motion representation (positions + 6D rotations, head-relative)
- Diffusion transformer with audio cross-attention
- Start small (~50M params, 8-bar windows)
  ⚠️ Detailed plan needed: architecture, window length, conditioning,
  training budget, evaluation metrics, what counts as "good enough".

## Phase 5 — Motion → map
Goal: playable maps derived from generated motion.
- Swing peak detection to note time, lane, layer, direction, color
- Difficulty control (note density, speed)
- Validator in the loop; first real VR playtests
  ⚠️ Detailed plan needed: extraction algorithm, how to handle dots, bombs,
  walls and stacks, playtest protocol.

## Phase 6 — Quality & control
Goal: maps that feel intentional.
- Structure-aware planning, motif reuse on repeated sections
- Style/mapper conditioning
- Critic-guided sampling or RL fine-tuning
- Section inpainting ("regenerate these 8 bars")
  ⚠️ Detailed plan needed: per feature; prioritize based on Phase 5
  playtest findings.

## Phase 7 — Product
Goal: usable by mappers.
- Model serving (FastAPI or ONNX)
- Editor integration (ChroMapper plugin or own web editor)
  ⚠️ Detailed plan needed: product direction decided after Phase 6.
