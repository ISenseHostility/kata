"""Paths and constants shared by the data pipeline."""

from __future__ import annotations

import os
from pathlib import Path

USER_AGENT = "kata-dataset/0.1 (+https://github.com/ISenseHostility/kata)"

BEATSAVER_API = "https://api.beatsaver.com"
BEATLEADER_API = "https://api.beatleader.com"
BEATLEADER_REPLAY_CDN = "https://cdn.replays.beatleader.com"

# Stop download stages when free space on the data drive drops below this.
MIN_FREE_BYTES = 50 * 1024**3


def data_root() -> Path:
    return Path(os.environ.get("KATA_DATA", r"E:\kata-data"))


class Paths:
    """Directory layout from docs/phases/phase-1-data.md §7."""

    def __init__(self, root: Path | None = None):
        self.root = root or data_root()

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def catalog_pages(self) -> Path:
        return self.raw / "beatsaver" / "catalog"

    @property
    def maps(self) -> Path:
        return self.raw / "maps"

    @property
    def bl_leaderboards(self) -> Path:
        return self.raw / "beatleader" / "leaderboards"

    @property
    def bl_scores(self) -> Path:
        return self.raw / "beatleader" / "scores"

    @property
    def replays(self) -> Path:
        return self.raw / "replays"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def replay_frames(self) -> Path:
        return self.processed / "replay_frames"

    @property
    def features(self) -> Path:
        """Phase 2 feature store root (docs/phases/phase-2-audio.md §9)."""
        return self.root / "features"

    @property
    def lowrated(self) -> Path:
        """Phase 3 Tier D tables (docs/phases/phase-3-validator.md §6.1)."""
        return self.processed / "lowrated"

    @property
    def validator(self) -> Path:
        """Phase 3 validator config and reference (docs/phases/phase-3-validator.md §9)."""
        return self.root / "validator"

    @property
    def generated(self) -> Path:
        """Phase 3 baseline maps (docs/phases/phase-3-validator.md §8.5)."""
        return self.root / "generated"

    @property
    def state_db(self) -> Path:
        return self.root / "state" / "pipeline.duckdb"

    @property
    def query_db(self) -> Path:
        return self.root / "kata.duckdb"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> "Paths":
        for p in (self.catalog_pages, self.maps, self.bl_leaderboards, self.bl_scores,
                  self.replays, self.processed, self.replay_frames, self.features,
                  self.lowrated, self.validator, self.generated, self.state_db.parent, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self
