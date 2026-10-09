"""BeatLeader: leaderboards per map hash, score pages, replay download."""

from __future__ import annotations

import json
import math
from pathlib import Path

import zstandard

from kata.config import BEATLEADER_API, BEATLEADER_REPLAY_CDN, Paths
from kata.data.http import HttpError, PoliteClient

SCORES_PAGE_SIZE = 100
# Sample replays from the top half of a leaderboard, but never page past this many scores.
MAX_SCORES_PER_LEADERBOARD = 500


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(zstandard.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"), 10))
    tmp.replace(path)


def load(path: Path):
    return json.loads(zstandard.decompress(path.read_bytes()).decode("utf-8"))


def leaderboards_path(paths: Paths, map_hash: str) -> Path:
    return paths.bl_leaderboards / f"{map_hash}.json.zst"


def scores_path(paths: Paths, leaderboard_id: str) -> Path:
    return paths.bl_scores / f"{leaderboard_id}.json.zst"


async def fetch_leaderboards(client: PoliteClient, paths: Paths, map_hash: str) -> list[dict]:
    """Returns flattened leaderboard rows; [] when BeatLeader has no leaderboard for the hash."""
    try:
        data = await client.get_json(f"{BEATLEADER_API}/leaderboards/hash/{map_hash}")
    except HttpError as e:
        if e.status == 404:
            data = {"leaderboards": []}
        else:
            raise
    _dump(leaderboards_path(paths, map_hash), data)
    return flatten_leaderboards(map_hash, data)


def flatten_leaderboards(map_hash: str, data: dict) -> list[dict]:
    rows = []
    for lb in data.get("leaderboards") or []:
        d = lb.get("difficulty") or {}
        rows.append({
            "leaderboard_id": lb["id"], "hash": map_hash, "characteristic": d.get("modeName"),
            "difficulty": d.get("difficultyName"), "bl_status": d.get("status"),
            "bl_stars": d.get("stars"), "bl_pass_rating": d.get("passRating"),
            "bl_acc_rating": d.get("accRating"), "bl_tech_rating": d.get("techRating"),
        })
    return rows


async def fetch_scores(client: PoliteClient, paths: Paths, leaderboard_id: str) -> dict:
    """Fetch score pages covering the top half of the leaderboard (capped)."""
    url = f"{BEATLEADER_API}/leaderboard/{leaderboard_id}"
    params = {"count": SCORES_PAGE_SIZE, "sortBy": "rank", "order": "desc"}
    first = await client.get_json(url, {**params, "page": 1})
    plays = int(first.get("plays") or 0)
    wanted = min(MAX_SCORES_PER_LEADERBOARD, max(1, math.ceil(plays / 2)))
    scores = list(first.get("scores") or [])
    page = 1
    while len(scores) < wanted and len(scores) == page * SCORES_PAGE_SIZE:
        page += 1
        scores.extend((await client.get_json(url, {**params, "page": page})).get("scores") or [])
    result = {"leaderboard_id": leaderboard_id, "plays": plays, "scores": scores[:wanted]}
    _dump(scores_path(paths, leaderboard_id), result)
    return result


def flatten_scores(result: dict) -> list[dict]:
    rows = []
    for s in result["scores"]:
        player = s.get("player") or {}
        rows.append({
            "score_id": s["id"], "leaderboard_id": result["leaderboard_id"],
            "player_id": str(s.get("playerId") or player.get("id")), "rank": s.get("rank"),
            "accuracy": s.get("accuracy"), "modifiers": s.get("modifiers") or "",
            "pauses": s.get("pauses"), "full_combo": s.get("fullCombo"),
            "missed": s.get("missedNotes"), "bad_cuts": s.get("badCuts"),
            "bomb_cuts": s.get("bombCuts"), "walls_hit": s.get("wallsHit"),
            "platform": s.get("platform"), "hmd": s.get("hmd"), "controller": s.get("controller"),
            "timeset": s.get("timeset"), "player_bot": bool(player.get("bot")),
            "player_temporary": bool(player.get("temporary")),
            "plays": result["plays"],
        })
    return rows


def replay_url(score_id: int, player_id: str, difficulty: str, mode: str, map_hash: str) -> str:
    return f"{BEATLEADER_REPLAY_CDN}/{score_id}-{player_id}-{difficulty}-{mode}-{map_hash.upper()}.bsor"


async def download_replay(client: PoliteClient, paths: Paths, row: dict) -> dict:
    dest = paths.replays / row["hash"] / f"{row['score_id']}.bsor"
    url = replay_url(row["score_id"], row["player_id"], row["difficulty"], row["characteristic"], row["hash"])
    try:
        sha = await client.download(url, dest)
        source = "pattern"
    except HttpError as e:
        if e.status not in (403, 404):
            raise
        score = await client.get_json(f"{BEATLEADER_API}/score/{row['score_id']}")
        if not score.get("replay"):
            raise HttpError(url, e.status, "no replay available") from e
        sha = await client.download(score["replay"], dest)
        source = "score_api"
    return {"sha256": sha, "bytes": dest.stat().st_size, "url_source": source}
