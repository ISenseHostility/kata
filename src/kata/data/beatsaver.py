"""BeatSaver: full metadata catalogue crawl, flattening, and map download/extraction."""

from __future__ import annotations

import json
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

import polars as pl
import zstandard

from kata.config import BEATSAVER_API, Paths
from kata.data.http import PoliteClient

PAGE_SIZE = 100
COMPLETE_MARKER = "_COMPLETE"
# Covers and videos are dropped; everything else is kept, because Info.dat may name
# difficulty or audio files with any extension (seen: `*.ben`, a song literally named `.egg`).
DROP_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".mp4", ".webm", ".mov", ".mkv"}
# Tier D (Phase 3) keeps no audio: the validator reads only the .dat files.
AUDIO_NAMES = {".egg", ".ogg", ".wav", ".mp3", ".flac", ".m4a", ".opus"}   # mappers sometimes leave the source file in


def is_audio(name: str) -> bool:
    n = name.lower()
    return PurePosixPath(n).suffix in AUDIO_NAMES or n in AUDIO_NAMES


def _write_page(path: Path, docs: list[dict]) -> None:
    data = "\n".join(json.dumps(d, ensure_ascii=False) for d in docs).encode("utf-8")
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(zstandard.compress(data, 10))
    tmp.replace(path)


def read_page(path: Path) -> list[dict]:
    text = zstandard.decompress(path.read_bytes()).decode("utf-8")
    # Split on "\n" only: str.splitlines() also breaks on U+2028 etc. inside JSON strings.
    return [json.loads(line) for line in text.split("\n") if line]


def current_crawl_dir(paths: Paths) -> Path:
    """Resume the newest unfinished crawl, otherwise start a new one dated today."""
    dirs = sorted(p for p in paths.catalog_pages.iterdir() if p.is_dir()) if paths.catalog_pages.exists() else []
    if dirs and not (dirs[-1] / COMPLETE_MARKER).exists():
        return dirs[-1]
    d = paths.catalog_pages / date.today().isoformat()
    if (d / COMPLETE_MARKER).exists():
        return d  # already completed today; crawl is a no-op
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_complete_crawl(paths: Paths) -> Path:
    done = sorted(p.parent for p in paths.catalog_pages.glob(f"*/{COMPLETE_MARKER}"))
    if not done:
        raise RuntimeError("no completed catalogue crawl; run `kata data crawl-catalog` first")
    return done[-1]


async def crawl_catalog(client: PoliteClient, paths: Paths, log=print) -> dict:
    out = current_crawl_dir(paths)
    if (out / COMPLETE_MARKER).exists():
        return {"dir": str(out), "status": "already complete"}
    pages = sorted(out.glob("page-*.jsonl.zst"))
    before = None
    if pages:
        last = read_page(pages[-1])
        before = last[-1]["createdAt"]
        log(f"resuming crawl in {out.name} after {len(pages)} pages (before={before})")
    n_page, n_maps = len(pages), 0
    while True:
        params = {"pageSize": PAGE_SIZE, "sort": "CREATED", "automapper": "true"}
        if before:
            params["before"] = before
        docs = (await client.get_json(f"{BEATSAVER_API}/maps/latest", params))["docs"]
        if not docs:
            break
        _write_page(out / f"page-{n_page:05d}.jsonl.zst", docs)
        n_page += 1
        n_maps += len(docs)
        before = docs[-1]["createdAt"]
        if n_page % 50 == 0:
            log(f"  {n_page} pages, oldest so far {before}")
        # Short pages happen mid-catalogue (server-side filtering), so only an empty page ends it.
    (out / COMPLETE_MARKER).write_text(f"{n_page} pages\n")
    return {"dir": str(out), "pages": n_page, "new_maps": n_maps}


def _latest_version(m: dict) -> dict | None:
    published = [v for v in m.get("versions") or [] if v.get("state") == "Published"]
    return max(published, key=lambda v: v.get("createdAt", "")) if published else None


def flatten_catalog(paths: Paths) -> tuple[pl.DataFrame, pl.DataFrame]:
    crawl = latest_complete_crawl(paths)
    seen: dict[str, dict] = {}
    for page in sorted(crawl.glob("page-*.jsonl.zst")):
        for m in read_page(page):
            seen[m["id"]] = m
    maps, diffs = [], []
    for m in seen.values():
        v = _latest_version(m)
        meta, stats = m.get("metadata") or {}, m.get("stats") or {}
        maps.append({
            "map_id": m["id"], "hash": v["hash"].lower() if v else None,
            "name": m.get("name"), "song_name": meta.get("songName"),
            "song_sub_name": meta.get("songSubName"), "song_author": meta.get("songAuthorName"),
            "level_author": meta.get("levelAuthorName"),
            "uploader_id": (m.get("uploader") or {}).get("id"),
            "uploader_name": (m.get("uploader") or {}).get("name"),
            "meta_bpm": meta.get("bpm"), "meta_duration_s": meta.get("duration"),
            "uploaded_at": m.get("uploaded"), "created_at": m.get("createdAt"),
            "curated_at": m.get("curatedAt"), "upvotes": stats.get("upvotes"),
            "downvotes": stats.get("downvotes"), "rating": stats.get("score"),
            "ranked_ss": bool(m.get("ranked")), "ranked_bl": bool(m.get("blRanked")),
            "qualified_ss": bool(m.get("qualified")), "qualified_bl": bool(m.get("blQualified")),
            "automapper": bool(m.get("automapper")), "declared_ai": m.get("declaredAi"),
            "sage_score": v.get("sageScore") if v else None, "tags": m.get("tags") or [],
            "version_state": v.get("state") if v else None,
            "download_url": v.get("downloadURL") if v else None,
            "raw_json": json.dumps(m, ensure_ascii=False),
        })
        for d in (v or {}).get("diffs") or []:
            parity = d.get("paritySummary") or {}
            diffs.append({
                "map_id": m["id"], "hash": v["hash"].lower(),
                "characteristic": d.get("characteristic"), "difficulty": d.get("difficulty"),
                "njs": d.get("njs"), "njs_offset": d.get("offset"), "notes": d.get("notes"),
                "bombs": d.get("bombs"), "obstacles": d.get("obstacles"), "nps": d.get("nps"),
                "length_beats": d.get("length"), "seconds": d.get("seconds"),
                "events": d.get("events"), "chroma": bool(d.get("chroma")), "me": bool(d.get("me")),
                "ne": bool(d.get("ne")), "cinema": bool(d.get("cinema")),
                "parity_errors": parity.get("errors"), "parity_warns": parity.get("warns"),
                "parity_resets": parity.get("resets"), "ss_stars": d.get("stars"),
                "bl_stars": d.get("blStars"), "max_score": d.get("maxScore"),
                "environment": d.get("environment"),
            })
    maps_df = pl.DataFrame(maps, infer_schema_length=None)
    diffs_df = pl.DataFrame(diffs, infer_schema_length=None)
    maps_df.write_parquet(paths.processed / "metadata_catalog.parquet")
    diffs_df.write_parquet(paths.processed / "metadata_catalog_diffs.parquet")
    return maps_df, diffs_df


def extract_map(zip_path: Path, dest: Path, drop_audio: bool = False) -> list[str]:
    """Extract the files next to Info.dat (flattened; no path traversal possible)."""
    with zipfile.ZipFile(zip_path) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        info = [i for i in members if PurePosixPath(i.filename).name.lower() == "info.dat"]
        if not info:
            raise ValueError("zip has no Info.dat")
        base = PurePosixPath(min(info, key=lambda i: len(i.filename)).filename).parent
        dest.mkdir(parents=True, exist_ok=True)
        kept = []
        for i in members:
            p = PurePosixPath(i.filename)
            if p.parent != base or p.suffix.lower() in DROP_SUFFIXES or (drop_audio and is_audio(p.name)):
                continue
            (dest / p.name).write_bytes(z.read(i))
            kept.append(p.name)
    return kept


async def download_map(client: PoliteClient, paths: Paths, map_hash: str, url: str,
                       drop_audio: bool = False) -> dict:
    zip_path = paths.maps / "_zips" / f"{map_hash}.zip"
    sha = await client.download(url, zip_path)
    size = zip_path.stat().st_size
    dest = paths.maps / map_hash
    try:
        files = extract_map(zip_path, dest, drop_audio)
    finally:
        zip_path.unlink(missing_ok=True)
    return {"zip_sha256": sha, "zip_bytes": size, "files": files}
