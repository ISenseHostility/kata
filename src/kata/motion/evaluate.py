"""Evaluation metrics (docs/phases/phase-4-motion.md §7.2) on de-normalised motion, plus the quick
windowed validation that picks checkpoints during training (§6.4).

Motion here is (T, 27): head, left, right as position + 6D rotation, on frames frame0 + i at FPS.
"""

from __future__ import annotations

import numpy as np

from kata.audio.beats import f_measure
from kata.motion.geometry import blade, rot6_to_matrix
from kata.motion.prepare import FPS, HIT, OFF, TRACK

TIMING_TOL_S = 0.03          # Phase 3's timing agreement tolerance (baseline TIMING_TOL_S)
AGREE_TOL_FRAMES = 2         # ±40 ms (§7.2.3)
MIN_HIT_SPEED = 1.0          # m/s
QUANTITIES = ("tip_speed_at_hits", "peak_speed_per_swing", "roll_range", "hand_head_dist", "head_disp_1s")


def timing_f(hit_times, map_times) -> float:
    """Phase 3's measure: F at ±30 ms between unique map note times and unique hit times."""
    return f_measure(np.unique(np.asarray(map_times, np.float64)), np.unique(np.asarray(hit_times, np.float64)),
                     TIMING_TOL_S)


def tips(motion: np.ndarray) -> np.ndarray:
    """(T, 27) -> saber tips (T, 2, 3)."""
    out = []
    for part in ("left", "right"):
        s = TRACK[part].start
        out.append(blade(motion[:, s:s + 3], rot6_to_matrix(motion[:, s + 3:s + 9]))[1])
    return np.stack(out, 1)


def tip_velocity(motion: np.ndarray) -> np.ndarray:
    """(T, 2, 3) in m/s, central differences."""
    return np.gradient(tips(motion), axis=0) * FPS


def _hit_frames(hits, frame0: int, n: int) -> np.ndarray:
    f = np.rint(np.asarray(hits, np.float64) * FPS - frame0).astype(np.int64)
    return f[(f >= 0) & (f < n)]


def quantities(motion: np.ndarray, hits: tuple, frame0: int = 0, valid: np.ndarray | None = None) -> dict:
    """The five plausibility distributions of §7.2.2 for one track. hits: (left times, right times)."""
    valid = np.ones(len(motion), bool) if valid is None else np.asarray(valid, bool)
    speed = np.linalg.norm(tip_velocity(motion), axis=-1)
    speed[~valid] = np.nan
    at_hits, peaks, rolls = [], [], []
    for h, part in enumerate(("left", "right")):
        f = np.sort(_hit_frames(hits[h], frame0, len(motion)))
        at_hits.append(speed[f, h])
        segs = [speed[a:b + 1, h] for a, b in zip(f[:-1], f[1:]) if b > a]
        peaks += [np.nanmax(g) for g in segs if np.isfinite(g).any()]
        s = TRACK[part].start
        rot = rot6_to_matrix(motion[valid, s + 3:s + 9])
        roll = np.degrees(np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0])))
        rolls.append(np.percentile(roll, 95) - np.percentile(roll, 5) if len(roll) else np.nan)
    head = motion[:, 0:3]
    reach = np.concatenate([np.linalg.norm(motion[valid, s:s + 3] - head[valid], axis=-1)[::5]
                            for s in (TRACK["left"].start, TRACK["right"].start)])
    ok = valid[:-FPS] & valid[FPS:] if len(motion) > FPS else np.zeros(0, bool)
    disp = np.linalg.norm(head[FPS:] - head[:-FPS], axis=-1)[ok][::25] if len(motion) > FPS else np.zeros(0)
    out = {"tip_speed_at_hits": np.concatenate(at_hits), "peak_speed_per_swing": np.asarray(peaks),
           "roll_range": np.asarray(rolls), "hand_head_dist": reach, "head_disp_1s": disp}
    return {k: v[np.isfinite(v)] for k, v in out.items()}


def agreement(motion: np.ndarray, hits: tuple, frame0: int = 0) -> tuple[int, int]:
    """(hits within ±40 ms of a local tip-speed maximum and moving >= 1 m/s, all hits) (§7.2.3)."""
    speed = np.linalg.norm(tip_velocity(motion), axis=-1)
    good = total = 0
    for h in (0, 1):
        s = speed[:, h]
        is_max = np.zeros(len(s), bool)
        is_max[1:-1] = (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:])
        for f in _hit_frames(hits[h], frame0, len(motion)):
            lo, hi = max(f - AGREE_TOL_FRAMES, 0), min(f + AGREE_TOL_FRAMES + 1, len(s))
            good += bool(is_max[lo:hi].any() and s[f] >= MIN_HIT_SPEED)
            total += 1
    return good, total


def tip_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean saber-tip distance (m) between two tracks on the same frames (§7.2.5); no alignment:
    real players on one map are already aligned by its notes."""
    n = min(len(a), len(b))
    return float(np.linalg.norm(tips(a[:n]) - tips(b[:n]), axis=-1).mean())


def velocity_jumps(motion: np.ndarray, frames=None) -> np.ndarray:
    """|Δ tip velocity| (m/s, worse hand) between frame f - 1 and f, for every frame or the given ones."""
    v = tip_velocity(motion)
    jump = np.linalg.norm(np.diff(v, axis=0), axis=-1).max(-1)       # jump[f - 1] is between f - 1 and f
    if frames is None:
        return jump
    f = np.asarray(frames, np.int64)
    return jump[f[(f >= 1) & (f <= len(jump))] - 1]


def real_hits(raw: np.ndarray, frame0: int) -> tuple[np.ndarray, np.ndarray]:
    """Hit times per hand from a raw track's flag and offset channels."""
    out = []
    for h in (0, 1):
        rows = np.flatnonzero(raw[:, HIT[h]] > 0)
        out.append((frame0 + rows + raw[rows, OFF[h]]) / FPS)
    return out[0], out[1]


# --------------------------------------------------------------------------- quick validation (training)

def quick_windows(sampler, n: int) -> list:
    """Up to n val difficulties, evenly spread: their first replay's middle window (if fully valid)."""
    firsts = [diff[0] for _, diffs in sampler.songs for diff in diffs]
    if not firsts:
        return []
    picks = [firsts[i] for i in sorted(set(np.linspace(0, len(firsts) - 1, min(n, len(firsts))).round().astype(int)))]
    out = []
    for t in picks:
        start = (t.n_frames - sampler.window) // 2
        if sampler.motion.valid(t.score_id)[start:start + sampler.window].all():
            out.append((t, start))
    return out


def quick_eval(model, diffusion, sampler, windows: list, guidance: float, steps: int, device: str,
               chunk: int = 16) -> dict:
    """Generate each window once (seeded by its index) and score hits against the replay's own hits:
    median timing F (±30 ms) and median generated/real hit-count ratio."""
    import torch
    from kata.motion.dataset import decode_hits
    fs, ratios = [], []
    for c0 in range(0, len(windows), chunk):
        part = windows[c0:c0 + chunk]
        ex = [sampler.example(t, s) for t, s in part]
        gens = [torch.Generator().manual_seed(c0 + i) for i in range(len(part))]
        x_shape = ex[0]["x"].shape
        with torch.inference_mode():
            tokens = model.encode_audio(torch.from_numpy(np.stack([e["audio"] for e in ex])).to(device))
            x = diffusion.ddim(model, tokens, torch.tensor([e["label"] for e in ex], device=device),
                               torch.tensor([e["nps"] for e in ex], dtype=torch.float32, device=device),
                               lambda: torch.stack([torch.randn(x_shape, generator=g) for g in gens]).to(device),
                               steps, guidance)
        x = x.float().cpu().numpy()
        for (t, s), xi in zip(part, x):
            g0 = t.frame0 + s
            gen = np.concatenate([decode_hits(xi[:, HIT[h]], xi[:, OFF[h]], g0) for h in (0, 1)])
            raw = np.asarray(sampler.motion.load(t.score_id)[s:s + sampler.window], np.float32)
            real = np.concatenate(real_hits(raw, g0))
            if len(real):
                fs.append(timing_f(gen, real))
                ratios.append(len(np.unique(np.round(gen, 3))) / len(np.unique(real)))
    return {"val_timing_f": float(np.median(fs)) if fs else None,
            "val_rate_ratio": float(np.median(ratios)) if ratios else None, "n": len(fs)}


# --------------------------------------------------------------------------- split evaluation (§7.1, §8)

def spearman(a, b) -> float:
    import polars as pl
    ra = pl.Series(np.asarray(a, np.float64)).rank("average").to_numpy()
    rb = pl.Series(np.asarray(b, np.float64)).rank("average").to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 2 else float("nan")


def _overlap(f0a: int, na: int, f0b: int, nb: int) -> tuple[slice, slice]:
    """Rows of two tracks that cover the same absolute frames."""
    lo, hi = max(f0a, f0b), max(min(f0a + na, f0b + nb), max(f0a, f0b))
    return slice(lo - f0a, hi - f0a), slice(lo - f0b, hi - f0b)


def evaluate_samples(paths, run_id: str, split: str, targets, sample_dir) -> tuple:
    """Rows for every generated sample in sample_dir and every real replay of the targets, plus the
    pooled plausibility percentiles and the real frame-to-frame velocity-jump p99.

    targets: sample.eval_targets() rows. Returns (rows DataFrame, pooled dict)."""
    import polars as pl
    from kata.motion.prepare import MotionStore
    from kata.motion.sample import load_sample, sample_path
    from kata.motion.train import Run, tracks_table
    from kata.validate.rules import nps as nps_of
    keys = ["hash", "characteristic", "difficulty"]
    store = MotionStore(paths.motion)
    meta = tracks_table(paths, Run(paths, run_id).config().pilot).select("score_id", "frame0")
    frame0 = dict(zip(meta["score_id"], meta["frame0"]))
    notes = (pl.scan_parquet(paths.processed / "notes.parquet").join(targets.select(keys).lazy(), on=keys, how="semi")
             .select(keys + ["time_s"]).collect())
    note_times = {k: g["time_s"].to_numpy() for k, g in notes.partition_by(keys, as_dict=True).items()}
    pooled = {"generated": {q: [] for q in QUANTITIES}, "real": {q: [] for q in QUANTITIES}}
    rows, real_jumps = [], []
    for r in targets.iter_rows(named=True):
        key = (r["hash"], r["characteristic"], r["difficulty"])
        base = {"run": run_id, "split": split, **dict(zip(keys, key)), "audio_sha256": r["audio_sha256"],
                "label": r["label"], "nps": r["nps"], "real_hps": r["real_hps"]}
        reals = []
        for sid in r["score_ids"]:
            raw = np.asarray(store.load(sid), np.float32)
            valid = store.valid(sid)
            hits = real_hits(raw, frame0[sid])
            for q, v in quantities(raw[:, :27], hits, frame0[sid], valid).items():
                pooled["real"][q].append(v)
            good, total = agreement(raw[:, :27], hits, frame0[sid])
            real_jumps.append(velocity_jumps(raw[:, :27])[valid[1:] & valid[:-1]])
            reals.append((raw[:, :27], frame0[sid], valid))
            rows.append({**base, "source": "real", "mode": "replay", "seed": None, "score_id": sid,
                         "timing_f": timing_f(np.concatenate(hits), note_times[key]),
                         "hit_rate": nps_of(np.unique(np.concatenate(hits))), "agree_good": good,
                         "agree_total": total})
        pair = []
        for i in range(len(reals)):
            for k in range(i + 1, len(reals)):
                (a, fa, va), (b, fb, vb) = reals[i], reals[k]
                sa, sb = _overlap(fa, len(a), fb, len(b))
                ok = va[sa] & vb[sb]
                if ok.sum() > FPS:
                    pair.append(tip_distance(a[sa][ok], b[sb][ok]))
        if pair:
            rows[-len(reals)]["diversity"] = float(np.mean(pair))
        gen = {}
        for mode, seed in (("full", 0), ("full", 1), ("label_only", 0)):
            path = sample_path(sample_dir, key, mode, seed)
            if not path.exists():
                continue
            s = load_sample(path)
            gen[(mode, seed)] = len(rows)
            hits = (s["hits_left"], s["hits_right"])
            if mode == "full":
                for q, v in quantities(s["motion"], hits).items():
                    pooled["generated"][q].append(v)
            good, total = agreement(s["motion"], hits)
            rows.append({**base, "source": "generated", "mode": mode, "seed": seed, "score_id": None,
                         "timing_f": timing_f(np.concatenate(hits), note_times[key]),
                         "hit_rate": nps_of(np.unique(np.round(np.concatenate(hits), 4))),
                         "agree_good": good, "agree_total": total,
                         "seam_jumps": velocity_jumps(s["motion"], s["joins"]).tolist(), "_motion": s["motion"]})
        if ("full", 0) in gen and ("full", 1) in gen:
            rows[gen[("full", 0)]]["diversity"] = tip_distance(rows[gen[("full", 0)]]["_motion"],
                                                               rows[gen[("full", 1)]]["_motion"])
    for row in rows:
        row.pop("_motion", None)
    jumps = np.concatenate(real_jumps) if real_jumps else np.zeros(0)
    out = {}
    for src, d in pooled.items():
        out[src] = {}
        for q, parts in d.items():
            v = np.concatenate(parts) if parts else np.zeros(0)
            out[src][q] = [float(np.percentile(v, 50)), float(np.percentile(v, 95))] if len(v) else None
    out["real_jump_p99"] = float(np.percentile(jumps, 99)) if len(jumps) else None
    df = pl.DataFrame(rows, infer_schema_length=None)
    for col, dtype in (("diversity", pl.Float64), ("seam_jumps", pl.List(pl.Float64))):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    return df, out


def criteria(rows, pooled: dict, baseline=None) -> list[dict]:
    """Acceptance criteria 1-6 (§8) from evaluate_samples' output. baseline: baseline_eval rows
    (audio_sha256, difficulty, timing_f) for criterion 1, or None to skip it."""
    import polars as pl
    from kata.validate.rules import LABELS
    gen = rows.filter(pl.col("source") == "generated")
    full = gen.filter(pl.col("mode") == "full")
    full0 = full.filter(pl.col("seed") == 0)
    real = rows.filter(pl.col("source") == "real")
    out = []
    if baseline is not None:                                    # 1. same songs and labels as the baseline
        ours = full0.sort("hash").group_by("audio_sha256", "difficulty").agg(pl.col("timing_f").first())
        both = ours.join(baseline.select("audio_sha256", "difficulty", pl.col("timing_f").alias("base_f")),
                         on=["audio_sha256", "difficulty"]).drop_nulls()
        per_label = both.group_by("difficulty").agg(pl.col("timing_f").median(), pl.col("base_f").median())
        wins = int((per_label["timing_f"] > per_label["base_f"]).sum())
        ok = both.height > 0 and both["timing_f"].median() > both["base_f"].median() \
            and wins >= min(4, per_label.height)
        value = (f"median F {both['timing_f'].median():.3f} vs. baseline {both['base_f'].median():.3f}; "
                 f"ahead in {wins}/{per_label.height} labels") if both.height else "no common songs"
        out.append({"criterion": 1, "name": "hit timing vs. baseline", "value": value, "pass": bool(ok)})
    worst, ok2 = 0.0, True                                      # 2. p50, p95 within ±15 % of real
    for q in QUANTITIES:
        g, r = pooled["generated"].get(q), pooled["real"].get(q)
        if not g or not r:
            ok2 = False
            continue
        for gi, ri in zip(g, r):
            dev = abs(gi / ri - 1) if ri else float("inf")
            worst, ok2 = max(worst, dev), ok2 and dev <= 0.15
    out.append({"criterion": 2, "name": "physical plausibility", "value": f"worst deviation {worst:.1%}",
                "pass": bool(ok2)})
    g_rate = full["agree_good"].sum() / max(full["agree_total"].sum(), 1)    # 3. hits agree with motion
    r_rate = real["agree_good"].sum() / max(real["agree_total"].sum(), 1)
    out.append({"criterion": 3, "name": "hits agree with motion", "value": f"{g_rate:.1%} vs. real {r_rate:.1%}",
                "pass": bool(g_rate >= r_rate - 0.05)})
    rho = spearman(full0["nps"], full0["hit_rate"])                          # 4. control
    ratio = float((full0["hit_rate"] / full0["real_hps"] - 1).abs().median() or 0.0) if full0.height else float("nan")
    by_label = dict(gen.filter(pl.col("mode") == "label_only").group_by("label").agg(pl.col("hit_rate").median())
                    .iter_rows())
    order = [by_label[lab] for lab in LABELS if lab in by_label]
    increasing = all(b > a for a, b in zip(order, order[1:]))
    out.append({"criterion": 4, "name": "control",
                "value": f"Spearman {rho:.2f}; median |rate/real - 1| {ratio:.2f}; label-only increasing: {increasing}",
                "pass": bool(rho >= 0.9 and ratio <= 0.15 and increasing)})
    g_div, r_div = full0["diversity"].drop_nulls().mean(), real["diversity"].drop_nulls().mean()  # 5. diversity
    out.append({"criterion": 5, "name": "diversity",
                "value": f"{g_div if g_div is not None else float('nan'):.3f} m vs. real "
                         f"{r_div if r_div is not None else float('nan'):.3f} m",
                "pass": bool(g_div is not None and r_div and g_div >= 0.5 * r_div)})
    lists = [np.asarray(j, np.float64) for j in gen["seam_jumps"].drop_nulls().to_list()]   # 6. seams
    jumps = np.concatenate(lists) if lists else np.zeros(0)
    p99 = pooled.get("real_jump_p99")
    share = float((jumps <= p99).mean()) if len(jumps) and p99 is not None else 1.0
    out.append({"criterion": 6, "name": "seams", "value": f"{share:.1%} of {len(jumps)} joins within real p99",
                "pass": bool(share >= 0.99)})
    return out
