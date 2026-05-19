# scripts/make_lads_split_specs.py
#
# DEPRECATED — DO NOT USE FOR MAIN RESULTS
# =========================================
# This script pre-computes split specs (wstart, gap_start) arrays and saves them
# as .pkl files.  It uses the old window-start-based disjoint split logic which is
# methodologically unsound (overlapping train/val/test windows at segment level).
#
# It is NOT imported by any training script and is NOT part of the clean segment-level
# pipeline introduced in src/segment_split.py.
#
# Kept for archival and diagnostic purposes only.
# For any new experiments, use src.segment_split.build_star_split().
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import zlib
import numpy as np

from src.io_gaps import load_star
from src.protocol import choose_big_gap_points


def stable_star_seed(base_seed: int, star_name: str) -> int:
    # Stable across file ordering (better than seed+1000*i)
    h = zlib.adler32(star_name.encode("utf-8"))  # deterministic
    return int(base_seed + (h % 1_000_000))


def make_specs_for_star(
    star_path: Path,
    dt_factor: float,
    which_gap: str,
    L: int,
    min_context: int,
    n_train: int,
    n_val: int,
    n_test: int,
    base_seed: int,
) -> dict:
    star = load_star(star_path, dt_factor=dt_factor)

    seg_lengths = [(e - s + 1, s, e) for (s, e) in star.segments]
    if len(seg_lengths) == 0:
        return {
            "star_file": star_path.name,
            "skipped": 1,
            "skip_reason": "no_segments",
        }

    seg_lengths.sort(reverse=True)
    max_len, s0, e0 = seg_lengths[0]
    if max_len < L:
        return {
            "star_file": star_path.name,
            "skipped": 1,
            "skip_reason": f"max_segment_len={max_len}<L={L}",
            "max_segment_len": int(max_len),
            "n_points": int(len(star.t)),
            "n_segments": int(len(star.segments)),
            "n_real_gaps": int(len(star.real_gap_idx)),
        }

    # choose gap length
    cap_points = int(0.5 * L)
    gap_choice = choose_big_gap_points(
        star,
        which=which_gap,
        min_points=200,
        default_points_if_no_gaps=800,
        cap_points=cap_points,
    )
    G = int(gap_choice.gap_points)

    # ensure fits with min_context
    max_fit = (L - 2 * min_context - 1)
    gap_adjusted = 0
    if G > max_fit:
        G = max_fit
        gap_adjusted = 1

    g0 = min_context
    g1 = L - min_context - G
    if g1 < g0:
        return {
            "star_file": star_path.name,
            "skipped": 1,
            "skip_reason": f"gap_too_big_even_after_adjust: G={G}, L={L}, min_context={min_context}",
        }

    seed = stable_star_seed(base_seed, star_path.name)

    # Try disjoint split on the longest segment
    starts = np.arange(s0, e0 - L + 2, dtype=np.int64)
    N = starts.size

    def gen_disjoint(seed_off: int, pool: np.ndarray, n: int):
        r = np.random.default_rng(seed + seed_off)
        wstarts = r.choice(pool, size=n, replace=True).astype(np.int64)
        gap_starts = r.integers(g0, g1 + 1, size=n, dtype=np.int64)
        return wstarts, gap_starts

    split_kind = "disjoint"
    if N < 10:
        split_kind = "random"

    if split_kind == "disjoint":
        train_frac = 0.70
        val_frac = 0.15
        n_train_range = max(1, int(train_frac * N))
        n_val_range = max(1, int(val_frac * N))
        if n_train_range + n_val_range >= N:
            # keep at least 1 for test
            n_train_range = max(1, N - 2)
            n_val_range = 1

        train_pool = starts[:n_train_range]
        val_pool = starts[n_train_range:n_train_range + n_val_range]
        test_pool = starts[n_train_range + n_val_range:]

        if train_pool.size == 0 or val_pool.size == 0 or test_pool.size == 0:
            split_kind = "random"
        else:
            tr_w, tr_g = gen_disjoint(0, train_pool, n_train)
            va_w, va_g = gen_disjoint(1, val_pool, n_val)
            te_w, te_g = gen_disjoint(2, test_pool, n_test)

    if split_kind == "random":
        # random over eligible segments (same idea as sample_one)
        eligible = [(s, e) for (s, e) in star.segments if (e - s + 1) >= L]
        if not eligible:
            return {
                "star_file": star_path.name,
                "skipped": 1,
                "skip_reason": "no_eligible_segments_for_random",
            }

        def gen_random(seed_off: int, n: int):
            r = np.random.default_rng(seed + seed_off)
            wstarts = np.empty((n,), dtype=np.int64)
            gap_starts = r.integers(g0, g1 + 1, size=n, dtype=np.int64)
            for i in range(n):
                seg_s, seg_e = eligible[int(r.integers(0, len(eligible)))]
                wstarts[i] = int(r.integers(seg_s, seg_e - L + 2))
            return wstarts, gap_starts

        tr_w, tr_g = gen_random(0, n_train)
        va_w, va_g = gen_random(1, n_val)
        te_w, te_g = gen_random(2, n_test)

    return {
        "star_file": star_path.name,
        "skipped": 0,
        "skip_reason": "",
        "seed": int(seed),
        "split_kind": split_kind,

        "L": int(L),
        "min_context": int(min_context),
        "which_gap": which_gap,
        "gap_percentile": float(gap_choice.percentile),
        "gap_points_used": int(G),
        "gap_adjusted": int(gap_adjusted),

        "dt_factor": float(dt_factor),

        "train_wstart": tr_w,
        "train_gap_start": tr_g,
        "val_wstart": va_w,
        "val_gap_start": va_g,
        "test_wstart": te_w,
        "test_gap_start": te_g,

        "n_points": int(len(star.t)),
        "n_segments": int(len(star.segments)),
        "n_real_gaps": int(len(star.real_gap_idx)),
        "max_segment_len": int(max_len),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/LADS")
    ap.add_argument("--out_pkl", type=str, default="outputs/splits/lads_splits_L2048_p95.pkl")

    ap.add_argument("--dt_factor", type=float, default=5.0)
    ap.add_argument("--which_gap", type=str, default="p95", choices=["p90", "p95"])
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--min_context", type=int, default=256)

    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)

    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.dat"))
    if not files:
        raise RuntimeError(f"No .dat files found in {data_dir}")

    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    specs = {}
    for i, star_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {star_path.name}")
        spec = make_specs_for_star(
            star_path=star_path,
            dt_factor=args.dt_factor,
            which_gap=args.which_gap,
            L=args.L,
            min_context=args.min_context,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            base_seed=args.seed,
        )
        specs[star_path.name] = spec

    with open(out_pkl, "wb") as f:
        pickle.dump(specs, f)

    n_skipped = sum(int(specs[k].get("skipped", 0)) for k in specs)
    print(f"\nSaved split specs to {out_pkl} (stars={len(specs)}, skipped={n_skipped})")


if __name__ == "__main__":
    main()
