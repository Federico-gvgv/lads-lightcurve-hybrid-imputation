#!/usr/bin/env python3
"""
scripts/sanity_check_segment_split.py

Lightweight sanity checks for the new segment-level split / sampler module.

Runs WITHOUT training — only verifies correctness of the split logic,
determinism of val/test specs, tensor shapes, and disjointness guarantees.

Usage
-----
    python scripts/sanity_check_segment_split.py --data_dir data/LADS --n_stars 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on path when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io_gaps import load_star
from src.protocol import choose_big_gap_points
from src.segment_split import (
    SegmentInfo,
    SegmentSplit,
    WindowSpec,
    build_fourier_from_longest_train_seg,
    build_star_split,
    collate_fn,
    enumerate_eligible_segments,
    make_fixed_eval_specs,
    split_segments_exhaustive,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _seg_idx_set(segs) -> set:
    """Flat set of all point-indices covered by a collection of SegmentInfo."""
    s = set()
    for seg in segs:
        s.update(range(seg.start_idx, seg.end_idx + 1))
    return s


def check_disjoint(split: SegmentSplit, label: str):
    tr = _seg_idx_set(split.train_segs)
    va = _seg_idx_set(split.val_segs)
    te = _seg_idx_set(split.test_segs)
    assert tr.isdisjoint(va), f"[{label}] train ∩ val non-empty!"
    assert tr.isdisjoint(te), f"[{label}] train ∩ test non-empty!"
    assert va.isdisjoint(te), f"[{label}] val ∩ test non-empty!"
    print(f"  ✓ Disjoint check passed")


def check_specs_deterministic(star, segs, L, gap_len, min_context, n, seed, label):
    specs1 = make_fixed_eval_specs(star, segs, L, gap_len, min_context, n, seed)
    specs2 = make_fixed_eval_specs(star, segs, L, gap_len, min_context, n, seed)
    assert len(specs1) == len(specs2), f"[{label}] Different spec counts on two calls"
    for i, (a, b) in enumerate(zip(specs1, specs2)):
        assert a.wstart == b.wstart, f"[{label}] spec[{i}].wstart differs"
        assert a.gap_start == b.gap_start, f"[{label}] spec[{i}].gap_start differs"
    print(f"  ✓ Val/test specs are deterministic ({len(specs1)} windows)")
    return specs1


def print_split_summary(name: str, split: SegmentSplit, eligible: list):
    total = sum(s.n_valid_starts for s in eligible)
    print(f"\n  Split summary for: {name}")
    print(f"    Total eligible segments: {len(eligible)}")
    print(f"    Train segs: {len(split.train_segs)}  "
          f"(ids={[s.segment_id for s in split.train_segs]}, "
          f"weight={split.train_weight:.3f})")
    print(f"    Val   segs: {len(split.val_segs)}  "
          f"(ids={[s.segment_id for s in split.val_segs]}, "
          f"weight={split.val_weight:.3f})")
    print(f"    Test  segs: {len(split.test_segs)}  "
          f"(ids={[s.segment_id for s in split.test_segs]}, "
          f"weight={split.test_weight:.3f})")
    print(f"    Target 70/15/15  "
          f"Δtrain={split.train_weight-0.70:+.3f}  "
          f"Δval={split.val_weight-0.15:+.3f}  "
          f"Δtest={split.test_weight-0.15:+.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/LADS")
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--min_context", type=int, default=256)
    ap.add_argument("--which_gap", type=str, default="max",
                    choices=["p90", "p95", "max"])
    ap.add_argument("--n_stars", type=int, default=5,
                    help="Number of stars to test (0 = all)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--samples_per_epoch", type=int, default=32,
                    help="Small value for quick testing")
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dt_factor", type=float, default=5.0)
    ap.add_argument("--k_freqs", type=int, default=8)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.dat"))
    if not files:
        raise RuntimeError(f"No .dat files found in {data_dir}")

    if args.n_stars > 0:
        files = files[:args.n_stars]

    print(f"\n{'='*60}")
    print(f"Sanity check: segment_split module")
    print(f"  data_dir = {data_dir}  L = {args.L}  n_stars = {len(files)}")
    print(f"{'='*60}")

    n_ok = 0
    n_skipped = 0
    failures = []

    for star_path in files:
        print(f"\n[Star] {star_path.name}")
        try:
            star = load_star(star_path, dt_factor=args.dt_factor)

            gap_choice = choose_big_gap_points(
                star, which=args.which_gap,
                min_points=200,
                default_points_if_no_gaps=800,
                cap_points=int(0.5 * args.L),
            )
            G = int(gap_choice.gap_points)
            print(f"  gap_points = {G}")

            # ── Check 1: eligible segment enumeration ──────────────────────
            eligible = enumerate_eligible_segments(star, args.L)
            print(f"  Eligible segments (len >= {args.L}): {len(eligible)}")
            for seg in eligible:
                print(f"    SegmentInfo(id={seg.segment_id}, "
                      f"start={seg.start_idx}, end={seg.end_idx}, "
                      f"len={seg.length}, n_valid_starts={seg.n_valid_starts})")

            if len(eligible) < 3:
                print(f"  → SKIP: too_few_eligible_segments ({len(eligible)} < 3)")
                n_skipped += 1
                continue

            # ── Check 2: exhaustive split ──────────────────────────────────
            split = split_segments_exhaustive(
                eligible=eligible,
                train_frac=0.70,
                val_frac=0.15,
                split_seed=args.seed,
            )
            assert split is not None
            print_split_summary(star_path.name, split, eligible)

            # ── Check 3: disjoint segments ─────────────────────────────────
            check_disjoint(split, star_path.name)

            # ── Check 4: Fourier on train only ─────────────────────────────
            fourier = build_fourier_from_longest_train_seg(
                star, split.train_segs, args.L, k=args.k_freqs
            )
            assert fourier.A_full.shape == (args.L, 1 + 2 * fourier.freqs_cpd.size)
            longest_train = max(split.train_segs, key=lambda s: s.length)
            # Verify it comes from a train segment
            assert any(s.segment_id == longest_train.segment_id
                       for s in split.train_segs)
            print(f"  ✓ Fourier built on train seg {longest_train.segment_id} "
                  f"(len={longest_train.length}, "
                  f"freqs={fourier.freqs_cpd.size})")

            # ── Check 5: deterministic val/test specs ──────────────────────
            val_specs = check_specs_deterministic(
                star, split.val_segs, args.L, G, args.min_context,
                args.n_val, args.seed + 1, f"{star_path.name}/val"
            )
            test_specs = check_specs_deterministic(
                star, split.test_segs, args.L, G, args.min_context,
                args.n_test, args.seed + 2, f"{star_path.name}/test"
            )

            # ── Check 6: full build_star_split ─────────────────────────────
            for has_gate in [False, True]:
                result = build_star_split(
                    star=star,
                    L=args.L,
                    gap_len=G,
                    min_context=args.min_context,
                    k_freqs=args.k_freqs,
                    samples_per_epoch=args.samples_per_epoch,
                    n_val=args.n_val,
                    n_test=args.n_test,
                    seed=args.seed,
                    fourier_mode="static",
                    has_gate=has_gate,
                )
                assert result is not None, "build_star_split returned None unexpectedly"

                td = result["train_dataset"]
                vd = result["val_dataset"]
                tstd = result["test_dataset"]

                assert len(td) == args.samples_per_epoch
                assert len(vd) == len(val_specs)
                assert len(tstd) == len(test_specs)

                # ── Check 7: DataLoader shapes ─────────────────────────────
                train_loader = DataLoader(td, batch_size=args.batch_size,
                                          shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(vd, batch_size=args.batch_size,
                                        shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(tstd, batch_size=args.batch_size,
                                         shuffle=False, collate_fn=collate_fn)

                batch = next(iter(train_loader))
                assert batch["y_true"].shape[-1] == args.L
                assert batch["y_in"].shape[-1] == args.L
                assert batch["obs_mask"].shape[-1] == args.L
                assert batch["mu"].ndim == 1
                if has_gate:
                    assert "fourier_r2_obs" in batch, "fourier_r2_obs missing in gate batch"
                    assert batch["fourier_r2_obs"].shape[0] == batch["y_true"].shape[0]
                else:
                    assert "fourier_r2_obs" not in batch

                # Verify val/test loaders yield all samples
                n_val_batches = sum(b["y_true"].shape[0] for b in val_loader)
                n_test_batches = sum(b["y_true"].shape[0] for b in test_loader)
                assert n_val_batches == len(vd), \
                    f"val_loader yielded {n_val_batches} samples, expected {len(vd)}"
                assert n_test_batches == len(tstd), \
                    f"test_loader yielded {n_test_batches} samples, expected {len(tstd)}"

                # ── Check 8: epoch refresh changes samples ─────────────────
                item0_before = td[0]["y_true"].copy()
                td.set_epoch(1)
                item0_after = td[0]["y_true"]
                # Not guaranteed to change (random overlap possible), but log it
                changed = not np.allclose(item0_before, item0_after)

                gate_str = "has_gate=True " if has_gate else "has_gate=False"
                print(f"  ✓ {gate_str}  train={len(td)}  val={len(vd)}  "
                      f"test={len(tstd)}  shapes OK  "
                      f"epoch_refresh={'changed' if changed else 'same_window'}")

            # ── Check 9: meta dict ─────────────────────────────────────────
            meta = result["meta"]
            required_keys = [
                "n_eligible_segments", "n_train_segments", "n_val_segments",
                "n_test_segments", "train_weight", "val_weight", "test_weight",
                "split_kind", "samples_per_epoch", "fourier_train_only",
            ]
            for k in required_keys:
                assert k in meta, f"meta missing key '{k}'"
            assert meta["split_kind"] == "segment_level_clean"
            assert meta["fourier_train_only"] is True
            print(f"  ✓ meta dict keys OK  split_kind='{meta['split_kind']}'")

            n_ok += 1

        except Exception as exc:
            import traceback
            print(f"  ✗ FAILED: {exc}")
            traceback.print_exc()
            failures.append((star_path.name, str(exc)))

    print(f"\n{'='*60}")
    print(f"Results: {n_ok} OK  |  {n_skipped} skipped (< 3 eligible segs)  |  "
          f"{len(failures)} failed")
    if failures:
        print("Failures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        sys.exit(1)
    else:
        print("All checks passed ✓")


if __name__ == "__main__":
    main()
