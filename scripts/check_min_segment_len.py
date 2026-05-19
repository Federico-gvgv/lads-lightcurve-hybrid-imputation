# scripts/check_min_segment_len.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.io_gaps import load_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/LADS")
    ap.add_argument("--dt_factor", type=float, default=5.0)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--show", type=int, default=10, help="How many failing stars to print")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.dat"))
    if not files:
        raise RuntimeError(f"No .dat files found in {data_dir}")

    ok = []
    fail = []
    noseg = []

    maxlens = []

    for fp in files:
        star = load_star(fp, dt_factor=args.dt_factor)
        if not star.segments:
            noseg.append(fp.name)
            continue

        seg_lengths = [e - s + 1 for (s, e) in star.segments]
        mx = int(max(seg_lengths))
        maxlens.append(mx)

        if mx >= args.L:
            ok.append((fp.name, mx))
        else:
            fail.append((fp.name, mx))

    maxlens_np = np.array(maxlens, dtype=int) if maxlens else np.array([], dtype=int)

    print(f"\nData dir: {data_dir}")
    print(f"dt_factor: {args.dt_factor}")
    print(f"L: {args.L}\n")

    print(f"Total stars: {len(files)}")
    print(f"Stars with no segments: {len(noseg)}")
    print(f"OK (max_segment_len >= L): {len(ok)}")
    print(f"FAIL (max_segment_len <  L): {len(fail)}")

    if maxlens_np.size > 0:
        print("\nMax segment length stats (over stars with >=1 segment):")
        print(f"  min={int(maxlens_np.min())}  p25={int(np.percentile(maxlens_np,25))}  "
              f"p50={int(np.percentile(maxlens_np,50))}  p75={int(np.percentile(maxlens_np,75))}  "
              f"max={int(maxlens_np.max())}")

    if noseg:
        print("\nStars with no segments (first few):")
        for n in noseg[:args.show]:
            print("  ", n)

    if fail:
        print(f"\nStars failing L={args.L} (showing {min(args.show, len(fail))}):")
        for name, mx in sorted(fail, key=lambda x: x[1])[:args.show]:
            print(f"  {name:25s}  max_segment_len={mx}")

    # Optional: suggest a safer L (e.g., p25 of max segment lengths)
    if maxlens_np.size > 0:
        suggested = int(np.percentile(maxlens_np, 25))
        print(f"\nSuggestion: if you want to keep ~75% of stars, try L ≈ p25 = {suggested}.")


if __name__ == "__main__":
    main()
