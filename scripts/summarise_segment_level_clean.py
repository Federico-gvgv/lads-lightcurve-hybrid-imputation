#!/usr/bin/env python
"""
scripts/summarise_segment_level_clean.py

Reads all per-star CSVs from the segment-level clean experiment run and
produces a compact summary CSV.

Called automatically at the end of run_segment_level_clean_all.sh, but can
also be run standalone after training finishes:

    python scripts/summarise_segment_level_clean.py \
        --out_dir outputs/segment_level_clean \
        --runtime_csv outputs/segment_level_clean/runtime_clean_segment_level.csv \
        --summary_csv outputs/segment_level_clean/summary_clean_segment_level.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median(values: List[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _safe_float(v: str) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Per-experiment summary
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    ("tcn_clean",                         "tcn_clean.csv"),
    ("transformer_clean",                  "transformer_clean.csv"),
    ("conv_transformer_clean",             "conv_transformer_clean.csv"),
    ("tcn_basegate_r2_clean",             "tcn_basegate_r2_clean.csv"),
    ("transformer_basegate_r2_clean",     "transformer_basegate_r2_clean.csv"),
    ("conv_transformer_basegate_r2_clean","conv_transformer_basegate_r2_clean.csv"),
]


def summarise_one(exp_name: str, csv_path: Path, skipped_csv: Path,
                  runtime_row: dict) -> dict:
    """Build one summary row from one experiment's results."""

    mse_vals, mape_vals, r2_vals = [], [], []
    train_w, val_w, test_w = [], [], []
    n_balance_warnings = 0
    n_stars = 0

    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                n_stars += 1
                v = _safe_float(row.get("hybrid_mse", ""))
                if v is not None:
                    mse_vals.append(v)
                v = _safe_float(row.get("hybrid_mape", ""))
                if v is not None:
                    mape_vals.append(v)
                v = _safe_float(row.get("hybrid_r2", ""))
                if v is not None:
                    r2_vals.append(v)
                v = _safe_float(row.get("train_weight", ""))
                if v is not None:
                    train_w.append(v)
                v = _safe_float(row.get("val_weight", ""))
                if v is not None:
                    val_w.append(v)
                v = _safe_float(row.get("test_weight", ""))
                if v is not None:
                    test_w.append(v)
                if str(row.get("split_balance_warning", "")).lower() in ("true", "1"):
                    n_balance_warnings += 1

    n_skipped = 0
    if skipped_csv.exists():
        with open(skipped_csv, newline="") as f:
            n_skipped = sum(1 for _ in csv.DictReader(f))

    elapsed = _safe_float(runtime_row.get("elapsed_seconds", ""))

    return {
        "experiment":              exp_name,
        "n_stars":                 n_stars,
        "n_skipped":               n_skipped,
        "mse_mean":                f"{_mean(mse_vals):.4g}"   if mse_vals  else "",
        "mse_median":              f"{_median(mse_vals):.4g}" if mse_vals  else "",
        "mape_mean":               f"{_mean(mape_vals):.4g}"  if mape_vals else "",
        "mape_median":             f"{_median(mape_vals):.4g}"if mape_vals else "",
        "r2_mean":                 f"{_mean(r2_vals):.4f}"    if r2_vals   else "",
        "r2_median":               f"{_median(r2_vals):.4f}"  if r2_vals   else "",
        "train_weight_median":     f"{_median(train_w):.4f}"  if train_w   else "",
        "val_weight_median":       f"{_median(val_w):.4f}"    if val_w     else "",
        "test_weight_median":      f"{_median(test_w):.4f}"   if test_w    else "",
        "n_split_balance_warnings":n_balance_warnings,
        "runtime_total_seconds":   int(elapsed) if elapsed is not None else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",     type=str, required=True,
                    help="Directory containing per-star CSVs")
    ap.add_argument("--runtime_csv", type=str, required=True,
                    help="Runtime tracking CSV produced by the shell script")
    ap.add_argument("--summary_csv", type=str, required=True,
                    help="Output summary CSV path")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # Load runtime rows indexed by experiment name
    runtime_by_exp: dict = {}
    runtime_csv = Path(args.runtime_csv)
    if runtime_csv.exists():
        with open(runtime_csv, newline="") as f:
            for row in csv.DictReader(f):
                runtime_by_exp[row.get("experiment", "")] = row

    fieldnames = [
        "experiment", "n_stars", "n_skipped",
        "mse_mean", "mse_median",
        "mape_mean", "mape_median",
        "r2_mean", "r2_median",
        "train_weight_median", "val_weight_median", "test_weight_median",
        "n_split_balance_warnings",
        "runtime_total_seconds",
    ]

    rows = []
    for exp_name, csv_file in EXPERIMENTS:
        csv_path     = out_dir / csv_file
        skipped_csv  = out_dir / csv_file.replace(".csv", "_skipped.csv")
        runtime_row  = runtime_by_exp.get(exp_name, {})
        row = summarise_one(exp_name, csv_path, skipped_csv, runtime_row)
        rows.append(row)
        print(
            f"  {exp_name}: n_stars={row['n_stars']}  n_skipped={row['n_skipped']}"
            f"  r2_median={row['r2_median']}  elapsed={row['runtime_total_seconds']}s"
        )

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSummary written to: {summary_csv}")


if __name__ == "__main__":
    main()
