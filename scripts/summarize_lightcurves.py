from pathlib import Path
import numpy as np
import csv

# Project root = one level above /scripts
ROOT = Path(__file__).resolve().parents[1]

# Read-only input dirs (do NOT write into data/)
DATASETS = {
    "LADS": ROOT / "data" / "LADS",
    "HADS": ROOT / "data" / "HADS",
}

# Output goes to /outputs (safe)
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def summarize_file(fp: Path):
    arr = np.loadtxt(fp)
    t = arr[:, 0]
    t = np.sort(t)

    dt = np.diff(t)
    med_dt = float(np.median(dt)) if len(dt) else float("nan")
    span = float(t[-1] - t[0]) if len(t) else 0.0
    n = int(len(t))

    # "gap" heuristic: anything > 5× median cadence
    gap_thresh = 5 * med_dt if np.isfinite(med_dt) else float("inf")
    gaps = dt[dt > gap_thresh] if len(dt) else np.array([])

    def to_minutes(x_days: float) -> float:
        return float(x_days) * 24.0 * 60.0

    return {
        "file": fp.name,
        "n_points": n,
        "span_days": span,
        "median_dt_days": med_dt,
        "median_dt_min": to_minutes(med_dt) if np.isfinite(med_dt) else float("nan"),
        "n_gaps": int(len(gaps)),
        "max_gap_min": to_minutes(gaps.max()) if len(gaps) else 0.0,
        "p95_gap_min": to_minutes(np.percentile(gaps, 95)) if len(gaps) else 0.0,
    }

def summarize_dataset(name: str, folder: Path):
    files = sorted(folder.glob("*.dat"))
    if not files:
        print(f"[{name}] No .dat files found in {folder}")
        return None

    rows = [summarize_file(fp) for fp in files]

    # Print quick stats
    n_points = np.array([r["n_points"] for r in rows])
    span_days = np.array([r["span_days"] for r in rows])
    dt_min = np.array([r["median_dt_min"] for r in rows])

    print(f"\n=== {name} ===")
    print(f"Folder: {folder}")
    print(f"Files: {len(rows)}")
    print("n_points: min/median/max =", int(n_points.min()), int(np.median(n_points)), int(n_points.max()))
    print("span_days: min/median/max =", float(span_days.min()), float(np.median(span_days)), float(span_days.max()))
    print("cadence (min): min/median/max =", float(np.nanmin(dt_min)), float(np.nanmedian(dt_min)), float(np.nanmax(dt_min)))

    # Save CSV to outputs/
    out_csv = OUT_DIR / f"lightcurve_summary_{name.lower()}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {out_csv}")

    # Show top 5 by length
    rows_sorted = sorted(rows, key=lambda r: r["n_points"], reverse=True)
    print("Top 5 longest by points:")
    for r in rows_sorted[:5]:
        print(" ", r["file"], "n=", r["n_points"], "span_days=", round(r["span_days"], 3),
              "cadence_min=", round(r["median_dt_min"], 3), "max_gap_min=", round(r["max_gap_min"], 1))

    return rows

def main():
    for name, folder in DATASETS.items():
        summarize_dataset(name, folder)

if __name__ == "__main__":
    main()
