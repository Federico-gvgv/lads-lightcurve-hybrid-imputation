# src/protocol.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

from .io_gaps import StarData


@dataclass(frozen=True)
class GapChoice:
    gap_points: int
    percentile: float
    n_real_gaps: int


def real_gaps_to_points(star: StarData) -> np.ndarray:
    """
    Convert real gap durations (dt where dt > thresh) into approximate missing sample counts (points).
    points ≈ round(dt / dt_med) - 1
    """
    if star.real_gap_idx.size == 0:
        return np.array([], dtype=np.int64)
    dt = np.diff(star.t)  # days
    gap_dt = dt[star.real_gap_idx]
    pts = np.rint(gap_dt / max(star.dt_med_days, 1e-12)).astype(np.int64) - 1
    pts = np.maximum(pts, 1)
    return pts


def choose_big_gap_points(
    star: StarData,
    which: Literal["p90", "p95", "max"] = "p90",
    min_points: int = 200,
    default_points_if_no_gaps: int = 800,
    cap_points: Optional[int] = None,
) -> GapChoice:
    """
    Pick big gap length for THIS star:
      - p90 / p95 of real gaps in points (robust vs outliers), OR
      - max real gap in points (stress-test),
      - clamp to [min_points, cap_points] if cap provided.
    """
    pts = real_gaps_to_points(star)
    if pts.size == 0:
        g = default_points_if_no_gaps
        if cap_points is not None:
            g = min(g, cap_points)
        return GapChoice(gap_points=int(max(g, min_points)), percentile=0.0, n_real_gaps=0)

    if which == "max":
        p = 100.0
        g = int(np.max(pts))
    else:
        p = 90.0 if which == "p90" else 95.0
        g = int(np.percentile(pts.astype(np.float64), p))

    g = max(g, min_points)
    if cap_points is not None:
        g = min(g, cap_points)
    return GapChoice(gap_points=g, percentile=p, n_real_gaps=int(pts.size))
