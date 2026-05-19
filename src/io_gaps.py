# src/io_gaps.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class StarData:
    t: np.ndarray  # (N,) days
    y: np.ndarray  # (N,) flux
    dt_med_days: float
    dt_med_min: float
    real_gap_idx: np.ndarray  # indices i where dt[i]=t[i+1]-t[i] is a real gap
    segments: List[Tuple[int, int]]  # inclusive [start,end]


def load_dat(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(str(path), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad format in {path}. Expected 2 columns time flux.")
    t = arr[:, 0]
    y = arr[:, 1]
    idx = np.argsort(t)
    return t[idx], y[idx]


def detect_real_gaps(t: np.ndarray, dt_factor: float = 5.0) -> tuple[np.ndarray, float, float]:
    dt = np.diff(t)
    if dt.size == 0:
        return np.array([], dtype=np.int64), np.nan, np.nan
    dt_med_days = float(np.median(dt))
    dt_med_min = dt_med_days * 24.0 * 60.0
    thresh = dt_factor * dt_med_days
    gap_idx = np.where(dt > thresh)[0].astype(np.int64)
    return gap_idx, dt_med_days, dt_med_min


def make_segments(n: int, gap_idx: np.ndarray) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    if gap_idx.size == 0:
        return [(0, n - 1)]
    segs: List[Tuple[int, int]] = []
    start = 0
    for i in gap_idx.tolist():
        end = i
        if end >= start:
            segs.append((start, end))
        start = i + 1
    if start <= n - 1:
        segs.append((start, n - 1))
    return segs


def load_star(path: str | Path, dt_factor: float = 5.0) -> StarData:
    t, y = load_dat(path)
    gap_idx, dt_med_days, dt_med_min = detect_real_gaps(t, dt_factor=dt_factor)
    segs = make_segments(len(t), gap_idx)
    return StarData(
        t=t, y=y,
        dt_med_days=dt_med_days,
        dt_med_min=dt_med_min,
        real_gap_idx=gap_idx,
        segments=segs
    )
