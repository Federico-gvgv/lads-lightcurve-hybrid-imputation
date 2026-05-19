# src/window_sampler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io_gaps import StarData
from .fourier_baseline import FourierModel, fit_predict_fourier
from .fourier_dynamic import fit_predict_fourier_dynamic


@dataclass
class Sample:
    y_true: np.ndarray   # (L,) normalized ground truth
    y_in: np.ndarray     # (L,) normalized input (Fourier-filled in gap)
    obs_mask: np.ndarray # (L,) 1 observed / 0 missing
    gap_start: int
    gap_len: int
    mu: float
    sd: float


def zscore(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """Standard z-score using all points (NOT suitable if part of x is 'missing')."""
    mu = float(np.mean(x))
    sd = float(np.std(x))
    sd = max(sd, eps)
    return (x - mu) / sd, mu, sd


def zscore_obs_only(y: np.ndarray, obs_mask: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """
    Normalize using ONLY observed points (obs_mask==1).
    Prevents leakage from synthetic missing region when the gap is masked.
    """
    obs = obs_mask.astype(bool)
    if obs.sum() < 2:
        mu = float(np.mean(y))
        sd = float(np.std(y))
    else:
        mu = float(np.mean(y[obs]))
        sd = float(np.std(y[obs]))
    sd = max(sd, eps)
    return (y - mu) / sd, mu, sd


def sample_one(
    star: StarData,
    L: int,
    gap_len: int,
    rng: np.random.Generator,
    min_context: int = 256,
    normalize: bool = True,
    fourier: Optional[FourierModel] = None,
    fourier_mode: str = "static",  # "static" or "dynamic"
) -> Sample:
    """
    Pick a segment long enough for L, extract length-L chunk, place a synthetic gap of length gap_len,
    fill gap with Fourier warm start (if provided), return normalized arrays.

    NOTE: normalization is done OBSERVED-ONLY (no leakage from the synthetic gap).
    """
    if gap_len >= L - 2 * min_context:
        raise ValueError(f"gap_len={gap_len} too big for L={L} with min_context={min_context}")

    eligible = [(s, e) for (s, e) in star.segments if (e - s + 1) >= L]
    if not eligible:
        raise RuntimeError("No continuous segment long enough. Reduce L or check segmentation.")

    seg_s, seg_e = eligible[int(rng.integers(0, len(eligible)))]
    wstart = int(rng.integers(seg_s, seg_e - L + 2))
    wend = wstart + L

    y = star.y[wstart:wend].astype(np.float64)

    # Place gap inside [min_context, L-min_context-gap_len]
    g0 = min_context
    g1 = L - min_context - gap_len
    gap_start = int(rng.integers(g0, g1 + 1))

    obs_mask = np.ones((L,), dtype=np.float32)
    obs_mask[gap_start:gap_start + gap_len] = 0.0

    if normalize:
        y_norm, mu, sd = zscore_obs_only(y, obs_mask)
    else:
        y_norm, mu, sd = y.copy(), 0.0, 1.0

    y_in = y_norm.copy()

    if fourier is not None:
        # Fit on observed points and predict all, then fill gap
        if fourier_mode == "dynamic":
            y_hat = fit_predict_fourier_dynamic(
                fourier.A_full,
                y_norm,
                obs_mask,
                gap_start=gap_start,
                gap_len=gap_len,
            )
        else:
            y_hat = fit_predict_fourier(fourier.A_full, y_norm, obs_mask)

        miss = obs_mask < 0.5
        y_in[miss] = y_hat[miss]
    else:
        # no warm start
        y_in[obs_mask < 0.5] = 0.0

    # Invariant check: outside the synthetic gap, y_in should equal y_true
    outside = np.ones(L, dtype=bool)
    outside[gap_start:gap_start + gap_len] = False
    assert np.mean(np.abs(y_in[outside] - y_norm[outside])) < 1e-6

    return Sample(
        y_true=y_norm.astype(np.float32),
        y_in=y_in.astype(np.float32),
        obs_mask=obs_mask.astype(np.float32),
        gap_start=gap_start,
        gap_len=gap_len,
        mu=mu,
        sd=sd,
    )


def make_split(
    star: StarData,
    L: int,
    gap_len: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    min_context: int,
    fourier: Optional[FourierModel],
    fourier_mode: str = "static",
) -> dict:
    """
    [LEGACY — DO NOT USE FOR MAIN RESULTS]

    Random-window split.  train/val/test windows can overlap in time (fully leaky).
    This function is kept only for backward compatibility.
    Use `src.segment_split.build_star_split()` for the clean segment-level protocol.
    """
    def gen(n: int, seed_off: int):
        r = np.random.default_rng(seed + seed_off)
        return [
            sample_one(
                star=star,
                L=L,
                gap_len=gap_len,
                rng=r,
                min_context=min_context,
                fourier=fourier,
                fourier_mode=fourier_mode,  # IMPORTANT: propagate mode
            )
            for _ in range(n)
        ]

    return {
        "train": gen(n_train, 0),
        "val":   gen(n_val,   1),
        "test":  gen(n_test,  2),
    }


def make_split_disjoint(
    star: StarData,
    L: int,
    gap_len: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    min_context: int,
    fourier: Optional[FourierModel],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    fourier_mode: str = "static",
) -> dict:
    """
    [LEGACY — DO NOT USE FOR MAIN RESULTS]

    Splits WINDOW START positions (not full temporal segments) of the longest
    continuous segment into 70/15/15 ranges.  Adjacent train/val boundary windows
    overlap in L-1 points, so this is NOT a clean disjoint split.
    Kept only for backward compatibility.
    Use `src.segment_split.build_star_split()` for the clean segment-level protocol.
    """
    eligible = [(s, e) for (s, e) in star.segments if (e - s + 1) >= L]
    if not eligible:
        raise RuntimeError(f"No continuous segment long enough for L={L}.")

    seg_s, seg_e = max(eligible, key=lambda x: x[1] - x[0])

    # All valid window starts for this segment (inclusive)
    starts = np.arange(seg_s, seg_e - L + 2, dtype=np.int64)
    N = starts.size
    if N < 10:
        raise RuntimeError("Not enough valid window starts for a disjoint split.")

    n_train_range = max(1, int(train_frac * N))
    n_val_range = max(1, int(val_frac * N))
    if n_train_range + n_val_range >= N:
        n_train_range = max(1, int(0.7 * (N - 2)))
        n_val_range = 1

    train_starts = starts[:n_train_range]
    val_starts   = starts[n_train_range:n_train_range + n_val_range]
    test_starts  = starts[n_train_range + n_val_range:]

    def sample_one_fixed_start(wstart: int, r: np.random.Generator) -> Sample:
        wend = wstart + L
        y = star.y[wstart:wend].astype(np.float64)

        g0 = min_context
        g1 = L - min_context - gap_len
        if g1 < g0:
            raise ValueError(f"gap_len={gap_len} too big for L={L} with min_context={min_context}")
        gap_start = int(r.integers(g0, g1 + 1))

        obs_mask = np.ones((L,), dtype=np.float32)
        obs_mask[gap_start:gap_start + gap_len] = 0.0

        y_norm, mu, sd = zscore_obs_only(y, obs_mask)

        y_in = y_norm.copy()
        if fourier is not None:
            if fourier_mode == "dynamic":
                y_hat = fit_predict_fourier_dynamic(
                    fourier.A_full,
                    y_norm,
                    obs_mask,
                    gap_start=gap_start,
                    gap_len=gap_len,
                )
            else:
                y_hat = fit_predict_fourier(fourier.A_full, y_norm, obs_mask)

            miss = obs_mask < 0.5
            y_in[miss] = y_hat[miss]
        else:
            y_in[obs_mask < 0.5] = 0.0

        outside = np.ones(L, dtype=bool)
        outside[gap_start:gap_start + gap_len] = False
        assert np.mean(np.abs(y_in[outside] - y_norm[outside])) < 1e-6

        return Sample(
            y_true=y_norm.astype(np.float32),
            y_in=y_in.astype(np.float32),
            obs_mask=obs_mask.astype(np.float32),
            gap_start=gap_start,
            gap_len=gap_len,
            mu=mu,
            sd=sd,
        )

    def gen(n: int, pool: np.ndarray, seed_off: int):
        if pool.size == 0:
            raise RuntimeError("Empty start pool for one split; reduce L or change split fractions.")
        r = np.random.default_rng(seed + seed_off)
        picks = r.choice(pool, size=n, replace=True)
        return [sample_one_fixed_start(int(ws), r) for ws in picks]

    return {
        "train": gen(n_train, train_starts, 0),
        "val":   gen(n_val,   val_starts,   1),
        "test":  gen(n_test,  test_starts,  2),
    }
