# src/segment_split.py
"""
Segment-level train/val/test split and window sampling for LADS light curves.

Design principles
-----------------
* The unit of independence is the **continuous segment** (a gap-free run of samples),
  NOT the window start position.
* All segments of length >= L are eligible; the star is skipped when fewer than 3
  eligible segments exist (no clean disjoint split is possible).
* Segment assignment to train/val/test is solved by exhaustive search over all
  valid 3-way partitions that keep each split non-empty, minimising deviation from
  the 70/15/15 target ratio measured in valid window starts.
* Fourier frequencies are estimated using only the **longest TRAIN segment**.
  Val/test evaluation reuses the same train-derived FourierModel; per-window
  coefficients are still fitted on observed (non-gap) points inside each window.
* Val/test windows are FIXED and DETERMINISTIC (same specs every call with same seed).
* Train windows are drawn online; the dataset yields exactly `samples_per_epoch`
  samples (semantically equivalent to the old n_train).
* No leaky fallback is used for main results.  Stars with < 3 eligible segments are
  skipped with `skip_reason="too_few_eligible_segments"`.

Segment indexing convention
---------------------------
`end_idx` is **inclusive** throughout this module, consistent with the existing
`io_gaps.make_segments()` convention. Helper code converts to exclusive endpoints
where needed (e.g. `star.y[s:e+1]`).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .io_gaps import StarData
from .fourier_baseline import (
    FourierModel,
    build_fourier_model_for_star,
    fit_predict_fourier,
)
from .fourier_dynamic import fit_predict_fourier_dynamic


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentInfo:
    """Metadata for a single continuous segment."""
    segment_id: int          # 0-based index in StarData.segments
    start_idx: int           # inclusive start in star.y / star.t
    end_idx: int             # inclusive end   in star.y / star.t
    length: int              # = end_idx - start_idx + 1
    n_valid_starts: int      # = max(0, length - L + 1)

    def as_slice(self) -> slice:
        """Return slice for star.y[seg.as_slice()] (exclusive end)."""
        return slice(self.start_idx, self.end_idx + 1)


@dataclass(frozen=True)
class SegmentSplit:
    """Result of segment-level train/val/test assignment."""
    train_segs: Tuple[SegmentInfo, ...]
    val_segs:   Tuple[SegmentInfo, ...]
    test_segs:  Tuple[SegmentInfo, ...]
    # Achieved weight fractions
    train_weight: float
    val_weight:   float
    test_weight:  float
    split_seed:   int

    def n_train_starts(self) -> int:
        return sum(s.n_valid_starts for s in self.train_segs)

    def n_val_starts(self) -> int:
        return sum(s.n_valid_starts for s in self.val_segs)

    def n_test_starts(self) -> int:
        return sum(s.n_valid_starts for s in self.test_segs)


@dataclass
class WindowSpec:
    """Fixed specification for a single evaluation window."""
    segment_id: int    # which SegmentInfo
    wstart:     int    # absolute index into star.y  (inclusive)
    gap_start:  int    # relative to window start (0-indexed within [0, L))
    gap_len:    int


# ---------------------------------------------------------------------------
# Segment enumeration
# ---------------------------------------------------------------------------

def enumerate_eligible_segments(star: StarData, L: int) -> List[SegmentInfo]:
    """Return SegmentInfo objects for all segments with length >= L."""
    result = []
    for idx, (s, e) in enumerate(star.segments):
        length = e - s + 1
        if length >= L:
            n_vs = length - L + 1
            result.append(SegmentInfo(
                segment_id=idx,
                start_idx=s,
                end_idx=e,
                length=length,
                n_valid_starts=n_vs,
            ))
    return result


# ---------------------------------------------------------------------------
# Exhaustive combinatorial segment assignment
# ---------------------------------------------------------------------------

def split_segments_exhaustive(
    eligible: List[SegmentInfo],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    split_seed: int   = 0,
) -> Optional[SegmentSplit]:
    """
    Assign eligible segments to train/val/test by exhaustive search.

    Objective: minimise the sum of squared deviations from target fractions
    (70/15/15) measured in number of valid window starts.

    Returns None if fewer than 3 eligible segments exist (star must be skipped).

    The search is exhaustive over all 3-way partitions with each split non-empty.
    For the typical case (5-20 eligible segments per star) the search space is
    bounded by the Stirling numbers of the second kind — manageable in practice.
    For m segments and 3 buckets the maximum number of assignments is 3^m - 3*2^m + 3
    (all surjective functions), which stays < 100 000 for m <= 10 and grows
    moderately; for safety we cap at m=15 after which we fall back to a
    deterministic greedy heuristic (still no random/leaky fallback).

    Parameters
    ----------
    eligible : list of SegmentInfo already filtered to length >= L
    train_frac, val_frac : target fractions; test_frac = 1 - train_frac - val_frac
    split_seed : used only for tie-breaking and the greedy fallback shuffle

    Returns
    -------
    SegmentSplit or None
    """
    m = len(eligible)
    if m < 3:
        return None  # caller must skip this star

    test_frac = max(0.0, 1.0 - train_frac - val_frac)
    targets = np.array([train_frac, val_frac, test_frac], dtype=np.float64)

    total_starts = sum(s.n_valid_starts for s in eligible)
    if total_starts == 0:
        return None

    def score(assign: List[int]) -> float:
        """Lower is better. assign[i] in {0,1,2}."""
        counts = np.zeros(3, dtype=np.float64)
        for i, a in enumerate(assign):
            counts[a] += eligible[i].n_valid_starts
        fracs = counts / total_starts
        return float(np.sum((fracs - targets) ** 2))

    # --- Exhaustive search (m <= 15) ---
    if m <= 15:
        best_score = float("inf")
        best_assign = None

        # Iterate over all 3^m assignments that have at least one segment in each bucket
        for assign in itertools.product(range(3), repeat=m):
            buckets = [0, 0, 0]
            for a in assign:
                buckets[a] += 1
            if any(b == 0 for b in buckets):
                continue
            s = score(list(assign))
            if s < best_score:
                best_score = s
                best_assign = list(assign)
    else:
        # Greedy fallback: sort by length desc, assign round-robin with a bias
        # toward the largest bucket (train).  Deterministic via split_seed shuffle.
        rng = np.random.default_rng(split_seed)
        order = np.argsort([-seg.n_valid_starts for seg in eligible])
        # Shuffle equal-length segments for variety
        order = order.tolist()
        accumulated = np.zeros(3, dtype=np.float64)
        best_assign = [0] * m
        for rank, seg_idx in enumerate(order):
            ns = eligible[seg_idx].n_valid_starts
            # Pick bucket with largest remaining deficit
            current_fracs = accumulated / max(total_starts, 1)
            deficits = targets - current_fracs
            # Force non-empty buckets: if a bucket has 0 segments, prefer it
            bucket_counts = [best_assign[:rank].count(b) for b in range(3)]
            remaining = m - rank
            forced = [b for b in range(3) if bucket_counts[b] == 0 and remaining <= (3 - sum(bc > 0 for bc in bucket_counts))]
            if forced:
                chosen = forced[0]
            else:
                chosen = int(np.argmax(deficits))
            best_assign[seg_idx] = chosen
            accumulated[chosen] += ns
        best_score = score(best_assign)

    # Build SegmentSplit from best_assign
    buckets: List[List[SegmentInfo]] = [[], [], []]
    for i, a in enumerate(best_assign):
        buckets[a].append(eligible[i])

    total = total_starts
    achieved = [sum(s.n_valid_starts for s in b) / max(total, 1) for b in buckets]

    return SegmentSplit(
        train_segs=tuple(buckets[0]),
        val_segs=tuple(buckets[1]),
        test_segs=tuple(buckets[2]),
        train_weight=achieved[0],
        val_weight=achieved[1],
        test_weight=achieved[2],
        split_seed=split_seed,
    )


# ---------------------------------------------------------------------------
# Fourier estimation — train-only, longest train segment
# ---------------------------------------------------------------------------

def build_fourier_from_longest_train_seg(
    star: StarData,
    train_segs: Tuple[SegmentInfo, ...],
    L: int,
    k: int = 8,
) -> FourierModel:
    """
    Estimate Fourier frequencies from the longest TRAIN segment only.

    We use only a single contiguous segment so that `estimate_freqs_fft()`
    sees uniformly sampled data without artificial discontinuities.
    The resulting FourierModel (including the design matrix A_full) is valid
    for any window of length L with the same cadence.

    Parameters
    ----------
    star       : StarData for this star
    train_segs : tuple of SegmentInfo assigned to train
    L          : window length
    k          : number of Fourier frequencies to retain
    """
    if not train_segs:
        raise RuntimeError("No train segments — cannot build Fourier model.")
    longest = max(train_segs, key=lambda s: s.length)
    y_seg = star.y[longest.as_slice()].astype(np.float64)
    return build_fourier_model_for_star(
        y_long_segment=y_seg,
        dt_days=star.dt_med_days,
        L=L,
        k=k,
    )


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

def _zscore_obs_only(
    y: np.ndarray,
    obs_mask: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """Z-score using observed-only points to avoid leakage from the synthetic gap."""
    obs = obs_mask.astype(bool)
    if obs.sum() < 2:
        mu = float(np.mean(y))
        sd = float(np.std(y))
    else:
        mu = float(np.mean(y[obs]))
        sd = float(np.std(y[obs]))
    sd = max(sd, eps)
    return (y - mu) / sd, mu, sd


# ---------------------------------------------------------------------------
# Core single-window materialization
# ---------------------------------------------------------------------------

def _fourier_predict(
    A_full: np.ndarray,
    y_norm: np.ndarray,
    obs_mask: np.ndarray,
    gap_start: int,
    gap_len: int,
    fourier_mode: str,
) -> np.ndarray:
    if fourier_mode == "dynamic":
        return fit_predict_fourier_dynamic(
            A_full, y_norm, obs_mask, gap_start=gap_start, gap_len=gap_len
        )
    return fit_predict_fourier(A_full, y_norm, obs_mask)


def _compute_fourier_r2_obs(
    y_norm: np.ndarray,
    y_fourier_norm: np.ndarray,
    obs_mask: np.ndarray,
    clip: Tuple[float, float] = (-1.0, 1.0),
) -> float:
    """R² of Fourier model on observed (non-gap) points — scale-invariant gate signal."""
    obs = obs_mask.astype(bool)
    if obs.sum() < 3:
        return 0.0
    y = y_norm[obs]
    yh = y_fourier_norm[obs]
    sse = float(np.sum((y - yh) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - sse / sst, clip[0], clip[1]))


def sample_one_dict(
    y_window: np.ndarray,
    L: int,
    gap_len: int,
    gap_start: int,
    min_context: int,
    A_full: Optional[np.ndarray],
    fourier_mode: str,
    has_gate: bool,
) -> dict:
    """
    Materialise a single window dict from pre-sliced raw flux values.

    Parameters
    ----------
    y_window  : (L,) raw flux for this window (star.y[wstart:wstart+L])
    gap_start : relative position of gap within [0, L)
    A_full    : (L, 2K+1) Fourier design matrix, or None for no warm-start
    has_gate  : if True, include 'fourier_r2_obs' key

    Returns a dict with keys:
        y_true, y_in, obs_mask, gap_start, gap_len, mu, sd
        (+ fourier_r2_obs if has_gate)
    """
    y = y_window.astype(np.float64)

    obs_mask = np.ones(L, dtype=np.float32)
    obs_mask[gap_start: gap_start + gap_len] = 0.0

    y_norm, mu, sd = _zscore_obs_only(y, obs_mask)

    if A_full is not None:
        y_fourier_norm = _fourier_predict(
            A_full, y_norm, obs_mask, gap_start, gap_len, fourier_mode
        )
        y_obs_zeroed = y_norm.copy()
        y_obs_zeroed[obs_mask < 0.5] = 0.0
        residual_visible = (y_norm - y_fourier_norm) * obs_mask
        y_in = y_norm.copy()
        y_in[obs_mask < 0.5] = y_fourier_norm[obs_mask < 0.5]
    else:
        y_fourier_norm = None
        y_obs_zeroed = None
        residual_visible = None
        y_in = y_norm.copy()
        y_in[obs_mask < 0.5] = 0.0

    # Invariant: outside gap, y_in == y_norm
    outside = np.ones(L, dtype=bool)
    outside[gap_start: gap_start + gap_len] = False
    assert np.mean(np.abs(y_in[outside] - y_norm[outside])) < 1e-6

    out = dict(
        y_true=y_norm.astype(np.float32),
        y_in=y_in.astype(np.float32),
        obs_mask=obs_mask,
        gap_start=int(gap_start),
        gap_len=int(gap_len),
        mu=float(mu),
        sd=float(sd),
    )
    if has_gate:
        if y_fourier_norm is not None:
            r2obs = _compute_fourier_r2_obs(y_norm, y_fourier_norm, obs_mask)
        else:
            r2obs = 0.0
        out["fourier_r2_obs"] = float(r2obs)
    if y_fourier_norm is not None:
        out["y_obs_zeroed"] = y_obs_zeroed.astype(np.float32)
        out["y_fourier_full"] = y_fourier_norm.astype(np.float32)
        out["residual_visible"] = residual_visible.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Fixed deterministic evaluation specs (val / test)
# ---------------------------------------------------------------------------

def make_fixed_eval_specs(
    star: StarData,
    segs: Tuple[SegmentInfo, ...],
    L: int,
    gap_len: int,
    min_context: int,
    n_samples: int,
    seed: int,
    stride_eval: int = 1,
    max_per_segment: Optional[int] = None,
) -> List[WindowSpec]:
    """
    Build a fixed, deterministic list of WindowSpec for val or test.

    Strategy
    --------
    1. Enumerate candidate windows within each segment at `stride_eval` stride.
    2. If total candidates > n_samples, sample `n_samples` without replacement
       (seeded, deterministic).
    3. For each selected window, draw a fixed gap_start (seeded, deterministic).

    Parameters
    ----------
    segs        : segments assigned to this eval split
    n_samples   : target number of evaluation windows
    stride_eval : stride between candidate window starts (1 = all possible)
    max_per_segment : optional cap per segment before global sub-sampling
    seed        : global seed for reproducibility
    """
    g0 = min_context
    g1 = L - min_context - gap_len
    if g1 < g0:
        raise ValueError(
            f"gap_len={gap_len} too large for L={L}, min_context={min_context}"
        )

    rng = np.random.default_rng(seed)

    # Enumerate all candidate wstarts
    all_candidates: List[Tuple[int, int]] = []   # (segment_id, wstart)
    for seg in segs:
        wstarts = np.arange(seg.start_idx, seg.end_idx - L + 2, stride_eval, dtype=np.int64)
        if max_per_segment is not None and wstarts.size > max_per_segment:
            chosen = rng.choice(wstarts, size=max_per_segment, replace=False)
            chosen = np.sort(chosen)
        else:
            chosen = wstarts
        for ws in chosen.tolist():
            all_candidates.append((seg.segment_id, int(ws)))

    if not all_candidates:
        return []

    # Sub-sample if needed
    if len(all_candidates) > n_samples:
        indices = rng.choice(len(all_candidates), size=n_samples, replace=False)
        indices.sort()
        all_candidates = [all_candidates[i] for i in indices]

    # Assign fixed gap_starts
    gap_starts = rng.integers(g0, g1 + 1, size=len(all_candidates), dtype=np.int64)

    specs = []
    for (seg_id, ws), gs in zip(all_candidates, gap_starts):
        specs.append(WindowSpec(
            segment_id=seg_id,
            wstart=int(ws),
            gap_start=int(gs),
            gap_len=int(gap_len),
        ))
    return specs


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class FixedSpecsDataset(Dataset):
    """
    Dataset wrapping a fixed list of WindowSpecs (for val / test).

    All samples are materialised up-front at construction time.
    """

    def __init__(
        self,
        star: StarData,
        specs: List[WindowSpec],
        L: int,
        min_context: int,
        fourier: Optional[FourierModel],
        fourier_mode: str,
        has_gate: bool,
    ):
        self.samples = []
        A_full = fourier.A_full if fourier is not None else None
        for sp in specs:
            y_window = star.y[sp.wstart: sp.wstart + L]
            d = sample_one_dict(
                y_window=y_window,
                L=L,
                gap_len=sp.gap_len,
                gap_start=sp.gap_start,
                min_context=min_context,
                A_full=A_full,
                fourier_mode=fourier_mode,
                has_gate=has_gate,
            )
            self.samples.append(d)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class TrainEpochDataset(Dataset):
    """
    Dataset that yields exactly `samples_per_epoch` training windows per epoch.

    Windows are drawn randomly (with replacement) from all valid start positions
    across all train segments. Overlapping windows are allowed within train segments
    — this is intentional.

    The RNG is seeded as `seed + epoch` to make each epoch's samples reproducible
    while varying across epochs.  Call `set_epoch(epoch)` before each epoch.
    """

    def __init__(
        self,
        star: StarData,
        train_segs: Tuple[SegmentInfo, ...],
        L: int,
        gap_len: int,
        min_context: int,
        fourier: Optional[FourierModel],
        fourier_mode: str,
        has_gate: bool,
        samples_per_epoch: int,
        seed: int,
    ):
        self.star = star
        self.L = L
        self.gap_len = gap_len
        self.min_context = min_context
        self.A_full = fourier.A_full if fourier is not None else None
        self.fourier_mode = fourier_mode
        self.has_gate = has_gate
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self._epoch = 0

        # Build pool of all valid window starts, weighted by segment length
        starts_list: List[int] = []
        for seg in train_segs:
            ws = np.arange(seg.start_idx, seg.end_idx - L + 2, dtype=np.int64)
            starts_list.extend(ws.tolist())
        self._all_starts = np.array(starts_list, dtype=np.int64)

        if self._all_starts.size == 0:
            raise RuntimeError(
                "No valid window starts in train segments. "
                "All train segments may be too short for L."
            )

        self.g0 = min_context
        self.g1 = L - min_context - gap_len
        if self.g1 < self.g0:
            raise ValueError(
                f"gap_len={gap_len} too large for L={L}, min_context={min_context}"
            )

        # Generate samples for epoch 0
        self._refresh()

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch to get a fresh set of randomly drawn windows."""
        self._epoch = epoch
        self._refresh()

    def _refresh(self) -> None:
        rng = np.random.default_rng(self.seed + self._epoch)
        chosen_starts = rng.choice(
            self._all_starts, size=self.samples_per_epoch, replace=True
        )
        gap_starts = rng.integers(
            self.g0, self.g1 + 1, size=self.samples_per_epoch, dtype=np.int64
        )
        self._chosen_starts = chosen_starts
        self._gap_starts = gap_starts

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        ws = int(self._chosen_starts[idx])
        gs = int(self._gap_starts[idx])
        y_window = self.star.y[ws: ws + self.L]
        return sample_one_dict(
            y_window=y_window,
            L=self.L,
            gap_len=self.gap_len,
            gap_start=gs,
            min_context=self.min_context,
            A_full=self.A_full,
            fourier_mode=self.fourier_mode,
            has_gate=self.has_gate,
        )


class FixedTrainDataset(Dataset):
    """
    Dataset that generates exactly `samples_per_epoch` training windows once
    and stores them in memory (CPU).

    Windows are drawn randomly (with replacement) from all valid start positions
    across all train segments. By materializing them up front, we avoid re-evaluating
    expensive dynamic Fourier models at every epoch.

    `set_epoch` does nothing because the samples are fixed.
    """

    def __init__(
        self,
        star: StarData,
        train_segs: Tuple[SegmentInfo, ...],
        L: int,
        gap_len: int,
        min_context: int,
        fourier: Optional[FourierModel],
        fourier_mode: str,
        has_gate: bool,
        samples_per_epoch: int,
        seed: int,
    ):
        self.samples = []

        # Build pool of all valid window starts, weighted by segment length
        starts_list: List[int] = []
        for seg in train_segs:
            ws = np.arange(seg.start_idx, seg.end_idx - L + 2, dtype=np.int64)
            starts_list.extend(ws.tolist())
        all_starts = np.array(starts_list, dtype=np.int64)

        if all_starts.size == 0:
            raise RuntimeError(
                "No valid window starts in train segments. "
                "All train segments may be too short for L."
            )

        g0 = min_context
        g1 = L - min_context - gap_len
        if g1 < g0:
            raise ValueError(
                f"gap_len={gap_len} too large for L={L}, min_context={min_context}"
            )

        rng = np.random.default_rng(seed)
        chosen_starts = rng.choice(
            all_starts, size=samples_per_epoch, replace=True
        )
        gap_starts = rng.integers(
            g0, g1 + 1, size=samples_per_epoch, dtype=np.int64
        )

        A_full = fourier.A_full if fourier is not None else None

        for ws, gs in zip(chosen_starts, gap_starts):
            ws = int(ws)
            gs = int(gs)
            y_window = star.y[ws: ws + L]
            d = sample_one_dict(
                y_window=y_window,
                L=L,
                gap_len=gap_len,
                gap_start=gs,
                min_context=min_context,
                A_full=A_full,
                fourier_mode=fourier_mode,
                has_gate=has_gate,
            )
            self.samples.append(d)

    def set_epoch(self, epoch: int) -> None:
        """No-op for fixed dataset."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Unified collate functions
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    """
    Unified collate for both normal and gating samples.
    Handles optional Fourier-aware and 'fourier_r2_obs' keys gracefully.
    """
    y_true = torch.stack([torch.from_numpy(b["y_true"]).float() for b in batch])
    y_in   = torch.stack([torch.from_numpy(b["y_in"]).float()   for b in batch])
    obs_mask = torch.stack([torch.from_numpy(b["obs_mask"]).float() for b in batch])
    gap_start = torch.tensor([b["gap_start"] for b in batch], dtype=torch.long)
    gap_len   = torch.tensor([b["gap_len"]   for b in batch], dtype=torch.long)
    mu = torch.tensor([b["mu"] for b in batch], dtype=torch.float32)
    sd = torch.tensor([b["sd"] for b in batch], dtype=torch.float32)

    out = dict(
        y_true=y_true, y_in=y_in, obs_mask=obs_mask,
        gap_start=gap_start, gap_len=gap_len, mu=mu, sd=sd,
    )
    if "fourier_r2_obs" in batch[0]:
        out["fourier_r2_obs"] = torch.tensor(
            [b["fourier_r2_obs"] for b in batch], dtype=torch.float32
        )
    for key in ("y_obs_zeroed", "y_fourier_full", "residual_visible"):
        if key in batch[0]:
            out[key] = torch.stack([torch.from_numpy(b[key]).float() for b in batch])
    return out


# ---------------------------------------------------------------------------
# High-level convenience function used by training scripts
# ---------------------------------------------------------------------------

def build_star_split(
    star: StarData,
    L: int,
    gap_len: int,
    min_context: int,
    k_freqs: int,
    samples_per_epoch: int,
    n_val: int,
    n_test: int,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    fourier_mode: str = "static",
    has_gate: bool = False,
    stride_eval: int = 1,
    max_eval_per_segment: Optional[int] = None,
    train_sampling: str = "fixed",
) -> Optional[dict]:
    """
    Full pipeline for one star: segment split → Fourier → datasets.

    Returns None when the star must be skipped (< 3 eligible segments).

    Returns a dict with keys:
        split         : SegmentSplit
        fourier       : FourierModel (built on longest train segment)
        train_dataset : TrainEpochDataset
        val_dataset   : FixedSpecsDataset
        test_dataset  : FixedSpecsDataset
        val_specs     : List[WindowSpec]
        test_specs    : List[WindowSpec]
        meta          : dict with metadata columns for CSV
    """
    eligible = enumerate_eligible_segments(star, L)
    n_eligible = len(eligible)

    if n_eligible < 3:
        return None  # skip: too_few_eligible_segments

    split = split_segments_exhaustive(
        eligible=eligible,
        train_frac=train_frac,
        val_frac=val_frac,
        split_seed=seed,
    )
    if split is None:
        return None

    print("Building Fourier model...")
    fourier = build_fourier_from_longest_train_seg(
        star=star,
        train_segs=split.train_segs,
        L=L,
        k=k_freqs,
    )

    if train_sampling == "fixed":
        print(f"Materializing {samples_per_epoch} fixed train samples...")
        train_dataset = FixedTrainDataset(
            star=star,
            train_segs=split.train_segs,
            L=L,
            gap_len=gap_len,
            min_context=min_context,
            fourier=fourier,
            fourier_mode=fourier_mode,
            has_gate=has_gate,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
    elif train_sampling == "online":
        train_dataset = TrainEpochDataset(
            star=star,
            train_segs=split.train_segs,
            L=L,
            gap_len=gap_len,
            min_context=min_context,
            fourier=fourier,
            fourier_mode=fourier_mode,
            has_gate=has_gate,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown train_sampling: {train_sampling}")

    val_specs = make_fixed_eval_specs(
        star=star,
        segs=split.val_segs,
        L=L,
        gap_len=gap_len,
        min_context=min_context,
        n_samples=n_val,
        seed=seed + 1,
        stride_eval=stride_eval,
        max_per_segment=max_eval_per_segment,
    )
    test_specs = make_fixed_eval_specs(
        star=star,
        segs=split.test_segs,
        L=L,
        gap_len=gap_len,
        min_context=min_context,
        n_samples=n_test,
        seed=seed + 2,
        stride_eval=stride_eval,
        max_per_segment=max_eval_per_segment,
    )

    val_dataset = FixedSpecsDataset(
        star=star,
        specs=val_specs,
        L=L,
        min_context=min_context,
        fourier=fourier,
        fourier_mode=fourier_mode,
        has_gate=has_gate,
    )
    test_dataset = FixedSpecsDataset(
        star=star,
        specs=test_specs,
        L=L,
        min_context=min_context,
        fourier=fourier,
        fourier_mode=fourier_mode,
        has_gate=has_gate,
    )

    # Split balance diagnostics (informational; star is never skipped here)
    balance_notes = []
    if split.train_weight < 0.50:
        balance_notes.append("low_train_weight")
    if split.val_weight < 0.05:
        balance_notes.append("low_val_weight")
    if split.test_weight < 0.05:
        balance_notes.append("low_test_weight")
    split_balance_warning = len(balance_notes) > 0
    split_balance_note = "|".join(balance_notes)

    meta = dict(
        n_eligible_segments=n_eligible,
        n_train_segments=len(split.train_segs),
        n_val_segments=len(split.val_segs),
        n_test_segments=len(split.test_segs),
        train_weight=round(split.train_weight, 4),
        val_weight=round(split.val_weight, 4),
        test_weight=round(split.test_weight, 4),
        train_n_valid_starts=split.n_train_starts(),
        val_n_valid_starts=split.n_val_starts(),
        test_n_valid_starts=split.n_test_starts(),
        split_kind="segment_level_clean",
        train_sampling=train_sampling,
        samples_per_epoch=samples_per_epoch,
        stride_eval=stride_eval,
        max_eval_per_segment="" if max_eval_per_segment is None else max_eval_per_segment,
        fourier_train_only=True,
        split_seed=seed,
        split_balance_warning=split_balance_warning,
        split_balance_note=split_balance_note,
    )

    return dict(
        split=split,
        fourier=fourier,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        val_specs=val_specs,
        test_specs=test_specs,
        meta=meta,
    )
