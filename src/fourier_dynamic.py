# src/fourier_dynamic.py
from __future__ import annotations

import numpy as np

from .fourier_baseline import fit_predict_fourier


def _weighted_ridge(A: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
    """
    Solve: argmin_c sum_i w_i (A_i c - y_i)^2 + ridge * ||c||^2
    A: (N,P), y: (N,), w: (N,)
    returns c: (P,)
    """
    w = np.clip(w.astype(np.float64), 0.0, None)
    sw = np.sqrt(w + 1e-12)
    Aw = A * sw[:, None]
    yw = y.astype(np.float64) * sw

    P = A.shape[1]
    M = Aw.T @ Aw
    if ridge > 0:
        M = M + ridge * np.eye(P, dtype=np.float64)
    b = Aw.T @ yw
    return np.linalg.solve(M, b)


def fit_predict_fourier_dynamic(
    A_full: np.ndarray,
    y: np.ndarray,
    obs_mask: np.ndarray,
    gap_start: int,
    gap_len: int,
    n_knots: int = 9,
    sigma: float | None = None,
    ridge: float = 1e-4,
    include_gap_mid_knot: bool = False,
) -> np.ndarray:
    """
    Time-varying Fourier coefficients via local weighted ridge regression at knots.

    A_full: (L,P) design matrix for fixed frequencies (same as FourierModel.A_full)
    y: (L,) normalized series
    obs_mask: (L,) 1 observed / 0 missing
    gap_start/gap_len: used for good defaults (sigma + extra knots near edges)
    Returns y_hat: (L,) prediction for all points
    """
    L, P = A_full.shape
    obs = obs_mask.astype(bool)

    # Not enough info -> fall back to stationary Fourier
    if obs.sum() < max(P + 2, 16):
        return fit_predict_fourier(A_full, y, obs_mask, ridge=max(1e-6, ridge))

    idx = np.arange(L, dtype=np.float64)

    # Base knots across window + extra near gap edges
    base_knots = np.linspace(0, L - 1, int(n_knots), dtype=np.float64)
    extra = np.array([gap_start - 1, gap_start + gap_len], dtype=np.float64)
    if include_gap_mid_knot:
        extra = np.concatenate([extra, np.array([gap_start + 0.5 * (gap_len - 1)], dtype=np.float64)])
    knots = np.unique(np.clip(np.concatenate([base_knots, extra]), 0, L - 1))

    # Default sigma tied to gap length (controls how quickly coefficients can drift)
    if sigma is None:
        sigma = max(64.0, 1.0 * float(gap_len))
    sigma = float(max(sigma, 8.0))

    # Fit local coefficients at each knot
    C = np.zeros((len(knots), P), dtype=np.float64)
    for j, t0 in enumerate(knots):
        w = np.exp(-0.5 * ((idx - float(t0)) / sigma) ** 2)
        w = w * obs_mask.astype(np.float64)  # only observed contribute

        if float(w.sum()) < 1e-6:
            w = obs_mask.astype(np.float64)  # fallback: uniform on observed

        C[j] = _weighted_ridge(A_full, y, w, ridge=ridge)

    # Interpolate coefficients for each time index
    y_hat = np.zeros((L,), dtype=np.float64)
    for t in range(L):
        j = int(np.searchsorted(knots, t, side="right") - 1)
        j = int(np.clip(j, 0, len(knots) - 1))

        if j == len(knots) - 1:
            c_t = C[j]
        else:
            tL = float(knots[j])
            tR = float(knots[j + 1])
            a = 0.0 if tR == tL else (float(t) - tL) / (tR - tL)
            c_t = (1.0 - a) * C[j] + a * C[j + 1]

        y_hat[t] = float(A_full[t] @ c_t)

    return y_hat
