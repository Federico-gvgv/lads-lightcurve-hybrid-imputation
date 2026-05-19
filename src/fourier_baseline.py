# src/fourier_baseline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class FourierModel:
    freqs_cpd: np.ndarray  # (K,) cycles per day
    dt_days: float
    L: int
    A_full: np.ndarray     # (L, 2K+1) design matrix with [1, sin, cos]


def build_design_matrix(L: int, dt_days: float, freqs_cpd: np.ndarray) -> np.ndarray:
    """
    A[n] = [1, sin(2π f t_n), cos(2π f t_n) for each f]
    t_n = n * dt_days
    """
    t = (np.arange(L, dtype=np.float64) * dt_days).reshape(-1, 1)  # (L,1)
    K = freqs_cpd.size
    A = np.ones((L, 1 + 2 * K), dtype=np.float64)
    if K > 0:
        w = 2.0 * np.pi * freqs_cpd.reshape(1, -1)  # (1,K)
        phase = t @ w  # (L,K)
        A[:, 1:1 + K] = np.sin(phase)
        A[:, 1 + K:1 + 2 * K] = np.cos(phase)
    return A


def estimate_freqs_fft(
    y: np.ndarray,
    dt_days: float,
    k: int = 8,
    fmin_cpd: float = 1.0,
    fmax_cpd: Optional[float] = None,
) -> np.ndarray:
    """
    Fast frequency estimate using FFT (assumes nearly regular cadence).
    Returns top-k frequencies in cycles/day excluding DC.
    """
    y0 = y.astype(np.float64) - float(np.mean(y))
    n = y0.size
    if n < 32:
        return np.array([], dtype=np.float64)

    # FFT frequencies in cycles/day
    # sampling frequency = 1/dt_days (samples per day)
    fs = 1.0 / max(dt_days, 1e-12)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)  # cycles/day
    spec = np.abs(np.fft.rfft(y0))**2

    # Remove DC
    freqs = freqs[1:]
    spec = spec[1:]

    if fmax_cpd is None:
        # Nyquist ~ fs/2
        fmax_cpd = 0.5 * fs

    keep = (freqs >= fmin_cpd) & (freqs <= fmax_cpd)
    freqs_k = freqs[keep]
    spec_k = spec[keep]
    if freqs_k.size == 0:
        return np.array([], dtype=np.float64)

    idx = np.argsort(spec_k)[::-1]
    top = freqs_k[idx[: min(k, idx.size)]]
    top = np.sort(top)
    return top.astype(np.float64)


def fit_predict_fourier(
    A_full: np.ndarray,
    y: np.ndarray,
    obs_mask: np.ndarray,
    ridge: float = 1e-6
) -> np.ndarray:
    """
    Fit linear coefficients on observed points and predict y_hat on all points.
    y: (L,) normalized
    obs_mask: (L,) 1 observed / 0 missing
    """
    obs = obs_mask.astype(bool)
    A = A_full[obs]
    b = y[obs].astype(np.float64)

    # Ridge via normal equations (small ridge for stability)
    # w = (A^T A + λI)^-1 A^T b
    AtA = A.T @ A
    Atb = A.T @ b
    AtA = AtA + ridge * np.eye(AtA.shape[0], dtype=np.float64)
    w = np.linalg.solve(AtA, Atb)
    y_hat = (A_full @ w).astype(np.float64)
    return y_hat


def build_fourier_model_for_star(
    y_long_segment: np.ndarray,
    dt_days: float,
    L: int,
    k: int = 8
) -> FourierModel:
    freqs = estimate_freqs_fft(y_long_segment, dt_days=dt_days, k=k)
    A = build_design_matrix(L=L, dt_days=dt_days, freqs_cpd=freqs)
    return FourierModel(freqs_cpd=freqs, dt_days=dt_days, L=L, A_full=A)
