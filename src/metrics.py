# src/metrics.py
from __future__ import annotations
import numpy as np


def denorm(y_norm: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return y_norm * sd + mu


def gap_indices(L: int, gap_start: int, gap_len: int) -> np.ndarray:
    idx = np.arange(L)
    return (idx >= gap_start) & (idx < gap_start + gap_len)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_true - y_pred
    return float(np.mean(e * e))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot
