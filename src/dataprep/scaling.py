"""Robust scaling and class weights, always fitted on training wells only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def fit_stats(df: pd.DataFrame, wells: list[str], features: list[str]) -> dict:
    """Median / IQR / percentiles per feature using only ``wells`` (the training wells)."""
    sub = df[df[C.WELL].isin(wells)]
    stats = {}
    for f in features:
        v = sub[f].dropna().to_numpy(dtype="float64")
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        if iqr <= 0:                       # degenerate curve: fall back to std, then 1
            iqr = float(v.std()) or 1.0
        stats[f] = {
            "median": float(med),
            "iqr": float(iqr),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "p01": float(np.percentile(v, 1)),
            "p99": float(np.percentile(v, 99)),
            "n_valid": int(v.size),
        }
    return stats


def apply_scaling(df: pd.DataFrame, stats: dict, features: list[str], clip: float = 10.0) -> pd.DataFrame:
    """(x - median) / IQR, clipped to +/- ``clip``. NaN stays NaN.

    After scaling, 0 equals the training median, so filling remaining NaN with 0
    is median imputation. The imputation itself is left to the model code, which
    also receives the validity mask.
    """
    out = pd.DataFrame(index=df.index)
    for f in features:
        s = stats[f]
        out[f] = ((df[f] - s["median"]) / s["iqr"]).clip(-clip, clip).astype("float32")
    return out


def class_weights(df: pd.DataFrame, wells: list[str], n_classes: int = 12) -> dict:
    """Class frequencies and inverse-frequency weights from training wells."""
    sub = df[df[C.WELL].isin(wells)]
    counts = sub["lith"].value_counts().reindex(range(n_classes), fill_value=0).astype(float)
    freq = counts / counts.sum()
    inv = (1.0 / freq.replace(0, np.nan))
    inv = (inv / inv.mean(skipna=True)).fillna(0.0)          # mean weight of present classes = 1
    sqrt_inv = (1.0 / np.sqrt(freq.replace(0, np.nan)))
    sqrt_inv = (sqrt_inv / sqrt_inv.mean(skipna=True)).fillna(0.0)
    return {
        "class_names": C.CLASS_NAMES,
        "counts": counts.astype(int).tolist(),
        "frequency": freq.round(6).tolist(),
        "weight_inverse_frequency": inv.round(4).tolist(),
        "weight_sqrt_inverse_frequency": sqrt_inv.round(4).tolist(),
    }
