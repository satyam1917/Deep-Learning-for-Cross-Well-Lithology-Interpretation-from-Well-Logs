"""Cut wells into fixed-length depth windows for sequence models.

NaNs are preserved; ``fill_and_mask`` converts them into (value, validity-mask)
pairs, which is the input format used by the missing-curve experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

PAD_LABEL = -1


def make_windows(
    scaled: pd.DataFrame,
    meta: pd.DataFrame,
    wells: list[str],
    window: int = 256,
    stride: int = 128,
    label_col: str = "lith",
):
    """Return X (n, window, F), y (n, window) and an index table.

    ``scaled`` holds the (already scaled) feature columns, ``meta`` holds WELL,
    DEPTH_MD and the label, with the same row index. Windows never cross a well
    boundary. A well shorter than ``window`` is padded at the end (label = -1,
    features = NaN); a longer one gets a final window aligned to its end so no
    sample is dropped.
    """
    feats = list(scaled.columns)
    xs, ys, idx = [], [], []
    well_rows = meta[meta[C.WELL].isin(wells)].groupby(C.WELL).indices
    for w in wells:
        rows = well_rows[w]
        order = rows[np.argsort(meta[C.DEPTH].to_numpy()[rows], kind="stable")]
        x = scaled.to_numpy(np.float32)[order]
        y = meta[label_col].to_numpy()[order].astype(np.int16)
        n = len(order)
        starts = list(range(0, max(n - window, 0) + 1, stride))
        if n > window and starts[-1] + window < n:
            starts.append(n - window)
        for s in starts:
            xw, yw = x[s : s + window], y[s : s + window]
            if len(xw) < window:
                pad = window - len(xw)
                xw = np.vstack([xw, np.full((pad, len(feats)), np.nan, np.float32)])
                yw = np.concatenate([yw, np.full(pad, PAD_LABEL, np.int16)])
            xs.append(xw)
            ys.append(yw)
            idx.append((w, s))
    X = np.stack(xs)
    y = np.stack(ys)
    index = pd.DataFrame(idx, columns=["WELL", "start"])
    return X, y, index


def fill_and_mask(X: np.ndarray, fill: float = 0.0):
    """Return (X with NaN -> fill, boolean validity mask of the same shape)."""
    mask = ~np.isnan(X)
    return np.where(mask, X, fill).astype(np.float32), mask
