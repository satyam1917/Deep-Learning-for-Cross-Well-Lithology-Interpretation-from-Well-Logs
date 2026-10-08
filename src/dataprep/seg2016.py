"""SEG 2016 machine-learning contest data (Hugoton / Panoma gas fields, Kansas).

Small, well-known benchmark used as an independent second dataset. It has 10
training wells with facies labels and 2 blind wells (STUART, CRAWFORD) whose
labels come from a separate core-description file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

WELLC = "WELL"
DEPTHC = "DEPTH_FT"

# Curves that mean (roughly) the same thing in FORCE 2020, for cross-dataset checks.
# Facies/lithology label sets are different and are NOT mapped.
CURVE_MAP_TO_FORCE = {"GR": "GR", "PE": "PEF", "ILD_log10": "log10_RDEP"}


def load_raw() -> pd.DataFrame:
    r = C.RAW / "seg2016"
    tr = pd.read_csv(r / "facies_vectors.csv")
    bl = pd.read_csv(r / "validation_data_nofacies.csv")
    lab = pd.read_csv(r / "blind_stuart_crawford_core_facies.csv")

    tr["origin"] = "train"
    bl["origin"] = "blind"

    lab = lab.rename(columns={"WellName": "Well Name", "Depth.ft": "Depth", "LithCode": "Facies"})
    lab["Depth"] = lab["Depth"].round(1)
    bl["Depth"] = bl["Depth"].round(1)
    bl = bl.merge(lab[["Well Name", "Depth", "Facies"]], on=["Well Name", "Depth"], how="left",
                  validate="one_to_one")

    df = pd.concat([tr, bl], ignore_index=True)
    return df.rename(columns={"Well Name": WELLC, "Depth": DEPTHC})


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Type the columns, add a 0-based facies index and drop duplicate depths.

    Returns (clean_df, audit). The raw training file contains 3 depths that appear
    twice (2 with a conflicting NM_M flag, at formation boundaries); the first
    occurrence is kept.
    """
    df = df.copy()
    df["Facies"] = df["Facies"].astype("float64")
    df["facies"] = (df["Facies"] - 1).astype("Int8")        # 0..8, <NA> if unlabelled
    for c in C.SEG_CURVES:
        df[c] = df[c].astype("float32")
    df["Formation"] = df["Formation"].fillna("UNKNOWN")
    df = df.drop(columns=["Facies"])
    df = df.sort_values([WELLC, DEPTHC], kind="stable").reset_index(drop=True)
    dup = df.duplicated([WELLC, DEPTHC], keep="first")
    audit = {
        "rows_raw": int(len(df)),
        "duplicate_depth_rows_removed": int(dup.sum()),
        "blind_rows": int((df.origin == "blind").sum()),
        "blind_rows_without_label": int(((df.origin == "blind") & df["facies"].isna()).sum()),
        "pct_PE_missing": round(float(100 * df["PE"].isna().mean()), 2),
    }
    df = df[~dup].reset_index(drop=True)
    audit["rows_clean"] = int(len(df))
    return df, audit


def well_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(WELLC)
    s = pd.DataFrame(
        {
            "origin": g["origin"].first(),
            "n_samples": g.size(),
            "top_ft": g[DEPTHC].min(),
            "bottom_ft": g[DEPTHC].max(),
            "n_labelled": g["facies"].apply(lambda x: int(x.notna().sum())),
            "pct_missing_PE": g["PE"].apply(lambda x: round(100 * x.isna().mean(), 1)),
        }
    )
    counts = df.groupby([WELLC, "facies"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(9), fill_value=0)
    counts.columns = [f"n_facies{i + 1}" for i in range(9)]
    return s.join(counts).reset_index()


SINGLE_FACIES_WELL = "Recruit F9"   # 80 samples, all facies 9: kept for training, never validated on


def build_splits(df: pd.DataFrame) -> dict:
    """official: train wells / blind wells.

    leave_one_well_out: one fold per training well, except ``Recruit F9`` (a tiny
    single-facies well) which always stays in the training side.
    """
    train = sorted(df.loc[df.origin == "train", WELLC].unique().tolist())
    blind = sorted(df.loc[df.origin == "blind", WELLC].unique().tolist())
    lowo = [{"val": [w], "train": [t for t in train if t != w]}
            for w in train if w != SINGLE_FACIES_WELL]
    assert not set(train) & set(blind)
    return {"official": {"train": train, "blind": blind}, "leave_one_well_out": lowo}
