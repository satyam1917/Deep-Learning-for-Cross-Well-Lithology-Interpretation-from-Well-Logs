"""FORCE 2020 well-log lithology data: load, clean, summarise, split.

Design rules (they matter for the project's goals):
  * No imputation here. Missing curves are the *subject* of the study, so NaNs are
    preserved and every downstream stage decides how to deal with them.
  * Splits are always by well, never by depth, to avoid leakage between
    neighbouring samples of the same well.
  * All statistics used for scaling are fitted on training wells only (see scaling.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C

ORIGINS = ["train", "open_test", "hidden_test"]
PRESENT_MIN_COVERAGE = 0.05   # a curve counts as "present" in a well if >= 5 % of rows are valid


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    """Return one table with all 118 wells and an ``origin`` column.

    origin: train (98 wells), open_test (10 wells, labels merged from the
    separate target file), hidden_test (10 blind wells).
    """
    r = C.RAW / "force2020"
    tr = pd.read_csv(r / "train.csv", sep=";")
    lf = pd.read_csv(r / "leaderboard_test_features.csv", sep=";")
    lt = pd.read_csv(r / "leaderboard_test_target.csv", sep=";")
    hd = pd.read_csv(r / "hidden_test.csv", sep=";")

    lf = lf.merge(lt, on=[C.WELL, C.DEPTH], how="left", validate="one_to_one")
    if lf[C.LABEL_RAW].isna().any():
        raise ValueError("open-test labels failed to align with features")

    tr["origin"], lf["origin"], hd["origin"] = "train", "open_test", "hidden_test"
    df = pd.concat([tr, lf, hd], ignore_index=True)
    if df.duplicated([C.WELL, C.DEPTH]).any():
        raise ValueError("duplicate (WELL, DEPTH_MD) rows")
    return df


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardise labels, remove sentinels/impossible values, add log-resistivity.

    Returns (clean_df, bounds_report). The report lists, per curve, how many values
    were already missing, how many were sentinels and how many fell outside the
    physical bounds, so every removal is auditable.
    """
    df = df.copy()
    rows = []
    for c in C.CURVES:
        v = df[c].astype("float64")
        already_nan = int(v.isna().sum())
        sentinel = v <= C.SENTINEL_THRESHOLD
        lo, hi = C.PHYSICAL_BOUNDS[c]
        out_of_bounds = ((v < lo) | (v > hi)) & ~sentinel & v.notna()
        df[c] = v.mask(sentinel | out_of_bounds).astype("float32")
        rows.append(
            dict(
                curve=c,
                lower=lo,
                upper=hi,
                n_rows=len(v),
                n_missing_raw=already_nan,
                n_sentinel=int(sentinel.sum()),
                n_out_of_bounds=int(out_of_bounds.sum()),
                n_missing_clean=int(df[c].isna().sum()),
                pct_missing_clean=round(100 * df[c].isna().mean(), 2),
            )
        )
    report = pd.DataFrame(rows)

    for c in C.RESISTIVITY_CURVES:
        df[f"log10_{c}"] = np.log10(df[c]).astype("float32")

    codes = df[C.LABEL_RAW]
    unknown = set(codes.unique()) - set(C.CODE_TO_IDX)
    if unknown:
        raise ValueError(f"unmapped lithology codes: {sorted(unknown)}")
    df["lith"] = codes.map(C.CODE_TO_IDX).astype("int8")
    df["conf"] = df[C.CONF].astype("float32")  # NaN = unknown (open-test wells carry no confidence)

    df["GROUP"] = df["GROUP"].fillna("UNKNOWN")
    df["FORMATION"] = df["FORMATION"].fillna("UNKNOWN")

    df = df.drop(columns=[C.LABEL_RAW, C.CONF])
    df = df.sort_values([C.WELL, C.DEPTH], kind="stable").reset_index(drop=True)
    return df, report


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def model_curves() -> list[str]:
    """Log curves used as model inputs.

    Raw resistivities are replaced by their log10; curves listed in
    ``config.UNRELIABLE_CURVES`` are excluded.
    """
    base = [c for c in C.CURVES if c not in C.RESISTIVITY_CURVES and c not in C.UNRELIABLE_CURVES]
    return base + [f"log10_{c}" for c in C.RESISTIVITY_CURVES]


def curve_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """wells x model-curves matrix of the fraction of valid (non-NaN) samples."""
    cov = df.groupby(C.WELL)[model_curves()].agg(lambda s: s.notna().mean())
    return cov


def assign_regions(df: pd.DataFrame) -> pd.Series:
    """Cluster wells into spatial regions (K-means on well X/Y).

    Wells without coordinates inherit the most common region of wells that share
    their NPD quadrant (the number before the first '/' in the well name).
    """
    xy = df.groupby(C.WELL)[["X_LOC", "Y_LOC"]].median()
    have = xy.dropna()
    z = (have - have.mean()) / have.std(ddof=0)
    km = KMeans(n_clusters=C.N_REGIONS, n_init=20, random_state=C.SEED).fit(z)
    # relabel regions 0..k-1 from west to east for readable, stable numbering
    order = np.argsort(km.cluster_centers_[:, 0])
    remap = {old: new for new, old in enumerate(order)}
    region = pd.Series(km.labels_, index=have.index).map(remap)

    wells = df[C.WELL].drop_duplicates()
    quad = wells.str.split("/").str[0]
    quad.index = wells.values
    quad_region = region.groupby(quad.reindex(region.index)).agg(lambda s: s.mode().iloc[0])
    out = region.reindex(wells.values)
    for w in out[out.isna()].index:
        out.loc[w] = quad_region.get(quad.loc[w], np.nan)
    out = out.fillna(-1).astype(int)
    return out.rename("region")


def well_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(C.WELL)
    s = pd.DataFrame(
        {
            "origin": g["origin"].first(),
            "region": g["region"].first(),
            "n_samples": g.size(),
            "top_md": g[C.DEPTH].min(),
            "bottom_md": g[C.DEPTH].max(),
            "x": g["X_LOC"].median(),
            "y": g["Y_LOC"].median(),
            "mean_confidence": g["conf"].mean(),
        }
    )
    s["thickness_m"] = s["bottom_md"] - s["top_md"]
    cov = curve_coverage(df)
    s["n_curves_present"] = (cov >= PRESENT_MIN_COVERAGE).sum(axis=1)
    frac = (
        df.groupby([C.WELL, "lith"]).size().unstack(fill_value=0).reindex(columns=range(12), fill_value=0)
    )
    frac = frac.div(frac.sum(axis=1), axis=0)
    frac.columns = [f"frac_{n}" for n in C.CLASS_NAMES]
    return s.join(frac).reset_index()


def feature_sets(df: pd.DataFrame, core_min_wells: float = 0.70) -> dict:
    """Define input sets from data availability.

    core  = curves present in >= 70 % of all wells (what a robust model can rely on)
    all   = every cleaned curve (the pool used when studying missing-curve robustness)
    """
    cov = curve_coverage(df)
    present_frac = (cov >= PRESENT_MIN_COVERAGE).mean().round(4)
    core = [c for c in model_curves() if present_frac[c] >= core_min_wells]
    return {
        "logs_core": core,
        "logs_all": model_curves(),
        "geometry": ["DEPTH_MD", "X_LOC", "Y_LOC", "Z_LOC"],
        "stratigraphy": ["GROUP", "FORMATION"],
        "excluded_unreliable": C.UNRELIABLE_CURVES,
        "fraction_of_wells_with_curve": present_frac.to_dict(),
        "core_threshold_fraction_of_wells": core_min_wells,
        "present_min_coverage_within_well": PRESENT_MIN_COVERAGE,
    }


def categorical_vocab(df: pd.DataFrame, train_wells: list[str]) -> dict:
    """Vocabulary for GROUP / FORMATION built from training wells only (0 = unseen)."""
    tr = df[df[C.WELL].isin(train_wells)]
    vocab = {}
    for col in C.STRAT:
        cats = sorted(tr[col].unique())
        vocab[col] = {name: i + 1 for i, name in enumerate(cats)}
    return vocab


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def build_splits(df: pd.DataFrame) -> dict:
    """Well-wise splits.

    official            : the competition's train (98) / open_test (10) / hidden_test (10)
    groupkfold          : stratified, well-grouped 5-fold CV over the 108 development wells
                          (train + open_test). hidden_test is never used in CV.
    leave_region_out    : one fold per spatial region (development wells only); the
                          harder, geography-shifted generalisation test
    """
    wells = df.drop_duplicates(C.WELL).set_index(C.WELL)[["origin", "region"]]
    official = {o: sorted(wells.index[wells.origin == o].tolist()) for o in ORIGINS}
    dev_wells = official["train"] + official["open_test"]
    dev = df[df[C.WELL].isin(dev_wells)]

    sgkf = StratifiedGroupKFold(n_splits=C.N_RANDOM_FOLDS, shuffle=True, random_state=C.SEED)
    gkf = []
    for k, (_, va) in enumerate(sgkf.split(dev, dev["lith"], groups=dev[C.WELL])):
        val_wells = sorted(dev.iloc[va][C.WELL].unique().tolist())
        gkf.append({"fold": k, "val": val_wells, "train": sorted(set(dev_wells) - set(val_wells))})

    lro = []
    for r in sorted(wells.loc[dev_wells, "region"].unique()):
        val_wells = sorted(wells.index[(wells.region == r) & wells.index.isin(dev_wells)].tolist())
        lro.append(
            {"region": int(r), "val": val_wells, "train": sorted(set(dev_wells) - set(val_wells))}
        )

    splits = {
        "seed": C.SEED,
        "official": official,
        "groupkfold": gkf,
        "leave_region_out": lro,
        "regions": {w: int(r) for w, r in wells["region"].items()},
    }
    _check_splits(splits)
    return splits


def _check_splits(s: dict) -> None:
    off = s["official"]
    sets = [set(off[o]) for o in ORIGINS]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "official split overlaps"
    dev = sets[0] | sets[1]
    for name in ("groupkfold", "leave_region_out"):
        seen = []
        for f in s[name]:
            assert not (set(f["train"]) & set(f["val"])), f"{name}: train/val well overlap"
            assert not (set(f["train"]) | set(f["val"])) & sets[2], f"{name}: hidden wells leaked"
            assert set(f["train"]) | set(f["val"]) == dev, f"{name}: fold does not cover dev wells"
            seen += f["val"]
        assert sorted(seen) == sorted(dev), f"{name}: validation folds must tile the dev wells once"


def split_class_table(df: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Sample counts per class in every validation fold (to confirm folds are usable)."""
    rows = []

    def add(label, wells):
        sub = df[df[C.WELL].isin(wells)]
        counts = sub["lith"].value_counts().reindex(range(12), fill_value=0)
        rows.append({"split": label, "n_wells": len(wells), "n_samples": len(sub),
                     **{n: int(counts[i]) for i, n in enumerate(C.CLASS_NAMES)}})

    for o in ORIGINS:
        add(f"official/{o}", splits["official"][o])
    for f in splits["groupkfold"]:
        add(f"groupkfold/val{f['fold']}", f["val"])
    for f in splits["leave_region_out"]:
        add(f"leave_region_out/region{f['region']}", f["val"])
    return pd.DataFrame(rows)
