"""Fast sanity tests on the prepared data (run after scripts 01-03)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.dataprep import config as C  # noqa: E402
from src.dataprep import force2020 as F  # noqa: E402
from src.dataprep.scaling import apply_scaling, fit_stats  # noqa: E402
from src.dataprep.windows import PAD_LABEL, fill_and_mask, make_windows  # noqa: E402

PQ = C.PROCESSED / "force2020_clean.parquet"
needs_data = pytest.mark.skipif(not PQ.exists(), reason="run scripts/01-03 first")


def _toy():
    rows = []
    for w, n in (("A", 300), ("B", 100)):
        d = pd.DataFrame({C.WELL: w, C.DEPTH: np.arange(n) * 0.5, "GR": np.random.rand(n),
                          "RHOB": np.where(np.arange(n) % 7 == 0, np.nan, 2.4), "lith": np.arange(n) % 3})
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_sentinels_and_bounds_removed():
    raw = pd.DataFrame({c: [1.0, -999.25, -9999.0, 1e9] for c in C.CURVES})
    raw[C.LABEL_RAW] = 30000; raw[C.CONF] = 1; raw["GROUP"] = None; raw["FORMATION"] = None
    raw[C.WELL] = "W"; raw[C.DEPTH] = [1, 2, 3, 4]; raw["origin"] = "train"
    out, _ = F.clean(raw)
    assert out["GR"].isna().sum() == 3          # two sentinels + one out-of-bounds


def test_label_map_matches_penalty_order():
    assert [v[0] for v in sorted(C.LITHOLOGY.values())] == list(range(12))
    assert C.CLASS_NAMES[2] == "Shale" and C.CLASS_NAMES[11] == "Basement"


def test_windows_keep_nan_and_pad():
    df = _toy()
    feats = ["GR", "RHOB"]
    X, y, idx = make_windows(df[feats], df[[C.WELL, C.DEPTH, "lith"]], ["A", "B"], window=64, stride=32)
    assert X.shape[1:] == (64, 2) and y.shape[1] == 64
    assert np.isnan(X).any(), "NaN must be preserved"
    X2, y2, _ = make_windows(df[feats], df[[C.WELL, C.DEPTH, "lith"]], ["B"], window=160, stride=80)
    assert (y2 == PAD_LABEL).sum() == 60, "100-sample well padded to 160"
    Xf, m = fill_and_mask(X)
    assert not np.isnan(Xf).any() and m.shape == X.shape


def test_scaling_uses_train_wells_only():
    df = _toy()
    s = fit_stats(df, ["A"], ["GR"])
    s2 = fit_stats(df, ["A", "B"], ["GR"])
    assert s["GR"]["n_valid"] == 300 and s2["GR"]["n_valid"] == 400
    sc = apply_scaling(df, s, ["GR"])
    assert sc["GR"].abs().max() <= 10


@needs_data
def test_splits_are_well_disjoint_and_hidden_untouched():
    s = json.loads((C.SPLITS / "force2020_splits.json").read_text())
    hidden = set(s["official"]["hidden_test"])
    assert len(hidden) == 10
    for k in ("groupkfold", "leave_region_out"):
        for f in s[k]:
            assert not set(f["train"]) & set(f["val"])
            assert not (set(f["train"]) | set(f["val"])) & hidden


@needs_data
def test_processed_table():
    df = pd.read_parquet(PQ)
    assert df[C.WELL].nunique() == 118
    assert df["lith"].between(0, 11).all()
    assert not df.duplicated([C.WELL, C.DEPTH]).any()
    assert df["GR"].min() >= 0 and df["RHOB"].dropna().between(1, 3.3).all()
    assert (df[[c for c in C.CURVES]].min() > C.SENTINEL_THRESHOLD).all()
