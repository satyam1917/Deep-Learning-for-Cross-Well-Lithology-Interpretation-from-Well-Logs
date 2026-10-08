"""Step 1b: clean FORCE 2020, build summaries, splits, scaling statistics.

Outputs
  data/processed/force2020_clean.parquet      (git-ignored, regenerate with this script)
  data/splits/force2020_splits.json           well-wise train / CV / leave-region-out splits
  data/stats/force2020_*.json                 feature sets, scaling stats, class weights, vocab
  reports/tables/force_*.csv                  audit tables used in the EDA report
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.dataprep import config as C  # noqa: E402
from src.dataprep import force2020 as F  # noqa: E402
from src.dataprep.scaling import class_weights, fit_stats  # noqa: E402


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def main():
    for d in (C.PROCESSED, C.SPLITS, C.STATS, C.TABLES, C.FIGURES):
        d.mkdir(parents=True, exist_ok=True)

    print("loading raw ...")
    raw = F.load_raw()
    print(f"  {len(raw):,} rows, {raw[C.WELL].nunique()} wells")

    print("cleaning ...")
    df, bounds = F.clean(raw)
    bounds.to_csv(C.TABLES / "force_bounds_check.csv", index=False)

    regions = F.assign_regions(df)
    df["region"] = df[C.WELL].map(regions).astype("int8")

    print("summaries ...")
    ws = F.well_summary(df)
    ws.to_csv(C.TABLES / "force_well_summary.csv", index=False)
    cov = F.curve_coverage(df)
    cov.round(4).to_csv(C.TABLES / "force_curve_coverage_by_well.csv")

    print("splits ...")
    splits = F.build_splits(df)
    dump(splits, C.SPLITS / "force2020_splits.json")
    F.split_class_table(df, splits).to_csv(C.TABLES / "force_split_class_counts.csv", index=False)

    print("feature sets, stats, weights ...")
    fs = F.feature_sets(df)
    dump(fs, C.STATS / "force2020_feature_sets.json")
    train_wells = splits["official"]["train"]
    feats = fs["logs_all"]
    dump(fit_stats(df, train_wells, feats), C.STATS / "force2020_train_scaling_stats.json")
    dump(class_weights(df, train_wells), C.STATS / "force2020_train_class_weights.json")
    vocab = F.categorical_vocab(df, train_wells)
    dump(vocab, C.STATS / "force2020_train_categorical_vocab.json")

    # how much of the held-out wells' stratigraphy is unseen in the training wells?
    rows = []
    for o in ("open_test", "hidden_test"):
        sub = df[df.origin == o]
        for col in C.STRAT:
            unseen = ~sub[col].isin(vocab[col].keys())
            rows.append({"origin": o, "column": col, "pct_rows_unseen_category": round(100 * unseen.mean(), 2)})
    pd.DataFrame(rows).to_csv(C.TABLES / "force_unseen_stratigraphy.csv", index=False)

    out = C.PROCESSED / "force2020_clean.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB), columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
