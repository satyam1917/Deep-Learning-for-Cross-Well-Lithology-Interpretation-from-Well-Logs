"""Step 1c: prepare the SEG 2016 (Hugoton / Panoma) benchmark.

Outputs
  data/processed/seg2016_clean.parquet   (git-ignored)
  data/splits/seg2016_splits.json
  data/stats/seg2016_*.json
  reports/tables/seg_*.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataprep import config as C  # noqa: E402
from src.dataprep import seg2016 as S  # noqa: E402
from src.dataprep.scaling import fit_stats  # noqa: E402


def main():
    for d in (C.PROCESSED, C.SPLITS, C.STATS, C.TABLES):
        d.mkdir(parents=True, exist_ok=True)

    raw = S.load_raw()
    print(f"  {len(raw):,} rows, {raw[S.WELLC].nunique()} wells")
    df, audit = S.clean(raw)
    (C.TABLES / "seg_cleaning_audit.json").write_text(json.dumps(audit, indent=2))
    print("  audit:", audit)

    ws = S.well_summary(df)
    ws.to_csv(C.TABLES / "seg_well_summary.csv", index=False)

    splits = S.build_splits(df)
    (C.SPLITS / "seg2016_splits.json").write_text(json.dumps(splits, indent=2))

    train_wells = splits["official"]["train"]
    stats = fit_stats(df.rename(columns={S.WELLC: C.WELL}), train_wells, C.SEG_CURVES)
    (C.STATS / "seg2016_train_scaling_stats.json").write_text(json.dumps(stats, indent=2))

    lab = df[df["facies"].notna() & df[S.WELLC].isin(train_wells)]
    counts = lab["facies"].astype(int).value_counts().reindex(range(9), fill_value=0)
    freq = counts / counts.sum()
    inv = 1 / freq
    (C.STATS / "seg2016_train_class_weights.json").write_text(json.dumps({
        "class_names": [C.SEG_FACIES_NAMES[i + 1] for i in range(9)],
        "counts": counts.tolist(),
        "frequency": freq.round(6).tolist(),
        "weight_inverse_frequency": (inv / inv.mean()).round(4).tolist(),
    }, indent=2))

    (C.STATS / "seg2016_to_force_curve_map.json").write_text(json.dumps({
        "map": S.CURVE_MAP_TO_FORCE,
        "note": "Only curves with a comparable meaning are mapped. ILD is an induction log, "
                "FORCE RDEP is a deep resistivity: similar, not identical. Facies and "
                "lithology label sets are different and are not mapped."}, indent=2))

    out = C.PROCESSED / "seg2016_clean.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    print(f"saved {out} ({len(df)} rows)")
    print(ws.to_string(index=False))
    print("PE missing overall: %.1f %%" % (100 * df["PE"].isna().mean()))
    assert np.isfinite(df[C.SEG_CURVES].drop(columns="PE").to_numpy()).all(), "unexpected NaN in SEG curves"


if __name__ == "__main__":
    main()
