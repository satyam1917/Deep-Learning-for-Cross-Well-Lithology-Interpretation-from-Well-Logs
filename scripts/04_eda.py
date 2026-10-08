"""Step 1d: exploratory analysis. Writes figures to reports/figures and numbers to
reports/tables, and assembles reports/eda_summary.md from the computed values."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataprep import config as C  # noqa: E402
from src.dataprep import eda as E  # noqa: E402
from src.dataprep import force2020 as F  # noqa: E402


def main():
    E.style()
    fig, tab = C.FIGURES, C.TABLES
    fig.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.PROCESSED / "force2020_clean.parquet")
    ws = pd.read_csv(tab / "force_well_summary.csv")
    cov = pd.read_csv(tab / "force_curve_coverage_by_well.csv", index_col=0)
    curves = F.model_curves()

    n, pct = E.class_distribution(df, fig / "force_class_distribution.png")
    n.assign(**{}).T.rename(index=dict(enumerate(C.CLASS_NAMES))).to_csv(tab / "force_class_counts_by_origin.csv")
    E.curve_availability(cov, ws, fig / "force_curve_availability.png")
    E.curve_missing(cov, fig / "force_curve_presence.png")
    E.well_map(ws, fig / "force_well_map.png")
    ks = E.distribution_shift(df, curves, fig / "force_curve_distributions.png")
    ks.to_csv(tab / "force_ks_shift_vs_train.csv", index=False)
    E.correlation(df, curves, fig / "force_correlation.png")
    conf = E.label_confidence(df, fig / "force_label_confidence.png")
    E.crossplot(df, fig / "force_nphi_rhob_crossplot.png")
    n_pat, top_pat = E.missingness_patterns(cov, fig / "force_missingness_patterns.png")

    core = json.loads((C.STATS / "force2020_feature_sets.json"))["logs_core"] if False else \
        json.loads((C.STATS / "force2020_feature_sets.json").read_text())["logs_core"]
    panel_curves = ["GR", "RHOB", "NPHI", "DTC", "log10_RDEP"]
    full = ws[(ws.origin == "train") & (ws.n_curves_present >= ws.n_curves_present.quantile(.8))]
    sel = full.sort_values("n_samples", ascending=False).WELL.iloc[[0, len(full) // 2]].tolist()
    E.well_panels(df, sel, panel_curves, fig / "force_example_wells.png")

    # numbers for the summary
    sampling = (df.groupby(C.WELL)[C.DEPTH].apply(lambda s: s.diff().median())).describe()
    stats = {
        "wells": int(df[C.WELL].nunique()), "rows": int(len(df)),
        "rows_by_origin": df.origin.value_counts().to_dict(),
        "median_depth_step_m": round(float(sampling["50%"]), 4),
        "curves_present_per_well_mean": round(float(ws.n_curves_present.mean()), 2),
        "curves_present_per_well_min": int(ws.n_curves_present.min()),
        "curves_present_per_well_max": int(ws.n_curves_present.max()),
        "n_distinct_availability_patterns": n_pat,
        "label_confidence_share_train": {str(k): v for k, v in conf.items()},
        "top_ks_shifts": ks.sort_values("ks", ascending=False).head(6).to_dict("records"),
        "pct_samples_train_imbalance_ratio": round(float(n.loc["train"].max() / n.loc["train"].min()), 1),
    }
    # SEG
    seg = pd.read_parquet(C.PROCESSED / "seg2016_clean.parquet")
    E.seg_figures(seg, fig)
    (C.REPORTS / "eda_numbers.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))
    print(pct.round(2).T.rename(index=dict(enumerate(C.CLASS_NAMES))).to_string())
    print(ks.pivot(index="curve", columns="split", values="ks").to_string())


if __name__ == "__main__":
    main()
