"""EDA figures and numbers for the data-preparation report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from . import config as C  # noqa: E402

# reference categorical palette (fixed order); origins use slots 1-3
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ORIGIN_COL = {"train": BLUE, "open_test": ORANGE, "hidden_test": AQUA}
# conventional lithology colours (12 classes exceed the 8-hue categorical limit; each is
# also named in a legend / direct label, so identity never relies on colour alone)
LITH_COL = ["#f2c14e", "#c9b36a", "#6b7f5e", "#b58bb0", "#7aa6c2", "#8ecae6", "#e8e2c9",
            "#e4a0b5", "#b9a7d8", "#9a9a9a", "#2b2b2b", "#c0504d"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110,
    })


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def class_distribution(df, path):
    n = df.groupby(["origin", "lith"]).size().unstack(fill_value=0).reindex(columns=range(12), fill_value=0)
    pct = n.div(n.sum(axis=1), axis=0) * 100
    order = pct.loc["train"].sort_values().index
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    h = 0.26
    for k, o in enumerate(["train", "open_test", "hidden_test"]):
        ax.barh(np.arange(12) + (k - 1) * h, pct.loc[o, order].values, height=h - 0.03,
                color=ORIGIN_COL[o], label=o.replace("_", " "))
    ax.set_yticks(range(12)); ax.set_yticklabels([C.CLASS_NAMES[i] for i in order])
    ax.set_xscale("symlog", linthresh=0.1); ax.set_xlim(0, 100)
    ax.set_xticks([0, 0.1, 1, 10, 50]); ax.set_xticklabels(["0", "0.1", "1", "10", "50"])
    ax.set_xlabel("share of samples in split (%, log-style axis)")
    ax.set_title("Lithology classes are strongly imbalanced and the prior shifts between splits")
    ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
    _save(fig, path)
    return n, pct


def curve_availability(cov, ws, path):
    order = ws.sort_values(["origin", "region", "WELL"]).WELL
    m = cov.loc[order]
    fig, ax = plt.subplots(figsize=(7.6, 9.5))
    im = ax.imshow(m.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize=4.8)
    ax.grid(False)
    ori = ws.set_index("WELL").loc[order, "origin"].values
    for lab in ax.get_yticklabels():
        pass
    for i, o in enumerate(ori):
        ax.add_patch(plt.Rectangle((-1.2, i - .5), .7, 1, color=ORIGIN_COL[o], clip_on=False))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("fraction of samples with valid values")
    ax.set_title("Curve availability per well (left bar: train / open test / hidden test)")
    ax.legend(handles=[Patch(color=c, label=k.replace("_", " ")) for k, c in ORIGIN_COL.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    _save(fig, path)


def curve_missing(cov, path):
    wells_with = (cov >= 0.05).mean().sort_values()
    miss = (1 - cov).mean().reindex(wells_with.index) * 100
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(range(len(miss)), wells_with.values * 100, color=BLUE, height=0.62)
    ax.set_yticks(range(len(miss))); ax.set_yticklabels(miss.index)
    for i, v in enumerate(wells_with.values * 100):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8, color=MUTED)
    ax.axvline(70, color=ORANGE, lw=1); ax.text(71, -0.9, "core threshold 70%", color=MUTED, fontsize=8)
    ax.set_xlim(0, 110); ax.grid(axis="y", visible=False)
    ax.set_xlabel("% of the 118 wells that record the curve (≥5% valid samples)")
    ax.set_title("Only a handful of curves are present in nearly every well")
    _save(fig, path)


def well_map(ws, path):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    for o, col in ORIGIN_COL.items():
        s = ws[ws.origin == o]
        ax.scatter(s.x, s.y, s=26, color=col, edgecolor=SURFACE, linewidth=0.8, label=o.replace("_", " "))
    for r, g in ws.groupby("region"):
        ax.text(g.x.mean(), g.y.max() + (ws.y.max() - ws.y.min()) * 0.03, f"R{r}", ha="center",
                color=MUTED, fontsize=9, fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title("Well locations and spatial regions (R0 west → R4 east)")
    ax.legend(); _save(fig, path)


def distribution_shift(df, curves, path, ref="train"):
    rows = []
    for c in curves:
        a = df.loc[df.origin == ref, c].dropna().to_numpy()
        for o in ("open_test", "hidden_test"):
            b = df.loc[df.origin == o, c].dropna().to_numpy()
            if len(a) > 50 and len(b) > 50:
                ks = ks_2samp(a[::5], b[::max(1, len(b) // 20000)] if len(b) > 20000 else b).statistic
                rows.append({"curve": c, "split": o, "ks": round(float(ks), 4)})
    t = pd.DataFrame(rows)
    n = len(curves); cols = 4; r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, cols, figsize=(11, 2.3 * r))
    for ax, c in zip(axes.ravel(), curves):
        v = df[c].dropna(); lo, hi = np.percentile(v, [0.5, 99.5]); bins = np.linspace(lo, hi, 50)
        for o in ORIGIN_COL:
            x = df.loc[df.origin == o, c].dropna()
            if len(x):
                ax.hist(x.clip(lo, hi), bins=bins, density=True, histtype="step", lw=1.3, color=ORIGIN_COL[o])
        ax.set_title(c, fontsize=9); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=1.5, label=k.replace("_", " ")) for k, c in ORIGIN_COL.items()],
               loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle("Curve distributions by split (density, 0.5–99.5 percentile)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); _save(fig, path)
    return t


def correlation(df, curves, path):
    sub = df.loc[df.origin == "train", curves].sample(300_000, random_state=C.SEED, replace=False) \
        if (df.origin == "train").sum() > 300_000 else df.loc[df.origin == "train", curves]
    cm = sub.corr(method="spearman", min_periods=1000)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = cm.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("Spearman ρ (pairwise, training wells)")
    ax.set_title("Curves are strongly inter-correlated, which is what makes reconstruction possible")
    _save(fig, path)
    return cm


def label_confidence(df, path):
    tr = df[df.origin == "train"]
    t = tr.groupby(["lith", "conf"]).size().unstack(fill_value=0)
    t = t.div(t.sum(axis=1), axis=0) * 100
    order = tr.groupby("lith").size().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 4.2))
    left = np.zeros(len(order))
    cols = {1.0: "#9bc3f0", 2.0: BLUE, 3.0: "#14396b"}
    for k in (1.0, 2.0, 3.0):
        if k in t.columns:
            v = t.loc[order, k].values
            ax.barh(range(len(order)), v, left=left, color=cols[k], height=0.65, label=f"confidence {int(k)}",
                    edgecolor=SURFACE, linewidth=1)
            left += v
    ax.set_yticks(range(len(order))); ax.set_yticklabels([C.CLASS_NAMES[i] for i in order])
    ax.set_xlabel("% of training samples"); ax.grid(axis="y", visible=False)
    ax.set_title("Label confidence by lithology (training wells)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15)); _save(fig, path)
    return tr["conf"].value_counts(normalize=True).sort_index().round(4).to_dict()


def crossplot(df, path):
    s = df[(df.origin == "train")].dropna(subset=["RHOB", "NPHI"])
    s = s.sample(min(len(s), 120_000), random_state=C.SEED)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for i in [2, 0, 1, 3, 5, 6, 4, 8, 7, 10]:
        g = s[s.lith == i]
        ax.scatter(g.NPHI, g.RHOB, s=3, alpha=.35, color=LITH_COL[i], label=C.CLASS_NAMES[i], rasterized=True)
    ax.set_ylim(3.0, 1.6); ax.set_xlim(-0.05, 0.8)
    ax.set_xlabel("NPHI (v/v)"); ax.set_ylabel("RHOB (g/cc)")
    ax.set_title("Neutron–density crossplot by lithology")
    ax.legend(markerscale=4, ncol=2, fontsize=7.5); _save(fig, path)


def well_panels(df, wells, curves, path):
    cmap = ListedColormap(LITH_COL)
    ncol = len(curves) + 1
    fig, axes = plt.subplots(1, ncol * len(wells), figsize=(1.25 * ncol * len(wells), 6.6), squeeze=False)
    axes = axes[0]
    for w_i, w in enumerate(wells):
        g = df[df[C.WELL] == w]
        base = w_i * ncol
        for k, c in enumerate(curves):
            ax = axes[base + k]
            ax.plot(g[c], g[C.DEPTH], lw=0.6, color=BLUE)
            lo, hi = np.nanpercentile(df[c], [0.5, 99.5])
            ax.set_xlim(lo, hi)
            ax.set_ylim(g[C.DEPTH].max(), g[C.DEPTH].min()); ax.set_title(c, fontsize=8)
            ax.tick_params(labelsize=6); ax.grid(axis="x", visible=False)
            if k:
                ax.set_yticklabels([])
        ax = axes[base + len(curves)]
        ax.imshow(g["lith"].to_numpy()[:, None], aspect="auto", cmap=cmap, vmin=0, vmax=11,
                  interpolation="nearest", extent=[0, 1, g[C.DEPTH].max(), g[C.DEPTH].min()])
        ax.set_xticks([]); ax.set_title("lith", fontsize=8); ax.grid(False); ax.set_yticklabels([])
        axes[base].set_ylabel(f"{w}\nDEPTH_MD (m)", fontsize=7)
    fig.legend(handles=[Patch(color=LITH_COL[i], label=C.CLASS_NAMES[i]) for i in range(12)],
               loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Example wells: curves with the lithology track", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97)); _save(fig, path)


def missingness_patterns(cov, path, top=12):
    pat = (cov >= 0.05).apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)
    vc = pat.value_counts()
    cols = list(cov.columns)
    top_p = vc.head(top)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    m = np.array([[int(ch) for ch in p] for p in top_p.index])
    ax.imshow(m, cmap=ListedColormap(["#eeedea", BLUE]), aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right")
    ax.set_yticks(range(len(top_p))); ax.set_yticklabels([f"{n} wells" for n in top_p.values])
    ax.set_title(f"The {top} most common curve-availability patterns (blue = present)")
    _save(fig, path)
    return int(vc.size), vc.head(top)


# --------------------------------------------------------------------------- #
def seg_figures(df, fig_dir):
    names = [C.SEG_FACIES_NAMES[i] for i in range(1, 10)]
    lab = df[df.facies.notna()].copy(); lab["facies"] = lab.facies.astype(int)
    t = lab.groupby(["WELL", "facies"]).size().unstack(fill_value=0).reindex(columns=range(9), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bottom = np.zeros(len(t))
    pal = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#7a7a76"]
    for k in range(9):
        ax.bar(range(len(t)), t[k].values, bottom=bottom, color=pal[k], label=names[k], width=0.7,
               edgecolor=SURFACE, linewidth=1)
        bottom += t[k].values
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t.index, rotation=40, ha="right")
    ax.set_ylabel("labelled samples"); ax.grid(axis="x", visible=False)
    ax.set_title("SEG 2016: facies composition per well"); ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.01, 1))
    _save(fig, fig_dir / "seg_facies_per_well.png")

    fig, axes = plt.subplots(1, len(C.SEG_CURVES), figsize=(12, 2.4))
    for ax, c in zip(axes, C.SEG_CURVES):
        for o, col in (("train", BLUE), ("blind", ORANGE)):
            x = df.loc[df.origin == o, c].dropna()
            ax.hist(x, bins=40, density=True, histtype="step", color=col, lw=1.3, label=o)
        ax.set_title(c, fontsize=9); ax.set_yticks([])
    axes[0].legend(fontsize=7)
    fig.tight_layout(); _save(fig, fig_dir / "seg_curve_distributions.png")
    return t
