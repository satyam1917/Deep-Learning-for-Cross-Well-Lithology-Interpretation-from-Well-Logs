# Deep Learning for Cross-Well Lithology Interpretation from Well Logs

Robust, well-to-well generalisable lithology prediction from conventional well logs, including
behaviour when logging curves are missing. This repository currently contains **Step 1: data preparation**.

## Roadmap

1. **Data preparation** (this stage): load, clean, split, scale, explore
2. Baseline models (gradient-boosted trees, competition penalty metric)
3. Sequence deep models (1D-CNN, BiLSTM, Transformer)
4. Missing-curve robustness (masked-curve pretraining, then fine-tuning)
5. Geology-aware features (wavelet / fractal descriptors)
6. Confidence / calibration, second dataset, unseen-well comparison

## Datasets

| Dataset | Content | Role |
|---|---|---|
| FORCE 2020 lithology | 118 North Sea wells (98 train, 10 open test, 10 hidden test), 12 lithology classes, label confidence 1-3, official 12x12 penalty matrix | primary |
| SEG 2016 Hugoton/Panoma | 12 Kansas wells, 9 facies, 7 features (10 train + 2 blind wells) | independent second dataset |

Both are public and downloaded straight from GitHub by `scripts/01_download_data.py` (SHA-256 checksums in `data/checksums.json`).

## Run

```bash
pip install -r requirements.txt
python scripts/01_download_data.py      # raw files -> data/raw (git-ignored)
python scripts/02_prepare_force2020.py  # clean, splits, stats  (~25 s)
python scripts/03_prepare_seg2016.py
python scripts/04_eda.py                # figures + tables in reports/
pytest tests
```

Raw and processed data (`data/raw`, `data/processed`) are git-ignored and fully regenerable.
Committed: code, `data/splits`, `data/stats`, `data/checksums.json`, `reports/`.

## What data preparation does (and why)

- **No imputation.** Missing curves are the object of study, so NaNs are preserved and validity masks are used downstream.
- **Cleaning is auditable.** Sentinels (any value <= -900) and physically impossible values (bounds in `src/dataprep/config.py`) become NaN; counts per curve are in `reports/tables/force_bounds_check.csv`. Resistivities are log10-transformed.
- **MUDWEIGHT excluded from model inputs**: mixed units across wells and 92 % missing. Kept in the cleaned table.
- **Splits are always by well**, never by depth. `data/splits/force2020_splits.json` holds the official split, 5-fold stratified well-grouped CV (108 development wells) and leave-region-out CV (5 spatial regions from K-means on well X/Y). Hidden test wells never appear in any CV fold (asserted).
- **Scaling statistics (median/IQR, clip +-10) and class weights are fitted on training wells only**; GROUP / FORMATION vocabularies likewise (0 = unseen).
- **Windowing** for sequence models (`src/dataprep/windows.py`): 256-sample windows, stride 128, NaN kept, pad label -1.
- **SEG 2016**: 3 duplicate-depth rows removed, blind-well labels attached from the core file (21 blind rows unlabelled), `Recruit F9` (80 samples, single facies) kept for training but never used for validation.
- Feature sets (`data/stats/force2020_feature_sets.json`): `logs_core` = curves recorded in >= 70 % of wells (CALI, RHOB, GR, NPHI, PEF, DTC, BS, DRHO, log10 RMED, log10 RDEP); `logs_all` = 19 curves.

## EDA findings (figures in `reports/figures`, numbers in `reports/eda_numbers.json`)

- **1.43 M samples, 118 wells**, median sampling 0.152 m.
- **Severe class imbalance**: Shale ~62 % of training samples; Basement 0.01 % (103 samples; absent from both test sets). Class prior shifts between splits (e.g. Halite 0.7 % train, 0 % open test, 5.3 % hidden test).
- **Curve availability is the main difficulty**: wells record 8-17 of 19 curves (mean ~13), with 68 distinct availability patterns. Only GR, RDEP, RMED, CALI, DRHO, RHOB, DTC are near-universal; SGR, RMIC, DTS, DCAL, RXO are present in a minority of wells. This motivates the missing-curve study.
- **Covariate shift**: largest train-vs-test KS statistics are for DCAL, SP, ROP, SGR (up to 0.48), i.e. tool/environment dependent curves; GR, NPHI, DTC, RHOB are comparatively stable (KS < 0.12). Per-well normalisation of SP and SGR is a candidate for later stages (SGR has a spike at 0 in training wells).
- **Labels are noisy**: 87 % of training samples have confidence 1 (the best), 10 % confidence 2, 3 % confidence 3. Open-test wells carry no confidence value.
- Curves are strongly inter-correlated (density / neutron / sonic / GR), which is what makes masked-curve reconstruction plausible.
- SEG 2016: 9 facies, strongly varying composition per well; PE is missing for 18.4 % of rows.

## Layout

```
src/dataprep/   config, download, force2020, seg2016, scaling, windows, eda
scripts/        01 download, 02 FORCE prep, 03 SEG prep, 04 EDA
tests/          sanity tests for cleaning, label map, windows, scaling, splits
data/           checksums, splits/, stats/   (raw/ and processed/ are git-ignored)
reports/        figures/, tables/, eda_numbers.json
```
