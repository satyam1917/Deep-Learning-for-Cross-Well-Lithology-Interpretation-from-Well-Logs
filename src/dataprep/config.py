"""Central configuration for the data-preparation stage.

Everything that is a *decision* (curve lists, label maps, physical bounds, split
settings) lives here so that it is visible, reviewable and easy to change.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
STATS = DATA / "stats"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

SEED = 42

# --------------------------------------------------------------------------- #
# Sources (all public, direct download, no registration)
# --------------------------------------------------------------------------- #
FORCE_BASE = (
    "https://raw.githubusercontent.com/bolgebrygg/"
    "Force-2020-Machine-Learning-competition/master/lithology_competition/data"
)
FORCE_FILES = [
    "train.zip",                      # 98 wells, features + labels
    "leaderboard_test_features.csv",  # 10 "open test" wells, features
    "leaderboard_test_target.csv",    # 10 "open test" wells, labels
    "hidden_test.csv",                # 10 "hidden/blind" wells, features + labels
    "penalty_matrix.npy",             # 12 x 12 competition scoring matrix
]

SEG_BASE = "https://raw.githubusercontent.com/seg/2016-ml-contest/master"
SEG_FILES = [
    "facies_vectors.csv",                  # training wells with facies labels
    "validation_data_nofacies.csv",        # two blind wells (features only)
    "blind_stuart_crawford_core_facies.csv",  # labels for the two blind wells
]

# --------------------------------------------------------------------------- #
# FORCE 2020: schema and label map
# --------------------------------------------------------------------------- #
WELL = "WELL"
DEPTH = "DEPTH_MD"
XYZ = ["X_LOC", "Y_LOC", "Z_LOC"]
STRAT = ["GROUP", "FORMATION"]
LABEL_RAW = "FORCE_2020_LITHOFACIES_LITHOLOGY"
CONF = "FORCE_2020_LITHOFACIES_CONFIDENCE"

CURVES = [
    "CALI", "RSHA", "RMED", "RDEP", "RHOB", "GR", "SGR", "NPHI", "PEF", "DTC",
    "SP", "BS", "ROP", "DTS", "DCAL", "DRHO", "MUDWEIGHT", "RMIC", "ROPA", "RXO",
]
RESISTIVITY_CURVES = ["RSHA", "RMED", "RDEP", "RMIC", "RXO"]

# raw competition code -> (class index, name). The index order is the row/column
# order of penalty_matrix.npy, so it must not be changed.
LITHOLOGY = {
    30000: (0, "Sandstone"),
    65030: (1, "Sandstone/Shale"),
    65000: (2, "Shale"),
    80000: (3, "Marl"),
    74000: (4, "Dolomite"),
    70000: (5, "Limestone"),
    70032: (6, "Chalk"),
    88000: (7, "Halite"),
    86000: (8, "Anhydrite"),
    99000: (9, "Tuff"),
    90000: (10, "Coal"),
    93000: (11, "Basement"),
}
CODE_TO_IDX = {k: v[0] for k, v in LITHOLOGY.items()}
CLASS_NAMES = [v[1] for v in sorted(LITHOLOGY.values())]

# Curves kept in the cleaned table but excluded from model inputs, with the reason.
# (Found during the audit in reports/tables/force_bounds_check.csv.)
UNRELIABLE_CURVES = {
    "MUDWEIGHT": "mixed units across wells: ~0.13-0.20 in 20 of the 28 wells that have it, "
                 "1.0-2.4 g/cc elsewhere; also 92 % missing",
}

# Any value at or below this is treated as a logging "no data" sentinel
# (-999, -999.25, -999.9, -9999 ...). No real log curve in this data goes this low.
SENTINEL_THRESHOLD = -900.0

# Physical plausibility bounds (inclusive). Values outside are set to NaN.
# Chosen to remove impossible readings, not to trim real geology; see
# reports/tables/force_bounds_check.csv for what each bound removed.
PHYSICAL_BOUNDS = {
    "CALI": (2.0, 40.0),        # inch
    "BS": (2.0, 40.0),          # inch (bit size)
    "DCAL": (-10.0, 25.0),      # inch (caliper minus bit size)
    "GR": (0.0, 500.0),         # API
    "SGR": (0.0, 500.0),        # API
    "RHOB": (1.0, 3.3),         # g/cc
    "DRHO": (-1.0, 1.0),        # g/cc
    "NPHI": (-0.15, 1.0),       # v/v
    "PEF": (0.0, 20.0),         # b/e
    "DTC": (30.0, 250.0),       # us/ft
    "DTS": (50.0, 500.0),       # us/ft
    "SP": (-300.0, 300.0),      # mV
    "ROP": (0.0, 500.0),        # m/h
    "ROPA": (0.0, 500.0),       # m/h
    "MUDWEIGHT": (0.8, 3.0),    # g/cc
    "RSHA": (0.01, 1e5),        # ohm.m
    "RMED": (0.01, 1e5),
    "RDEP": (0.01, 1e5),
    "RMIC": (0.01, 1e5),
    "RXO": (0.01, 1e5),
}

# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
N_REGIONS = 5          # spatial clusters of wells used for leave-region-out CV
N_RANDOM_FOLDS = 5     # well-wise GroupKFold

# --------------------------------------------------------------------------- #
# SEG 2016 (Hugoton / Panoma) schema
# --------------------------------------------------------------------------- #
SEG_LABEL = "Facies"
SEG_CURVES = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]
SEG_FACIES_NAMES = {
    1: "Nonmarine sandstone",
    2: "Nonmarine coarse siltstone",
    3: "Nonmarine fine siltstone",
    4: "Marine siltstone and shale",
    5: "Mudstone",
    6: "Wackestone",
    7: "Dolomite",
    8: "Packstone-grainstone",
    9: "Phylloid-algal bafflestone",
}
