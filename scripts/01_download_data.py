"""Step 1a: download raw datasets (FORCE 2020, SEG 2016)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataprep.download import download_all  # noqa: E402

if __name__ == "__main__":
    download_all()
