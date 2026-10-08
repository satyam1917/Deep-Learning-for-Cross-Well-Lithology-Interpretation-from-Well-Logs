"""Reproducible download of the raw datasets, with SHA-256 integrity records."""
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

from . import config as C

CHECKSUM_FILE = C.DATA / "checksums.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not download {url}")


def download_all(force: bool = False) -> dict:
    """Download every raw file. Existing files are kept unless ``force`` is set.

    Returns {relative_path: sha256}. On first run the hashes are stored in
    ``data/checksums.json`` (committed to git); on later runs a mismatch raises,
    so silent upstream changes cannot slip into the experiments.
    """
    known = json.loads(CHECKSUM_FILE.read_text()) if CHECKSUM_FILE.exists() else {}
    out = {}
    jobs = [("force2020", C.FORCE_BASE, C.FORCE_FILES), ("seg2016", C.SEG_BASE, C.SEG_FILES)]
    for folder, base, files in jobs:
        for name in files:
            dest = C.RAW / folder / name
            if force or not dest.exists():
                print(f"downloading {folder}/{name}")
                _fetch(f"{base}/{name}", dest)
            digest = _sha256(dest)
            key = f"{folder}/{name}"
            if key in known and known[key] != digest:
                raise RuntimeError(f"Checksum mismatch for {key}: file changed upstream or is corrupt")
            out[key] = digest
            print(f"  ok {key}  {dest.stat().st_size/1e6:8.2f} MB  {digest[:12]}")
    if not known:
        CHECKSUM_FILE.write_text(json.dumps(out, indent=2, sort_keys=True))
    zpath = C.RAW / "force2020" / "train.zip"
    target = C.RAW / "force2020" / "train.csv"
    if not target.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(C.RAW / "force2020")
    return out


if __name__ == "__main__":
    download_all()
