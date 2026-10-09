"""Download the three raw listings files into data/raw/ (only needed if they are missing).

The files are the Inside Airbnb June/July 2026 snapshots, fetched from a public GitHub mirror.
Newer snapshots can be downloaded by hand from https://insideairbnb.com/get-the-data/.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from airbnb import config as C  # noqa: E402

MIRROR = "https://raw.githubusercontent.com/aryanmittal2107/Airbnb-Multi-City-Price-Prediction-Project/main/data"
SOURCES = {
    "nyc.csv.gz": f"{MIRROR}/NYC/listings.csv.gz",
    "paris.csv.gz": f"{MIRROR}/Paris/listings%20(1).csv.gz",
    "berlin.csv.gz": f"{MIRROR}/berlin/listings%20(2).csv.gz",
}


def main():
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = C.RAW_DIR / name
        if dest.exists():
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading ...")
        urllib.request.urlretrieve(url, dest)
        print(f"   {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
