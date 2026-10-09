"""Step 1: raw Inside Airbnb files -> one clean table (data/processed/listings_clean.csv.gz)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb import config as C  # noqa: E402
from airbnb.data import build_dataset  # noqa: E402


def main():
    df = build_dataset()
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.PROCESSED_FILE, index=False, compression="gzip")
    print(f"\nSaved {len(df):,} listings x {df.shape[1]} columns -> {C.PROCESSED_FILE.relative_to(C.ROOT)}")
    print(df.groupby("city")["price"].describe().round(1))


if __name__ == "__main__":
    main()
