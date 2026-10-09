# Data

Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/) detailed listings files
(`listings.csv.gz`), snapshots scraped in June/July 2026:

| City | File | Scraped | Listings | With a price | Currency |
|---|---|---|---|---|---|
| New York City | `raw/nyc.csv.gz` | 2026-06-15 | 30,259 | 21,515 | USD |
| Paris | `raw/paris.csv.gz` | 2026-06-21 | 77,679 | 48,402 | EUR |
| Berlin | `raw/berlin.csv.gz` | 2026-07-03 | 12,776 | 8,441 | EUR |

`processed/listings_clean.csv.gz` is produced by `python scripts/prepare_data.py`
(75,222 listings, prices converted to USD, 10 to 1000 USD per night kept).

If the raw files are missing, run `python scripts/download_data.py` or download the three
`listings.csv.gz` files for New York City, Paris and Berlin from the Inside Airbnb
"Get the data" page and save them under `data/raw/` with the names above.
Inside Airbnb data is licensed CC BY 4.0.
