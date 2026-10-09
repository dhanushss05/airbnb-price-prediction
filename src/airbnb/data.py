"""Loading and cleaning the raw Inside Airbnb `listings.csv.gz` files."""
import html
import json
import re

import numpy as np
import pandas as pd

from . import config as C

# Columns we read from the raw files (the rest are URLs, ids or empty).
RAW_COLUMNS = [
    "id", "name", "description", "last_scraped", "property_type", "room_type",
    "accommodates", "bathrooms", "bathrooms_text", "bedrooms", "beds", "amenities",
    "price", "price_quote_checkin_date", "price_quote_checkout_date",
    "minimum_nights", "maximum_nights", "availability_30", "availability_90",
    "availability_365", "number_of_reviews", "number_of_reviews_ltm",
    "reviews_per_month", "first_review", "last_review",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "host_listings_count", "calculated_host_listings_count",
    "hosts_time_as_host_years", "host_is_superhost", "host_identity_verified",
    "host_has_profile_pic", "license", "estimated_occupancy_l365d",
    "neighbourhood_cleansed", "latitude", "longitude",
]
# Deliberately NOT used because they leak the label: price_quote_total_price,
# price_quote_price_per_night, price_quote_raw, estimated_revenue_l365d.

TAG_RE = re.compile(r"<[^>]+>")


def parse_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,€£]", "", regex=True), errors="coerce")


def parse_bathrooms(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Number of bathrooms (falls back to `bathrooms_text`) and a shared-bath flag."""
    text = df["bathrooms_text"].fillna("").str.lower()
    from_text = pd.to_numeric(text.str.extract(r"([\d.]+)")[0], errors="coerce")
    from_text[text.str.contains("half") & from_text.isna()] = 0.5
    baths = df["bathrooms"].fillna(from_text)
    shared = text.str.contains("shared").astype(int)
    return baths, shared


def haversine_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def clean_text(name: pd.Series, desc: pd.Series) -> pd.Series:
    t = name.fillna("") + ". " + desc.fillna("")
    t = t.map(html.unescape).str.replace(TAG_RE, " ", regex=True)
    return t.str.replace(r"\s+", " ", regex=True).str.strip()


def parse_amenities(s: pd.Series) -> pd.Series:
    """JSON list -> '|'-joined lower-case string (compact to store in a CSV)."""
    def _one(x):
        try:
            items = json.loads(x)
        except (TypeError, ValueError):
            return ""
        return "|".join(sorted({i.strip().lower() for i in items}))
    return s.map(_one)


def load_city(city: str) -> pd.DataFrame:
    meta = C.CITIES[city]
    df = pd.read_csv(C.RAW_DIR / meta["file"], usecols=RAW_COLUMNS, low_memory=False)
    out = pd.DataFrame({"id": df["id"], C.CITY_COLUMN: city})

    price = parse_price(df["price"])
    if meta["currency"] == "EUR":
        price = price * C.EUR_TO_USD
    out["price"] = price

    out["bathrooms"], out["shared_bath"] = parse_bathrooms(df)
    for col in ["accommodates", "bedrooms", "beds", "availability_30", "availability_90",
                "availability_365", "number_of_reviews", "number_of_reviews_ltm",
                "reviews_per_month", "review_scores_rating", "review_scores_accuracy",
                "review_scores_cleanliness", "review_scores_checkin",
                "review_scores_communication", "review_scores_location",
                "review_scores_value", "host_listings_count",
                "calculated_host_listings_count", "hosts_time_as_host_years",
                "estimated_occupancy_l365d", "latitude", "longitude"]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    # Long minimum/maximum stays are capped so a few 1000-night values do not dominate.
    out["minimum_nights"] = df["minimum_nights"].clip(upper=365)
    out["maximum_nights"] = df["maximum_nights"].clip(upper=1125)
    # Length of the stay Airbnb quoted the price for (monthly quotes are discounted).
    # Kept for analysis only, not used as a model feature: it describes how the price
    # was scraped rather than the listing (minimum_nights carries the same signal).
    out["quote_nights"] = (pd.to_datetime(df["price_quote_checkout_date"], errors="coerce")
                           - pd.to_datetime(df["price_quote_checkin_date"], errors="coerce")).dt.days

    for col in ["host_is_superhost", "host_identity_verified", "host_has_profile_pic"]:
        out[col] = df[col].map({"t": 1, "f": 0})
    out["has_license"] = df["license"].notna().astype(int)

    scraped = pd.to_datetime(df["last_scraped"], errors="coerce")
    out["days_since_first_review"] = (scraped - pd.to_datetime(df["first_review"], errors="coerce")).dt.days
    out["days_since_last_review"] = (scraped - pd.to_datetime(df["last_review"], errors="coerce")).dt.days

    out["room_type"] = df["room_type"]
    out["property_type"] = df["property_type"]
    out["neighbourhood"] = df["neighbourhood_cleansed"]
    out["amenity_list"] = parse_amenities(df["amenities"])
    out["n_amenities"] = out["amenity_list"].str.count(r"\|") + (out["amenity_list"] != "").astype(int)
    out["text"] = clean_text(df["name"], df["description"])
    out["desc_length"] = df["description"].fillna("").str.len()

    lat0, lon0 = meta["centre"]
    out["dist_centre_km"] = haversine_km(out["latitude"], out["longitude"], lat0, lon0)
    return out


def build_dataset(verbose: bool = True) -> pd.DataFrame:
    """Load all cities, drop listings without a usable price, return one table."""
    frames = []
    for city in C.CITIES:
        d = load_city(city)
        n0 = len(d)
        d = d[d["price"].notna()]
        n1 = len(d)
        d = d[(d["price"] >= C.MIN_PRICE) & (d["price"] <= C.MAX_PRICE)]
        if verbose:
            print(f"{city:7s} raw={n0:6d}  with price={n1:6d}  "
                  f"kept {C.MIN_PRICE}-{C.MAX_PRICE} USD={len(d):6d} ({1 - len(d) / n1:.1%} dropped)")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["log_price"] = np.log(df["price"])
    return df


def load_processed() -> pd.DataFrame:
    df = pd.read_csv(C.PROCESSED_FILE, low_memory=False)
    df["amenity_list"] = df["amenity_list"].fillna("")
    df["text"] = df["text"].fillna("")
    return df
