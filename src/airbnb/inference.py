"""Helpers to predict the price of a new (hand-entered) listing with the saved model."""
import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data import haversine_km

MODEL_FILE = C.MODEL_DIR / "xgb_all_cities.joblib"


def city_defaults(df: pd.DataFrame) -> dict:
    """A 'typical listing' per city: median of numeric columns, mode of categoricals."""
    out = {}
    for city, d in df.groupby(C.CITY_COLUMN):
        row = d[C.NUMERIC_FEATURES + C.GEO_FEATURES].median(numeric_only=True).to_dict()
        for col in C.CATEGORICAL_FEATURES:
            row[col] = d[col].mode().iloc[0]
        row[C.AMENITY_COLUMN] = "wifi|kitchen|essentials|heating|hot water|smoke alarm"
        row[C.TEXT_COLUMN] = ""
        row[C.CITY_COLUMN] = city
        # Most common property type for each room type, so "Private room" does not
        # end up paired with "Entire rental unit".
        row["property_by_room"] = d.groupby("room_type")["property_type"].agg(lambda s: s.mode().iloc[0]).to_dict()
        out[city] = row
    return out


def load_bundle(path=MODEL_FILE) -> dict:
    return joblib.load(path)


def make_listing(defaults: dict, city: str, amenities=None, **overrides) -> pd.DataFrame:
    """Start from the typical listing of `city` and overwrite whatever the user gave."""
    row = dict(defaults[city])
    by_room = row.pop("property_by_room", {})
    row.update({k: v for k, v in overrides.items() if v is not None})
    if overrides.get("room_type") and not overrides.get("property_type"):
        row["property_type"] = by_room.get(row["room_type"], row["property_type"])
    if amenities is not None:
        row[C.AMENITY_COLUMN] = "|".join(a.strip().lower() for a in amenities if a.strip())
        row["n_amenities"] = max(len(amenities), 1)
    lat0, lon0 = C.CITIES[city]["centre"]
    row["dist_centre_km"] = float(haversine_km(row["latitude"], row["longitude"], lat0, lon0))
    row["desc_length"] = len(row.get(C.TEXT_COLUMN, ""))
    return pd.DataFrame([row])


def predict_price(bundle: dict, listing: pd.DataFrame) -> np.ndarray:
    """Predicted nightly price in USD (model outputs log-price)."""
    return np.exp(bundle["pipeline"].predict(listing))
