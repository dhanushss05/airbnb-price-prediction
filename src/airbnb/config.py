"""Project-wide settings: paths, cities, cleaning thresholds and feature lists."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"
PROCESSED_FILE = PROCESSED_DIR / "listings_clean.csv.gz"

RANDOM_STATE = 42

# Inside Airbnb snapshots (June/July 2026). Paris and Berlin prices are in EUR.
CITIES = {
    "nyc": {"name": "New York City", "file": "nyc.csv.gz", "currency": "USD",
            "centre": (40.7580, -73.9855)},      # Times Square
    "paris": {"name": "Paris", "file": "paris.csv.gz", "currency": "EUR",
              "centre": (48.8566, 2.3522)},      # Hotel de Ville
    "berlin": {"name": "Berlin", "file": "berlin.csv.gz", "currency": "EUR",
               "centre": (52.5200, 13.4050)},    # Alexanderplatz / Mitte
}

# Fixed conversion so all cities share one price unit (USD per night).
# A single constant only shifts log(price) by a constant, so within-city
# results do not depend on it; it matters only for cross-city transfer.
EUR_TO_USD = 1.15

# Label cleaning: drop obviously broken or extreme nightly prices (USD).
MIN_PRICE = 10
MAX_PRICE = 1000

# ---- Feature groups --------------------------------------------------------
# "Continuous" listing features (numeric, imputed + scaled).
NUMERIC_FEATURES = [
    "accommodates", "bathrooms", "bedrooms", "beds", "shared_bath",
    "minimum_nights", "maximum_nights",
    "availability_30", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "host_listings_count", "calculated_host_listings_count",
    "hosts_time_as_host_years", "host_is_superhost", "host_identity_verified",
    "host_has_profile_pic", "has_license", "estimated_occupancy_l365d",
    "days_since_first_review", "days_since_last_review",
    "n_amenities", "desc_length", "dist_centre_km",
]
# Raw coordinates only make sense inside one city, so they are dropped for
# the cross-city transfer experiment (dist_centre_km is kept).
GEO_FEATURES = ["latitude", "longitude"]
CATEGORICAL_FEATURES = ["room_type", "property_type"]
AMENITY_COLUMN = "amenity_list"
TEXT_COLUMN = "text"
CITY_COLUMN = "city"

TOP_AMENITIES = 60
TEXT_SVD_DIM = 50
