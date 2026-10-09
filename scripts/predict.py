"""Predict the nightly price of a listing from the command line (live-demo script).

Examples
    python scripts/predict.py --city paris --room-type "Entire home/apt" --accommodates 4 --bedrooms 2
    python scripts/predict.py --city nyc --room-type "Private room" --accommodates 2 \
        --amenities "wifi,kitchen,air conditioning" --text "Cozy room near Central Park"
    python scripts/predict.py --sample 5      # 5 random real test listings: predicted vs actual
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb import config as C  # noqa: E402
from airbnb.data import load_processed  # noqa: E402
from airbnb.evaluate import split_city  # noqa: E402
from airbnb.inference import load_bundle, make_listing, predict_price  # noqa: E402


def show_samples(bundle, n):
    df = load_processed()
    for city in C.CITIES:
        _, test = split_city(df, city)  # same held-out split used in training
        s = test.sample(n, random_state=7)
        pred = predict_price(bundle, s)
        print(f"\n{C.CITIES[city]['name']} - held-out listings")
        print(f"{'room type':18s} {'guests':>6s} {'bedrm':>5s} {'actual $':>9s} {'predicted $':>11s}")
        for (_, r), p in zip(s.iterrows(), pred):
            bedrooms = "-" if r.bedrooms != r.bedrooms else f"{r.bedrooms:.0f}"  # NaN -> "-"
            print(f"{r.room_type:18s} {r.accommodates:6.0f} {bedrooms:>5s} {r.price:9.0f} {p:11.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", choices=list(C.CITIES), default="nyc")
    ap.add_argument("--room-type", default=None,
                    choices=["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    ap.add_argument("--accommodates", type=float)
    ap.add_argument("--bedrooms", type=float)
    ap.add_argument("--beds", type=float)
    ap.add_argument("--bathrooms", type=float)
    ap.add_argument("--min-nights", type=float, dest="minimum_nights")
    ap.add_argument("--lat", type=float, dest="latitude")
    ap.add_argument("--lon", type=float, dest="longitude")
    ap.add_argument("--rating", type=float, dest="review_scores_rating", help="0-5")
    ap.add_argument("--reviews", type=float, dest="number_of_reviews")
    ap.add_argument("--superhost", action="store_true")
    ap.add_argument("--amenities", help="comma separated, e.g. 'wifi,kitchen,pool'")
    ap.add_argument("--text", default=None, help="listing title / description")
    ap.add_argument("--sample", type=int, help="show N real test listings per city instead")
    a = ap.parse_args()

    bundle = load_bundle()
    if a.sample:
        return show_samples(bundle, a.sample)

    listing = make_listing(
        bundle["defaults"], a.city,
        amenities=a.amenities.split(",") if a.amenities else None,
        room_type=a.room_type, accommodates=a.accommodates, bedrooms=a.bedrooms, beds=a.beds,
        bathrooms=a.bathrooms, minimum_nights=a.minimum_nights, latitude=a.latitude,
        longitude=a.longitude, review_scores_rating=a.review_scores_rating,
        number_of_reviews=a.number_of_reviews, host_is_superhost=1 if a.superhost else None,
        text=a.text)
    usd = predict_price(bundle, listing)[0]
    msg = f"Predicted nightly price in {C.CITIES[a.city]['name']}: ${usd:,.0f}"
    if C.CITIES[a.city]["currency"] == "EUR":
        msg += f"  (about EUR {usd / bundle['eur_to_usd']:,.0f})"
    print(msg)


if __name__ == "__main__":
    main()
