"""Interactive demo: `streamlit run app/app.py`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from airbnb import config as C  # noqa: E402
from airbnb.inference import load_bundle, make_listing, predict_price  # noqa: E402

st.set_page_config(page_title="Airbnb price predictor", layout="wide")


@st.cache_resource
def bundle():
    return load_bundle()


b = bundle()
st.title("Airbnb nightly price predictor")
st.caption("XGBoost trained on Inside Airbnb listings from New York City, Paris and Berlin "
           "(2026 snapshots). Prices in USD; EUR converted at a fixed rate.")
result = st.container(border=True)  # filled at the end, shown at the top

left, right = st.columns([1, 1])
with left:
    city = st.selectbox("City", list(C.CITIES), format_func=lambda c: C.CITIES[c]["name"])
    d = b["defaults"][city]
    room = st.selectbox("Room type", ["Entire home/apt", "Private room", "Hotel room", "Shared room"])
    guests = st.slider("Guests (accommodates)", 1, 16, 2)
    bedrooms = st.slider("Bedrooms", 0, 8, 1)
    baths = st.slider("Bathrooms", 0.0, 5.0, 1.0, 0.5)
    min_n = st.slider("Minimum nights", 1, 60, int(d["minimum_nights"]))
    rating = st.slider("Review rating (0-5)", 0.0, 5.0, 4.8, 0.05)
    reviews = st.number_input("Number of reviews", 0, 2000, 20)
    superhost = st.checkbox("Superhost")
with right:
    lat = st.number_input("Latitude", value=float(C.CITIES[city]["centre"][0]), format="%.4f")
    lon = st.number_input("Longitude", value=float(C.CITIES[city]["centre"][1]), format="%.4f")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=11)
    amen = st.multiselect("Amenities", ["wifi", "kitchen", "air conditioning", "washer", "dryer",
                                        "dishwasher", "elevator", "free parking on premises",
                                        "pool", "gym", "hot tub", "tv", "dedicated workspace",
                                        "self check-in", "private patio or balcony"],
                          default=["wifi", "kitchen"])
    text = st.text_area("Listing title / description", "Bright apartment close to the metro")

listing = make_listing(b["defaults"], city, amenities=amen, room_type=room, accommodates=guests,
                       bedrooms=bedrooms, beds=max(bedrooms, 1), bathrooms=baths,
                       minimum_nights=min_n, review_scores_rating=rating,
                       number_of_reviews=reviews, host_is_superhost=int(superhost),
                       latitude=lat, longitude=lon, text=text)
usd = predict_price(b, listing)[0]
with result:
    eur = f"  (about EUR {usd / b['eur_to_usd']:,.0f})" if C.CITIES[city]["currency"] == "EUR" else ""
    st.metric(f"Predicted price per night in {C.CITIES[city]['name']}", f"${usd:,.0f}{eur}")
