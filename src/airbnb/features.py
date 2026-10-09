"""Feature pipeline: one sklearn ColumnTransformer that turns a listings table into a matrix.

Feature groups (switchable, used for the ablation study):
    C  continuous   numeric listing / host / review features (+ lat/lon if `geo`)
    O  categorical  room type, property type (one-hot) and amenities (multi-hot)
    T  text         TF-IDF (uni+bi-grams) of title + description, reduced by truncated SVD
    city            one-hot city indicator (only useful when training on several cities)
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def _split_amenities(s):
    return [a for a in s.split("|") if a]


class AmenityEncoder(BaseEstimator, TransformerMixin):
    """Multi-hot encoding of the `top_k` most common amenities seen in training data."""

    def __init__(self, top_k=C.TOP_AMENITIES):
        self.top_k = top_k

    def fit(self, X, y=None):
        self.vec_ = CountVectorizer(tokenizer=_split_amenities, lowercase=False,
                                    token_pattern=None, binary=True, max_features=self.top_k)
        self.vec_.fit(_as_series(X))
        return self

    def transform(self, X):
        return self.vec_.transform(_as_series(X)).toarray().astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"amenity: {a}" for a in self.vec_.get_feature_names_out()])


class TextEncoder(BaseEstimator, TransformerMixin):
    """TF-IDF on unigrams+bigrams followed by truncated SVD (same idea as the reference report)."""

    def __init__(self, n_components=C.TEXT_SVD_DIM, max_features=20000):
        self.n_components = n_components
        self.max_features = max_features

    def fit(self, X, y=None):
        self.tfidf_ = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=self.max_features,
                                      sublinear_tf=True, strip_accents="unicode")
        tf = self.tfidf_.fit_transform(_as_series(X))
        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=C.RANDOM_STATE)
        self.svd_.fit(tf)
        return self

    def transform(self, X):
        return self.svd_.transform(self.tfidf_.transform(_as_series(X))).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"text_svd_{i}" for i in range(self.n_components)])


def _as_series(X):
    # ColumnTransformer passes a one-column DataFrame for list-style column selectors.
    if hasattr(X, "iloc") and X.ndim == 2:
        X = X.iloc[:, 0]
    return X.fillna("").astype(str)


def numeric_columns(geo=True):
    return C.NUMERIC_FEATURES + (C.GEO_FEATURES if geo else [])


def build_preprocessor(groups=("C", "O", "T"), geo=True, city=False):
    """Return a ColumnTransformer for the requested feature groups."""
    parts = []
    if "C" in groups:
        num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
        parts.append(("num", num, numeric_columns(geo)))
    if "O" in groups:
        cat = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20,
                            sparse_output=False)
        parts.append(("cat", cat, C.CATEGORICAL_FEATURES))
        parts.append(("amen", AmenityEncoder(), [C.AMENITY_COLUMN]))
    if "T" in groups:
        parts.append(("text", TextEncoder(), [C.TEXT_COLUMN]))
    if city:
        parts.append(("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                      [C.CITY_COLUMN]))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0)
