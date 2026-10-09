"""Optional: small random search for XGBoost hyper-parameters on the NYC training split.

The best setting found is hard-coded in src/airbnb/models.py (XGB_PARAMS); rerun this to check it.
The test split is never touched here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from airbnb import config as C  # noqa: E402
from airbnb.data import load_processed  # noqa: E402
from airbnb.evaluate import split_city  # noqa: E402
from airbnb.features import build_preprocessor  # noqa: E402


def main(n_iter=10):
    df = load_processed()
    train, _ = split_city(df, "nyc")
    X = build_preprocessor().fit_transform(train)
    y = train["log_price"].values
    space = {
        "n_estimators": randint(300, 1200),
        "learning_rate": loguniform(0.02, 0.15),
        "max_depth": randint(4, 10),
        "min_child_weight": randint(1, 8),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.4, 0.5),
    }
    search = RandomizedSearchCV(XGBRegressor(tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE),
                                space, n_iter=n_iter, cv=3, scoring="r2", random_state=C.RANDOM_STATE,
                                verbose=1)
    search.fit(X, y)
    print("best CV R2 (log price):", round(search.best_score_, 4))
    print(json.dumps(search.best_params_, indent=2, default=float))
    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "xgb_tuning.json", "w") as f:
        json.dump({"best_cv_r2": search.best_score_, "best_params": search.best_params_}, f,
                  indent=2, default=float)


if __name__ == "__main__":
    main()
