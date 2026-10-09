"""Step 2: run every experiment, write results/*.csv + figures, and save the demo model.

Experiments
    E1  model comparison inside each city (NYC, Paris, Berlin), all features
    E2  feature-group ablation with XGBoost (NYC, Paris)
    E3  one model trained on NYC + Paris together vs. one model per city
    E4  transfer: models trained on NYC / Paris / NYC+Paris predict Berlin, a city never seen
Final model: XGBoost on all three cities -> models/xgb_all_cities.joblib (used by predict.py and the app)

Usage:  python scripts/train.py            (everything, ~10-15 min on a laptop)
        python scripts/train.py --quick    (skip slow models in E1, for a fast check)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from airbnb import config as C  # noqa: E402
from airbnb import plots  # noqa: E402
from airbnb.data import load_processed  # noqa: E402
from airbnb.evaluate import regression_metrics, split_city  # noqa: E402
from airbnb.inference import city_defaults  # noqa: E402
from airbnb.models import MODEL_NAMES, make_pipeline  # noqa: E402

TARGET = "log_price"


def fit_score(pipe, train, tests: dict) -> tuple:
    t0 = time.time()
    pipe.fit(train, train[TARGET])
    fit_s = time.time() - t0
    rows = []
    train_pred = pipe.predict(train)
    train_r2 = regression_metrics(train[TARGET], train_pred)["R2_log"]
    for name, test in tests.items():
        m = regression_metrics(test[TARGET], pipe.predict(test))
        rows.append({"test_city": name, "train_R2_log": train_r2, **m, "fit_seconds": fit_s})
    return pipe, rows


def main(quick=False):
    df = load_processed()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_DIR.mkdir(exist_ok=True)
    splits = {c: split_city(df, c) for c in C.CITIES}
    train = {c: s[0] for c, s in splits.items()}
    test = {c: s[1] for c, s in splits.items()}

    summary = {"n_listings": df.groupby("city").size().to_dict(),
               "n_train": {c: len(v) for c, v in train.items()},
               "n_test": {c: len(v) for c, v in test.items()}}
    plots.price_distribution(df)

    # ---- E1: model comparison per city ---------------------------------------------
    models = ["Ridge (linear)", "XGBoost"] if quick else MODEL_NAMES
    e1, best_preds = [], {}
    for city in C.CITIES:
        for name in models:
            pipe, rows = fit_score(make_pipeline(name), train[city], {city: test[city]})
            for r in rows:
                e1.append({"experiment": "E1", "train_city": city, "model": name, **r})
            print(f"E1 {city:7s} {name:18s} test R2(log)={rows[0]['R2_log']:.3f}  "
                  f"MAE=${rows[0]['MAE_usd']:.0f}  ({rows[0]['fit_seconds']:.0f}s)")
            if name == "XGBoost":
                best_preds[city] = (test[city][TARGET].values, pipe.predict(test[city]))
                if city == "nyc":
                    plots.feature_importance(pipe, "nyc")
    e1 = pd.DataFrame(e1)
    e1.to_csv(C.RESULTS_DIR / "e1_model_comparison.csv", index=False)
    plots.model_comparison(e1)
    plots.pred_vs_actual(best_preds)

    # ---- E2: feature ablation (XGBoost) --------------------------------------------
    e2 = []
    for city in ["nyc", "paris"]:
        for groups in [("C",), ("O",), ("T",), ("C", "O"), ("C", "O", "T")]:
            label = "+".join(groups)
            _, rows = fit_score(make_pipeline("XGBoost", groups=groups), train[city], {city: test[city]})
            e2.append({"experiment": "E2", "train_city": city, "features": label, **rows[0]})
            print(f"E2 {city:7s} {label:7s} test R2(log)={rows[0]['R2_log']:.3f}")
    e2 = pd.DataFrame(e2)
    e2.to_csv(C.RESULTS_DIR / "e2_feature_ablation.csv", index=False)
    plots.ablation(e2)

    # ---- E3 + E4: combined training and transfer to Berlin ---------------------------
    # Raw lat/lon are city-specific, so all E3/E4 models use distance-to-centre instead
    # and no city indicator (Berlin would be an unseen category anyway).
    combos = {"nyc": ["nyc"], "paris": ["paris"], "nyc+paris": ["nyc", "paris"], "berlin": ["berlin"]}
    e34 = []
    for tag, cities in combos.items():
        tr = pd.concat([train[c] for c in cities])
        _, rows = fit_score(make_pipeline("XGBoost", geo=False), tr,
                            {"nyc": test["nyc"], "paris": test["paris"], "berlin": test["berlin"]})
        for r in rows:
            e34.append({"experiment": "E3/E4", "train_on": tag, **r})
            print(f"E3/E4 train={tag:10s} test={r['test_city']:7s} R2(log)={r['R2_log']:.3f}  "
                  f"MAE=${r['MAE_usd']:.0f}")
    e34 = pd.DataFrame(e34)
    e34.to_csv(C.RESULTS_DIR / "e3_e4_cross_city.csv", index=False)
    plots.transfer(e34)

    # ---- Final demo model: XGBoost on all cities, all features + city indicator -------
    all_train = pd.concat(train.values())
    final, rows = fit_score(make_pipeline("XGBoost", city=True), all_train, test)
    final_df = pd.DataFrame(rows)
    final_df.to_csv(C.RESULTS_DIR / "final_model_test.csv", index=False)
    print("Final model (all cities):\n", final_df[["test_city", "R2_log", "R2_usd", "MAE_usd", "MedAPE_%"]].round(3))
    bundle = {"pipeline": final, "defaults": city_defaults(df), "eur_to_usd": C.EUR_TO_USD,
              "test_metrics": final_df.to_dict("records")}
    joblib.dump(bundle, C.MODEL_DIR / "xgb_all_cities.joblib", compress=3)

    summary["final_model"] = final_df.set_index("test_city")[["R2_log", "MAE_usd", "MedAPE_%"]].round(3).to_dict("index")
    with open(C.RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("Done. Results in results/, model in models/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
