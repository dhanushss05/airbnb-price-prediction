"""Model zoo. Every model predicts log(price); see evaluate.py for the metrics."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from . import config as C
from .features import build_preprocessor

# Tuned once on the NYC training split with 3-fold CV, 10 random candidates
# (scripts/tune_xgb.py, best CV R2 on log price = 0.848; see results/xgb_tuning.json).
XGB_PARAMS = dict(n_estimators=1170, learning_rate=0.0225, max_depth=8, min_child_weight=4,
                  subsample=0.88, colsample_bytree=0.48, reg_lambda=1.0)


def make_regressor(name: str):
    if name == "Ridge (linear)":
        return Ridge(alpha=10.0)
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.33,
                                     n_jobs=-1, random_state=C.RANDOM_STATE)
    if name == "XGBoost":
        return XGBRegressor(**XGB_PARAMS, tree_method="hist", n_jobs=-1,
                            random_state=C.RANDOM_STATE)
    if name == "Neural net (MLP)":
        # Two small hidden layers + early stopping, as the reference report found
        # bigger networks overfit.
        return MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                            batch_size=128, max_iter=300, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=15,
                            random_state=C.RANDOM_STATE)
    raise ValueError(name)


MODEL_NAMES = ["Ridge (linear)", "KNN", "Random Forest", "XGBoost", "Neural net (MLP)"]


def make_pipeline(model_name: str, groups=("C", "O", "T"), geo=True, city=False) -> Pipeline:
    return Pipeline([("features", build_preprocessor(groups, geo=geo, city=city)),
                     ("model", make_regressor(model_name))])
