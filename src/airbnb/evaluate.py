"""Metrics. Models are trained on y = log(price); we report both log-scale and dollar-scale errors."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from . import config as C


def regression_metrics(y_log_true, y_log_pred) -> dict:
    p_true, p_pred = np.exp(y_log_true), np.exp(y_log_pred)
    return {
        "R2_log": r2_score(y_log_true, y_log_pred),
        "MSE_log": mean_squared_error(y_log_true, y_log_pred),
        "R2_usd": r2_score(p_true, p_pred),
        "MAE_usd": mean_absolute_error(p_true, p_pred),
        "MedAPE_%": float(np.median(np.abs(p_pred - p_true) / p_true) * 100),
    }


def split_city(df, city, test_size=0.2):
    """Fixed 80/20 split per city so every experiment is scored on the same test listings."""
    d = df[df[C.CITY_COLUMN] == city]
    return train_test_split(d, test_size=test_size, random_state=C.RANDOM_STATE)
