"""Reference models (Linear Regression, Random Forest) and the default-XGBoost baseline.

Why XGBoost is the main model:
* the data is tabular with mixed categorical / numeric features, where gradient-boosted
  trees are the strongest standard choice;
* it captures non-linear interactions (station x hour x user history) and handles missing
  values natively;
* it has many interacting continuous and integer hyperparameters whose best values depend
  on the data, which makes it an ideal target for a continuous metaheuristic such as SMA.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor


def xgb_fixed_params(seed: int) -> dict:
    """Settings shared by every XGBoost model in the project (never tuned)."""
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
        "enable_categorical": True,
    }


def make_linear_regression(meta: dict) -> Pipeline:
    """One-hot categoricals + StandardScaler on numerics (both fitted on train only)."""
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), meta["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              categories=[meta["categories"][c] for c in meta["categorical"]]),
         meta["categorical"]),
    ])
    return Pipeline([("pre", pre), ("lr", LinearRegression())])


def make_random_forest(meta: dict, cfg: dict, seed: int) -> Pipeline:
    """Default-parameter Random Forest; categoricals as integer codes."""
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), meta["numeric"]),
        ("cat", OrdinalEncoder(categories=[meta["categories"][c] for c in meta["categorical"]],
                               handle_unknown="use_encoded_value", unknown_value=-1),
         meta["categorical"]),
    ])
    rf = RandomForestRegressor(**cfg["baselines"]["random_forest"], n_jobs=-1, random_state=seed)
    return Pipeline([("pre", pre), ("rf", rf)])


def make_default_xgb(seed: int) -> XGBRegressor:
    """XGBoost with library-default hyperparameters: the baseline without optimisation."""
    return XGBRegressor(**xgb_fixed_params(seed))


def fit_xgb(params: dict, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
            seed: int, early_stopping_rounds: int | None, track_train: bool = False) -> XGBRegressor:
    """Train an XGBoost model with the given hyperparameters.

    Early stopping monitors the *last* eval set, which is always the validation set.
    ``track_train=True`` also records train RMSE per round (for learning curves).
    """
    model = XGBRegressor(**xgb_fixed_params(seed), **params,
                         early_stopping_rounds=early_stopping_rounds, eval_metric="rmse")
    eval_set = [(X_train, y_train), (X_val, y_val)] if track_train else [(X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model
