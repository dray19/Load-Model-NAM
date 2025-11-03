from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import save_columns


def compute_error(df: pd.DataFrame, pred_col: str) -> Tuple[float, float, float, float, float]:
    """
    Compute error metrics comparing the 'bias' column to predictions in pred_col.

    Returns:
        mae, rmse, pct_mae, pct_rmse, mean_error (mean(actual - predicted))
    """
    if "bias" not in df.columns:
        raise KeyError("Input dataframe must include a 'bias' column")

    y_true = df["bias"].values
    y_pred = df[pred_col].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # avoid division by zero
    mean_bias = np.mean(y_true)
    if mean_bias == 0:
        pct_mae = float("nan")
        pct_rmse = float("nan")
    else:
        pct_mae = mae / mean_bias * 100
        pct_rmse = rmse / mean_bias * 100

    mean_error = float(np.mean(y_true - y_pred))

    return mae, rmse, pct_mae, pct_rmse, mean_error


def train_xgb_model(
    anal: pd.DataFrame,
    feat_list: Iterable[str],
    feat_name: str,
    out_path: str,
    model_num: int,
    ty: str,
    params: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train an XGBoost regression model on provided training DataFrame.

    - anal: dataframe with at least 'init_date' (datetime-like) and 'bias' columns.
    - feat_list: list of columns to keep from anal for training (must include 'bias').
    - feat_name: descriptive name for the feature set used (for reporting).
    - out_path: directory where model and auxiliary files will be saved.
    - model_num: integer identifier used to build filenames.
    - ty: string tag used in filenames.
    - num_boost_round: number of boosting rounds for xgboost.train.
    - params: xgboost parameters dict. Defaults provided when None.

    Returns:
        (res_train_df, summary_df)
        - res_train_df: original anal with a new column 'XGB' containing predictions.
        - summary_df: single-row DataFrame with metrics for the trained model.
    """
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # validate presence of required columns
    if "init_date" not in anal.columns:
        raise KeyError("DataFrame 'anal' must contain 'init_date' column (datetime-like).")
    if "bias" not in anal.columns:
        raise KeyError("DataFrame 'anal' must contain 'bias' column.")

    # ensure init_date is datetime
    anal = anal.copy()
    anal["init_date"] = pd.to_datetime(anal["init_date"])

    # compute month_distance and simple weighting scheme
    T = anal["init_date"].max()
    anal["month_distance"] = (T.year - anal["init_date"].dt.year) * 12 + (T.month - anal["init_date"].dt.month)
    anal["weights"] = (anal["month_distance"].max() - anal["month_distance"]) / anal["month_distance"].max()
    w_train = anal["weights"].values

    # save training dataframe for traceability
    train_csv_path = out_dir / f"ML_train_data_{ty}.csv"
    anal.to_csv(train_csv_path, index=False)

    # prepare features
    cols = list(feat_list)
    if "bias" not in cols:
        cols.append("bias")

    train_data = anal.loc[:, cols].copy()

    # categorical encoding - keep behavior of original code (drop_first=True)
    # guard against missing columns in case user didn't include them in feat_list
    if "EST day name" in train_data.columns:
        train_data = pd.get_dummies(
            train_data, prefix="EST_day_name", columns=["EST day name"], drop_first=True
        )
    if "LZ" in train_data.columns:
        train_data = pd.get_dummies(train_data, prefix="LZ", columns=["LZ"], drop_first=True)

    X = train_data.drop(columns=["bias"])
    y = train_data["bias"].values

    # persist column order so downstream code can apply same ordering when predicting
    save_columns(out_dir.as_posix(), f"Model{model_num}_{ty}_col_order", list(X.columns))

    # default xgboost parameters (can be overridden via params argument)
    if params is None:
        params = {"objective": "reg:squarederror", "eval_metric": "mae", 'max_depth': 5, 'eta': 0.1,"subsample": 0.8, 
        "colsample_bytree": 0.7}

    # prepare DMatrix
    dtrain = xgb.DMatrix(X, label=y, weight=w_train)
    num_boost_round = 520
    # train model
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    # predict on training data
    y_pred_train = model.predict(dtrain)

    # attach predictions to original DataFrame (use a copy of anal to avoid side-effects)
    res_train = anal.copy()
    res_train["XGB"] = y_pred_train

    # compute metrics
    mae, rmse, pct_mae, pct_rmse, mean_error = compute_error(res_train, "XGB")

    summary = pd.DataFrame(
        [
            {
                "Type": feat_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAE %": pct_mae,
                "RMSE %": pct_rmse,
                "Bias (Actual - Predicted)": mean_error,
            }
        ]
    )

    # save model to disk
    model_file = out_dir / f"model{model_num}_{ty}.ubj"
    model.save_model(model_file.as_posix())

    # optional sanity-check (load & compare first prediction)
    try:
        loaded = xgb.Booster()
        loaded.load_model(model_file.as_posix())
        # compare first prediction value to ensure saved model loads properly
        same_first_pred = np.isclose(loaded.predict(dtrain)[0], model.predict(dtrain)[0])
        if not same_first_pred:
            # best-effort warning; do not raise
            print("Warning: first prediction differs after saving/loading the model.")
    except Exception as exc:  # pragma: no cover - non-critical
        print(f"Warning: unable to sanity-check loaded model: {exc}")

    return res_train, summary

